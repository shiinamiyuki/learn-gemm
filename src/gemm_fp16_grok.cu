#include <cuda_runtime.h>
#include <cstdint>
#include <assert.h>
#include "matrix.h"

struct FragmentA {
    unsigned int regs[2];
};

struct FragmentB {
    unsigned int regs[1];
};

struct FragmentC {
    float regs[4];
};

struct SharedStorage {
    Half a[2][64 * 32];
    Half b[2][32 * 64];
};

__device__ int sh_a_index(int row, int col) {
    int m_group = row / 16;
    int local_row = row % 16;
    int k_group = col / 8;
    int local_col = col % 8;
    return (m_group * (32 / 8) + k_group) * 128 + local_row * 8 + local_col;
}

__device__ int sh_b_index(int row, int col) {
    int k_group = row / 8;
    int local_k = row % 8;
    int n_group = col / 8;
    int local_n = col % 8;
    return (k_group * (64 / 8) + n_group) * 64 + local_k * 8 + local_n;
}

__device__ Half *sh_a_ptr(SharedStorage &shared, int buf, int row, int col) {
    return &shared.a[buf][sh_a_index(row, col)];
}

__device__ Half *sh_b_ptr(SharedStorage &shared, int buf, int row, int col) {
    return &shared.b[buf][sh_b_index(row, col)];
}

__device__ void load_a_to_shared(SharedStorage &shared, int buf, const Half *A, uint32_t M, uint32_t K, int k_base) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int num_loads = 64 * 32 / 8;
    if (tid < num_loads) {
        int load_id = tid;
        int col_group = load_id % (32 / 8);
        int row = load_id / (32 / 8);
        int col = col_group * 8;
        const Half *g_ptr = A + (blockIdx.y * 64 + row) * K + (k_base + col);
        unsigned int s_ptr = __cvta_generic_to_shared(sh_a_ptr(shared, buf, row, col));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(s_ptr), "l"(g_ptr));
    }
}

__device__ void load_b_to_shared(SharedStorage &shared, int buf, const Half *B, uint32_t N, uint32_t K, int k_base) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int num_loads = 32 * 64 / 8;
    if (tid < num_loads) {
        int load_id = tid;
        int col_group = load_id % (64 / 8);
        int row = load_id / (64 / 8);
        int col = col_group * 8;
        const Half *g_ptr = B + (k_base + row) * N + (blockIdx.x * 64 + col);
        unsigned int s_ptr = __cvta_generic_to_shared(sh_b_ptr(shared, buf, row, col));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(s_ptr), "l"(g_ptr));
    }
}

__device__ void store_c(FragmentC &frag_c, Half *C, uint32_t M, uint32_t N, int m_offset, int n_offset) {
    int lane = threadIdx.x % 32;
    int row_base = lane % 4;
    int col_base = (lane / 16) * 4 + ((lane % 16) / 4);
    for (int i = 0; i < 4; ++i) {
        int row = row_base + i * 4;
        int col = col_base;
        Half val = __float2half(frag_c.regs[i]);
        int g_row = m_offset + row;
        int g_col = n_offset + col;
        if (g_row < M && g_col < N) {
            C[g_row * N + g_col] = val;
        }
    }
}

__global__ void gemm_grok_fp16_kernel(const Half *A, const Half *B, Half *C, uint32_t M, uint32_t N, uint32_t K) {
    if (M == 0 || N == 0) return;
    extern __shared__ Half shmem[];
    SharedStorage &shared = *reinterpret_cast<SharedStorage *>(shmem);

    int warp_id = (threadIdx.y * (blockDim.x / 32)) + (threadIdx.x / 32);
    int warp_m = warp_id / 8;
    int warp_n = warp_id % 8;
    int lane = threadIdx.x % 32;

    FragmentC accum;
    accum.regs[0] = 0.f;
    accum.regs[1] = 0.f;
    accum.regs[2] = 0.f;
    accum.regs[3] = 0.f;

    int buf = 0;
    load_a_to_shared(shared, buf, A, M, K, 0);
    load_b_to_shared(shared, buf, B, N, K, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int k_base = 0;
    for (int k_tile = 0; k_tile < K / 32; ++k_tile) {
        buf = k_tile % 2;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for (int kk = 0; kk < 32; kk += 8) {
            FragmentA frag_a;
            unsigned int smem_a = __cvta_generic_to_shared(sh_a_ptr(shared, buf, warp_m * 16, kk));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(frag_a.regs[0]), "=r"(frag_a.regs[1])
                         : "r"(smem_a));

            FragmentB frag_b;
            unsigned int smem_b = __cvta_generic_to_shared(sh_b_ptr(shared, buf, kk, warp_n * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                         : "=r"(frag_b.regs[0])
                         : "r"(smem_b));

            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                         : "+f"(accum.regs[0]), "+f"(accum.regs[1]), "+f"(accum.regs[2]), "+f"(accum.regs[3])
                         : "r"(frag_a.regs[0]), "r"(frag_a.regs[1]), "r"(frag_b.regs[0]));
        }

        k_base += 32;
        if (k_tile < (K / 32) - 1) {
            load_a_to_shared(shared, 1 - buf, A, M, K, k_base);
            load_b_to_shared(shared, 1 - buf, B, N, K, k_base);
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }

    int m_offset = blockIdx.y * 64 + warp_m * 16;
    int n_offset = blockIdx.x * 64 + warp_n * 8;
    store_c(accum, C, M, N, m_offset, n_offset);
}

static void launch_gemm(cudaStream_t stream, const Half *A, const Half *B, Half *C, uint32_t M, uint32_t N, uint32_t K) {
    assert(M % 64 == 0);
    assert(N % 64 == 0);
    assert(K % 8 == 0);
    dim3 block(256, 4);
    dim3 grid(N / 64, M / 64);
    size_t shared_size = 2 * (64 * 32 + 32 * 64) * sizeof(Half);
    gemm_grok_fp16_kernel<<<grid, block, shared_size, stream>>>(A, B, C, M, N, K);
}
void grok_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
    uint32_t M = static_cast<uint32_t>(A.rows);
    uint32_t N = static_cast<uint32_t>(B.cols);
    uint32_t K = static_cast<uint32_t>(A.cols);
    cudaMemsetAsync(C.data, 0, sizeof(Half) * M * N, stream);
    launch_gemm(stream, A.data, B.data, C.data, M, N, K);
}