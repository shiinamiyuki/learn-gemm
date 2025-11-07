#include "matrix.h"
struct Half8 {
    __half2 r0, r1, r2, r3;
};

__device__ Half8 load_half8_global(const Half *ptr) {
    uint32_t b0, b1, b2, b3;
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                 : "l"(ptr));
    return Half8{
        *reinterpret_cast<__half2 *>(&b0),
        *reinterpret_cast<__half2 *>(&b1),
        *reinterpret_cast<__half2 *>(&b2),
        *reinterpret_cast<__half2 *>(&b3),
    };
}
__device__ void store_half8_shared(const Half8 &h8, Half *ptr) {
    uint32_t b0 = *reinterpret_cast<const uint32_t *>(&h8.r0);
    uint32_t b1 = *reinterpret_cast<const uint32_t *>(&h8.r1);
    uint32_t b2 = *reinterpret_cast<const uint32_t *>(&h8.r2);
    uint32_t b3 = *reinterpret_cast<const uint32_t *>(&h8.r3);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(b0), "r"(b1), "r"(b2), "r"(b3));
}

__device__ void load_matrix_m8n8_x2_b16(const Half *p, __half2 &a0a1, __half2 &a2a3) {
    uint32_t r0_u32, r1_u32;
    uint64_t p_val = reinterpret_cast<uint64_t>(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(r0_u32), "=r"(r1_u32)
                 : "l"(p_val));
    a0a1 = *reinterpret_cast<__half2 *>(&r0_u32);
    a2a3 = *reinterpret_cast<__half2 *>(&r1_u32);
}
__device__ void load_matrix_m8n8_b16(const Half *p, __half2 &a0a1) {
    uint32_t r0_u32;
    uint64_t p_val = reinterpret_cast<uint64_t>(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.b16 {%0}, [%1];"
                 : "=r"(r0_u32)
                 : "l"(p_val));
    a0a1 = *reinterpret_cast<__half2 *>(&r0_u32);
}
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t BLOCK_SIZE_X, uint32_t BLOCK_SIZE_Y, class IndexFn>
__device__ void load_matrix_view_to_shared_vectorized(MatrixView<Half> &gm_mat, MatrixView<Half, IndexFn> &shared_mat, uint32_t row_offset, uint32_t col_offset) {
    // load data into shared memory
    for (uint32_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
        uint32_t row = row_offset + y;
        if (row >= gm_mat.rows) continue;
        constexpr uint32_t vector_width = 8;
        for (uint32_t x = threadIdx.x * vector_width; x < TILE_N; x += BLOCK_SIZE_X * vector_width) {
            // vectorized load
            auto x_hi = x + vector_width;
            uint32_t col = col_offset + x;
            uint32_t col_hi = col_offset + x_hi;
            if (col_hi <= gm_mat.cols) {
                Half8 h8 = load_half8_global(&gm_mat(row, col));
                store_half8_shared(h8, &shared_mat(y, x));
            } else {
#pragma unroll
                for (uint32_t xi = 0; xi < vector_width; ++xi) {
                    uint32_t c = col + xi;
                    shared_mat(y, x + xi) = (c < gm_mat.cols) ? gm_mat(row, c) : Half{0.0f};
                }
            }
        }
    }
}

__global__ void naive_gemm_kernel(MatrixView<Half> A, MatrixView<Half> B, MatrixView<Half> C) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C.rows && col < C.cols) {
        float sum{};
        for (uint32_t k = 0; k < A.cols; ++k) {
            float a = A(row, k);
            float b = B(k, col);
            sum += a * b;
        }
        C(row, col) = sum;
    }
}

void naive_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
    uint32_t M = A.rows;
    uint32_t N = B.cols;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    naive_gemm_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C);
}

constexpr uint32_t MMA_M = 16;
constexpr uint32_t MMA_N = 8;
constexpr uint32_t MMA_K = 8;
