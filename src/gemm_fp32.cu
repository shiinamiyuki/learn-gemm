#include "matrix.h"
#include <cuda/barrier>
#include <cooperative_groups/memcpy_async.h>
__global__ void naive_gemm_kernel(const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    if (row < M && col < N) {
        for (uint32_t k = 0; k < K; ++k) {
            // value += A[row * K + k] * B[k * N + col];
            value = fmaf(A[row * K + k], B[k * N + col], value);
        }
        C[row * N + col] = value;
    }
}
__device__ float4 load_float4(const float *ptr) {
    float x, y, z, w;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "l"(ptr));
    return float4{x, y, z, w};
}
__device__ void store_float4(float *ptr, const float4 &value) {
    asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

__device__ float4 load_float4_shared(uint32_t shared_addr) {
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(shared_addr));
    return float4{x, y, z, w};
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t BLOCK_SIZE_X, uint32_t BLOCK_SIZE_Y, class IndexFn>
__device__ void load_matrix_view_to_shared_vectorized(MatrixView<float> &gm_mat, MatrixView<float, IndexFn> &shared_mat, uint32_t row_offset, uint32_t col_offset) {
    // load data into shared memory
    for (uint32_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
        uint32_t row = row_offset + y;
        if (row >= gm_mat.rows) continue;
        constexpr uint32_t vector_width = 4;
        for (uint32_t x = threadIdx.x * vector_width; x < TILE_N; x += BLOCK_SIZE_X * vector_width) {
            // vectorized load
            uint32_t col = col_offset + x;
            if (col >= gm_mat.cols) continue;
            float4 f4 = load_float4(&gm_mat(row, col));
            shared_mat(y, x + 0) = f4.x;
            shared_mat(y, x + 1) = f4.y;
            shared_mat(y, x + 2) = f4.z;
            shared_mat(y, x + 3) = f4.w;
        }
    }
}
template<uint32_t M, uint32_t N, class IndexFn>
inline __device__ StaticMatrix<float, M, N> load_from_matrix_view_vectorized(const MatrixView<float, IndexFn> &mat, uint32_t row_offset, uint32_t col_offset) {
    StaticMatrix<float, M, N> result{};
    constexpr uint32_t vector_width = 4;
#pragma unroll
    for (uint32_t i = 0; i < M; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < N; j += vector_width) {
            // result(i, j) = mat(row_offset + i, col_offset + j);
            auto addr = reinterpret_cast<uint64_t>(&mat(row_offset + i, col_offset + j));
            float4 f4 = load_float4_shared(addr);
            result(i, j + 0) = f4.x;
            result(i, j + 1) = f4.y;
            result(i, j + 2) = f4.z;
            result(i, j + 3) = f4.w;
        }
    }
    return result;
}
void naive_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {
    uint32_t M = A.rows;
    uint32_t K = A.cols;
    uint32_t N = B.cols;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    naive_gemm_kernel<<<gridSize, blockSize, 0, stream>>>(A.data, B.data, C.data, M, N, K);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
__global__ void tiled_gemm_kernel(MatrixView<float> A, MatrixView<float> B, MatrixView<float> C, uint32_t M, uint32_t N, uint32_t K) {
    // blockDim: (TILE_N, TILE_M)
    constexpr uint32_t BLOCK_DIM_X = TILE_N;
    constexpr uint32_t BLOCK_DIM_Y = TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    MatrixView<float> A_tile(TILE_M, TILE_K, shared_A);
    MatrixView<float> B_tile(TILE_K, TILE_N, shared_B);

    uint32_t row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    uint32_t col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

    uint32_t num_tiles = (K + TILE_K - 1) / TILE_K;

    float value = 0.0f;
    for (uint32_t k_start = 0; k_start < num_tiles; ++k_start) {
        // load tiles of A and B into shared memory

        load_matrix_view_to_shared_vectorized<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(A, A_tile, blockIdx.y * BLOCK_DIM_Y, k_start * TILE_K);
        load_matrix_view_to_shared_vectorized<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(B, B_tile, k_start * TILE_K, blockIdx.x * BLOCK_DIM_X);

        __syncthreads();

// compute partial results
#pragma unroll
        for (uint32_t k = 0; k < TILE_K; ++k) {
            float a_val = A_tile(threadIdx.y, k);
            float b_val = B_tile(k, threadIdx.x);
            value += a_val * b_val;
        }
        __syncthreads();
    }
    // write back results
    if (row < M && col < N) {
        C(row, col) = value;
    }
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t REG_TILE_M, uint32_t REG_TILE_N, uint32_t REG_TILE_K>
__global__ void tiled_reg_gemm_kernel(MatrixView<float> A, MatrixView<float> B, MatrixView<float> C, uint32_t M, uint32_t N, uint32_t K) {
    // blockDim: (TILE_N / REG_TILE_N, TILE_M / REG_TILE_M)
    static_assert(TILE_M % REG_TILE_M == 0, "TILE_M must be divisible by REG_TILE_M");
    static_assert(TILE_N % REG_TILE_N == 0, "TILE_N must be divisible by REG_TILE_N");
    static_assert(TILE_K % REG_TILE_K == 0, "TILE_K must be divisible by REG_TILE_K");
    constexpr uint32_t BLOCK_DIM_X = TILE_N / REG_TILE_N;
    constexpr uint32_t BLOCK_DIM_Y = TILE_M / REG_TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    MatrixView A_tile(TILE_M, TILE_K, shared_A);
    MatrixView B_tile(TILE_K, TILE_N, shared_B);

    uint32_t row = blockIdx.y * TILE_M + threadIdx.y * REG_TILE_M;
    uint32_t col = blockIdx.x * TILE_N + threadIdx.x * REG_TILE_N;

    uint32_t num_tiles = (K + TILE_K - 1) / TILE_K;

    using RegMatrixA = StaticMatrix<float, REG_TILE_M, REG_TILE_K>;
    using RegMatrixB = StaticMatrix<float, REG_TILE_K, REG_TILE_N>;
    using RegMatrixC = StaticMatrix<float, REG_TILE_M, REG_TILE_N>;
    RegMatrixC reg_C{};
    for (uint32_t k_start = 0; k_start < num_tiles; ++k_start) {
        // load tiles of A and B into shared memory
        load_matrix_view_to_shared_vectorized<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(A, A_tile, blockIdx.y * TILE_M, k_start * TILE_K);
        load_matrix_view_to_shared_vectorized<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(B, B_tile, k_start * TILE_K, blockIdx.x * TILE_N);

        __syncthreads();

// compute partial results
#pragma unroll
        for (uint32_t k = 0; k < TILE_K / REG_TILE_K; k++) {
            RegMatrixA reg_A = RegMatrixA::load_from_matrix_view(A_tile, threadIdx.y * REG_TILE_M, k * REG_TILE_K);
            RegMatrixB reg_B = RegMatrixB::load_from_matrix_view(B_tile, k * REG_TILE_K, threadIdx.x * REG_TILE_N);
            RegMatrixC::mma<true>(reg_A, reg_B, reg_C);
        }
        __syncthreads();
    }
// write back results
#pragma unroll
    for (uint32_t i = 0; i < REG_TILE_M; ++i) {
        uint32_t global_row = row + i;
        if (global_row < M) {
#pragma unroll
            for (uint32_t j = 0; j < REG_TILE_N; ++j) {
                uint32_t global_col = col + j;
                if (global_col < N) {
                    C(global_row, global_col) = reg_C(i, j);
                }
            }
        }
    }
}

void tiled_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_K = 32;
    dim3 blockSize(TILE_N, TILE_M);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    tiled_gemm_kernel<TILE_M, TILE_N, TILE_K><<<gridSize, blockSize, 0, stream>>>(MatrixView(A), MatrixView(B), MatrixView(C), M, N, K);

    CHECK_CUDA(cudaGetLastError());
}

void tiled_reg_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    constexpr size_t TILE_M = 128;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 32;
    constexpr size_t REG_TILE_M = 4;
    constexpr size_t REG_TILE_N = 4;
    constexpr size_t REG_TILE_K = 4;
    if (M % TILE_M != 0 || N % TILE_N != 0 || K % TILE_K != 0) {
        throw std::runtime_error("tiled_gemm_fp32: M, N, K must be multiples of TILE_M, TILE_N, TILE_K respectively");
    }
    if (M % REG_TILE_M != 0 || N % REG_TILE_N != 0 || K % REG_TILE_K != 0) {
        throw std::runtime_error("tiled_gemm_fp32: M, N, K must be multiples of REG_TILE_M, REG_TILE_N, REG_TILE_K respectively");
    }
    dim3 blockSize(TILE_N / REG_TILE_N, TILE_M / REG_TILE_M);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    tiled_reg_gemm_kernel<TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, REG_TILE_K>
        <<<gridSize, blockSize, 0, stream>>>(MatrixView(A), MatrixView(B), MatrixView(C), M, N, K);

    CHECK_CUDA(cudaGetLastError());
}



__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
template<uint32_t group_id>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(group_id));
}
__device__ constexpr uint32_t cdiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}
/// shared_mat.data must be in shared address space
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t BLOCK_SIZE_X, uint32_t BLOCK_SIZE_Y, class IndexFn>
__device__ void load_matrix_view_to_shared_vectorized_async(MatrixView<float> &gm_mat, MatrixView<float, IndexFn> &shared_mat, uint32_t row_offset, uint32_t col_offset) {
    // static_assert(TILE_M % BLOCK_SIZE_Y == 0, "TILE_M must be divisible by BLOCK_SIZE_Y");
    // static_assert(TILE_N % BLOCK_SIZE_X == 0, "TILE_N must be divisible by BLOCK_SIZE_X");
    constexpr bool is_row_major = IndexFn::is_row_major;
    constexpr uint32_t vector_width = 4;
    if constexpr (is_row_major) {
        // for (uint32_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
        //     uint32_t row = row_offset + y;
        //     if (row >= gm_mat.rows) continue;

        //     for (uint32_t x = threadIdx.x * vector_width; x < TILE_N; x += BLOCK_SIZE_X * vector_width) {
        //         // vectorized load
        //         uint32_t col = col_offset + x;
        //         if (col >= gm_mat.cols) continue;
        //         auto global_ptr = &gm_mat(row, col);
        //         auto shared_ptr = reinterpret_cast<uint64_t>(&shared_mat(y, x));
        //         cp_async_global_to_shared<sizeof(float) * vector_width>(shared_ptr, global_ptr);
        //     }
        // }
        // rewrite the make loop bounds statically known
        constexpr uint32_t ITER_Y = cdiv(TILE_M, BLOCK_SIZE_Y);
        constexpr uint32_t ITER_X = cdiv(TILE_N, BLOCK_SIZE_X * vector_width);

        for (uint32_t iy = 0; iy < ITER_Y; ++iy) {
            uint32_t y = threadIdx.y + iy * BLOCK_SIZE_Y;
            uint32_t row = row_offset + y;
            if (row >= gm_mat.rows) continue;
            for (uint32_t ix = 0; ix < ITER_X; ++ix) {
                uint32_t x = threadIdx.x * vector_width + ix * BLOCK_SIZE_X * vector_width;
                uint32_t col = col_offset + x;
                if (col >= gm_mat.cols) continue;
                if (x + vector_width <= TILE_N) {
                    auto global_ptr = &gm_mat(row, col);
                    auto shared_ptr = reinterpret_cast<uint64_t>(&shared_mat(y, x));
                    cp_async_global_to_shared<sizeof(float) * vector_width>(shared_ptr, global_ptr);
                } else {
// handle the tail case
#pragma unroll
                    for (uint32_t v = 0; v < vector_width; ++v) {
                        if (x + v < TILE_N) {
                            auto global_ptr = &gm_mat(row, col + v);
                            auto shared_ptr = reinterpret_cast<uint64_t>(&shared_mat(y, x + v));
                            cp_async_global_to_shared<sizeof(float)>(shared_ptr, global_ptr);
                        }
                    }
                }
            }
        }
    } else {
        for (uint32_t x = threadIdx.x; x < TILE_N; x += BLOCK_SIZE_X) {
            uint32_t col = col_offset + x;
            if (col >= gm_mat.cols) continue;

            for (uint32_t y = threadIdx.y * vector_width; y < TILE_M; y += BLOCK_SIZE_Y * vector_width) {
                // vectorized load
                uint32_t row = row_offset + y;
                if (row >= gm_mat.rows) continue;
                auto global_ptr = &gm_mat(row, col);
                auto shared_ptr = reinterpret_cast<uint64_t>(&shared_mat(y, x));
                cp_async_global_to_shared<sizeof(float) * vector_width>(shared_ptr, global_ptr);
            }
        }
    }
}

template<uint32_t Rows, uint32_t Cols>
struct SwizzledIndexFn {
    static constexpr bool is_row_major = true;
    __device__ __host__ SwizzledIndexFn(uint32_t, uint32_t) {}
    __device__ __host__ size_t operator()(uint32_t r, uint32_t c) const {
        return Swizzle2D<float, Rows, Cols, 4>::apply(r, c);
    }
};
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t REG_TILE_M, uint32_t REG_TILE_N, uint32_t REG_TILE_K>
__global__ void tiled_cp_async_gemm_kernel(MatrixView<float> A, MatrixView<float> B, MatrixView<float> C, uint32_t M, uint32_t N, uint32_t K) {
    // blockDim: (TILE_N / REG_TILE_N, TILE_M / REG_TILE_M)
    static_assert(TILE_M % REG_TILE_M == 0, "TILE_M must be divisible by REG_TILE_M");
    static_assert(TILE_N % REG_TILE_N == 0, "TILE_N must be divisible by REG_TILE_N");
    static_assert(TILE_K % REG_TILE_K == 0, "TILE_K must be divisible by REG_TILE_K");
    constexpr uint32_t BLOCK_DIM_X = TILE_N / REG_TILE_N;
    constexpr uint32_t BLOCK_DIM_Y = TILE_M / REG_TILE_M;

    __shared__ float shared_A_buf[2][TILE_M * TILE_K];
    __shared__ float shared_B_buf[2][TILE_K * TILE_N];

    uint32_t num_tiles = K / TILE_K;

    using RegMatrixA = StaticMatrix<float, REG_TILE_M, REG_TILE_K>;
    using RegMatrixB = StaticMatrix<float, REG_TILE_K, REG_TILE_N>;
    using RegMatrixC = StaticMatrix<float, REG_TILE_M, REG_TILE_N>;

    // using MatrixViewA = MatrixView<float, SwizzledIndexFn<TILE_M, TILE_K>>;
    // using MatrixViewB = MatrixView<float, SwizzledIndexFn<TILE_K, TILE_N>>;

    using MatrixViewA = MatrixView<float, DefaultIndexFn<true>>;
    using MatrixViewB = MatrixView<float, DefaultIndexFn<true>>;

    RegMatrixC reg_C{};

    // load the first tile of A and B
    auto prefetch_tile = [&](uint32_t k) {
        uint32_t shared_A_ptrs[2] = {
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_A_buf[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_A_buf[1])),
        };
        uint32_t shared_B_ptrs[2] = {
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_B_buf[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_B_buf[1])),
        };

        auto buffer_index = k & 1;
        MatrixViewA A_tile(TILE_M, TILE_K, reinterpret_cast<float *>(shared_A_ptrs[buffer_index]));
        MatrixViewB B_tile(TILE_K, TILE_N, reinterpret_cast<float *>(shared_B_ptrs[buffer_index]));
        load_matrix_view_to_shared_vectorized_async<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(A, A_tile, blockIdx.y * TILE_M, k * TILE_K);
        load_matrix_view_to_shared_vectorized_async<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(B, B_tile, k * TILE_K, blockIdx.x * TILE_N);
        cp_async_commit_group();
    };

    prefetch_tile(0);

    for (uint32_t k_start = 0; k_start < num_tiles; ++k_start) {
        cp_async_wait_group<0>();
        __syncthreads();
        if (k_start + 1 < num_tiles) {
            prefetch_tile(k_start + 1);
        }
        uint32_t shared_A_ptrs[2] = {
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_A_buf[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_A_buf[1])),
        };
        uint32_t shared_B_ptrs[2] = {
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_B_buf[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_B_buf[1])),
        };
        auto buffer_index = k_start & 1;
        MatrixViewA A_tile(TILE_M, TILE_K, reinterpret_cast<float *>(shared_A_ptrs[buffer_index]));
        MatrixViewB B_tile(TILE_K, TILE_N, reinterpret_cast<float *>(shared_B_ptrs[buffer_index]));
        // MatrixViewA A_tile(TILE_M, TILE_K, &shared_A_buf[buffer_index][0]);
        // MatrixViewB B_tile(TILE_K, TILE_N, &shared_B_buf[buffer_index][0]);
// compute partial results
#pragma unroll
        for (uint32_t k = 0; k < TILE_K / REG_TILE_K; k++) {
            RegMatrixA reg_A = load_from_matrix_view_vectorized<REG_TILE_M, REG_TILE_K>(A_tile, threadIdx.y * REG_TILE_M, k * REG_TILE_K);
            RegMatrixB reg_B = load_from_matrix_view_vectorized<REG_TILE_K, REG_TILE_N>(B_tile, k * REG_TILE_K, threadIdx.x * REG_TILE_N);
            RegMatrixC::mma<true>(reg_A, reg_B, reg_C);
        }
    }
    // write back results
    uint32_t row = blockIdx.y * TILE_M + threadIdx.y * REG_TILE_M;
    uint32_t col = blockIdx.x * TILE_N + threadIdx.x * REG_TILE_N;
#pragma unroll
    for (uint32_t i = 0; i < REG_TILE_M; ++i) {
        uint32_t global_row = row + i;
        if (global_row < M) {
#pragma unroll
            for (uint32_t j = 0; j < REG_TILE_N; j += 4) {
                uint32_t global_col = col + j;
                if (global_col < N) {
                    float4 f4;
                    f4.x = reg_C(i, j + 0);
                    f4.y = reg_C(i, j + 1);
                    f4.z = reg_C(i, j + 2);
                    f4.w = reg_C(i, j + 3);
                    auto C_ptr = &C(global_row, global_col);
                    store_float4(C_ptr, f4);
                }
            }
        }
    }
}

void tile_cp_async_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    constexpr size_t TILE_M = 64;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 32;
    constexpr size_t REG_TILE_M = 4;
    constexpr size_t REG_TILE_N = 4;
    constexpr size_t REG_TILE_K = 4;
    if (M % TILE_M != 0 || N % TILE_N != 0 || K % TILE_K != 0) {
        throw std::runtime_error("tile_cp_async_gemm_fp32: M, N, K must be multiples of TILE_M, TILE_N, TILE_K respectively");
    }
    if (M % REG_TILE_M != 0 || N % REG_TILE_N != 0 || K % REG_TILE_K != 0) {
        throw std::runtime_error("tile_cp_async_gemm_fp32: M, N, K must be multiples of REG_TILE_M, REG_TILE_N, REG_TILE_K respectively");
    }
    dim3 blockSize(TILE_N / REG_TILE_N, TILE_M / REG_TILE_M);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    tiled_cp_async_gemm_kernel<TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, REG_TILE_K>
        <<<gridSize, blockSize, 0, stream>>>(MatrixView(A), MatrixView(B), MatrixView(C), M, N, K);

    CHECK_CUDA(cudaGetLastError());
}