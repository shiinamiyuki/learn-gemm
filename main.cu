#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <string_view>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <random>
#include <cublas_v2.h>
#define CHECK_CUDA(call) [&]() { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } }()
#define CHECK_CUBLAS(call) [&]() { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS Error: %d\n", status); \
        exit(EXIT_FAILURE); \
    } }()
enum class MatrixStorageType {
    Device,
    Host,
};
inline float relative_error(float a, float b) {
    return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-8f);
}
struct Matrix {
    int rows;
    int cols;
    float *data;
    MatrixStorageType storage_type;
    Matrix(int r, int c, MatrixStorageType st = MatrixStorageType::Device) : rows(r), cols(c), storage_type(st) {
        allocate();
    }
    Matrix(const Matrix &other) {
        free();
        rows = other.rows;
        cols = other.cols;
        storage_type = other.storage_type;
        allocate();
        copy_from(other);
    }
    Matrix &operator=(const Matrix &other) {
        if (this != &other) {
            free();
            rows = other.rows;
            cols = other.cols;
            storage_type = other.storage_type;
            allocate();
            copy_from(other);
        }
        return *this;
    }
    Matrix(Matrix &&other) noexcept : rows(other.rows), cols(other.cols), data(other.data), storage_type(other.storage_type) {
        other.data = nullptr;
    }
    ~Matrix() {
        free();
    }
    void copy_from(const Matrix &other) {
        if (other.rows != rows || other.cols != cols || other.storage_type != storage_type) {
            throw std::runtime_error("Matrix copy_from: dimension or storage type mismatch");
        }
        size_t size = rows * cols * sizeof(float);
        if (storage_type == MatrixStorageType::Host) {
            std::memcpy(data, other.data, size);
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice));
        }
    }
    void init_random() {
        std::random_device rd;
        if (storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("init_random only supports Host storage");
        }
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = dis(gen);
        }
    }
    void init_ones() {
        if (storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("init_ones only supports Host storage");
        }
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = 1.0f;
        }
    }
    Matrix to(MatrixStorageType st) const {
        Matrix result(rows, cols, st);
        size_t size = rows * cols * sizeof(float);
        if (storage_type == MatrixStorageType::Device && st == MatrixStorageType::Host) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(result.data, data, size, cudaMemcpyDeviceToHost));
        } else if (storage_type == MatrixStorageType::Host && st == MatrixStorageType::Device) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(result.data, data, size, cudaMemcpyHostToDevice));
        } else if (storage_type == st) {
            if (st == MatrixStorageType::Host) {
                std::memcpy(result.data, data, size);
            } else {
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(result.data, data, size, cudaMemcpyDeviceToDevice));
            }
        }
        return result;
    }
    bool allclose(const Matrix &other, float tol) const {
        if (other.storage_type != MatrixStorageType::Host || storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("allclose requires both matrices to be in Host storage");
        }
        if (rows != other.rows || cols != other.cols) {
            return false;
        }
        for (int i = 0; i < rows * cols; ++i) {
            if (relative_error(data[i], other.data[i]) > tol) {
                return false;
            }
        }
        return true;
    }
    __device__ __host__ float &operator()(int r, int c) {
        return data[r * cols + c];
    }
    __device__ __host__ const float &operator()(int r, int c) const {
        return data[r * cols + c];
    }

private:
    void allocate() {
        size_t size = rows * cols * sizeof(float);
        if (storage_type == MatrixStorageType::Device) {
            CHECK_CUDA(cudaMalloc(&data, size));
        } else {
            data = (float *)malloc(size);
        }
    }
    void free() {
        if (storage_type == MatrixStorageType::Device) {
            CHECK_CUDA(cudaFree(data));
        } else {
            std::free(data);
        }
    }
};
struct MatrixView {
    int rows;
    int cols;
    float *data;
    __device__ __host__ MatrixView() : rows(0), cols(0), data(nullptr) {}
    __device__ __host__ MatrixView(int r, int c) : rows(r), cols(c), data(nullptr) {}
    __device__ __host__ MatrixView(const Matrix &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(const MatrixView &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(int r, int c, float *d) : rows(r), cols(c), data(d) {}
    __device__ __host__ float &operator()(int r, int c) {
        return data[r * cols + c];
    }
    __device__ __host__ const float &operator()(int r, int c) const {
        return data[r * cols + c];
    }
};
template<size_t M, size_t N>
struct StaticMatrix {
    float data[M * N]{};
    __device__ __host__ float &operator()(size_t r, size_t c) {
        return data[r * N + c];
    }
    __device__ __host__ const float &operator()(size_t r, size_t c) const {
        return data[r * N + c];
    }
    template<bool ACC, size_t P>
    __device__ __host__ static void mma(const StaticMatrix<M, P> &A, const StaticMatrix<P, N> &B, StaticMatrix<M, N> &C) {
#pragma unroll
        for (size_t i = 0; i < M; ++i) {
#pragma unroll
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
#pragma unroll
                for (size_t k = 0; k < P; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                if constexpr (ACC) {
                    C(i, j) += sum;
                } else {
                    C(i, j) = sum;
                }
            }
        }
    }
    static __device__ StaticMatrix<M, N> load_from_matrix_view(const MatrixView &mat, size_t row_offset, size_t col_offset) {
        StaticMatrix<M, N> result{};
#pragma unroll
        for (size_t i = 0; i < M; ++i) {
#pragma unroll
            for (size_t j = 0; j < N; ++j) {
                result(i, j) = mat(row_offset + i, col_offset + j);
            }
        }
        return result;
    }
    static __device__ void store_to_matrix_view(const StaticMatrix<M, N> &smat, MatrixView &mat, size_t row_offset, size_t col_offset) {
#pragma unroll
        for (size_t i = 0; i < M; ++i) {
#pragma unroll
            for (size_t j = 0; j < N; ++j) {
                mat(row_offset + i, col_offset + j) = smat(i, j);
            }
        }
    }
};
__global__ void naive_gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void naive_gemm(cudaStream_t stream, const Matrix &A, const Matrix &B, Matrix &C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    naive_gemm_kernel<<<gridSize, blockSize, 0, stream>>>(A.data, B.data, C.data, M, N, K);
}
__device__ float4 load_float4(const float *ptr) {
    float x, y, z, w;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "l"(ptr));
    return float4{x, y, z, w};
}

template<size_t TILE_M, size_t TILE_N, size_t BLOCK_SIZE_X, size_t BLOCK_SIZE_Y, class AddrFn, class MaskFn>
__device__ void load_matrix_view_to_shared(MatrixView &shared_mat, AddrFn addr_fn, MaskFn mask_fn) {
// load data into shared memory
#pragma unroll
    for (size_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
#pragma unroll
        for (size_t x = threadIdx.x; x < TILE_N; x += BLOCK_SIZE_X) {
            if (mask_fn(y, x)) {
                shared_mat(y, x) = addr_fn(y, x);
            } else {
                shared_mat(y, x) = 0.0f;
            }
        }
    }
}
template<size_t TILE_M, size_t TILE_N, size_t BLOCK_SIZE_X, size_t BLOCK_SIZE_Y, class AddrFn, class MaskFn>
__device__ void load_matrix_view_to_shared_vectorized(MatrixView &shared_mat, AddrFn addr_fn, MaskFn mask_fn) {
// load data into shared memory
#pragma unroll
    for (size_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
        if (!mask_fn(y, 0)) {
            continue;
        }
#pragma unroll
        for (size_t x = threadIdx.x * 4; x < TILE_N; x += BLOCK_SIZE_X * 4) {
            // vectorized load
            auto x_hi = x + 4;
            if (mask_fn(y, x_hi - 1)) {
                float4 vec = load_float4(addr_fn(y, x));
                shared_mat(y, x + 0) = vec.x;
                shared_mat(y, x + 1) = vec.y;
                shared_mat(y, x + 2) = vec.z;
                shared_mat(y, x + 3) = vec.w;
            } else {
#pragma unroll
                for (size_t xi = 0; xi < 4; ++xi) {
                    size_t x_curr = x + xi;
                    shared_mat(y, x_curr) = mask_fn(y, x_curr) ? *addr_fn(y, x_curr) : 0.0f;
                }
            }
        }
    }
}
template<size_t TILE_M, size_t TILE_N, size_t TILE_K>
__global__ void tiled_gemm_kernel(MatrixView A, MatrixView B, MatrixView C, size_t M, size_t N, size_t K) {
    // blockDim: (TILE_N, TILE_M)
    constexpr size_t BLOCK_DIM_X = TILE_N;
    constexpr size_t BLOCK_DIM_Y = TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    MatrixView A_tile(TILE_M, TILE_K, shared_A);
    MatrixView B_tile(TILE_K, TILE_N, shared_B);

    size_t row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

    size_t num_tiles = (K + TILE_K - 1) / TILE_K;

    float value = 0.0f;
    for (size_t k_start = 0; k_start < num_tiles; ++k_start) {
        // load tiles of A and B into shared memory

        load_matrix_view_to_shared<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(
            A_tile,
            [&](size_t y, size_t x) {
                size_t a_row = blockIdx.y * BLOCK_DIM_Y + y;
                size_t a_col = k_start * TILE_K + x;
                return A(a_row, a_col);
            },
            [&](size_t y, size_t x) {
                size_t a_row = blockIdx.y * BLOCK_DIM_Y + y;
                size_t a_col = k_start * TILE_K + x;
                return (a_row < M) && (a_col < K);
            });
        load_matrix_view_to_shared<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(
            B_tile,
            [&](size_t y, size_t x) {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * BLOCK_DIM_X + x;
                return B(b_row, b_col);
            },
            [&](size_t y, size_t x) {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * BLOCK_DIM_X + x;
                return (b_row < K) && (b_col < N);
            });

        __syncthreads();

// compute partial results
#pragma unroll
        for (size_t k = 0; k < TILE_K; ++k) {
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

void tiled_gemm(cudaStream_t stream, const Matrix &A, const Matrix &B, Matrix &C) {
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

template<size_t TILE_M, size_t TILE_N, size_t TILE_K, size_t REG_TILE_M, size_t REG_TILE_N, size_t REG_TILE_K>
__global__ void tiled_reg_gemm_kernel(MatrixView A, MatrixView B, MatrixView C, size_t M, size_t N, size_t K) {
    // blockDim: (TILE_N / REG_TILE_N, TILE_M / REG_TILE_M)
    static_assert(TILE_M % REG_TILE_M == 0, "TILE_M must be divisible by REG_TILE_M");
    static_assert(TILE_N % REG_TILE_N == 0, "TILE_N must be divisible by REG_TILE_N");
    static_assert(TILE_K % REG_TILE_K == 0, "TILE_K must be divisible by REG_TILE_K");
    constexpr size_t BLOCK_DIM_X = TILE_N / REG_TILE_N;
    constexpr size_t BLOCK_DIM_Y = TILE_M / REG_TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    MatrixView A_tile(TILE_M, TILE_K, shared_A);
    MatrixView B_tile(TILE_K, TILE_N, shared_B);

    size_t row = blockIdx.y * TILE_M + threadIdx.y * REG_TILE_M;
    size_t col = blockIdx.x * TILE_N + threadIdx.x * REG_TILE_N;

    size_t num_tiles = (K + TILE_K - 1) / TILE_K;

    using RegMatrixA = StaticMatrix<REG_TILE_M, REG_TILE_K>;
    using RegMatrixB = StaticMatrix<REG_TILE_K, REG_TILE_N>;
    using RegMatrixC = StaticMatrix<REG_TILE_M, REG_TILE_N>;
    RegMatrixC reg_C{};
    for (size_t k_start = 0; k_start < num_tiles; ++k_start) {
        // load tiles of A and B into shared memory

        load_matrix_view_to_shared<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(
            A_tile,
            [&](size_t y, size_t x) {
                size_t a_row = blockIdx.y * TILE_M + y;
                size_t a_col = k_start * TILE_K + x;
                return A(a_row, a_col);
            },
            [&](size_t y, size_t x) {
                size_t a_row = blockIdx.y * TILE_M + y;
                size_t a_col = k_start * TILE_K + x;
                return (a_row < M) && (a_col < K);
            });
        load_matrix_view_to_shared<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(
            B_tile,
            [&](size_t y, size_t x) {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * TILE_N + x;
                return B(b_row, b_col);
            },
            [&](size_t y, size_t x) {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * TILE_N + x;
                return (b_row < K) && (b_col < N);
            });

        __syncthreads();

// compute partial results
#pragma unroll
        for (size_t k = 0; k < TILE_K / REG_TILE_K; k++) {
            RegMatrixA reg_A = RegMatrixA::load_from_matrix_view(A_tile, threadIdx.y * REG_TILE_M, k * REG_TILE_K);
            RegMatrixB reg_B = RegMatrixB::load_from_matrix_view(B_tile, k * REG_TILE_K, threadIdx.x * REG_TILE_N);
            RegMatrixC::mma<true>(reg_A, reg_B, reg_C);
        }
        __syncthreads();
    }
// write back results
#pragma unroll
    for (size_t i = 0; i < REG_TILE_M; ++i) {
        size_t global_row = row + i;
        if (global_row < M) {
#pragma unroll
            for (size_t j = 0; j < REG_TILE_N; ++j) {
                size_t global_col = col + j;
                if (global_col < N) {
                    C(global_row, global_col) = reg_C(i, j);
                }
            }
        }
    }
}
void tiled_reg_gemm(cudaStream_t stream, const Matrix &A, const Matrix &B, Matrix &C) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    constexpr size_t TILE_M = 64;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 32;
    constexpr size_t REG_TILE_M = 4;
    constexpr size_t REG_TILE_N = 4;
    constexpr size_t REG_TILE_K = 4;
    dim3 blockSize(TILE_N / REG_TILE_N, TILE_M / REG_TILE_M);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    tiled_reg_gemm_kernel<TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, REG_TILE_K>
        <<<gridSize, blockSize, 0, stream>>>(MatrixView(A), MatrixView(B), MatrixView(C), M, N, K);

    CHECK_CUDA(cudaGetLastError());
}

template<size_t TILE_M, size_t TILE_N, size_t TILE_K, size_t REG_TILE_M, size_t REG_TILE_N, size_t REG_TILE_K>
__global__ void tiled_reg_vectorized_gemm_kernel(MatrixView A, MatrixView B, MatrixView C, size_t M, size_t N, size_t K) {
    // blockDim: (TILE_N / REG_TILE_N, TILE_M / REG_TILE_M)
    static_assert(TILE_M % REG_TILE_M == 0, "TILE_M must be divisible by REG_TILE_M");
    static_assert(TILE_N % REG_TILE_N == 0, "TILE_N must be divisible by REG_TILE_N");
    static_assert(TILE_K % REG_TILE_K == 0, "TILE_K must be divisible by REG_TILE_K");
    constexpr size_t BLOCK_DIM_X = TILE_N / REG_TILE_N;
    constexpr size_t BLOCK_DIM_Y = TILE_M / REG_TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    MatrixView A_tile(TILE_M, TILE_K, shared_A);
    MatrixView B_tile(TILE_K, TILE_N, shared_B);

    size_t row = blockIdx.y * TILE_M + threadIdx.y * REG_TILE_M;
    size_t col = blockIdx.x * TILE_N + threadIdx.x * REG_TILE_N;

    size_t num_tiles = (K + TILE_K - 1) / TILE_K;

    using RegMatrixA = StaticMatrix<REG_TILE_M, REG_TILE_K>;
    using RegMatrixB = StaticMatrix<REG_TILE_K, REG_TILE_N>;
    using RegMatrixC = StaticMatrix<REG_TILE_M, REG_TILE_N>;
    RegMatrixC reg_C{};
    for (size_t k_start = 0; k_start < num_tiles; ++k_start) {
        // load tiles of A and B into shared memory

        load_matrix_view_to_shared_vectorized<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(
            A_tile,
            [&](size_t y, size_t x) -> float * {
                size_t a_row = blockIdx.y * TILE_M + y;
                size_t a_col = k_start * TILE_K + x;
                return &A(a_row, a_col);
            },
            [&](size_t y, size_t x) {
                size_t a_row = blockIdx.y * TILE_M + y;
                size_t a_col = k_start * TILE_K + x;
                return (a_row < M) && (a_col < K);
            });
        load_matrix_view_to_shared_vectorized<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(
            B_tile,
            [&](size_t y, size_t x) -> float * {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * TILE_N + x;
                return &B(b_row, b_col);
            },
            [&](size_t y, size_t x) {
                size_t b_row = k_start * TILE_K + y;
                size_t b_col = blockIdx.x * TILE_N + x;
                return (b_row < K) && (b_col < N);
            });

        __syncthreads();

// compute partial results
#pragma unroll
        for (size_t k = 0; k < TILE_K / REG_TILE_K; k++) {
            RegMatrixA reg_A = RegMatrixA::load_from_matrix_view(A_tile, threadIdx.y * REG_TILE_M, k * REG_TILE_K);
            RegMatrixB reg_B = RegMatrixB::load_from_matrix_view(B_tile, k * REG_TILE_K, threadIdx.x * REG_TILE_N);
            RegMatrixC::mma<true>(reg_A, reg_B, reg_C);
        }
        __syncthreads();
    }
// write back results
#pragma unroll
    for (size_t i = 0; i < REG_TILE_M; ++i) {
        size_t global_row = row + i;
        if (global_row < M) {
#pragma unroll
            for (size_t j = 0; j < REG_TILE_N; ++j) {
                size_t global_col = col + j;
                if (global_col < N) {
                    C(global_row, global_col) = reg_C(i, j);
                }
            }
        }
    }
}

void tiled_reg_vectorized_gemm(cudaStream_t stream, const Matrix &A, const Matrix &B, Matrix &C) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    constexpr size_t TILE_M = 64;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 32;
    constexpr size_t REG_TILE_M = 4;
    constexpr size_t REG_TILE_N = 4;
    constexpr size_t REG_TILE_K = 4;
    dim3 blockSize(TILE_N / REG_TILE_N, TILE_M / REG_TILE_M);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    tiled_reg_vectorized_gemm_kernel<TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, REG_TILE_K>
        <<<gridSize, blockSize, 0, stream>>>(MatrixView(A), MatrixView(B), MatrixView(C), M, N, K);

    CHECK_CUDA(cudaGetLastError());
}

void cublas_gemm(cublasHandle_t handle, const Matrix &A, const Matrix &B, Matrix &C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, N,
                A.data, K,
                &beta,
                C.data, N);
}
void print_cuda_info() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Using CUDA Device %d: %s\n", device, prop.name);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
}
void check_result(const Matrix &ref, const Matrix &test) {
    auto tol = 1e-4f;
    if (!ref.allclose(test, tol)) {
        printf("Result mismatch!\n");
        auto cnt = 0;
        for (int i = 0; i < ref.rows; ++i) {
            for (int j = 0; j < ref.cols; ++j) {
                float v1 = ref(i, j);
                float v2 = test(i, j);
                if (relative_error(v1, v2) > tol) {
                    printf("C[%d, %d]: ref=%f, test=%f\n", i, j, v1, v2);
                    cnt++;
                    if (cnt >= 10) {
                        printf("More than 10 mismatches, aborting print.\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        exit(EXIT_FAILURE);
    }
}
int main(int argc, char **argv) {
    size_t M{0};
    size_t N{0};
    size_t K{0};

    // read M, N, K from command line arguments
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return 1;
    }
    size_t total_flops = 2ull * M * N * K;
    printf("Matrix dimensions: M=%zu, N=%zu, K=%zu, Total FLOPs=%zu\n", M, N, K, total_flops);
    print_cuda_info();
    Matrix host_A(M, K, MatrixStorageType::Host);
    Matrix host_B(K, N, MatrixStorageType::Host);
    Matrix ref_C(M, N), test_C(M, N);
    host_A.init_ones();
    host_B.init_random();
    auto A = host_A.to(MatrixStorageType::Device);
    auto B = host_B.to(MatrixStorageType::Device);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    naive_gemm(stream, A, B, ref_C);
    tiled_gemm(stream, A, B, test_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto host_ref_C = ref_C.to(MatrixStorageType::Host);
    auto host_test_C = test_C.to(MatrixStorageType::Host);
    printf("Checking tiled_gemm result...\n");
    check_result(host_ref_C, host_test_C);
    tiled_reg_gemm(stream, A, B, test_C);
    host_test_C = test_C.to(MatrixStorageType::Host);
    printf("Checking tiled_reg_gemm result...\n");
    check_result(host_ref_C, host_test_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    tiled_reg_vectorized_gemm(stream, A, B, test_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    host_test_C = test_C.to(MatrixStorageType::Host);
    printf("Checking tiled_reg_vectorized_gemm result...\n");
    check_result(host_ref_C, host_test_C);

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    cublas_gemm(cublas_handle, A, B, test_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    host_test_C = test_C.to(MatrixStorageType::Host);
    printf("Checking cublas_gemm result...\n");
    check_result(host_ref_C, host_test_C);

    auto bench = [&]<typename F>(const std::string_view name, F &&f, cudaStream_t stream, size_t warm_up, size_t repeats) {
        // warm up
        for (size_t i = 0; i < warm_up; ++i) {
            f(stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        // benchmark
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (size_t i = 0; i < repeats; ++i) {
            f(stream);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        auto avg_ms = milliseconds / repeats;
        double flops_per_s = static_cast<double>(total_flops) / (avg_ms / 1e3);
        printf("%30s: %10.5f ms GFLOPs/s: %10.3f\n", name.data(), avg_ms, flops_per_s / 1e9);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    };
    auto warm_up = 10;
    auto repeats = 10;
    bench("naive_gemm", [&](cudaStream_t s) { naive_gemm(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tiled_gemm", [&](cudaStream_t s) { tiled_gemm(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tiled_reg_gemm", [&](cudaStream_t s) { tiled_reg_gemm(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tiled_reg_vectorized_gemm", [&](cudaStream_t s) { tiled_reg_vectorized_gemm(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("cublas_gemm", [&](cudaStream_t s) { cublas_gemm(cublas_handle, A, B, test_C); }, stream, warm_up, repeats);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}