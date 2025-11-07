#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <string_view>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <random>
#include <thread>
#include <cublas_v2.h>
using Half = __half;
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
inline float relative_error(Half a, Half b) {
    auto fa = static_cast<float>(a);
    auto fb = static_cast<float>(b);
    return std::abs(fa - fb) / (std::abs(fa) + std::abs(fb) + 1e-6f);
}
template<class F>
void parallel_for(uint32_t count, F &&f) {
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint32_t chunk_size = (count + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < n_threads; ++t) {
        uint32_t start = t * chunk_size;
        uint32_t end = std::min(start + chunk_size, count);
        if (start >= end) break;
        threads.emplace_back([=]() {
            for (uint32_t i = start; i < end; ++i) {
                f(i);
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }
}
template<class T>
struct Matrix {
    int rows;
    int cols;
    T *data;
    MatrixStorageType storage_type;
    Matrix(int r, int c, MatrixStorageType st = MatrixStorageType::Device) : rows(r), cols(c), storage_type(st) {
        allocate();
    }
    Matrix(const Matrix &other) {
        rows = other.rows;
        cols = other.cols;
        storage_type = other.storage_type;
        allocate();
        copy_from(other);
    }
    template<class U>
    Matrix(const Matrix<U> &other) {
        if (other.storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("Matrix conversion constructor only supports Host storage");
        }
        rows = other.rows;
        cols = other.cols;
        storage_type = other.storage_type;
        allocate();
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = static_cast<T>(other.data[i]);
        }
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
        size_t size = rows * cols * sizeof(T);
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
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = dis(gen);
        }
    }
    void init_random_ones() {
        std::random_device rd;
        if (storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("init_random only supports Host storage");
        }
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = dis(gen) > 0.5f ? 1.0f : -1.0f;
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
        size_t size = rows * cols * sizeof(T);
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
    __device__ __host__ T &operator()(uint32_t r, uint32_t c) {
        return data[r * cols + c];
    }
    __device__ __host__ const T &operator()(uint32_t r, uint32_t c) const {
        return data[r * cols + c];
    }
    void mma(const Matrix &A, const Matrix &B, bool accumulate = false) {
        if (A.cols != B.rows || rows != A.rows || cols != B.cols) {
            throw std::runtime_error("Matrix mma: dimension mismatch");
        }

        parallel_for(rows, [&](uint32_t r) {
            for (uint32_t c = 0; c < cols; ++c) {
                float sum = accumulate ? static_cast<float>((*this)(r, c)) : 0.0f;
                for (uint32_t k = 0; k < A.cols; ++k) {
                    sum += static_cast<float>(A(r, k)) * static_cast<float>(B(k, c));
                }
                (*this)(r, c) = static_cast<T>(sum);
            }
        });
    }
    void print() const {
        if (storage_type != MatrixStorageType::Host) {
            throw std::runtime_error("print only supports Host storage");
        }
        uint32_t max_print_cols = 8, max_print_rows = 8;
        for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(rows), max_print_rows); ++i) {
            for (uint32_t j = 0; j < std::min(static_cast<uint32_t>(cols), max_print_cols); ++j) {
                printf("%10.4f ", static_cast<float>((*this)(i, j)));
            }
            if (cols > max_print_cols) {
                printf("... %u more columns", cols - max_print_cols);
            }
            printf("\n");
            if (i == max_print_rows - 1 && rows > max_print_rows) {
                printf("...\n");
                printf("%u more rows\n", rows - max_print_rows);
            }
        }
    }
private:
    void allocate() {
        auto row_bytes = static_cast<size_t>(rows) * sizeof(T);
        if (row_bytes % 16 != 0) {
            throw std::runtime_error("Matrix allocation: rows must be aligned to 16 bytes");
        }
        size_t size = rows * cols * sizeof(T);
        if (storage_type == MatrixStorageType::Device) {
            CHECK_CUDA(cudaMalloc(&data, size));
        } else {
            data = (T *)malloc(size);
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
struct DefaultIndexFn {
    uint32_t rows, cols;
    __device__ __host__ DefaultIndexFn(uint32_t r, uint32_t c) : rows(r), cols(c) {}
    __device__ __host__ size_t operator()(uint32_t r, uint32_t c) const {
        return r * cols + c;
    }
};
template<class T, class IndexFn = DefaultIndexFn>
struct MatrixView {
    uint32_t rows;
    uint32_t cols;
    T *data;
    __device__ __host__ MatrixView() : rows(0), cols(0), data(nullptr) {}
    __device__ __host__ MatrixView(uint32_t r, uint32_t c) : rows(r), cols(c), data(nullptr) {}
    __device__ __host__ MatrixView(const Matrix<T> &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(const MatrixView &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(uint32_t r, uint32_t c, T *d) : rows(r), cols(c), data(d) {}
    __device__ __host__ T &operator()(uint32_t r, uint32_t c) {
        return data[IndexFn(rows, cols)(r, c)];
    }
    __device__ __host__ const T &operator()(uint32_t r, uint32_t c) const {
        return data[IndexFn(rows, cols)(r, c)];
    }
};
template<class T, uint32_t M, uint32_t N>
struct StaticMatrix {
    T data[M * N]{};
    __device__ __host__ T &operator()(uint32_t r, uint32_t c) {
        return data[r * N + c];
    }
    __device__ __host__ const T &operator()(uint32_t r, uint32_t c) const {
        return data[r * N + c];
    }
    template<bool ACC, uint32_t P>
    __device__ __host__ static void mma(const StaticMatrix<T, M, P> &A, const StaticMatrix<T, P, N> &B, StaticMatrix<T, M, N> &C) {
#pragma unroll
        for (uint32_t i = 0; i < M; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < N; ++j) {
                T sum = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < P; ++k) {
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
    template<class IndexFn>
    static __device__ StaticMatrix<T, M, N> load_from_matrix_view(const MatrixView<IndexFn> &mat, uint32_t row_offset, uint32_t col_offset) {
        StaticMatrix<T, M, N> result{};
        // #pragma unroll
        for (uint32_t i = 0; i < M; ++i) {
            // #pragma unroll
            for (uint32_t j = 0; j < N; ++j) {
                result(i, j) = mat(row_offset + i, col_offset + j);
            }
        }
        return result;
    }
    //     static __device__ void store_to_matrix_view(const StaticMatrix<M, N> &smat, MatrixView &mat, uint32_t row_offset, uint32_t col_offset) {
    // #pragma unroll
    //         for (uint32_t i = 0; i < M; ++i) {
    // #pragma unroll
    //             for (uint32_t j = 0; j < N; ++j) {
    //                 mat(row_offset + i, col_offset + j) = smat(i, j);
    //             }
    //         }
    //     }
};

void naive_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C);
void tiled_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C);
void tiled_reg_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C);
void tile_cp_async_gemm_fp32(cudaStream_t stream, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C);
void naive_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C);
void gemini_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C);
void grok_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C);