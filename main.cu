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
template<class IndexFn = DefaultIndexFn>
struct MatrixView {
    uint32_t rows;
    uint32_t cols;
    Half *data;
    __device__ __host__ MatrixView() : rows(0), cols(0), data(nullptr) {}
    __device__ __host__ MatrixView(uint32_t r, uint32_t c) : rows(r), cols(c), data(nullptr) {}
    __device__ __host__ MatrixView(const Matrix<Half> &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(const MatrixView &mat) : rows(mat.rows), cols(mat.cols), data(mat.data) {}
    __device__ __host__ MatrixView(uint32_t r, uint32_t c, Half *d) : rows(r), cols(c), data(d) {}
    __device__ __host__ Half &operator()(uint32_t r, uint32_t c) {
        return data[IndexFn(rows, cols)(r, c)];
    }
    __device__ __host__ const Half &operator()(uint32_t r, uint32_t c) const {
        return data[IndexFn(rows, cols)(r, c)];
    }
};
template<uint32_t M, uint32_t N>
struct StaticMatrix {
    Half data[M * N]{};
    __device__ __host__ Half &operator()(uint32_t r, uint32_t c) {
        return data[r * N + c];
    }
    __device__ __host__ const Half &operator()(uint32_t r, uint32_t c) const {
        return data[r * N + c];
    }
    template<bool ACC, uint32_t P>
    __device__ __host__ static void mma(const StaticMatrix<M, P> &A, const StaticMatrix<P, N> &B, StaticMatrix<M, N> &C) {
#pragma unroll
        for (uint32_t i = 0; i < M; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < N; ++j) {
                Half sum = 0.0f;
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
    static __device__ StaticMatrix<M, N> load_from_matrix_view(const MatrixView<IndexFn> &mat, uint32_t row_offset, uint32_t col_offset) {
        StaticMatrix<M, N> result{};
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

__global__ void naive_gemm_kernel(MatrixView<> A, MatrixView<> B, MatrixView<> C) {
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

void naive_gemm(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
    uint32_t M = A.rows;
    uint32_t N = B.cols;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    naive_gemm_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C);
}
constexpr uint32_t MMA_M = 16;
constexpr uint32_t MMA_N = 8;
constexpr uint32_t MMA_K = 8;

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
__device__ void load_matrix_view_to_shared_vectorized(MatrixView<> &gm_mat, MatrixView<IndexFn> &shared_mat, uint32_t row_offset, uint32_t col_offset) {
    // load data into shared memory
    for (uint32_t y = threadIdx.y; y < TILE_M; y += BLOCK_SIZE_Y) {
        uint32_t row = row_offset + y;
        if (row >= gm_mat.rows) continue;
        constexpr uint32_t vector_width = 8;
        for (uint32_t x = threadIdx.x * vector_width; x < TILE_N; x += BLOCK_SIZE_X * vector_width4) {
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

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARP_M, uint32_t WARP_N, uint32_t WARP_K>
__global__ void tiled_gemm_kernel(MatrixView<> A, MatrixView<> B, MatrixView<> C) {
    constexpr uint32_t BLOCK_DIM_X = TILE_N;
    constexpr uint32_t BLOCK_DIM_Y = TILE_M;

    __shared__ float shared_A[TILE_M * TILE_K];
    __shared__ float shared_B[TILE_K * TILE_N];
    __shared__ float shared_C[TILE_M * TILE_N];
    MatrixView A_tile(TILE_M, TILE_K, shared_A);
    MatrixView B_tile(TILE_K, TILE_N, shared_B);
    MatrixView C_tile(TILE_M, TILE_N, shared_C);

    // set C_tile to zero
#pragma unroll
    for (uint32_t i = 0; i < TILE_M; i += BLOCK_DIM_Y) {
#pragma unroll
        for (uint32_t j = 0; j < TILE_N; j += BLOCK_DIM_X) {
            if (i + threadIdx.y < TILE_M && j + threadIdx.x < TILE_N) {
                C_tile(i + threadIdx.y, j + threadIdx.x) = Half{0.0f};
            }
        }
    }
    __syncthreads();

    uint32_t row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    uint32_t col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

    uint32_t num_iters = (K + TILE_K - 1) / TILE_K;

    for (uint32_t k_start = 0; k_start < num_tiles; ++k_start) {
        load_matrix_view_to_shared_vectorized<TILE_M, TILE_K, BLOCK_DIM_X, BLOCK_DIM_Y>(A, A_tile, blockIdx.y * BLOCK_DIM_Y, k_start * TILE_K);
        load_matrix_view_to_shared_vectorized<TILE_K, TILE_N, BLOCK_DIM_X, BLOCK_DIM_Y>(B, B_tile, k_start * TILE_K, blockIdx.x * BLOCK_DIM_X);
        __syncthreads();
        for (uint32_t wm = 0; wm < TILE_M; wm += BLOCK_DIM_Y) {
            for (uint32_t wn = 0; wn < TILE_N; wn += BLOCK_DIM_X) {
            }
        }
    }
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
void check_result(const Matrix<Half> &ref, const Matrix<Half> &cublas, const Matrix<Half> &test) {
    float tol{0.05f};
    if (!ref.allclose(test, tol)) {
        printf("Result mismatch!\n");
        auto cnt = 0;
        for (int i = 0; i < ref.rows; ++i) {
            for (int j = 0; j < ref.cols; ++j) {
                Half v1 = ref(i, j);
                Half v2 = test(i, j);
                auto err_cublas = relative_error(v1, cublas(i, j));
                auto err_test = relative_error(v1, v2);
                if (err_test > tol && err_test >= 2.0 * err_cublas) {
                    printf("C[%d, %d]: ref=%f, cublas=%f test=%f\n", i, j, float(v1), float(cublas(i, j)), float(v2));
                    cnt++;
                    if (cnt >= 10) {
                        printf("More than 10 mismatches, aborting print.\n");
                        // exit(EXIT_FAILURE);
                        return;
                    }
                }
            }
        }
        // exit(EXIT_FAILURE);
    }
}

void cublas_gemm(cublasHandle_t handle, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
    uint32_t M = A.rows;
    uint32_t K = A.cols;
    uint32_t N = B.cols;
    const Half alpha = 1.0f;
    const Half beta = 0.0f;
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, N,
                A.data, K,
                &beta,
                C.data, N);
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

    Matrix<float> test_host_A(M, K, MatrixStorageType::Host);
    Matrix<float> test_host_B(K, N, MatrixStorageType::Host);
    Matrix<float> test_ref_C(M, N, MatrixStorageType::Host);
    printf("Initializing matrices...\n");
    test_host_A.init_random_ones();
    test_host_B.init_random_ones();
    test_ref_C.mma(test_host_A, test_host_B);
    Matrix<Half> A = Matrix<Half>(test_host_A).to(MatrixStorageType::Device);
    Matrix<Half> B = Matrix<Half>(test_host_B).to(MatrixStorageType::Device);
    Matrix<Half> host_ref_C = Matrix<Half>(test_ref_C);
    Matrix<Half> test_C(M, N), cublas_ref_C(M, N);

    printf("Matrix A:\n");
    test_host_A.print();
    printf("Matrix B:\n");
    test_host_B.print();
    printf("Reference Matrix C:\n");
    host_ref_C.print();
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    cublas_gemm(cublas_handle, A, B, cublas_ref_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto cublas_ref_C_host = cublas_ref_C.to(MatrixStorageType::Host);
    naive_gemm(stream, A, B, test_C);

    auto host_test_C = test_C.to(MatrixStorageType::Host);
    printf("Checking naive_gemm result...\n");
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);

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
        printf("%40s: %10.5f ms GFLOPs/s: %10.3f\n", name.data(), avg_ms, flops_per_s / 1e9);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    };
    auto warm_up = 10;
    auto repeats = 10;
    bench("cublas_gemm", [&](cudaStream_t s) { cublas_gemm(cublas_handle, A, B, test_C); }, stream, warm_up, repeats);
    bench("naive_gemm", [&](cudaStream_t s) { naive_gemm(s, A, B, test_C); }, stream, warm_up, repeats);
    return 0;
}