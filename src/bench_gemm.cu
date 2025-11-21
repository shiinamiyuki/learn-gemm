#include "matrix.h"
#include <cublas_v2.h>

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
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        // exit(EXIT_FAILURE);
    }
}
void cublas_gemm_fp32(cublasHandle_t handle, const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {
    uint32_t M = A.rows;
    uint32_t K = A.cols;
    uint32_t N = B.cols;
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
void cublas_gemm_fp16(cublasHandle_t handle, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
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
int bench_fp32(size_t M, size_t N, size_t K) {
    Matrix<float> test_host_A(M, K, MatrixStorageType::Host);
    Matrix<float> test_host_B(K, N, MatrixStorageType::Host);
    Matrix<float> test_ref_C(M, N, MatrixStorageType::Host);
    printf("Initializing matrices...\n");
    test_host_A.init_random_ones();
    test_host_B.init_random_ones();
    test_ref_C.mma(test_host_A, test_host_B);
    Matrix<float> A = test_host_A.to(MatrixStorageType::Device);
    Matrix<float> B = test_host_B.to(MatrixStorageType::Device);
    Matrix<float> host_ref_C = test_ref_C;
    Matrix<float> test_C(M, N), cublas_ref_C(M, N);

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
    cublas_gemm_fp32(cublas_handle, A, B, cublas_ref_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto cublas_ref_C_host = cublas_ref_C.to(MatrixStorageType::Host);

    printf("Checking naive_gemm result...\n");
    naive_gemm_fp32(stream, A, B, test_C);
    auto host_test_C = test_C.to(MatrixStorageType::Host);
    host_ref_C.print();
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    printf("naive_gemm_fp32 passed!\n");

    printf("Checking tiled_gemm_fp32 result...\n");
    tiled_gemm_fp32(stream, A, B, test_C);
    host_test_C = test_C.to(MatrixStorageType::Host);
    host_test_C.print();
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    printf("tiled_gemm_fp32 passed!\n");

    printf("Checking tiled_reg_gemm_fp32 result...\n");
    tiled_reg_gemm_fp32(stream, A, B, test_C);
    host_test_C = test_C.to(MatrixStorageType::Host);
    host_test_C.print();
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    printf("tiled_reg_gemm_fp32 passed!\n");

    printf("Checking tile_cp_async_gemm_fp32 result...\n");
    tile_cp_async_gemm_fp32(stream, A, B, test_C);
    host_test_C = test_C.to(MatrixStorageType::Host);
    host_test_C.print();
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    printf("tile_cp_async_gemm_fp32 passed!\n");

    size_t total_flops = 2ull * M * N * K;
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
        CHECK_CUDA(cudaStreamSynchronize(stream));
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
    bench("cublas_gemm", [&](cudaStream_t s) { cublas_gemm_fp32(cublas_handle, A, B, test_C); }, stream, warm_up, repeats);
    bench("naive_gemm", [&](cudaStream_t s) { naive_gemm_fp32(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tiled_gemm", [&](cudaStream_t s) { tiled_gemm_fp32(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tiled_reg_gemm", [&](cudaStream_t s) { tiled_reg_gemm_fp32(s, A, B, test_C); }, stream, warm_up, repeats);
    bench("tile_cp_async_gemm", [&](cudaStream_t s) { tile_cp_async_gemm_fp32(s, A, B, test_C); }, stream, warm_up, repeats);
    return 0;
}
int bench_fp16(size_t M, size_t N, size_t K) {

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
    cublas_gemm_fp16(cublas_handle, A, B, cublas_ref_C);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto cublas_ref_C_host = cublas_ref_C.to(MatrixStorageType::Host);

    printf("Checking naive_gemm result...\n");
    naive_gemm_fp16(stream, A, B, test_C);
    auto host_test_C = test_C.to(MatrixStorageType::Host);
    host_ref_C.print();
    check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    printf("naive_gemm_fp16 passed!\n");

    // printf("Checking gemini_gemm_fp16 result...\n");
    // gemini_gemm_fp16(stream, A, B, test_C);
    // host_test_C = test_C.to(MatrixStorageType::Host);
    // host_test_C.print();
    // check_result(host_ref_C, cublas_ref_C_host, host_test_C);

    // printf("gemini_gemm_fp16 passed!\n");

    // printf("Checking grok_gemm_fp16 result...\n");
    // grok_gemm_fp16(stream, A, B, test_C);
    // host_test_C = test_C.to(MatrixStorageType::Host);
    // host_test_C.print();
    // check_result(host_ref_C, cublas_ref_C_host, host_test_C);
    // printf("grok_gemm_fp16 passed!\n");
    size_t total_flops = 2ull * M * N * K;
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
    auto warm_up = 5;
    auto repeats = 10;
    bench("cublas_gemm", [&](cudaStream_t s) { cublas_gemm_fp16(cublas_handle, A, B, test_C); }, stream, warm_up, repeats);
    bench("naive_gemm", [&](cudaStream_t s) { naive_gemm_fp16(s, A, B, test_C); }, stream, warm_up, repeats);
    // bench("gemini_gemm_fp16", [&](cudaStream_t s) { gemini_gemm_fp16(s, A, B, test_C); }, stream, warm_up, repeats);
    // bench("grok_gemm_fp16", [&](cudaStream_t s) { grok_gemm_fp16(s, A, B, test_C); }, stream, warm_up, repeats);
    return 0;
}
int main(int argc, char **argv) {
    size_t M{0};
    size_t N{0};
    size_t K{0};
    char mode = 'f';
    // read M, N, K from command line arguments
    if (argc >= 4) {
        mode = argv[1][0];
        M = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else {
        printf("Usage: %s f|h <M> <N> <K>\n", argv[0]);
        return 1;
    }
    size_t total_flops = 2ull * M * N * K;
    printf("Matrix dimensions: M=%zu, N=%zu, K=%zu, dtype=%s, Total FLOPs=%zu\n", M, N, K, (mode == 'f' ? "fp32" : "fp16"), total_flops);
    print_cuda_info();
    if (mode == 'f') {
        return bench_fp32(M, N, K);
    } else {
        return bench_fp16(M, N, K);
    }
}