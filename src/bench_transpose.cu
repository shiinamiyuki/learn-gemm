#include "matrix.h"
#include <iostream>

void check_results(const Matrix<float> &A, const Matrix<float> &B_transposed) {
    if (A.rows != B_transposed.cols || A.cols != B_transposed.rows) {
        throw std::runtime_error("check_results: dimension mismatch");
    }
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            float a_val = A(i, j);
            float b_val = B_transposed(j, i);
            if (a_val != b_val) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << "A = " << a_val << ", B_transposed = " << b_val << "\n";
                throw std::runtime_error("check_results: values do not match");
            }
        }
    }
    std::cout << "Results match!\n";
}

int main(int argc, char **argv) {
    uint32_t M{};
    uint32_t N{};
    if (argc != 3 ||
        (M = static_cast<uint32_t>(std::stoul(argv[1]))) == 0 ||
        (N = static_cast<uint32_t>(std::stoul(argv[2]))) == 0) {
        std::cerr << "Usage: " << argv[0] << " <M> <N>\n";
        std::cerr << "  where <M> and <N> are positive integers representing the matrix dimensions.\n";
        return EXIT_FAILURE;
    }
    Matrix<float> test_host_A(M, N, MatrixStorageType::Host);
    test_host_A.init_random();
    Matrix<float> test_device_A = test_host_A.to(MatrixStorageType::Device);
    Matrix<float> test_device_B(N, M, MatrixStorageType::Device);
    cudaStream_t stream;

    CHECK_CUDA(cudaStreamCreate(&stream));

    std::cout << "checking naive_transpose_fp32...\n";
    naive_transpose_fp32(stream, test_device_A, test_device_B);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    Matrix<float> test_host_B = test_device_B.to(MatrixStorageType::Host);
    check_results(test_host_A, test_host_B);
    
    cudaMemset(test_device_B.data, 0, sizeof(float) * N * M);
    
    std::cout << "checking tiled_transpose_fp32_gemini...\n";
    tiled_transpose_fp32_gemini(stream, test_device_A, test_device_B);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    test_host_B = test_device_B.to(MatrixStorageType::Host);
    check_results(test_host_A, test_host_B);

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
        printf("%40s: %10.5f ms\n", name.data(), avg_ms);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    };

    auto warm_up = 10;
    auto repeats = 10;
    bench("naive_transpose_fp32", [&](cudaStream_t stream) {
        naive_transpose_fp32(stream, test_device_A, test_device_B);
    }, stream, warm_up, repeats);
    bench("tiled_transpose_fp32_gemini", [&](cudaStream_t stream) {
        tiled_transpose_fp32_gemini(stream, test_device_A, test_device_B);
    }, stream, warm_up, repeats);
}