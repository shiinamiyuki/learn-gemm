project(learn-gemm)
cmake_minimum_required(VERSION 3.26)

enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 23)
find_package(CUDAToolkit REQUIRED)
add_executable(learn-gemm main.cu)
set_target_properties(learn-gemm PROPERTIES CUDA_ARCHITECTURES 89)
target_link_libraries(learn-gemm CUDA::cudart CUDA::cublas)