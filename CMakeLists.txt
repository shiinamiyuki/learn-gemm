cmake_minimum_required(VERSION 3.31)
project(learn-gemm)

enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 23)
find_package(CUDAToolkit REQUIRED)
add_executable(learn-gemm src/main.cu src/matrix.h src/gemm_fp32.cu src/gemm_fp16.cu src/gemm_fp16_gemini.cu src/gemm_fp16_grok.cu)
set_target_properties(learn-gemm PROPERTIES CUDA_ARCHITECTURES 89)
target_link_libraries(learn-gemm CUDA::cudart CUDA::cublas)
# add_custom_command(TARGET learn-gemm POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/ptx
#     COMMAND ${CMAKE_CUDA_COMPILER} -std=c++20 --ptx ${CMAKE_CURRENT_SOURCE_DIR}/main.cu -o ${CMAKE_CURRENT_BINARY_DIR}/ptx/learn-gemm.ptx -arch=sm_89
#     COMMENT "Generating PTX for learn-gemm"
# )