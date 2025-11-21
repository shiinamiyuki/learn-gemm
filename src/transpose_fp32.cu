#include "matrix.h"
__global__ void transpose_kernel(const float *A, float *B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}
void naive_transpose_fp32(cudaStream_t stream, const Matrix<float> &A, Matrix<float> &B) {
    if (A.rows != B.cols || A.cols != B.rows) {
        throw std::runtime_error("naive_transpose_fp32: dimension mismatch");
    }
    dim3 blockSize(16, 16);
    dim3 gridSize((A.cols + blockSize.x - 1) / blockSize.x, (A.rows + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<gridSize, blockSize, 0, stream>>>(A.data, B.data, A.rows, A.cols);
    CHECK_CUDA(cudaGetLastError());
}

namespace gemini {
static constexpr int TILE_DIM = 64;
static constexpr int BLOCK_ROWS = 4;

__global__ void tiled_transpose_kernel_optimized_gemini(const float *__restrict__ A,
                                                        float *__restrict__ B,
                                                        int rows, int cols) {
    // 64 * (64 + 1) = 4160 floats = 16640 bytes.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // --- READ PHASE (Coalesced from A) ---
    // x = global column index, y = global row index
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_base = blockIdx.y * TILE_DIM;

    // A is (rows x cols)
    const float *A_ptr = &A[y_base * cols + x];

// Each thread (tx, ty) loads a column from the tile.
// threadIdx.y goes from 0..3. j increments by 4.
#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // No boundary check needed.
        // A_ptr is already at the correct column (x).
        // We read rows (y_base + threadIdx.y + j).
        tile[threadIdx.y + j][threadIdx.x] = A_ptr[(threadIdx.y + j) * cols];
    }

    __syncthreads();

    // --- WRITE PHASE (Coalesced to B) ---
    // Target block in B is (blockIdx.y, blockIdx.x)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_base_B = blockIdx.x * TILE_DIM;

    // B is (cols x rows)
    float *B_ptr = &B[y_base_B * rows + x];

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // Read from shared memory transposed.
        // The padded dimension (TILE_DIM + 1) prevents bank conflicts
        // as 32 threads in a warp access tile[0..31][col_idx].
        B_ptr[(threadIdx.y + j) * rows] = tile[threadIdx.x][threadIdx.y + j];
    }
}
}// namespace gemini

void tiled_transpose_fp32_gemini(cudaStream_t stream, const Matrix<float> &A, Matrix<float> &B) {
    if (A.rows != B.cols || A.cols != B.rows) {
        throw std::runtime_error("tiled_transpose_fp32: dimension mismatch");
    }
    using namespace gemini;
    // Host-side check, as requested.
    if (A.rows % TILE_DIM != 0 || A.cols % TILE_DIM != 0) {
        throw std::runtime_error("tiled_transpose_fp32: Matrix dimensions are not multiples of TILE_DIM (64).");
    }

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);// (64, 4)
    dim3 gridSize(A.cols / TILE_DIM, A.rows / TILE_DIM);

    tiled_transpose_kernel_optimized_gemini<<<gridSize, blockSize, 0, stream>>>(A.data, B.data, A.rows, A.cols);
    CHECK_CUDA(cudaGetLastError());
}

// __global__ void tiled_transpose_kernel(const float *A, float *B, int rows, int cols) {
// }

// void tiled_transpose_fp32(cudaStream_t stream, const Matrix<float> &A, Matrix<float> &B) {
//     if (A.rows != B.cols || A.cols != B.rows) {
//         throw std::runtime_error("naive_transpose_fp32: dimension mismatch");
//     }
//     dim3 blockSize(...);
//     dim3 gridSize(...);
//     tiled_transpose_kernel<<<gridSize, blockSize, 0, stream>>>(A.data, B.data, A.rows, A.cols);
//     CHECK_CUDA(cudaGetLastError());
// }