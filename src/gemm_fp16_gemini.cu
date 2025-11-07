#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "matrix.h"

////////////////////////////////////////////////////////////////////////////////
// KERNEL IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

namespace gemini {

// Tiling and MMA configuration constants
constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BLOCK_K = 64;

constexpr uint32_t MMA_M = 16;
constexpr uint32_t MMA_N = 8;
constexpr uint32_t MMA_K = 8;

// 1024 threads per block
constexpr uint32_t CTA_SIZE_X = 32;
constexpr uint32_t CTA_SIZE_Y = 32;
constexpr uint32_t CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y;// 1024

// 32 warps per block
constexpr uint32_t WARPS_M = BLOCK_M / MMA_M;// 64 / 16 = 4
constexpr uint32_t WARPS_N = BLOCK_N / MMA_N;// 64 / 8 = 8
static_assert(WARPS_M * WARPS_N == 32, "Warp count must be 32");

// --- PTX Assembly Helpers ---

/**
 * @brief Issues an 8-byte asynchronous copy from global to shared memory.
 * @param smem_ptr 32-bit shared memory address (from __cvta_generic_to_shared)
 * @param gmem_ptr 64-bit global memory address (from cvta_to_global)
 */
__forceinline__ __device__ void cp_async_load_8b(uint32_t smem_ptr, uint64_t gmem_ptr) {
    // cp.async.ca.shared.global [dst], [src], 8;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), "l"(gmem_ptr), "n"(8));
}

/**
 * @brief Commits all outstanding cp.async operations in the group.
 */
__forceinline__ __device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

/**
 * @brief Waits for all outstanding cp.async operations in the group to complete.
 */
__forceinline__ __device__ void cp_async_wait() {
    // Waits until the count of outstanding operations drops to 0
    asm volatile("cp.async.wait_group %0;\n" : : "n"(0));
}

/**
 * @brief Performs one m16n8k8 MMA.sync operation.
 * d = a * b + c
 */
__forceinline__ __device__ void mma_sync_m16n8k8(
    unsigned int &d0, unsigned int &d1,
    const unsigned int a0, const unsigned int a1,
    const unsigned int b0,
    const unsigned int c0, const unsigned int c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
}

// --- Fragment Load/Store Helpers ---

/**
 * @brief Helper to pack two __half values into one unsigned int
 * (This is a NEW function)
 */
__forceinline__ __device__ unsigned int __halves2uint(const __half h0, const __half h1) {
    const __half2 v = __halves2half2(h0, h1);
    return *reinterpret_cast<const unsigned int*>(&v);
}


/**
 * @brief Loads a 16x8 .row fragment for matrix A from SRAM into registers.
 * (This function is correct, no change)
 */
__forceinline__ __device__ void load_a_fragment(const __half* sA_ptr, uint32_t laneId,
                                                unsigned int& a0, unsigned int& a1) {
    const uint32_t sA_row = laneId % 16;
    const uint32_t sA_col_offset = (laneId / 16) * 4;
    
    a0 = *reinterpret_cast<const unsigned int*>(&sA_ptr[sA_row * BLOCK_K + sA_col_offset]);
    a1 = *reinterpret_cast<const unsigned int*>(&sA_ptr[sA_row * BLOCK_K + sA_col_offset + 2]);
}

/**
 * @brief Loads an 8x8 .col fragment for matrix B from SRAM into registers.
 * (This function is REWRITTEN for the .col layout)
 */
__forceinline__ __device__ void load_b_fragment(const __half* sB_ptr, uint32_t laneId,
                                                unsigned int& b0) {
    // sB is [BLOCK_K][BLOCK_N] = [64][64], row-major
    // We need to load an 8x8 .col fragment for the MMA.
    // Each thread (32) loads one __half2 (2 elements). Total 64 elements.
    // A .col fragment register (b0) holds sB[k, n] and sB[k+1, n].
    
    // Map 32 threads to an 8x4 grid. Each thread loads a 2x1 chunk.
    const uint32_t sB_col = laneId % 8;     // Column within 8x8 tile (0..7)
    const uint32_t sB_row = (laneId / 8) * 2; // Row base within 8x8 tile (0, 2, 4, 6)
    
    // Pointer to sB[sB_row, sB_col]
    const __half* b_smem_ptr = &sB_ptr[sB_row * BLOCK_N + sB_col];

    // Load sB[row, col]
    __half h0 = *b_smem_ptr;
    // Load sB[row+1, col]
    __half h1 = *(b_smem_ptr + BLOCK_N); // +BLOCK_N moves one row down
    
    // Pack {h0, h1} into the register b0
    b0 = __halves2uint(h0, h1);
}

/**
 * @brief Loads a 16x8 .row fragment (A) from SRAM using ldmatrix.
 * This loads 4x b16 elements per thread, packing them into 2x 32-bit registers.
 */
__forceinline__ __device__ void ldmatrix_sync_m16n8k8_row(
    unsigned int& a0, unsigned int& a1, 
    const void* smem_ptr
) {
    uint64_t ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(a0), "=r"(a1) 
        : "l"(ptr)
    );
}

/**
 * @brief Loads an 8x8 .col fragment (B) from SRAM using ldmatrix.
 * This loads 2x b16 elements per thread, packing them into 1x 32-bit register.
 */
__forceinline__ __device__ void ldmatrix_sync_m16n8k8_col(
    unsigned int& b0, 
    const void* smem_ptr
) {
    uint64_t ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0}, [%1];\n"
        : "=r"(b0) 
        : "l"(ptr)
    );
}

__forceinline__ __device__ __half2 auto_cast(const unsigned int &reg) {
    return *reinterpret_cast<const __half2 *>(&reg);
}
/**
 * @brief Stores a 16x8 .row fragment from registers to global memory C.
 */
__forceinline__ __device__ void store_c_fragment(__half *gC_ptr, uint32_t N_stride, uint32_t laneId,
                                                 const unsigned int c0, const unsigned int c1) {
    // C is 16x8 .row layout
    // Each thread stores 4 __half elements (2 __half2 regs)
    // Row mapping: (laneId % 16)
    // Col mapping: (laneId / 16) * 4 (cols 0, 4)
    const uint32_t gC_row_offset = laneId % 16;
    const uint32_t gC_col_offset = (laneId / 16) * 4;

    // Pointer to the start of the thread's store location
    __half *gC_thread_ptr = gC_ptr + gC_row_offset * N_stride + gC_col_offset;

    // Store the 4 __half values
    *reinterpret_cast<__half2 *>(gC_thread_ptr) = auto_cast(c0);
    *reinterpret_cast<__half2 *>(gC_thread_ptr + 2) = auto_cast(c1);
}

/**
 * @brief Kernel core: Performs the MMA accumulation loop for one CTA tile.
 * (This function is UPDATED)
 */
__forceinline__ __device__ void compute_mma_tile(
    const __half* sA, const __half* sB,
    uint32_t warp_m, uint32_t warp_n, uint32_t laneId,
    unsigned int& c0, unsigned int& c1
) {
    // Pointers to the start of this warp's tiles in SRAM
    const __half* sA_warp_ptr = &sA[warp_m * MMA_M * BLOCK_K];
    // Base pointer for this warp's Kx8 tile slice in sB
    const __half* sB_warp_ptr = &sB[warp_n * MMA_N]; 

    #pragma unroll
    for (int k_inner = 0; k_inner < BLOCK_K; k_inner += MMA_K) {
        unsigned int a0, a1, b0;

        // 1. Load fragments from SRAM to Registers
        
        // Load A (row-major)
        load_a_fragment(sA_warp_ptr + k_inner, laneId, a0, a1);
        
        // Load B (col-major)
        // Pass the pointer to the top-left of the 8x8 tile
        load_b_fragment(sB_warp_ptr + k_inner * BLOCK_N, laneId, b0);

        // 2. Perform MMA
        mma_sync_m16n8k8(c0, c1, a0, a1, b0, c0, c1);
    }
}
__forceinline__ __device__ void cp_async_load_16b(uint32_t smem_ptr, uint64_t gmem_ptr) {
    // cp.async.ca.shared.global [dst], [src], 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), "l"(gmem_ptr), "n"(16));
}

/**
 * @brief Kernel core: Loads the next tile from HBM to SRAM using cp.async.
 */
__forceinline__ __device__ void load_sram_tile(
    const __half *gA, const __half *gB,
    __half *sA, __half *sB,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t cta_row, uint32_t cta_col, uint32_t k_tile,
    uint32_t tid) {
    // Use 1024 threads to load both tiles (512 for A, 512 for B)
    // Each thread loads 8 bytes (__half4)

    // --- Load Tile A (64x64, 4096 __half, 8192 bytes) ---
    // 8192 bytes / 512 threads = 16 bytes/thread. Let's load 2x 8-byte chunks.
    const uint32_t load_threads_A = CTA_SIZE / 2;// 512 threads
    if (tid < load_threads_A) {
        // Map 512 threads to a 64x64 matrix, 16 bytes (__half8) per thread
        // Total elements = 64 * 64 = 4096. 4096 / 512 = 8 __half elements per thread.
        const uint32_t row = tid / (BLOCK_K / 8);// 64 / 8 = 8 -> 512/8 = 64 rows
        const uint32_t col = (tid % 8) * 8;      // 8 cols per thread

        const __half *gA_ptr = &gA[(cta_row + row) * K + (k_tile + col)];
        __half *sA_ptr = &sA[row * BLOCK_K + col];

        // cp.async must use converted pointers
        uint64_t gA_u64 = __cvta_generic_to_global(gA_ptr);
        uint32_t sA_u32 = __cvta_generic_to_shared(sA_ptr);

        cp_async_load_16b(sA_u32, gA_u64);
    }

    // --- Load Tile B (64x64, 4096 __half, 8192 bytes) ---
    else if (tid < CTA_SIZE) {
        const uint32_t local_tid = tid - load_threads_A;// 0..511

        // Map 512 threads to a 64x64 matrix, 8 __half elements per thread
        const uint32_t row = local_tid / (BLOCK_N / 8);// 64 / 8 = 8 -> 512/8 = 64 rows. Wait...
        // sB is [BLOCK_K][BLOCK_N] = [64][64]
        const uint32_t row_b = local_tid / 8;      // 512/8 = 64 rows
        const uint32_t col_b = (local_tid % 8) * 8;// 8 cols per thread

        const __half *gB_ptr = &gB[(k_tile + row_b) * N + (cta_col + col_b)];
        __half *sB_ptr = &sB[row_b * BLOCK_N + col_b];

        uint64_t gB_u64 = __cvta_generic_to_global(gB_ptr);
        uint32_t sB_u32 = __cvta_generic_to_shared(sB_ptr);

        cp_async_load_16b(sB_u32, gB_u64);
    }
}

}// namespace gemini

/**
 * @brief Main GEMM kernel
 * Assumes M, N, K are multiples of 64.
 */
__global__ void gemm_gemini_fp16_kernel(
    const __half *A, const __half *B, __half *C,
    uint32_t M, uint32_t N, uint32_t K) {
    using namespace gemini;

    // Shared memory for 2x buffers (A and B)
    extern __shared__ __half sram_buffer[];

    // Pointers for double buffering
    __half *sA_buffers[2] = {
        sram_buffer,
        sram_buffer + (BLOCK_M * BLOCK_K)};
    __half *sB_buffers[2] = {
        sram_buffer + 2 * (BLOCK_M * BLOCK_K),
        sram_buffer + 2 * (BLOCK_M * BLOCK_K) + (BLOCK_K * BLOCK_N)};

    // --- Thread/Warp/Block Identification ---
    const uint32_t tid = threadIdx.x + threadIdx.y * CTA_SIZE_X;
    const uint32_t warpId = tid / 32;
    const uint32_t laneId = tid % 32;

    // Map 2D block to 1D grid
    const uint32_t cta_col = blockIdx.x * BLOCK_N;
    const uint32_t cta_row = blockIdx.y * BLOCK_M;

    // This warp's tile within the CTA
    const uint32_t warp_m = warpId / WARPS_N;// 0..3
    const uint32_t warp_n = warpId % WARPS_N;// 0..7

    // --- Register Accumulator Initialization ---
    // Each thread holds 4 __half elements for its 16x8 C-tile
    unsigned int c0 = 0;
    unsigned int c1 = 0;

    // --- Asynchronous Pipelining Setup ---
    int k_tile = 0;
    int sram_buf_idx = 0;// Current buffer to read from

    // 1. Prologue: Load first tile
    load_sram_tile(A, B, sA_buffers[sram_buf_idx], sB_buffers[sram_buf_idx],
                   M, N, K, cta_row, cta_col, k_tile, tid);
    cp_async_commit();// Commit the loads
    k_tile += BLOCK_K;

    // 2. Main K-Loop: Overlap compute and memory
    for (; k_tile < K; k_tile += BLOCK_K) {
        const int compute_buf_idx = sram_buf_idx;
        const int load_buf_idx = 1 - sram_buf_idx;

        // 2a. Start loading next tile
        load_sram_tile(A, B, sA_buffers[load_buf_idx], sB_buffers[load_buf_idx],
                       M, N, K, cta_row, cta_col, k_tile, tid);

        // 2b. Wait for current tile to finish loading
        cp_async_wait();
        __syncthreads();// Ensure all threads see the loaded data

        // 2c. Compute on current tile
        compute_mma_tile(sA_buffers[compute_buf_idx], sB_buffers[compute_buf_idx],
                         warp_m, warp_n, laneId, c0, c1);

        // 2d. Commit next load
        cp_async_commit();

        // 2e. Wait for compute to finish before next loop iteration
        // (which will overwrite the sram)
        __syncthreads();

        // 2f. Flip buffers
        sram_buf_idx = load_buf_idx;
    }

    // 3. Epilogue: Compute on last tile
    cp_async_wait();
    __syncthreads();
    compute_mma_tile(sA_buffers[sram_buf_idx], sB_buffers[sram_buf_idx],
                     warp_m, warp_n, laneId, c0, c1);

    // --- Store Results to Global Memory ---

    // Since we assume M, N are multiples of 64, no boundary checks needed

    // Pointer to this warp's 16x8 tile in C
    __half *gC_warp_ptr = &C[(cta_row + warp_m * MMA_M) * N +
                             (cta_col + warp_n * MMA_N)];

    store_c_fragment(gC_warp_ptr, N, laneId, c0, c1);
}

/**
 * @brief Host-side launcher function
 */
static void launch_gemm(
    cudaStream_t stream,
    const __half *A,
    const __half *B,
    __half *C,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    // --- Checks ---
    // This high-performance kernel assumes M, N, and K are multiples
    // of the block tile dimensions for simplicity and speed (no tail handling).
    if ((M % gemini::BLOCK_M != 0) ||
        (N % gemini::BLOCK_N != 0) ||
        (K % gemini::BLOCK_K != 0)) {
        printf("Error: M, N, and K must be multiples of 64 for this kernel.\n");
        printf("M=%u (req %u), N=%u (req %u), K=%u (req %u)\n",
               M, gemini::BLOCK_M, N, gemini::BLOCK_N, K, gemini::BLOCK_K);
        return;
    }

    // --- Launch Configuration ---

    // CTA grid
    dim3 grid((N + gemini::BLOCK_N - 1) / gemini::BLOCK_N,
              (M + gemini::BLOCK_M - 1) / gemini::BLOCK_M);

    // 1024 threads per CTA, in 2D
    dim3 block(gemini::CTA_SIZE_X, gemini::CTA_SIZE_Y);

    // Shared memory size for double buffering
    size_t sram_size = 2 * (gemini::BLOCK_M * gemini::BLOCK_K + gemini::BLOCK_K * gemini::BLOCK_N) * sizeof(__half);
    // sram_size = 2 * (64*64 + 64*64) * 2 = 2 * (4096 + 4096) * 2 = 32768 bytes
    cudaMemsetAsync(C, 0, sizeof(Half) * M * N, stream);
    gemm_gemini_fp16_kernel<<<grid, block, sram_size, stream>>>(A, B, C, M, N, K);
}

/**
 * @brief User-provided entry point
 */
void gemini_gemm_fp16(cudaStream_t stream, const Matrix<Half> &A, const Matrix<Half> &B, Matrix<Half> &C) {
    launch_gemm(
        stream,
        A.data,
        B.data,
        C.data,
        A.rows,
        B.cols,
        A.cols);
}