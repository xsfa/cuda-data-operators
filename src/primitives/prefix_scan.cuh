#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Block-level exclusive prefix sum using shared memory.
 *
 * Blelloch scan algorithm:
 * 1. Upsweep: build reduction tree
 * 2. Set last element to 0 (identity for exclusive scan)
 * 3. Downsweep: propagate prefix sums
 *
 * This version handles one block. For large arrays, use hierarchical scan.
 */

constexpr int SCAN_BLOCK_SIZE = 256;

/**
 * In-place exclusive prefix sum within a block.
 * Input/output in shared memory.
 * Returns total sum (reduction) of the block.
 */
__device__ inline uint32_t block_exclusive_scan(uint32_t* sdata, int tid, int block_size) {
    // Upsweep (reduce)
    for (int stride = 1; stride < block_size; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < block_size) {
            sdata[idx] += sdata[idx - stride];
        }
        __syncthreads();
    }

    // Save total and clear last element
    uint32_t total = 0;
    if (tid == 0) {
        total = sdata[block_size - 1];
        sdata[block_size - 1] = 0;
    }
    __syncthreads();

    // Broadcast total to all threads
    __shared__ uint32_t block_total;
    if (tid == 0) block_total = total;
    __syncthreads();

    // Downsweep
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < block_size) {
            uint32_t temp = sdata[idx - stride];
            sdata[idx - stride] = sdata[idx];
            sdata[idx] += temp;
        }
        __syncthreads();
    }

    return block_total;
}

/**
 * Large array exclusive prefix scan.
 *
 * Three-phase algorithm:
 * 1. Per-block scan, store block totals
 * 2. Scan of block totals
 * 3. Add block offsets to each block
 *
 * For simplicity, this version requires n <= SCAN_BLOCK_SIZE^2
 */
__global__ void scan_phase1_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ block_sums,
    int n
) {
    __shared__ uint32_t sdata[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input (0 for out-of-bounds)
    sdata[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    // Block scan
    uint32_t block_total = block_exclusive_scan(sdata, tid, blockDim.x);

    // Write output
    if (gid < n) {
        output[gid] = sdata[tid];
    }

    // Write block total
    if (tid == 0) {
        block_sums[blockIdx.x] = block_total;
    }
}

__global__ void scan_phase2_kernel(
    uint32_t* __restrict__ block_sums,
    int num_blocks
) {
    __shared__ uint32_t sdata[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load block sums
    sdata[tid] = (tid < num_blocks) ? block_sums[tid] : 0;
    __syncthreads();

    // Scan block sums
    block_exclusive_scan(sdata, tid, blockDim.x);

    // Write back
    if (tid < num_blocks) {
        block_sums[tid] = sdata[tid];
    }
}

__global__ void scan_phase3_kernel(
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ block_sums,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        output[gid] += block_sums[blockIdx.x];
    }
}

/**
 * Host function: exclusive prefix sum on GPU array.
 * Supports arbitrary array sizes via iterative hierarchical scan.
 *
 * @param input  Device pointer to input array
 * @param output Device pointer to output array (can be same as input)
 * @param n      Number of elements
 * @param temp   Temporary storage (caller provides, size >= 2 * ceil(n/256))
 *
 * Max supported: 256^3 = 16M elements (3 levels). Extend levels[] for more.
 */
inline void exclusive_scan(
    const uint32_t* input,
    uint32_t* output,
    int n,
    uint32_t* temp_block_sums
) {
    if (n == 0) return;

    constexpr int MAX_LEVELS = 4;  // Supports up to 256^4 = 4B elements

    int num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

    // Phase 1: per-block scan of input
    scan_phase1_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(
        input, output, temp_block_sums, n
    );

    if (num_blocks == 1) {
        cudaDeviceSynchronize();
        return;
    }

    // Build hierarchy: track block counts and temp offsets at each level
    int level_blocks[MAX_LEVELS];
    uint32_t* level_data[MAX_LEVELS];
    int num_levels = 0;

    level_blocks[0] = num_blocks;
    level_data[0] = temp_block_sums;
    num_levels = 1;

    // Compute hierarchy levels (iterative, bounded by MAX_LEVELS)
    uint32_t* temp_ptr = temp_block_sums + num_blocks;
    int blocks_at_level = num_blocks;

    while (blocks_at_level > SCAN_BLOCK_SIZE && num_levels < MAX_LEVELS) {
        int next_blocks = (blocks_at_level + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

        // Scan this level's block sums into themselves, producing next level sums
        scan_phase1_kernel<<<next_blocks, SCAN_BLOCK_SIZE>>>(
            level_data[num_levels - 1],
            level_data[num_levels - 1],
            temp_ptr,
            blocks_at_level
        );

        level_blocks[num_levels] = next_blocks;
        level_data[num_levels] = temp_ptr;
        num_levels++;

        temp_ptr += next_blocks;
        blocks_at_level = next_blocks;
    }

    // Scan the top level (fits in one block)
    if (blocks_at_level > 1) {
        scan_phase2_kernel<<<1, SCAN_BLOCK_SIZE>>>(level_data[num_levels - 1], blocks_at_level);
    }

    // Propagate down: add each level's scanned sums to the level below
    for (int lvl = num_levels - 1; lvl > 0; lvl--) {
        scan_phase3_kernel<<<level_blocks[lvl], SCAN_BLOCK_SIZE>>>(
            level_data[lvl - 1],
            level_data[lvl],
            level_blocks[lvl - 1]
        );
    }

    // Final phase: add level 0 block sums to output
    scan_phase3_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(
        output, temp_block_sums, n
    );

    cudaDeviceSynchronize();
}

/**
 * Convenience: compute total (reduction) during scan.
 * Returns sum of all elements.
 */
inline uint32_t exclusive_scan_with_total(
    const uint32_t* input,
    uint32_t* output,
    int n,
    uint32_t* temp_block_sums
) {
    exclusive_scan(input, output, n, temp_block_sums);

    // Total = last prefix sum + last input element
    uint32_t last_prefix, last_input;
    cudaMemcpy(&last_prefix, output + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_input, input + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    return last_prefix + last_input;
}
