#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../primitives/prefix_scan.cuh"
#include "../column.cuh"

/**
 * Filter operator: stream compaction based on predicate.
 *
 * Given a predicate mask, produces dense output containing only matching rows.
 *
 * Algorithm:
 * 1. Evaluate predicate → bitmask (or use provided mask)
 * 2. Exclusive prefix sum on mask → output indices
 * 3. Scatter matching elements to output positions
 *
 * Example:
 *   input:      [10, 20, 30, 40, 50]
 *   mask:       [ 1,  0,  1,  0,  1]
 *   prefix_sum: [ 0,  1,  1,  2,  2]  (exclusive)
 *   output:     [10, 30, 50]
 */

constexpr int FILTER_BLOCK_SIZE = 256;

// Comparison operators for predicates
enum class CompareOp : uint8_t {
    EQ,   // ==
    NE,   // !=
    LT,   // <
    LE,   // <=
    GT,   // >
    GE    // >=
};

/**
 * Evaluate predicate: column[i] <op> value
 * Writes 1 to mask[i] if true, 0 if false.
 */
template<typename T>
__global__ void evaluate_predicate_kernel(
    const T* __restrict__ column,
    T value,
    CompareOp op,
    uint32_t* __restrict__ mask,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    T col_val = column[gid];
    bool result = false;

    switch (op) {
        case CompareOp::EQ: result = (col_val == value); break;
        case CompareOp::NE: result = (col_val != value); break;
        case CompareOp::LT: result = (col_val <  value); break;
        case CompareOp::LE: result = (col_val <= value); break;
        case CompareOp::GT: result = (col_val >  value); break;
        case CompareOp::GE: result = (col_val >= value); break;
    }

    mask[gid] = result ? 1 : 0;
}

/**
 * Scatter elements where mask=1 to output positions determined by prefix sum.
 */
template<typename T>
__global__ void scatter_kernel(
    const T* __restrict__ input,
    const uint32_t* __restrict__ prefix_sum,
    const uint32_t* __restrict__ mask,
    T* __restrict__ output,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    if (mask[gid]) {
        output[prefix_sum[gid]] = input[gid];
    }
}

/**
 * Multi-column scatter: scatter multiple columns using same mask.
 * Each thread handles one row, scatters all columns for that row.
 */
__global__ void scatter_multi_column_kernel(
    const Column* __restrict__ input_columns,
    const uint32_t* __restrict__ prefix_sum,
    const uint32_t* __restrict__ mask,
    Column* __restrict__ output_columns,
    int num_columns,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    if (!mask[gid]) return;

    int out_idx = prefix_sum[gid];

    for (int c = 0; c < num_columns; c++) {
        DataType type = input_columns[c].type;
        switch (type) {
            case DataType::INT32: {
                auto* in = static_cast<const int32_t*>(input_columns[c].data);
                auto* out = static_cast<int32_t*>(output_columns[c].data);
                out[out_idx] = in[gid];
                break;
            }
            case DataType::INT64: {
                auto* in = static_cast<const int64_t*>(input_columns[c].data);
                auto* out = static_cast<int64_t*>(output_columns[c].data);
                out[out_idx] = in[gid];
                break;
            }
            case DataType::FLOAT32: {
                auto* in = static_cast<const float*>(input_columns[c].data);
                auto* out = static_cast<float*>(output_columns[c].data);
                out[out_idx] = in[gid];
                break;
            }
            case DataType::FLOAT64: {
                auto* in = static_cast<const double*>(input_columns[c].data);
                auto* out = static_cast<double*>(output_columns[c].data);
                out[out_idx] = in[gid];
                break;
            }
            default:
                break;
        }
    }
}

/**
 * Host function: filter a single column by predicate.
 *
 * @param input      Input column (device memory)
 * @param value      Value to compare against
 * @param op         Comparison operator
 * @param output     Output column (device memory, pre-allocated to input.length)
 * @param temp_mask  Temporary storage for mask (size = input.length)
 * @param temp_scan  Temporary storage for prefix sum (size = input.length)
 * @param temp_block_sums  Temporary storage for scan (size = ceil(n/256))
 * @return           Number of elements in output
 */
template<typename T>
uint32_t filter_column(
    const Column& input,
    T value,
    CompareOp op,
    Column& output,
    uint32_t* temp_mask,
    uint32_t* temp_scan,
    uint32_t* temp_block_sums
) {
    int n = input.length;
    int num_blocks = (n + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;

    // Step 1: Evaluate predicate
    evaluate_predicate_kernel<T><<<num_blocks, FILTER_BLOCK_SIZE>>>(
        input.as<T>(), value, op, temp_mask, n
    );

    // Step 2: Prefix sum on mask
    uint32_t output_size = exclusive_scan_with_total(temp_mask, temp_scan, n, temp_block_sums);

    // Step 3: Scatter
    scatter_kernel<T><<<num_blocks, FILTER_BLOCK_SIZE>>>(
        input.as<T>(), temp_scan, temp_mask, output.as<T>(), n
    );

    cudaDeviceSynchronize();

    output.length = output_size;
    return output_size;
}

/**
 * Compound predicate: AND of two predicates.
 * mask_out[i] = mask_a[i] AND mask_b[i]
 */
__global__ void and_masks_kernel(
    const uint32_t* __restrict__ mask_a,
    const uint32_t* __restrict__ mask_b,
    uint32_t* __restrict__ mask_out,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    mask_out[gid] = mask_a[gid] & mask_b[gid];
}

/**
 * Compound predicate: OR of two predicates.
 */
__global__ void or_masks_kernel(
    const uint32_t* __restrict__ mask_a,
    const uint32_t* __restrict__ mask_b,
    uint32_t* __restrict__ mask_out,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    mask_out[gid] = mask_a[gid] | mask_b[gid];
}
