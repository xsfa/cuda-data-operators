#include <cuda_runtime.h>
#include <cstdint>

#include "memory_pool.cuh"
#include "column.cuh"
#include "primitives/prefix_scan.cuh"
#include "operators/filter.cuh"
#include "operators/aggregate.cuh"

// =============================================================================
// C API exports for Python bindings
// =============================================================================

extern "C" {

// --- Memory Pool ---

void* pool_create(size_t capacity) {
    return new MemoryPool(capacity);
}

void pool_destroy(void* pool) {
    delete static_cast<MemoryPool*>(pool);
}

void* pool_allocate(void* pool, size_t bytes) {
    return static_cast<MemoryPool*>(pool)->allocate(bytes);
}

void pool_reset(void* pool) {
    static_cast<MemoryPool*>(pool)->reset();
}

size_t pool_used(void* pool) {
    return static_cast<MemoryPool*>(pool)->used();
}

size_t pool_capacity(void* pool) {
    return static_cast<MemoryPool*>(pool)->capacity();
}

// --- Prefix Scan ---

void prefix_scan_exclusive_uint32(
    const uint32_t* input,
    uint32_t* output,
    int n,
    uint32_t* temp_block_sums
) {
    exclusive_scan(input, output, n, temp_block_sums);
}

uint32_t prefix_scan_exclusive_uint32_total(
    const uint32_t* input,
    uint32_t* output,
    int n,
    uint32_t* temp_block_sums
) {
    return exclusive_scan_with_total(input, output, n, temp_block_sums);
}

// --- Filter ---

uint32_t filter_int32_gt(
    const int32_t* input,
    int n,
    int32_t threshold,
    int32_t* output,
    uint32_t* temp_mask,
    uint32_t* temp_scan,
    uint32_t* temp_block_sums
) {
    Column in_col{const_cast<int32_t*>(input), static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    Column out_col{output, static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    return filter_column<int32_t>(in_col, threshold, CompareOp::GT, out_col, temp_mask, temp_scan, temp_block_sums);
}

uint32_t filter_int32_eq(
    const int32_t* input,
    int n,
    int32_t value,
    int32_t* output,
    uint32_t* temp_mask,
    uint32_t* temp_scan,
    uint32_t* temp_block_sums
) {
    Column in_col{const_cast<int32_t*>(input), static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    Column out_col{output, static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    return filter_column<int32_t>(in_col, value, CompareOp::EQ, out_col, temp_mask, temp_scan, temp_block_sums);
}

uint32_t filter_int32(
    const int32_t* input,
    int n,
    int32_t value,
    int op,  // 0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE
    int32_t* output,
    uint32_t* temp_mask,
    uint32_t* temp_scan,
    uint32_t* temp_block_sums
) {
    Column in_col{const_cast<int32_t*>(input), static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    Column out_col{output, static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    return filter_column<int32_t>(in_col, value, static_cast<CompareOp>(op), out_col, temp_mask, temp_scan, temp_block_sums);
}

// --- Aggregates ---

int64_t agg_sum_int32(const int32_t* input, int n) {
    Column col{const_cast<int32_t*>(input), static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    int64_t* d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    int64_t result = sum_column<int32_t, int64_t>(col, d_result);
    cudaFree(d_result);
    return result;
}

int64_t agg_sum_int64(const int64_t* input, int n) {
    Column col{const_cast<int64_t*>(input), static_cast<size_t>(n), DataType::INT64, nullptr, nullptr};
    int64_t* d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    int64_t result = sum_column<int64_t, int64_t>(col, d_result);
    cudaFree(d_result);
    return result;
}

double agg_sum_float64(const double* input, int n) {
    Column col{const_cast<double*>(input), static_cast<size_t>(n), DataType::FLOAT64, nullptr, nullptr};
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    double result = sum_column<double, double>(col, d_result);
    cudaFree(d_result);
    return result;
}

uint64_t agg_count(int n) {
    Column col{nullptr, static_cast<size_t>(n), DataType::INT32, nullptr, nullptr};
    unsigned long long* d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));
    unsigned long long result = count_column(col, d_result);
    cudaFree(d_result);
    return result;
}

}  // extern "C"
