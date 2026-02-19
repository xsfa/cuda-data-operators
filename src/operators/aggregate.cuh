#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include "../column.cuh"

/**
 * Aggregate operators: SUM, COUNT, MIN, MAX, AVG
 *
 * Uses parallel reduction with shared memory.
 * For ungrouped aggregates (full table → single value).
 */

constexpr int AGG_BLOCK_SIZE = 256;

enum class AggOp : uint8_t {
    SUM,
    COUNT,
    MIN,
    MAX,
    AVG  // Computed as SUM / COUNT
};

// =============================================================================
// SUM
// =============================================================================

template<typename T, typename AccT = T>
__global__ void sum_kernel(
    const T* __restrict__ input,
    AccT* __restrict__ block_results,
    int n
) {
    __shared__ AccT sdata[AGG_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride accumulation
    AccT thread_sum = 0;
    for (int i = gid; i < n; i += stride) {
        thread_sum += static_cast<AccT>(input[i]);
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(block_results, sdata[0]);
    }
}

template<typename T, typename AccT = T>
AccT sum_column(const Column& col, AccT* d_result) {
    int n = col.length;
    int num_blocks = min((n + AGG_BLOCK_SIZE - 1) / AGG_BLOCK_SIZE, 256);

    cudaMemset(d_result, 0, sizeof(AccT));
    sum_kernel<T, AccT><<<num_blocks, AGG_BLOCK_SIZE>>>(col.as<T>(), d_result, n);
    cudaDeviceSynchronize();

    AccT result;
    cudaMemcpy(&result, d_result, sizeof(AccT), cudaMemcpyDeviceToHost);
    return result;
}

// =============================================================================
// COUNT
// =============================================================================

__global__ void count_non_null_kernel(
    const uint8_t* __restrict__ validity,
    uint64_t* __restrict__ block_results,
    int n
) {
    __shared__ uint64_t sdata[AGG_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint64_t thread_count = 0;
    for (int i = gid; i < n; i += stride) {
        if (!validity) {
            thread_count++;  // No validity mask = all valid
        } else {
            thread_count += (validity[i / 8] >> (i % 8)) & 1;
        }
    }

    sdata[tid] = thread_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(block_results, sdata[0]);
    }
}

inline uint64_t count_column(const Column& col, uint64_t* d_result) {
    int n = col.length;
    int num_blocks = min((n + AGG_BLOCK_SIZE - 1) / AGG_BLOCK_SIZE, 256);

    cudaMemset(d_result, 0, sizeof(uint64_t));
    count_non_null_kernel<<<num_blocks, AGG_BLOCK_SIZE>>>(col.validity, d_result, n);
    cudaDeviceSynchronize();

    uint64_t result;
    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return result;
}

// =============================================================================
// MIN
// =============================================================================

template<typename T>
__device__ T atomic_min_wrapper(T* addr, T val);

template<>
__device__ int32_t atomic_min_wrapper<int32_t>(int32_t* addr, int32_t val) {
    return atomicMin(addr, val);
}

template<>
__device__ uint32_t atomic_min_wrapper<uint32_t>(uint32_t* addr, uint32_t val) {
    return atomicMin(addr, val);
}

// For types without native atomicMin, use CAS loop
template<>
__device__ float atomic_min_wrapper<float>(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (old_val <= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
}

template<typename T>
__global__ void min_kernel(
    const T* __restrict__ input,
    T* __restrict__ result,
    int n
) {
    __shared__ T sdata[AGG_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize to max value
    T thread_min;
    if constexpr (sizeof(T) == 4 && !__is_same(T, float)) {
        thread_min = INT32_MAX;
    } else if constexpr (__is_same(T, float)) {
        thread_min = FLT_MAX;
    } else if constexpr (__is_same(T, double)) {
        thread_min = DBL_MAX;
    } else {
        thread_min = INT64_MAX;
    }

    for (int i = gid; i < n; i += stride) {
        T val = input[i];
        if (val < thread_min) thread_min = val;
    }

    sdata[tid] = thread_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_min_wrapper(result, sdata[0]);
    }
}

// =============================================================================
// MAX
// =============================================================================

template<typename T>
__device__ T atomic_max_wrapper(T* addr, T val);

template<>
__device__ int32_t atomic_max_wrapper<int32_t>(int32_t* addr, int32_t val) {
    return atomicMax(addr, val);
}

template<>
__device__ uint32_t atomic_max_wrapper<uint32_t>(uint32_t* addr, uint32_t val) {
    return atomicMax(addr, val);
}

template<>
__device__ float atomic_max_wrapper<float>(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (old_val >= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
}

template<typename T>
__global__ void max_kernel(
    const T* __restrict__ input,
    T* __restrict__ result,
    int n
) {
    __shared__ T sdata[AGG_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize to min value
    T thread_max;
    if constexpr (sizeof(T) == 4 && !__is_same(T, float)) {
        thread_max = INT32_MIN;
    } else if constexpr (__is_same(T, float)) {
        thread_max = -FLT_MAX;
    } else if constexpr (__is_same(T, double)) {
        thread_max = -DBL_MAX;
    } else {
        thread_max = INT64_MIN;
    }

    for (int i = gid; i < n; i += stride) {
        T val = input[i];
        if (val > thread_max) thread_max = val;
    }

    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_max_wrapper(result, sdata[0]);
    }
}

// =============================================================================
// Aggregate result container
// =============================================================================

struct AggregateResult {
    double sum = 0;
    uint64_t count = 0;
    double min = DBL_MAX;
    double max = -DBL_MAX;

    double avg() const { return count > 0 ? sum / count : 0; }
};
