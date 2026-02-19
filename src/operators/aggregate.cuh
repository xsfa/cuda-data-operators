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
// Atomic add helper (CUDA lacks atomicAdd for signed 64-bit)
// =============================================================================

// Device-compatible bit_cast (std::bit_cast is host-only without --expt-relaxed-constexpr)
template<typename To, typename From>
__device__ __host__ inline To bit_cast(From val) {
    static_assert(sizeof(To) == sizeof(From), "bit_cast requires same size types");
    To result;
    memcpy(&result, &val, sizeof(To));
    return result;
}

template<typename T>
__device__ inline void atomicAddAny(T* addr, T val);

template<>
__device__ inline void atomicAddAny<int>(int* addr, int val) {
    atomicAdd(addr, val);
}

template<>
__device__ inline void atomicAddAny<unsigned int>(unsigned int* addr, unsigned int val) {
    atomicAdd(addr, val);
}

template<>
__device__ inline void atomicAddAny<unsigned long long>(unsigned long long* addr, unsigned long long val) {
    atomicAdd(addr, val);
}

template<>
__device__ inline void atomicAddAny<long long>(long long* addr, long long val) {
    atomicAdd(bit_cast<unsigned long long*>(addr), bit_cast<unsigned long long>(val));
}

template<>
__device__ inline void atomicAddAny<int64_t>(int64_t* addr, int64_t val) {
    atomicAdd(bit_cast<unsigned long long*>(addr), bit_cast<unsigned long long>(val));
}

template<>
__device__ inline void atomicAddAny<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template<>
__device__ inline void atomicAddAny<double>(double* addr, double val) {
    atomicAdd(addr, val);
}

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
        atomicAddAny(block_results, sdata[0]);
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
    unsigned long long* __restrict__ block_results,
    int n
) {
    __shared__ unsigned long long sdata[AGG_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned long long thread_count = 0;
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

inline unsigned long long count_column(const Column& col, unsigned long long* d_result) {
    int n = col.length;
    int num_blocks = min((n + AGG_BLOCK_SIZE - 1) / AGG_BLOCK_SIZE, 256);

    cudaMemset(d_result, 0, sizeof(unsigned long long));
    count_non_null_kernel<<<num_blocks, AGG_BLOCK_SIZE>>>(col.validity, d_result, n);
    cudaDeviceSynchronize();

    unsigned long long result;
    cudaMemcpy(&result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
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
