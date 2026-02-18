#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 256;

/**
 * Predicate scan kernel: sums values where region_id matches target.
 * Uses shared memory reduction within blocks, atomicAdd for final accumulation.
 */
__global__ void predicate_sum_kernel(
    const int* __restrict__ values,
    const int* __restrict__ region_ids,
    int target_region,
    int n,
    long long* __restrict__ result
) {
    __shared__ long long sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread accumulates multiple elements
    long long thread_sum = 0;
    for (int i = gid; i < n; i += stride) {
        if (region_ids[i] == target_region) {
            thread_sum += values[i];
        }
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block leader writes to global result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

/**
 * Host wrapper for predicate sum.
 * extern "C" enables calling from Python via ctypes.
 */
extern "C" long long predicate_sum(
    const int* d_values,
    const int* d_region_ids,
    int target_region,
    int n
) {
    long long* d_result;
    cudaMalloc(&d_result, sizeof(long long));
    cudaMemset(d_result, 0, sizeof(long long));

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 256);  // cap blocks to avoid excessive atomics

    predicate_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_values, d_region_ids, target_region, n, d_result
    );

    long long result;
    cudaMemcpy(&result, d_result, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

// Example usage
int main() {
    constexpr int N = 1000000;

    int* h_values = new int[N];
    int* h_region_ids = new int[N];

    // Generate test data: region_ids from 0-9, values from 1-100
    long long expected = 0;
    for (int i = 0; i < N; i++) {
        h_region_ids[i] = i % 10;
        h_values[i] = (i % 100) + 1;
        if (h_region_ids[i] == 3) {
            expected += h_values[i];
        }
    }

    int *d_values, *d_region_ids;
    cudaMalloc(&d_values, N * sizeof(int));
    cudaMalloc(&d_region_ids, N * sizeof(int));
    cudaMemcpy(d_values, h_values, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_region_ids, h_region_ids, N * sizeof(int), cudaMemcpyHostToDevice);

    long long result = predicate_sum(d_values, d_region_ids, 3, N);

    printf("Sum where region_id=3: %lld (expected: %lld)\n", result, expected);

    cudaFree(d_values);
    cudaFree(d_region_ids);
    delete[] h_values;
    delete[] h_region_ids;

    return 0;
}
