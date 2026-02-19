#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <bit>

// Option 1: with --expt-relaxed-constexpr flag
// __device__ inline void atomicAddInt64(int64_t* addr, int64_t val) {
//     atomicAdd(
//         reinterpret_cast<unsigned long long*>(addr),
//         std::bit_cast<unsigned long long>(val)
//     );
// }

// Option 2: manual bit_cast for device code (no flag needed)
template<typename To, typename From>
__device__ __host__ inline To device_bit_cast(From val) {
    static_assert(sizeof(To) == sizeof(From), "Size mismatch");
    To result;
    memcpy(&result, &val, sizeof(To));
    return result;
}

__device__ inline void atomicAddInt64(int64_t* addr, int64_t val) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(addr),
        device_bit_cast<unsigned long long>(val)
    );
}

__global__ void test_kernel(int64_t* result) {
    atomicAddInt64(result, 42);
}

int main() {
    int64_t* d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    cudaMemset(d_result, 0, sizeof(int64_t));

    test_kernel<<<1, 1>>>(d_result);

    int64_t h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);

    printf("Result: %lld (expected 42)\n", h_result);

    cudaFree(d_result);
    return h_result == 42 ? 0 : 1;
}
