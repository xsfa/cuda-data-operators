#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

/**
 * Simple bump allocator over pre-allocated GPU memory.
 *
 * Design:
 * - Single cudaMalloc at construction
 * - Allocations bump a pointer forward
 * - No individual free — reset() reclaims all memory
 * - Intended for per-query lifetime: allocate intermediates, reset after query
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t capacity_bytes) : capacity_(capacity_bytes), offset_(0) {
        cudaError_t err = cudaMalloc(&base_, capacity_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "MemoryPool: cudaMalloc failed for %zu bytes: %s\n",
                    capacity_bytes, cudaGetErrorString(err));
            base_ = nullptr;
            capacity_ = 0;
        }
    }

    ~MemoryPool() {
        if (base_) {
            cudaFree(base_);
        }
    }

    // Non-copyable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // Movable
    MemoryPool(MemoryPool&& other) noexcept
        : base_(other.base_), capacity_(other.capacity_), offset_(other.offset_) {
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    /**
     * Allocate bytes from the pool.
     * Returns nullptr if insufficient space.
     * Alignment defaults to 256 bytes (cache line friendly).
     */
    void* allocate(size_t bytes, size_t alignment = 256) {
        if (!base_ || bytes == 0) return nullptr;

        // Align the current offset
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + bytes > capacity_) {
            fprintf(stderr, "MemoryPool: out of memory. Requested %zu, available %zu\n",
                    bytes, capacity_ - aligned_offset);
            return nullptr;
        }

        void* ptr = static_cast<char*>(base_) + aligned_offset;
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    /**
     * Allocate typed array.
     */
    template<typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T) > 256 ? alignof(T) : 256));
    }

    /**
     * Reset pool — all prior allocations become invalid.
     * Call between queries to reuse memory.
     */
    void reset() {
        offset_ = 0;
    }

    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }
    size_t available() const { return capacity_ - offset_; }

private:
    void* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};
