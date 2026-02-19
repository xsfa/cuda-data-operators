#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/**
 * Supported column data types.
 */
enum class DataType : uint8_t {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    // Strings stored as offset array + char buffer (Arrow-style)
    STRING
};

/**
 * Returns size in bytes for fixed-width types.
 * Returns 0 for variable-width types (STRING).
 */
inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::INT32:   return 4;
        case DataType::INT64:   return 8;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        case DataType::STRING:  return 0;  // variable width
        default:                return 0;
    }
}

/**
 * A single column of data residing in GPU memory.
 *
 * For fixed-width types:
 *   - data points to contiguous array of values
 *   - length is number of elements
 *
 * For STRING:
 *   - data points to char buffer
 *   - offsets points to int32 array of length (length + 1)
 *   - string[i] spans offsets[i] to offsets[i+1]
 */
struct Column {
    void* data = nullptr;
    size_t length = 0;
    DataType type = DataType::INT32;

    // Optional null bitmask: bit i = 1 means row i is valid (not null)
    // Bit-packed: byte[i/8] & (1 << (i%8))
    uint8_t* validity = nullptr;

    // For STRING type only
    int32_t* offsets = nullptr;

    // Convenience accessors (device code should use these carefully)
    template<typename T>
    __host__ __device__ T* as() { return static_cast<T*>(data); }

    template<typename T>
    __host__ __device__ const T* as() const { return static_cast<const T*>(data); }

    __host__ __device__ bool is_valid(size_t idx) const {
        if (!validity) return true;  // no nulls
        return (validity[idx / 8] >> (idx % 8)) & 1;
    }

    size_t size_bytes() const {
        if (type == DataType::STRING) {
            // Approximate: would need to read offsets[length] for exact size
            return 0;
        }
        return length * dtype_size(type);
    }
};

/**
 * A table is a collection of columns with the same row count.
 */
struct Table {
    Column* columns = nullptr;
    size_t num_columns = 0;
    size_t num_rows = 0;

    Column& operator[](size_t idx) { return columns[idx]; }
    const Column& operator[](size_t idx) const { return columns[idx]; }
};

/**
 * Helper to create a column from host data.
 * Allocates GPU memory and copies data.
 */
template<typename T>
Column make_column(const T* host_data, size_t length, DataType type) {
    Column col;
    col.length = length;
    col.type = type;

    size_t bytes = length * sizeof(T);
    cudaMalloc(&col.data, bytes);
    cudaMemcpy(col.data, host_data, bytes, cudaMemcpyHostToDevice);

    return col;
}

/**
 * Free GPU memory for a column.
 */
inline void free_column(Column& col) {
    if (col.data) cudaFree(col.data);
    if (col.validity) cudaFree(col.validity);
    if (col.offsets) cudaFree(col.offsets);
    col.data = nullptr;
    col.validity = nullptr;
    col.offsets = nullptr;
    col.length = 0;
}
