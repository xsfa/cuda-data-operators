#!/usr/bin/env python3
"""
Test runner for GPU data operators.

Usage in Colab:
    !git clone https://github.com/xsfa/cuda-data-operators.git
    %cd cuda-data-operators
    !git checkout tesfashenkute/feat-bench-cudf-comparison
    !python test_runner.py --setup   # compile everything
    !python test_runner.py           # run all tests
    !python test_runner.py --test filter  # run specific test
"""

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

# Will be loaded after compilation
lib = None


def setup():
    """Compile CUDA code into shared library."""
    print("=" * 60)
    print("COMPILING CUDA OPERATORS")
    print("=" * 60)

    # Find nvcc
    nvcc = None
    for path in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"]:
        if Path(path).exists():
            nvcc = path
            break

    if not nvcc:
        result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc = result.stdout.strip()

    if not nvcc:
        # Try to find it
        result = subprocess.run(
            ["find", "/usr", "-name", "nvcc", "-type", "f"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            nvcc = result.stdout.strip().split("\n")[0]

    if not nvcc:
        print("ERROR: nvcc not found. Make sure CUDA is installed.")
        sys.exit(1)

    print(f"Using nvcc: {nvcc}")

    # Write test kernels that export C functions
    test_cu = Path("test_ops.cu")
    test_cu.write_text('''
#include <cuda_runtime.h>
#include <cstdint>
#include "src/memory_pool.cuh"
#include "src/column.cuh"
#include "src/primitives/prefix_scan.cuh"
#include "src/operators/filter.cuh"
#include "src/operators/aggregate.cuh"

// =============================================================================
// Exported test functions
// =============================================================================

extern "C" {

// --- Memory Pool ---
void* create_pool(size_t capacity) {
    return new MemoryPool(capacity);
}

void destroy_pool(void* pool) {
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

// --- Prefix Scan ---
void test_prefix_scan(const uint32_t* input, uint32_t* output, int n, uint32_t* temp) {
    exclusive_scan(input, output, n, temp);
}

// --- Filter ---
uint32_t test_filter_int32(
    const int32_t* input,
    int n,
    int32_t value,
    int op,  // 0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE
    int32_t* output,
    uint32_t* temp_mask,
    uint32_t* temp_scan,
    uint32_t* temp_block_sums
) {
    Column in_col;
    in_col.data = const_cast<int32_t*>(input);
    in_col.length = n;
    in_col.type = DataType::INT32;

    Column out_col;
    out_col.data = output;
    out_col.length = n;
    out_col.type = DataType::INT32;

    return filter_column<int32_t>(
        in_col, value, static_cast<CompareOp>(op),
        out_col, temp_mask, temp_scan, temp_block_sums
    );
}

// --- Aggregates ---
int64_t test_sum_int32(const int32_t* input, int n) {
    Column col;
    col.data = const_cast<int32_t*>(input);
    col.length = n;
    col.type = DataType::INT32;

    int64_t* d_result;
    cudaMalloc(&d_result, sizeof(int64_t));
    int64_t result = sum_column<int32_t, int64_t>(col, d_result);
    cudaFree(d_result);
    return result;
}

double test_sum_float64(const double* input, int n) {
    Column col;
    col.data = const_cast<double*>(input);
    col.length = n;
    col.type = DataType::FLOAT64;

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    double result = sum_column<double, double>(col, d_result);
    cudaFree(d_result);
    return result;
}

uint64_t test_count(int n) {
    Column col;
    col.data = nullptr;
    col.length = n;
    col.validity = nullptr;  // All valid

    uint64_t* d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    uint64_t result = count_column(col, d_result);
    cudaFree(d_result);
    return result;
}

// --- Predicate Sum (original kernel) ---
}  // extern "C"

// Include original predicate sum for comparison
#include "predicate_sum.cu"
''')

    # Compile
    cmd = [
        nvcc,
        "-O3",
        "-arch=sm_75",
        "-Xcompiler", "-fPIC",
        "-shared",
        "test_ops.cu",
        "-o", "libtest_ops.so",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("COMPILATION FAILED:")
        print(result.stderr)
        sys.exit(1)

    print("Compilation successful!")
    print("=" * 60)


def load_lib():
    """Load the compiled shared library."""
    global lib
    lib_path = Path("libtest_ops.so")
    if not lib_path.exists():
        print("Library not found. Run with --setup first.")
        sys.exit(1)
    lib = ctypes.CDLL(str(lib_path))
    _setup_signatures()
    return lib


def _setup_signatures():
    """Set up ctypes function signatures."""
    # Memory pool
    lib.create_pool.argtypes = [ctypes.c_size_t]
    lib.create_pool.restype = ctypes.c_void_p

    lib.destroy_pool.argtypes = [ctypes.c_void_p]
    lib.destroy_pool.restype = None

    lib.pool_allocate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.pool_allocate.restype = ctypes.c_void_p

    lib.pool_reset.argtypes = [ctypes.c_void_p]
    lib.pool_reset.restype = None

    lib.pool_used.argtypes = [ctypes.c_void_p]
    lib.pool_used.restype = ctypes.c_size_t

    # Prefix scan
    lib.test_prefix_scan.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]
    lib.test_prefix_scan.restype = None

    # Filter
    lib.test_filter_int32.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.test_filter_int32.restype = ctypes.c_uint32

    # Aggregates
    lib.test_sum_int32.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.test_sum_int32.restype = ctypes.c_int64

    lib.test_sum_float64.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.test_sum_float64.restype = ctypes.c_double

    lib.test_count.argtypes = [ctypes.c_int]
    lib.test_count.restype = ctypes.c_uint64

    # Original predicate sum
    lib.predicate_sum.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.predicate_sum.restype = ctypes.c_longlong


# =============================================================================
# Test functions
# =============================================================================

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not available. Install with: pip install cupy-cuda12x")


def test_memory_pool():
    """Test memory pool allocation."""
    print("\n[TEST] Memory Pool")

    pool = lib.create_pool(1024 * 1024)  # 1MB
    assert pool is not None, "Failed to create pool"

    ptr1 = lib.pool_allocate(pool, 1024)
    assert ptr1 != 0, "First allocation failed"
    print(f"  Allocated 1KB, used: {lib.pool_used(pool)} bytes")

    ptr2 = lib.pool_allocate(pool, 4096)
    assert ptr2 != 0, "Second allocation failed"
    print(f"  Allocated 4KB, used: {lib.pool_used(pool)} bytes")

    lib.pool_reset(pool)
    assert lib.pool_used(pool) == 0, "Reset failed"
    print(f"  After reset, used: {lib.pool_used(pool)} bytes")

    lib.destroy_pool(pool)
    print("  PASSED")


def test_prefix_scan():
    """Test exclusive prefix scan."""
    print("\n[TEST] Prefix Scan")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    # Test case
    input_data = np.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=np.uint32)
    expected = np.array([0, 3, 4, 11, 11, 15, 16, 22], dtype=np.uint32)

    n = len(input_data)
    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(256, dtype=cp.uint32)  # Block sums

    lib.test_prefix_scan(
        d_input.data.ptr,
        d_output.data.ptr,
        n,
        d_temp.data.ptr
    )

    result = cp.asnumpy(d_output)
    print(f"  Input:    {input_data}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

    assert np.array_equal(result, expected), f"Mismatch!"
    print("  PASSED")


def test_prefix_scan_large():
    """Test prefix scan on larger array."""
    print("\n[TEST] Prefix Scan (large)")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 10000
    input_data = np.ones(n, dtype=np.uint32)
    expected = np.arange(n, dtype=np.uint32)

    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(256, dtype=cp.uint32)

    lib.test_prefix_scan(d_input.data.ptr, d_output.data.ptr, n, d_temp.data.ptr)

    result = cp.asnumpy(d_output)
    assert np.array_equal(result, expected), f"Mismatch at large scale"
    print(f"  {n} elements: PASSED")


def test_filter():
    """Test filter with stream compaction."""
    print("\n[TEST] Filter (stream compaction)")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    # Test: filter where value > 50
    input_data = np.array([10, 80, 30, 90, 50, 70, 20, 60], dtype=np.int32)
    expected = np.array([80, 90, 70, 60], dtype=np.int32)  # values > 50

    n = len(input_data)
    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.int32)
    d_mask = cp.zeros(n, dtype=cp.uint32)
    d_scan = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(256, dtype=cp.uint32)

    # op=4 is GT (greater than)
    output_size = lib.test_filter_int32(
        d_input.data.ptr, n, 50, 4,  # GT 50
        d_output.data.ptr,
        d_mask.data.ptr,
        d_scan.data.ptr,
        d_temp.data.ptr
    )

    result = cp.asnumpy(d_output[:output_size])
    print(f"  Input:    {input_data}")
    print(f"  Filter:   > 50")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

    assert np.array_equal(result, expected), f"Mismatch!"
    print("  PASSED")


def test_filter_large():
    """Test filter on large array."""
    print("\n[TEST] Filter (large)")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 1_000_000
    np.random.seed(42)
    input_data = np.random.randint(0, 100, size=n, dtype=np.int32)

    # CPU reference
    expected = input_data[input_data > 50]

    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.int32)
    d_mask = cp.zeros(n, dtype=cp.uint32)
    d_scan = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(1024, dtype=cp.uint32)

    output_size = lib.test_filter_int32(
        d_input.data.ptr, n, 50, 4,
        d_output.data.ptr,
        d_mask.data.ptr,
        d_scan.data.ptr,
        d_temp.data.ptr
    )

    result = cp.asnumpy(d_output[:output_size])

    assert len(result) == len(expected), f"Size mismatch: {len(result)} vs {len(expected)}"
    assert np.array_equal(result, expected), "Content mismatch"
    print(f"  {n:,} elements, {output_size:,} matched: PASSED")


def test_sum():
    """Test SUM aggregation."""
    print("\n[TEST] SUM aggregation")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    # Int32 sum
    input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    expected = input_data.sum()

    d_input = cp.asarray(input_data)
    result = lib.test_sum_int32(d_input.data.ptr, len(input_data))

    print(f"  Input:    {input_data}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

    assert result == expected, f"Mismatch!"
    print("  PASSED (int32)")

    # Float64 sum
    input_data = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    expected = input_data.sum()

    d_input = cp.asarray(input_data)
    result = lib.test_sum_float64(d_input.data.ptr, len(input_data))

    assert abs(result - expected) < 1e-10, f"Mismatch: {result} vs {expected}"
    print("  PASSED (float64)")


def test_sum_large():
    """Test SUM on large array."""
    print("\n[TEST] SUM (large)")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 10_000_000
    input_data = np.ones(n, dtype=np.int32)
    expected = n

    d_input = cp.asarray(input_data)
    result = lib.test_sum_int32(d_input.data.ptr, n)

    assert result == expected, f"Mismatch: {result} vs {expected}"
    print(f"  {n:,} elements, sum={result:,}: PASSED")


def test_count():
    """Test COUNT aggregation."""
    print("\n[TEST] COUNT aggregation")

    result = lib.test_count(1000)
    assert result == 1000, f"Mismatch: {result} vs 1000"
    print(f"  count(1000 rows) = {result}: PASSED")

    result = lib.test_count(10_000_000)
    assert result == 10_000_000, f"Mismatch"
    print(f"  count(10M rows) = {result:,}: PASSED")


def test_predicate_sum():
    """Test original predicate sum kernel."""
    print("\n[TEST] Predicate Sum (original)")

    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 100_000
    values = np.random.randint(1, 101, size=n, dtype=np.int32)
    region_ids = np.random.randint(0, 10, size=n, dtype=np.int32)
    target = 3

    # CPU reference
    expected = values[region_ids == target].sum()

    d_values = cp.asarray(values)
    d_region_ids = cp.asarray(region_ids)

    result = lib.predicate_sum(
        d_values.data.ptr,
        d_region_ids.data.ptr,
        target,
        n
    )

    print(f"  {n:,} rows, target region={target}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

    assert result == expected, f"Mismatch!"
    print("  PASSED")


# =============================================================================
# Main
# =============================================================================

ALL_TESTS = [
    ("memory_pool", test_memory_pool),
    ("prefix_scan", test_prefix_scan),
    ("prefix_scan_large", test_prefix_scan_large),
    ("filter", test_filter),
    ("filter_large", test_filter_large),
    ("sum", test_sum),
    ("sum_large", test_sum_large),
    ("count", test_count),
    ("predicate_sum", test_predicate_sum),
]


def main():
    parser = argparse.ArgumentParser(description="Test GPU data operators")
    parser.add_argument("--setup", action="store_true", help="Compile CUDA code")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--list", action="store_true", help="List available tests")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name, _ in ALL_TESTS:
            print(f"  {name}")
        return

    if args.setup:
        setup()
        return

    # Load library and run tests
    load_lib()

    if args.test:
        for name, fn in ALL_TESTS:
            if name == args.test:
                fn()
                return
        print(f"Unknown test: {args.test}")
        sys.exit(1)

    # Run all tests
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"  ERROR: {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
