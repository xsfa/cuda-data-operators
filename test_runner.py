#!/usr/bin/env python3
"""
Test runner for GPU data operators.

Usage in Colab:
    !git clone https://github.com/xsfa/cuda-data-operators.git
    %cd cuda-data-operators
    !git checkout tesfashenkute/feat-bench-cudf-comparison
    !python test_runner.py --setup
    !python test_runner.py
"""

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

lib = None


def find_nvcc() -> str:
    """Find nvcc compiler."""
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    result = subprocess.run(
        ["find", "/usr", "-name", "nvcc", "-type", "f"], capture_output=True, text=True
    )
    if result.stdout.strip():
        return result.stdout.strip().split("\n")[0]

    return None


def setup():
    """Compile CUDA code into shared library."""
    print("=" * 60)
    print("COMPILING CUDA OPERATORS")
    print("=" * 60)

    nvcc = find_nvcc()
    if not nvcc:
        print("ERROR: nvcc not found")
        sys.exit(1)

    print(f"Using: {nvcc}")

    cmd = [
        nvcc,
        "-O3",
        "-arch=sm_75",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        "src/lib.cu",
        "-o",
        "libdataops.so",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("COMPILATION FAILED:")
        print(result.stderr)
        sys.exit(1)

    print("Compiled: libdataops.so")
    print("=" * 60)


def load_lib():
    """Load the compiled shared library."""
    global lib
    if not Path("libdataops.so").exists():
        print("Library not found. Run: python test_runner.py --setup")
        sys.exit(1)

    lib = ctypes.CDLL("./libdataops.so")

    # Memory pool
    lib.pool_create.argtypes = [ctypes.c_size_t]
    lib.pool_create.restype = ctypes.c_void_p
    lib.pool_destroy.argtypes = [ctypes.c_void_p]
    lib.pool_reset.argtypes = [ctypes.c_void_p]
    lib.pool_used.argtypes = [ctypes.c_void_p]
    lib.pool_used.restype = ctypes.c_size_t

    # Prefix scan
    lib.prefix_scan_exclusive_uint32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    # Filter
    lib.filter_int32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.filter_int32.restype = ctypes.c_uint32

    # Aggregates
    lib.agg_sum_int32.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.agg_sum_int32.restype = ctypes.c_int64
    lib.agg_sum_float64.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.agg_sum_float64.restype = ctypes.c_double
    lib.agg_count.argtypes = [ctypes.c_int]
    lib.agg_count.restype = ctypes.c_uint64

    return lib


# =============================================================================
# Tests
# =============================================================================

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def test_memory_pool():
    """Test memory pool allocation."""
    print("\n[TEST] Memory Pool")

    pool = lib.pool_create(1024 * 1024)
    assert pool, "Failed to create pool"

    lib.pool_reset(pool)
    assert lib.pool_used(pool) == 0

    lib.pool_destroy(pool)
    print("  PASSED")


def test_prefix_scan():
    """Test exclusive prefix scan."""
    print("\n[TEST] Prefix Scan")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    input_data = np.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=np.uint32)
    expected = np.array([0, 3, 4, 11, 11, 15, 16, 22], dtype=np.uint32)

    d_input = cp.asarray(input_data)
    d_output = cp.zeros_like(d_input)
    d_temp = cp.zeros(256, dtype=cp.uint32)

    lib.prefix_scan_exclusive_uint32(
        d_input.data.ptr, d_output.data.ptr, len(input_data), d_temp.data.ptr
    )

    result = cp.asnumpy(d_output)
    assert np.array_equal(result, expected), f"Got {result}"
    print(f"  Input:  {input_data}")
    print(f"  Output: {result}")
    print("  PASSED")


def test_prefix_scan_large():
    """Test prefix scan on 10K elements."""
    print("\n[TEST] Prefix Scan (10K)")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 10_000
    d_input = cp.ones(n, dtype=cp.uint32)
    d_output = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(256, dtype=cp.uint32)

    lib.prefix_scan_exclusive_uint32(d_input.data.ptr, d_output.data.ptr, n, d_temp.data.ptr)

    result = cp.asnumpy(d_output)
    expected = np.arange(n, dtype=np.uint32)
    assert np.array_equal(result, expected)
    print(f"  {n:,} elements: PASSED")


def test_filter():
    """Test filter with stream compaction."""
    print("\n[TEST] Filter")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    input_data = np.array([10, 80, 30, 90, 50, 70, 20, 60], dtype=np.int32)
    expected = np.array([80, 90, 70, 60], dtype=np.int32)

    n = len(input_data)
    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.int32)
    d_mask = cp.zeros(n, dtype=cp.uint32)
    d_scan = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(256, dtype=cp.uint32)

    # op=4 is GT
    count = lib.filter_int32(
        d_input.data.ptr,
        n,
        50,
        4,
        d_output.data.ptr,
        d_mask.data.ptr,
        d_scan.data.ptr,
        d_temp.data.ptr,
    )

    result = cp.asnumpy(d_output[:count])
    assert np.array_equal(result, expected), f"Got {result}"
    print(f"  Input:  {input_data}")
    print("  Filter: > 50")
    print(f"  Output: {result}")
    print("  PASSED")


def test_filter_large():
    """Test filter on 1M elements."""
    print("\n[TEST] Filter (1M)")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 1_000_000
    np.random.seed(42)
    input_data = np.random.randint(0, 100, n, dtype=np.int32)
    expected = input_data[input_data > 50]

    d_input = cp.asarray(input_data)
    d_output = cp.zeros(n, dtype=cp.int32)
    d_mask = cp.zeros(n, dtype=cp.uint32)
    d_scan = cp.zeros(n, dtype=cp.uint32)
    d_temp = cp.zeros(1024, dtype=cp.uint32)

    count = lib.filter_int32(
        d_input.data.ptr,
        n,
        50,
        4,
        d_output.data.ptr,
        d_mask.data.ptr,
        d_scan.data.ptr,
        d_temp.data.ptr,
    )

    result = cp.asnumpy(d_output[:count])
    assert len(result) == len(expected)
    assert np.array_equal(result, expected)
    print(f"  {n:,} → {count:,} rows: PASSED")


def test_sum():
    """Test SUM aggregation."""
    print("\n[TEST] SUM")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    data = np.arange(1, 11, dtype=np.int32)  # 1..10
    d_data = cp.asarray(data)

    result = lib.agg_sum_int32(d_data.data.ptr, len(data))
    assert result == 55, f"Got {result}"
    print(f"  sum(1..10) = {result}: PASSED")


def test_sum_large():
    """Test SUM on 10M elements."""
    print("\n[TEST] SUM (10M)")
    if not HAS_CUPY:
        print("  SKIPPED (no CuPy)")
        return

    n = 10_000_000
    d_data = cp.ones(n, dtype=cp.int32)

    result = lib.agg_sum_int32(d_data.data.ptr, n)
    assert result == n, f"Got {result}"
    print(f"  sum(10M ones) = {result:,}: PASSED")


def test_count():
    """Test COUNT aggregation."""
    print("\n[TEST] COUNT")

    assert lib.agg_count(1000) == 1000
    assert lib.agg_count(10_000_000) == 10_000_000
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
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Compile CUDA")
    parser.add_argument("--test", help="Run specific test")
    parser.add_argument("--list", action="store_true", help="List tests")
    args = parser.parse_args()

    if args.list:
        for name, _ in ALL_TESTS:
            print(f"  {name}")
        return

    if args.setup:
        setup()
        return

    load_lib()

    if args.test:
        for name, fn in ALL_TESTS:
            if name == args.test:
                fn()
                return
        print(f"Unknown test: {args.test}")
        sys.exit(1)

    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)

    passed = failed = 0
    for name, fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED ({name}): {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
