#!/usr/bin/env python3
"""
Benchmark: Raw CUDA kernel vs cuDF for predicate sum.

Run in Google Colab:
    !pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
    !nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -shared predicate_sum.cu -o libpredicate_sum.so
    !python benchmark.py

Or on any NVIDIA GPU machine with CUDA toolkit and RAPIDS installed.
"""

import ctypes
import time
from pathlib import Path

import cudf
import cupy as cp
import numpy as np


def load_cuda_kernel() -> ctypes.CDLL:
    """Load the compiled CUDA shared library."""
    lib_path = Path(__file__).parent / "libpredicate_sum.so"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} not found. Compile with:\n"
            "nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -shared predicate_sum.cu -o libpredicate_sum.so"
        )
    return ctypes.CDLL(str(lib_path))


def benchmark_cudf(
    values: np.ndarray, region_ids: np.ndarray, target: int, warmup: int = 3, runs: int = 10
) -> tuple[int, float]:
    """Benchmark cuDF predicate sum."""
    df = cudf.DataFrame({"values": values, "region_id": region_ids})

    # Warmup
    for _ in range(warmup):
        _ = df[df["region_id"] == target]["values"].sum()

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        result = df[df["region_id"] == target]["values"].sum()
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - start) / runs

    return int(result), elapsed


def benchmark_raw_cuda(
    values: np.ndarray, region_ids: np.ndarray, target: int, warmup: int = 3, runs: int = 10
) -> tuple[int, float]:
    """Benchmark raw CUDA kernel via ctypes."""
    lib = load_cuda_kernel()

    lib.predicate_sum.argtypes = [
        ctypes.c_void_p,  # d_values
        ctypes.c_void_p,  # d_region_ids
        ctypes.c_int,  # target_region
        ctypes.c_int,  # n
    ]
    lib.predicate_sum.restype = ctypes.c_longlong

    # Transfer to GPU via CuPy
    d_values = cp.asarray(values, dtype=cp.int32)
    d_region_ids = cp.asarray(region_ids, dtype=cp.int32)
    n = len(values)

    # Warmup
    for _ in range(warmup):
        _ = lib.predicate_sum(d_values.data.ptr, d_region_ids.data.ptr, target, n)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        result = lib.predicate_sum(d_values.data.ptr, d_region_ids.data.ptr, target, n)
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - start) / runs

    return result, elapsed


def main():
    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]
    target_region = 3
    num_regions = 10

    print(f"{'N':>12} | {'cuDF (ms)':>10} | {'CUDA (ms)':>10} | {'Speedup':>8} | {'Match':>5}")
    print("-" * 60)

    for n in sizes:
        np.random.seed(42)
        values = np.random.randint(1, 101, size=n, dtype=np.int32)
        region_ids = np.random.randint(0, num_regions, size=n, dtype=np.int32)

        # CPU reference
        expected = values[region_ids == target_region].sum()

        cudf_result, cudf_time = benchmark_cudf(values, region_ids, target_region)

        try:
            cuda_result, cuda_time = benchmark_raw_cuda(values, region_ids, target_region)
            cuda_ms = cuda_time * 1000
            speedup = cudf_time / cuda_time
            match = "yes" if cuda_result == expected else "NO"
        except FileNotFoundError:
            cuda_ms = float("nan")
            speedup = float("nan")
            match = "N/A"
            cuda_result = None

        print(
            f"{n:>12,} | {cudf_time * 1000:>10.3f} | {cuda_ms:>10.3f} | {speedup:>7.2f}x | {match}"
        )

    print("\nNote: Speedup > 1 means raw CUDA is faster than cuDF.")


if __name__ == "__main__":
    main()
