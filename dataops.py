"""
Python bindings for cuda-data-operators library.

Usage:
    from dataops import DataOps

    ops = DataOps()  # loads libdataops.so
    temp_size = ops.scan_temp_size(n)
    result = ops.filter_int32(input_ptr, n, value, op, output_ptr, ...)
"""

import ctypes
from pathlib import Path


class DataOps:
    """Wrapper for libdataops.so CUDA operators."""

    def __init__(self, lib_path: str | None = None):
        if lib_path is None:
            path = Path(__file__).parent / "libdataops.so"
        else:
            path = Path(lib_path)
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python test_runner.py --setup")

        self._lib = ctypes.CDLL(str(path))
        self._setup_signatures()

    def _setup_signatures(self):
        lib = self._lib

        # --- Memory Pool ---
        lib.pool_create.argtypes = [ctypes.c_size_t]
        lib.pool_create.restype = ctypes.c_void_p

        lib.pool_destroy.argtypes = [ctypes.c_void_p]
        lib.pool_destroy.restype = None

        lib.pool_reset.argtypes = [ctypes.c_void_p]
        lib.pool_reset.restype = None

        lib.pool_used.argtypes = [ctypes.c_void_p]
        lib.pool_used.restype = ctypes.c_size_t

        lib.pool_capacity.argtypes = [ctypes.c_void_p]
        lib.pool_capacity.restype = ctypes.c_size_t

        # --- Prefix Scan ---
        lib.prefix_scan_temp_size.argtypes = [ctypes.c_int]
        lib.prefix_scan_temp_size.restype = ctypes.c_size_t

        lib.prefix_scan_num_levels.argtypes = [ctypes.c_int]
        lib.prefix_scan_num_levels.restype = ctypes.c_int

        lib.prefix_scan_exclusive_uint32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.prefix_scan_exclusive_uint32.restype = None

        # --- Filter ---
        lib.filter_int32.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_int,  # n
            ctypes.c_int32,  # value
            ctypes.c_int,  # op (0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE)
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # temp_mask
            ctypes.c_void_p,  # temp_scan
            ctypes.c_void_p,  # temp_block_sums
        ]
        lib.filter_int32.restype = ctypes.c_uint32

        # --- Aggregates ---
        lib.agg_sum_int32.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.agg_sum_int32.restype = ctypes.c_int64

        lib.agg_sum_int64.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.agg_sum_int64.restype = ctypes.c_int64

        lib.agg_sum_float64.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.agg_sum_float64.restype = ctypes.c_double

        lib.agg_count.argtypes = [ctypes.c_int]
        lib.agg_count.restype = ctypes.c_uint64

    # --- Public API ---

    def scan_temp_size(self, n: int) -> int:
        return self._lib.prefix_scan_temp_size(n)

    def scan_num_levels(self, n: int) -> int:
        return self._lib.prefix_scan_num_levels(n)

    def prefix_scan(self, input_ptr: int, output_ptr: int, n: int, temp_ptr: int):
        self._lib.prefix_scan_exclusive_uint32(input_ptr, output_ptr, n, temp_ptr)

    def filter_int32(
        self,
        input_ptr: int,
        n: int,
        value: int,
        op: int,
        output_ptr: int,
        mask_ptr: int,
        scan_ptr: int,
        temp_ptr: int,
    ) -> int:
        return self._lib.filter_int32(
            input_ptr, n, value, op, output_ptr, mask_ptr, scan_ptr, temp_ptr
        )

    def sum_int32(self, input_ptr: int, n: int) -> int:
        return self._lib.agg_sum_int32(input_ptr, n)

    def sum_int64(self, input_ptr: int, n: int) -> int:
        return self._lib.agg_sum_int64(input_ptr, n)

    def sum_float64(self, input_ptr: int, n: int) -> float:
        return self._lib.agg_sum_float64(input_ptr, n)

    def count(self, n: int) -> int:
        return self._lib.agg_count(n)

    # --- Memory Pool ---

    def pool_create(self, capacity: int) -> int:
        return self._lib.pool_create(capacity)

    def pool_destroy(self, pool: int):
        self._lib.pool_destroy(pool)

    def pool_reset(self, pool: int):
        self._lib.pool_reset(pool)

    def pool_used(self, pool: int) -> int:
        return self._lib.pool_used(pool)


# Comparison operators for filter
class CompareOp:
    EQ = 0  # ==
    NE = 1  # !=
    LT = 2  # <
    LE = 3  # <=
    GT = 4  # >
    GE = 5  # >=
