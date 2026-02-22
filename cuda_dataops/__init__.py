"""
cuda_dataops — GPU-accelerated columnar data operators.

Provides:
  - Column / Table  : GPU-resident data model
  - DType           : supported column types
  - DataOps         : low-level CUDA kernel bindings
  - CompareOp       : comparison operator constants for filter kernels
"""

from .data import Column, DType, Table
from .ops import CompareOp, DataOps

__all__ = [
    "Column",
    "Table",
    "DType",
    "DataOps",
    "CompareOp",
]
