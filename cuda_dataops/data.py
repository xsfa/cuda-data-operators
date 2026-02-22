"""
GPU-resident columnar data structures.

Column and Table are the data model for the query engine. All arrays live
in GPU memory (CuPy) and are never copied to CPU unless explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import cupy as cp
import numpy as np


class DType(Enum):
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


_DTYPE_TO_CP: dict[DType, type] = {
    DType.INT32: cp.int32,
    DType.INT64: cp.int64,
    DType.FLOAT32: cp.float32,
    DType.FLOAT64: cp.float64,
}

_NP_STR_TO_DTYPE: dict[str, DType] = {
    "int32": DType.INT32,
    "int64": DType.INT64,
    "float32": DType.FLOAT32,
    "float64": DType.FLOAT64,
}


@dataclass
class Column:
    """A single GPU-resident typed array with a name.

    The underlying CuPy array is always contiguous and matches the declared
    dtype. Use the class methods for convenient construction.
    """

    name: str
    dtype: DType
    data: cp.ndarray

    def __post_init__(self) -> None:
        expected = _DTYPE_TO_CP[self.dtype]
        if self.data.dtype != expected:
            self.data = self.data.astype(expected)
        if not self.data.flags["C_CONTIGUOUS"]:
            self.data = cp.ascontiguousarray(self.data)

    # --- Constructors ---

    @classmethod
    def from_numpy(cls, name: str, arr: np.ndarray) -> Column:
        """Copy a NumPy array to GPU memory."""
        dtype_str = str(arr.dtype)
        if dtype_str not in _NP_STR_TO_DTYPE:
            raise ValueError(
                f"Unsupported dtype {dtype_str!r}. "
                f"Supported: {list(_NP_STR_TO_DTYPE)}"
            )
        return cls(name=name, dtype=_NP_STR_TO_DTYPE[dtype_str], data=cp.asarray(arr))

    @classmethod
    def from_list(cls, name: str, values: list, dtype: DType) -> Column:
        """Create a GPU column from a Python list."""
        return cls(
            name=name,
            dtype=dtype,
            data=cp.array(values, dtype=_DTYPE_TO_CP[dtype]),
        )

    @classmethod
    def zeros(cls, name: str, n: int, dtype: DType) -> Column:
        """Allocate a zero-filled column of length n."""
        return cls(name=name, dtype=dtype, data=cp.zeros(n, dtype=_DTYPE_TO_CP[dtype]))

    @classmethod
    def arange(cls, name: str, n: int, dtype: DType = DType.INT32) -> Column:
        """Allocate a column filled with 0..n-1."""
        return cls(
            name=name,
            dtype=dtype,
            data=cp.arange(n, dtype=_DTYPE_TO_CP[dtype]),
        )

    # --- Properties ---

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def ptr(self) -> int:
        """Raw device pointer — pass to CUDA kernels."""
        return self.data.data.ptr

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    # --- Utilities ---

    def to_numpy(self) -> np.ndarray:
        """Copy column data back to CPU."""
        return cp.asnumpy(self.data)

    def __repr__(self) -> str:
        mb = self.nbytes / 1024 / 1024
        return f"Column({self.name!r}, {self.dtype.value}, {self.n_rows:,} rows, {mb:.2f}MB)"


class Table:
    """An ordered collection of GPU-resident columns sharing a row count.

    Columns are added via ``add()``, which validates row count consistency.
    Indexing by column name returns the Column object.

    Example::

        t = Table("sales")
        t.add(Column.from_numpy("revenue", np.array([100, 200, 150], dtype=np.int32)))
        t.add(Column.from_numpy("region",  np.array([0,   1,   0  ], dtype=np.int32)))
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._columns: dict[str, Column] = {}

    def add(self, col: Column) -> Table:
        """Add a column. Returns self for chaining."""
        if self._columns and col.n_rows != self.n_rows:
            raise ValueError(
                f"Column {col.name!r} has {col.n_rows:,} rows, "
                f"expected {self.n_rows:,}"
            )
        self._columns[col.name] = col
        return self

    # --- Access ---

    @property
    def columns(self) -> dict[str, Column]:
        return self._columns

    def __getitem__(self, name: str) -> Column:
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(f"No column {name!r} in table {self.name!r}")

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __iter__(self) -> Iterator[Column]:
        return iter(self._columns.values())

    # --- Metadata ---

    @property
    def n_rows(self) -> int:
        if not self._columns:
            return 0
        return next(iter(self._columns.values())).n_rows

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self._columns.values())

    @property
    def schema(self) -> dict[str, DType]:
        """Column name → DType mapping."""
        return {name: col.dtype for name, col in self._columns.items()}

    def schema_prompt(self) -> str:
        """Human-readable schema for injecting into an LLM system prompt."""
        lines = [f"Table: {self.name}  ({self.n_rows:,} rows)"]
        for name, col in self._columns.items():
            lines.append(f"  {name:<20} {col.dtype.value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        mb = self.nbytes / 1024 / 1024
        col_summary = ", ".join(
            f"{n}:{c.dtype.value}" for n, c in self._columns.items()
        )
        return f"Table({self.name!r}, {self.n_rows:,} rows, {mb:.1f}MB, [{col_summary}])"
