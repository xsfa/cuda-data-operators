# cuda-data-operators

GPU-native data operators for SQL-like operations directly on GPU memory.

## Quick Start (Google Colab)

```python
!git clone https://github.com/xsfa/cuda-data-operators.git
%cd cuda-data-operators
!pip install --quiet cupy-cuda12x
!python test_runner.py --setup
!python test_runner.py
```

**Requirements**: GPU runtime (Runtime → Change runtime type → T4 GPU)

## Local Development (with uv)

```bash
uv pip install -e ".[gpu]"
uv run python test_runner.py --setup
uv run python test_runner.py
```

## Operators

| Operator | Status | Description |
|----------|--------|-------------|
| Filter | ✅ | Predicate evaluation + stream compaction |
| SUM | ✅ | Parallel reduction |
| COUNT | ✅ | Parallel reduction |
| MIN/MAX | ✅ | Parallel reduction |
| GROUP BY | 🔄 | Hash-based grouping |
| Hash Join | 🔄 | Build + probe |
| Sort | 🔄 | Radix sort |

## Project Structure

```
src/
├── memory_pool.cuh      # Arena allocator for GPU memory
├── column.cuh           # Typed columnar arrays
├── primitives/
│   └── prefix_scan.cuh  # Blelloch scan algorithm
└── operators/
    ├── filter.cuh       # Stream compaction
    └── aggregate.cuh    # SUM, COUNT, MIN, MAX
```

## Running Individual Tests

```bash
python test_runner.py --list
python test_runner.py --test filter
python test_runner.py --test sum_large
```
