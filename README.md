# cuda-data-operators

GPU-native data operators for SQL-like operations directly on GPU memory.

## Quick Start (Google Colab)

```python
# 1. Clone and setup
!git clone https://github.com/xsfa/cuda-data-operators.git
%cd cuda-data-operators

# 2. Install dependencies
!pip install --quiet cupy-cuda12x

# 3. Compile and test
!python test_runner.py --setup
!python test_runner.py
```

**Requirements**: GPU runtime (Runtime → Change runtime type → T4 GPU)

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
# List tests
!python test_runner.py --list

# Run specific test
!python test_runner.py --test filter
!python test_runner.py --test sum_large
```

## Benchmarking vs cuDF

```bash
!bash colab_setup.sh
!python benchmark.py
```


The goal: execute analytical queries without CPU round-trips, keeping data in HBM alongside model weights.
