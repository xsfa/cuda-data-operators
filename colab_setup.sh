#!/bin/bash
# Run this in Google Colab to set up the environment.

set -e

# Install RAPIDS (cuDF + CuPy)
pip install --quiet cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# Compile the CUDA kernel as a shared library
nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -shared predicate_sum.cu -o libpredicate_sum.so

echo "Setup complete. Run: python benchmark.py"
