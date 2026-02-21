#!/bin/bash
# Run this in Google Colab or EC2 to set up the environment.

set -e

# Check for GPU runtime
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. In Colab, go to Runtime > Change runtime type > GPU"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

# Find nvcc
NVCC=$(find /usr -name "nvcc" -type f 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc not found. Installing CUDA toolkit..."
    apt-get update && apt-get install -y cuda-toolkit-12-2
    NVCC=$(find /usr -name "nvcc" -type f 2>/dev/null | head -1)
fi
echo "Using nvcc: $NVCC"

# Install cupy
pip install --quiet cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# Compile CUDA library
echo "Compiling libdataops.so..."
$NVCC -O3 -arch=sm_75 -Xcompiler -fPIC -shared src/lib.cu -o libdataops.so
echo "Compiled: libdataops.so"

echo "Setup complete. Run: python test_runner.py"
