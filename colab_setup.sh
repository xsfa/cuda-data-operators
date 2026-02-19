#!/bin/bash
# Run this in Google Colab to set up the environment.

set -e

# Check for GPU runtime
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. In Colab, go to Runtime > Change runtime type > GPU"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

# Find nvcc
NVCC=$(find /usr -name "nvcc" 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc not found. Installing CUDA toolkit..."
    apt-get update && apt-get install -y cuda-toolkit-12-2
    NVCC=$(find /usr -name "nvcc" 2>/dev/null | head -1)
fi
echo "Using nvcc: $NVCC"

# Install cupy
pip install --quiet cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

echo "Setup complete."
