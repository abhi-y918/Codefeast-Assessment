#!/bin/bash
set -e

# ==============================================================================
# Build Script for Vulnerability Benchmark Ops
# Compiles the PyTorch C++ / CUDA extension.
# Requires: python, nvcc (CUDA toolkit), PyTorch installed.
# ==============================================================================

echo "--------------------------------------------------------"
if [ "$PATCH_APPLIED" = "1" ]; then
    echo " BUILDING SECURE (PATCHED) EXTENSION"
else
    echo " BUILDING VULNERABLE EXTENSION (WITH ASAN)"
fi
echo "--------------------------------------------------------"

# Ensure we're in the correct directory (project root)
cd "$(dirname "$0")/.."

# Clean up previous build artifacts to ensure a fresh compile
rm -rf build/ dist/ vulnerability_benchmark_ops_cuda.egg-info/

# Execute setup.py using pip or setuptools
# The setup.py script handles ASAN compiler flags dynamically
python setup.py install

echo ""
echo "[+] Build completed successfully."
echo "--------------------------------------------------------"
