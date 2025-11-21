#!/bin/bash
#
# Wrapper script to run Python scripts with proper CUDA library paths
# This ensures onnxruntime-gpu can find CUDA 12 libraries from pip packages
#
# Usage: ./run_with_cuda.sh python script.py [args...]
#

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Build LD_LIBRARY_PATH with NVIDIA CUDA libraries from pip packages
NVIDIA_LIBS=$(python3 << 'PYEOF'
import os
import sys

# Base path for nvidia packages
base_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")

# List of library directories to add
lib_dirs = [
    "cublas/lib",
    "cudnn/lib",
    "cuda_runtime/lib",
    "cuda_nvrtc/lib",
    "curand/lib",
    "cufft/lib",
    "cusparse/lib",
    "cusolver/lib",
    "nvjitlink/lib",
]

# Collect existing directories
paths = []
for lib_dir in lib_dirs:
    full_path = os.path.join(base_path, lib_dir)
    if os.path.exists(full_path):
        paths.append(full_path)

# Print colon-separated paths
print(":".join(paths))
PYEOF
)

# Check if we found any NVIDIA libraries
if [ -z "$NVIDIA_LIBS" ]; then
    echo "Warning: Could not find NVIDIA CUDA libraries in virtual environment"
    echo "Expected location: $VIRTUAL_ENV/lib/python*/site-packages/nvidia/"
    exit 1
fi

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"

echo "=================================================="
echo "Running with CUDA library path configured"
echo "=================================================="
echo "NVIDIA Libraries added to LD_LIBRARY_PATH:"
echo "$NVIDIA_LIBS" | tr ':' '\n' | sed 's/^/  - /'
echo ""
echo "Command: $@"
echo "=================================================="
echo ""

# Execute the command
exec "$@"
