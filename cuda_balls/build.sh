#!/usr/bin/env bash
# Builds the headless benchmark (no window, no OpenGL) on Linux, including Google Colab.
# The interactive 3D version is Windows-only: use build.bat there.
#
#   bash build.sh                      compile for the GPU in this machine
#   ARCH=-arch=sm_75 bash build.sh     compile for a T4 (sm_80 for an A100)
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v nvcc >/dev/null; then
    echo "nvcc not found. Install the CUDA Toolkit, or use a Colab or Kaggle GPU runtime." >&2
    exit 1
fi
ARCH="${ARCH:--arch=native}"
mkdir -p build
nvcc -O2 -std=c++17 $ARCH -I src src/bench_main.cpp src/physics.cu -o build/cuda_balls_bench
echo "Built build/cuda_balls_bench"
