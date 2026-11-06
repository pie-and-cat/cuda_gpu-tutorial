# Getting started with CUDA

Teaching materials for a short session on installing CUDA and seeing where GPUs beat CPUs.

| Folder | What it is |
| --- | --- |
| `cuda_examples/` | Four examples (vector addition, matrix multiplication, image blur, summation), each with a sequential CPU version and a parallel CUDA version, timing and correctness checks. Build with `make`. |
| `notebooks/cuda_benchmarks.ipynb` | The same code as a notebook: compiles and benchmarks everything on a free Colab or Kaggle T4 GPU, then draws the charts. |
| `thread-lab/` | Thread Lab, a block-based browser game that teaches thread indexing, 2-D grids, race conditions, atomics and `__syncthreads()`, then has students build the CUDA Balls physics engine: the kernel (level 5) and the CPU-side time loop (level 6). |
| `cuda_balls/` | CUDA Balls, a 3D box of bouncing balls simulated on the GPU (Windows), with a benchmark of one CPU core against the GPU that also runs on Linux and Colab. |

## Run the benchmarks in Colab

Open `notebooks/cuda_benchmarks.ipynb` in Colab (File → Open notebook → GitHub, or use
`https://colab.research.google.com/github/pie-and-cat/cuda_gpu-tutorial/blob/main/notebooks/cuda_benchmarks.ipynb`),
choose Runtime → Change runtime type → T4 GPU, then Run all.

## Run the examples locally

```bash
cd cuda_examples
make                      # or: make ARCH=-arch=sm_75 (T4), make ARCH=-arch=sm_80 (A100)
./run_all.sh              # writes results.csv
```

## Play Thread Lab

Open `thread-lab/index.html` in a browser, or enable GitHub Pages for this repository and visit
`https://pie-and-cat.github.io/cuda_gpu-tutorial/thread-lab/`.

| Level | What students build | What it teaches |
|---|---|---|
| 1 | Vector addition | Thread indexing and the bounds check |
| 2 | Image blur | 2-D grids and blocks |
| 3 | A shared total | Race conditions and `atomicAdd` |
| 4 | Tree reduction | Shared memory and `__syncthreads()` |
| 5 | The CUDA Balls physics kernel | One thread per ball, forces, and double buffering (writing in place is a race) |
| 6 | The frame loop on the CPU | Launch and swap 16 times per frame, then one `cudaMemcpy` back and one draw |

## Run CUDA Balls

On Windows with an NVIDIA GPU: `cuda_balls\build.bat`, then `cuda_balls\build\cuda_balls.exe`.
On Linux or Colab, only the benchmark runs: `bash cuda_balls/build.sh && ./cuda_balls/build/cuda_balls_bench`.
See `cuda_balls/README.md`.
