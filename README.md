# Getting started with CUDA

Teaching materials for a 30-minute session on installing CUDA and seeing where GPUs beat CPUs. The hands-on part
builds one project, CUDA Balls, a piece at a time in the Thread Lab game, then times it against one CPU core.

| Folder | What it is |
| --- | --- |
| `cuda_examples/` | Four examples (vector addition, matrix multiplication, image blur, summation), each with a sequential CPU version and a parallel CUDA version, timing and correctness checks. Build with `make`. |
| `notebooks/cuda_benchmarks.ipynb` | The same code as a notebook: compiles and benchmarks everything on a free Colab or Kaggle T4 GPU, then draws the charts. |
| `thread-lab/` | Thread Lab, a block-based browser game. The main track builds the CUDA Balls physics engine in five levels; a second track turns the four classic examples into levels. |
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

The **Build CUDA Balls** track is the session (about 3 minutes a level, 14 runs in testing):

| Level | What students build | What it teaches |
|---|---|---|
| 1 Move the balls | A kernel that moves each ball by its velocity | Thread indexing and the bounds check |
| 2 Gravity and walls | Load, forces, velocity, position, store | Each thread's own variables; a kernel launched every substep |
| 3 Collisions | A loop over every other ball | Races between blocks and double buffering |
| 4 The time loop | CPU code: launch, swap, copy back, draw | Host code and the cost of `cudaMemcpy` |
| 5 Total energy | The sum of every ball's energy | Race conditions and `atomicAdd` |

The **Classic examples** track (links `#classic-1` to `#classic-4`) covers vector addition, image blur
(2-D grids), a shared total (atomics) and tree reduction (shared memory and `__syncthreads()`).

## Run CUDA Balls

On Windows with an NVIDIA GPU: `cuda_balls\build.bat`, then `cuda_balls\build\cuda_balls.exe`.
On Linux or Colab, only the benchmark runs: `bash cuda_balls/build.sh && ./cuda_balls/build/cuda_balls_bench`.
See `cuda_balls/README.md`.
