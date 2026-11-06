# CUDA Balls

A prototype where a box of bouncing balls is simulated on the GPU with CUDA. You fly around them with
a WASD camera. The ball-to-ball and ball-to-wall forces all run in a CUDA kernel (`src/physics.cu`).
Each frame the positions are copied back to the CPU and drawn with OpenGL (`src/main.cpp`).

## Controls

| Input | Action |
| --- | --- |
| W A S D | Move the camera |
| Q / E | Move down / up |
| Hold right mouse | Look around |
| Space | Kick all the balls upward |
| R | Reset the balls |
| Esc | Quit |

## Build and run on Windows

Needs Visual Studio with the C++ tools, an NVIDIA GPU, and the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). The toolkit installer sets `CUDA_PATH`,
which `build.bat` uses.

    build.bat
    build\cuda_balls.exe                 (24 balls)
    build\cuda_balls.exe 200             (any count from 1 to 500)
    build\cuda_balls.exe --selftest      (headless physics check)
    build\cuda_balls.exe --bench         (one CPU core vs the GPU, see below)

If `CUDA_PATH` is not set, `build.bat` falls back to a toolkit unpacked into `tools\cuda`. That folder
is not in the repository: NVIDIA's licence only lets us redistribute the runtime libraries, not the
compiler.

### Which GPUs it runs on

`build.bat` builds ready-made machine code for every GPU generation that CUDA 13 supports: RTX 20,
GTX 16 and T4 (`sm_75`), A100 (`sm_80`), RTX 30 (`sm_86`), RTX 40 (`sm_89`), H100 (`sm_90`) and
RTX 50 (`sm_120`). Because nothing has to be JIT-compiled, a driver older than the toolkit still works,
such as a 13.1 driver with the 13.5 toolkit. GTX 10-series and V100 GPUs are not supported by CUDA 13;
they need a CUDA 12 toolkit.

## Benchmark: one CPU core vs the GPU

`--bench` runs the same substep for 100 to 20,000 balls, first on one CPU core and then on the GPU. Both
call the same `stepBall` function, so only the scheduling differs. It prints the time per substep, the
speedup, and whether each one keeps up with real time. The simulation needs 960 substeps per simulated
second, so a substep must take at most 1.04 ms. It also checks that the CPU and the GPU compute the
same result.

CPU times measured on a 2.1 GHz Intel Xeon (one core):

| Balls | Pair checks per substep | CPU ms per substep | Keeps up with real time? |
| --- | --- | --- | --- |
| 100 | 9,900 | 0.018 | yes |
| 500 | 249,500 | 0.55 | yes |
| 1,000 | 999,000 | 1.57 | no |
| 5,000 | 24,995,000 | 38 | no |
| 20,000 | 399,980,000 | 652 | no |

Every ball checks every other ball, so the work grows with the square of the ball count. One CPU core
keeps up to about 500 balls, which is why the interactive version's maximum doesn't show off the GPU.
Run `--bench` on your GPU to see where it pulls ahead.

## Headless on Linux or Google Colab

The window and the OpenGL drawing are Windows-only, but the physics and the benchmark run anywhere
CUDA does. In a Colab notebook with a T4 GPU runtime:

    !git clone https://github.com/pie-and-cat/cuda_gpu-tutorial
    %cd cuda_gpu-tutorial/cuda_balls
    !bash build.sh
    !./build/cuda_balls_bench
    !./build/cuda_balls_bench 1000 8000 30000    (your own ball counts)

`./build/cuda_balls_bench --cpu-only` times the CPU alone and needs no GPU.

## How the code works

| File | What it does |
| --- | --- |
| `src/physics.cu` | `stepBall` updates one ball for one substep. `stepKernel` runs it with one GPU thread per ball; `stepCpu` runs it in a loop on one core for the benchmark. |
| `src/main.cpp` | Window, camera, OpenGL drawing and the `--selftest` and `--bench` options (Windows) |
| `src/bench_main.cpp` | The benchmark on its own, for Linux and Colab |

The ideas from the workshop that it uses:

- **One thread per ball**, with the usual `if (i >= count) return;` for the threads past the end.
- **Double buffering.** Every ball reads the old positions and writes to a second buffer, which is then
  swapped in. If balls updated in place, some would see neighbours that had already moved this
  substep and others wouldn't.
- **Data stays on the GPU.** About 16 substeps run back to back on the GPU each frame, and only the
  positions come back for drawing. This is the opposite of the vector-addition example, where the
  copies cost more than the work.
