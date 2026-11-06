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

## Build and run

Needs Visual Studio with the C++ tools, and an NVIDIA GPU.

    build.bat
    build\cuda_balls.exe            (24 balls)
    build\cuda_balls.exe 200        (any count from 1 to 500)
    build\cuda_balls.exe --selftest (headless physics check)

`build.bat` uses the toolkit in `CUDA_PATH` if that is set. Otherwise it uses `tools\cuda`, a copy of
the CUDA 13.5 toolkit unpacked there without running NVIDIA's installer.

`build.bat` compiles for RTX 40 series GPUs only (`-arch=sm_89`), because the 13.1 driver can't JIT-compile
PTX from the newer 13.5 toolkit. To run on another GPU, change that flag to match it, or update the driver
and remove the flag.
