# CUDA examples: sequential CPU vs parallel GPU

Four classic problems, each written twice (a CPU loop and a CUDA kernel), timed, and checked against each other.

| File | What it shows |
| --- | --- |
| `vector_add.cu` | One thread per element; why copies over PCIe can cancel the GPU's advantage |
| `matmul.cu` | CPU triple loop vs naive kernel vs shared-memory tiled kernel vs cuBLAS |
| `image_blur.cu` | 2-D grid, one thread per pixel; the same `__host__ __device__` function on both sides |
| `reduction.cu` | A race condition, atomics, and a shared-memory tree reduction (plus a float-accuracy surprise) |
| `common.h` | Error-check macros, CPU and GPU (CUDA event) timers, CSV output |

## Build and run

```bash
make                     # uses -arch=native (needs a GPU in this machine)
make ARCH=-arch=sm_75    # T4 (Colab, Kaggle)
make ARCH=-arch=sm_80    # A100 (LSU SuperMike-III)
./run_all.sh             # runs everything, writes results.csv
./run_all.sh --cpu-only  # CPU timings only, no GPU needed
```

Every program accepts its own problem sizes, for example `./matmul 512 4096` or `./reduction 100000000`.
`image_blur` also takes `--radius R`, `--in photo.pgm` (binary PGM) and `--out prefix`.

## Output

One CSV row per measurement: `example,variant,size,time_ms,throughput,unit,error,check`.

- `time_ms` is the median of 5 runs after one warm-up (CPU runs over 2 s are timed once).
- `gpu_*_kernel` rows assume the data is already on the GPU; `*_incl_transfers` rows include the copies in and out.
- `check` is `PASS` when the GPU result matches the CPU result. `WRONG_RACE` (the racy sum) and `INACCURATE` (the CPU float sum at 67 M values) are expected.

The notebook `cuda_benchmarks.ipynb` contains the same code and adds tables and charts.
