// vector_add.cu - c[i] = a[i] + b[i]
//
// CPU: one core walks the array from start to end.
// GPU: one thread per element; all of them run at once.
//
// Build: nvcc -O3 -std=c++17 -arch=native vector_add.cu -o vector_add
// Run:   ./vector_add                 (default sizes)
//        ./vector_add 1000000 50000000 (your own sizes)
#include "common.h"

// Sequential version: a plain loop.
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// Parallel version: the loop is gone. Each thread works out which element it
// owns from its block number and its position inside the block.
__global__ void vector_add_gpu(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) c[i] = a[i] + b[i];                  // the last block may overhang n
}

static void run(int n, bool cpu_only) {
    const size_t bytes = size_t(n) * sizeof(float);
    const double gb_moved = 3.0 * bytes / 1e9;  // read a, read b, write c

    std::vector<float> a(n), b(n), c_cpu(n), c_gpu(n);
    fill_random(a, -1.0f, 1.0f, 1);
    fill_random(b, -1.0f, 1.0f, 2);

    // ---- CPU -------------------------------------------------------------
    const double cpu_ms = median_ms([&] {
        CpuTimer t;
        t.start();
        vector_add_cpu(a.data(), b.data(), c_cpu.data(), n);
        return t.stop_ms();
    });
    print_row("vector_add", "cpu_sequential", n, cpu_ms, gb_moved / (cpu_ms / 1e3),
              "GB/s", 0.0, "PASS");
    if (cpu_only) return;

    // ---- GPU -------------------------------------------------------------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));  // memory on the GPU
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    const int threads = 256;                           // threads per block
    const int blocks = (n + threads - 1) / threads;    // enough blocks to cover n

    // (1) Kernel only: the inputs are already in GPU memory.
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    const double kernel_ms = median_ms([&] {
        GpuTimer t;
        t.start();
        vector_add_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK_LAUNCH();
        return t.stop_ms();
    });

    // (2) End to end: copy inputs over PCIe, compute, copy the result back.
    const double total_ms = median_ms([&] {
        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        vector_add_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        return t.stop_ms();
    });

    // Both sides perform the identical float addition, so results must match exactly.
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        err = std::max(err, (double)std::fabs(c_gpu[i] - c_cpu[i]));
    const char* check = (err == 0.0) ? "PASS" : "FAIL";

    print_row("vector_add", "gpu_kernel", n, kernel_ms, gb_moved / (kernel_ms / 1e3),
              "GB/s", err, check);
    print_row("vector_add", "gpu_incl_transfers", n, total_ms,
              gb_moved / (total_ms / 1e3), "GB/s", err, check);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main(int argc, char** argv) {
    const bool cpu_only = has_flag(argc, argv, "--cpu-only");
    if (!cpu_only) print_device_info();
    const auto sizes = sizes_from_args(
        argc, argv, {1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26});
    print_csv_header();
    for (long long n : sizes) run(int(n), cpu_only);
    return 0;
}
