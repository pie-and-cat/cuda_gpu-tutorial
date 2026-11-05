// reduction.cu - sum of n floats
//
// Summation is the classic example of a problem that is NOT naturally parallel:
// every addition depends on the running total. This file shows four ways:
//
// CPU:              one running total, n additions in a row.
// GPU racy:         every thread does `*total += x[i]`. Fast and WRONG: a race condition.
// GPU atomic:       every thread does atomicAdd(total, x[i]). Correct, but the additions
//                   to one memory address are serialized, so it is slow.
// GPU shared tree:  each block sums its share in shared memory as a tree of pairwise
//                   additions; one atomicAdd per block. Correct and fast.
//
// The reference answer is computed in double precision (not timed).
//
// Build: nvcc -O3 -std=c++17 -arch=native reduction.cu -o reduction
// Run:   ./reduction              (default sizes)
//        ./reduction 100000000    (your own size)
#include "common.h"

constexpr int BLOCK = 256;       // threads per block (a power of two for the tree)
constexpr int MAX_BLOCKS = 1024; // the shared-tree kernel loops if n needs more

// ---- CPU ---------------------------------------------------------------------
float sum_cpu(const float* x, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) total += x[i];
    return total;
}

double sum_reference(const float* x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += x[i];
    return total;
}

// ---- GPU: racy (wrong on purpose) ------------------------------------------------
// Thousands of threads read the same old value of *total, add to it, and write it
// back, overwriting each other's updates. Most additions are lost.
__global__ void sum_racy(const float* x, float* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *total += x[i];
}

// ---- GPU: one atomic per element -----------------------------------------------
// atomicAdd makes each read-add-write indivisible, so no update is lost. But all n
// threads queue for the same address.
__global__ void sum_atomic(const float* x, float* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(total, x[i]);
}

// ---- GPU: shared-memory tree reduction -----------------------------------------
//  1. Each thread adds up a strided slice of x in a private register (grid-stride loop).
//  2. The block's 256 partial sums go into shared memory.
//  3. Pairwise tree: 256 -> 128 -> 64 -> ... -> 1, halving the active threads each step.
//  4. Thread 0 adds the block's total to the global result: one atomic per block.
__global__ void sum_shared_tree(const float* x, float* total, int n) {
    __shared__ float partial[BLOCK];

    float local = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += BLOCK * gridDim.x)
        local += x[i];
    partial[threadIdx.x] = local;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();  // finish this level of the tree before starting the next
    }
    if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

static void run(int n, bool cpu_only) {
    const size_t bytes = size_t(n) * sizeof(float);
    const double gb = bytes / 1e9;
    const double tol = 1e-3;  // allowed relative error

    std::vector<float> x(n);
    fill_random(x, 0.0f, 1.0f, 5);
    const double ref = sum_reference(x.data(), n);
    auto rel = [ref](double v) { return std::fabs(v - ref) / std::fabs(ref); };

    // ---- CPU -------------------------------------------------------------
    float cpu_result = 0.0f;
    const double cpu_ms = median_ms([&] {
        CpuTimer t;
        t.start();
        cpu_result = sum_cpu(x.data(), n);
        return t.stop_ms();
    });
    const double cpu_err = rel(cpu_result);
    print_row("reduction", "cpu_sequential", n, cpu_ms, gb / (cpu_ms / 1e3), "GB/s",
              cpu_err, cpu_err <= tol ? "PASS" : "INACCURATE");
    if (cpu_only) return;

    // ---- GPU -------------------------------------------------------------
    float *d_x = nullptr, *d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));

    const int one_per_thread_blocks = (n + BLOCK - 1) / BLOCK;
    const int tree_blocks = std::min(one_per_thread_blocks, MAX_BLOCKS);

    // Times one variant (reset total + kernel) and reports its result.
    auto time_variant = [&](const char* variant, auto launch, const char* bad_label) {
        const double ms = median_ms([&] {
            GpuTimer t;
            t.start();
            CUDA_CHECK(cudaMemsetAsync(d_total, 0, sizeof(float)));
            launch();
            CUDA_CHECK_LAUNCH();
            return t.stop_ms();
        });
        float result = 0.0f;
        CUDA_CHECK(cudaMemcpy(&result, d_total, sizeof(float), cudaMemcpyDeviceToHost));
        const double err = rel(result);
        print_row("reduction", variant, n, ms, gb / (ms / 1e3), "GB/s", err,
                  err <= tol ? "PASS" : bad_label);
    };

    time_variant("gpu_racy_kernel", [&] {
        sum_racy<<<one_per_thread_blocks, BLOCK>>>(d_x, d_total, n);
    }, "WRONG_RACE");
    time_variant("gpu_atomic_kernel", [&] {
        sum_atomic<<<one_per_thread_blocks, BLOCK>>>(d_x, d_total, n);
    }, "INACCURATE");
    time_variant("gpu_shared_tree_kernel", [&] {
        sum_shared_tree<<<tree_blocks, BLOCK>>>(d_x, d_total, n);
    }, "INACCURATE");

    // Shared tree again, counting the copy of x to the GPU and the result back.
    float result = 0.0f;
    const double total_ms = median_ms([&] {
        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_total, 0, sizeof(float)));
        sum_shared_tree<<<tree_blocks, BLOCK>>>(d_x, d_total, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(&result, d_total, sizeof(float), cudaMemcpyDeviceToHost));
        return t.stop_ms();
    });
    const double err = rel(result);
    print_row("reduction", "gpu_shared_tree_incl_transfers", n, total_ms,
              gb / (total_ms / 1e3), "GB/s", err, err <= tol ? "PASS" : "INACCURATE");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_total));
}

int main(int argc, char** argv) {
    const bool cpu_only = has_flag(argc, argv, "--cpu-only");
    if (!cpu_only) print_device_info();
    const auto sizes =
        sizes_from_args(argc, argv, {1 << 10, 1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26});
    print_csv_header();
    for (long long n : sizes) run(int(n), cpu_only);
    return 0;
}
