// matmul.cu - C = A x B for square n x n matrices of floats
//
// CPU:  triple loop on one core (i-k-j order, the cache-friendly one).
// GPU1: naive kernel, one thread per output element, reads straight from global memory.
// GPU2: tiled kernel, threads in a block share 16 x 16 tiles through fast shared memory.
// GPU3: cuBLAS, NVIDIA's tuned library, as the "what experts get" reference.
//
// Build: nvcc -O3 -std=c++17 -arch=native matmul.cu -o matmul -lcublas
// Run:   ./matmul               (default sizes)
//        ./matmul 1024 2048     (your own sizes)
#include "common.h"

#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st_ = (call);                                         \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_),     \
                         __FILE__, __LINE__);                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int TILE = 16;  // 16 x 16 = 256 threads per block

// ---- CPU: 2 n^3 floating-point operations, one after another ----------------
void matmul_cpu(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + size_t(n) * n, 0.0f);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const float a = A[size_t(i) * n + k];
            for (int j = 0; j < n; ++j)  // walks rows of B and C contiguously
                C[size_t(i) * n + j] += a * B[size_t(k) * n + j];
        }
}

// ---- GPU 1: naive -----------------------------------------------------------
// Thread (row, col) computes one dot product. Every operand comes from global
// memory, and neighbouring threads re-read the same values many times.
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

// ---- GPU 2: tiled with shared memory ----------------------------------------
// The block walks across A's row band and down B's column band one TILE x TILE
// tile at a time. Each thread loads ONE element of each tile into shared memory,
// the block waits (__syncthreads), then every thread reuses the whole tile.
// Global-memory traffic drops by a factor of TILE.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + tx;  // column of A this thread loads
        const int b_row = t * TILE + ty;  // row of B this thread loads
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();  // the whole tile is loaded before anyone uses it

        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();  // everyone is done with the tile before it is overwritten
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

// Largest |gpu - cpu|, relative to the largest |cpu| value. Float sums done in a
// different order differ slightly, so we allow a small tolerance.
static double rel_error(const std::vector<float>& gpu, const std::vector<float>& cpu) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        max_diff = std::max(max_diff, (double)std::fabs(gpu[i] - cpu[i]));
        max_ref = std::max(max_ref, (double)std::fabs(cpu[i]));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

static void run(int n, bool cpu_only, cublasHandle_t handle) {
    const size_t elems = size_t(n) * n, bytes = elems * sizeof(float);
    const double gflop = 2.0 * double(n) * n * n / 1e9;
    const double tol = 1e-3;

    std::vector<float> A(elems), B(elems), C_cpu(elems), C_gpu(elems);
    fill_random(A, -1.0f, 1.0f, 3);
    fill_random(B, -1.0f, 1.0f, 4);

    // ---- CPU -------------------------------------------------------------
    const double cpu_ms = median_ms([&] {
        CpuTimer t;
        t.start();
        matmul_cpu(A.data(), B.data(), C_cpu.data(), n);
        return t.stop_ms();
    }, 3);
    print_row("matmul", "cpu_sequential", n, cpu_ms, gflop / (cpu_ms / 1e3), "GFLOP/s",
              0.0, "PASS");
    if (cpu_only) return;

    // ---- GPU -------------------------------------------------------------
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto report = [&](const char* variant, double ms) {
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
        const double err = rel_error(C_gpu, C_cpu);
        print_row("matmul", variant, n, ms, gflop / (ms / 1e3), "GFLOP/s", err,
                  err <= tol ? "PASS" : "FAIL");
    };

    // GPU 1: naive
    double ms = median_ms([&] {
        GpuTimer t;
        t.start();
        matmul_naive<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK_LAUNCH();
        return t.stop_ms();
    });
    report("gpu_naive_kernel", ms);

    // GPU 2: tiled
    ms = median_ms([&] {
        GpuTimer t;
        t.start();
        matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK_LAUNCH();
        return t.stop_ms();
    });
    report("gpu_tiled_kernel", ms);

    // GPU 3: cuBLAS. cuBLAS assumes column-major storage; computing B x A in
    // column-major order gives exactly A x B in our row-major layout.
    const float alpha = 1.0f, beta = 0.0f;
    ms = median_ms([&] {
        GpuTimer t;
        t.start();
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                 dB, n, dA, n, &beta, dC, n));
        return t.stop_ms();
    });
    report("gpu_cublas", ms);

    // Tiled kernel again, this time counting the copies in and out.
    ms = median_ms([&] {
        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
        matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
        return t.stop_ms();
    });
    report("gpu_tiled_incl_transfers", ms);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(int argc, char** argv) {
    const bool cpu_only = has_flag(argc, argv, "--cpu-only");
    cublasHandle_t handle = nullptr;
    if (!cpu_only) {
        print_device_info();
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    const auto sizes = sizes_from_args(argc, argv, {128, 256, 512, 1024, 2048});
    print_csv_header();
    for (long long n : sizes) run(int(n), cpu_only, handle);
    if (handle) cublasDestroy(handle);
    return 0;
}
