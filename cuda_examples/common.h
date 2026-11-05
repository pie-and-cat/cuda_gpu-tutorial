// common.h - small helpers shared by the four CUDA tutorial examples.
//
// Every example prints one CSV row per measurement:
//   example,variant,size,time_ms,throughput,unit,error,check
// Lines that start with '#' are comments (for example, which GPU was used).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Abort with a readable message if a CUDA API call fails.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in: %s\n",        \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// A kernel launch returns no error code, so ask for the last error right after it.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Wall-clock timer for CPU code.
struct CpuTimer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

// GPU timer built on CUDA events. Kernel launches return immediately, so a CPU
// timer would only measure the launch. Events are timestamped by the GPU itself.
struct GpuTimer {
    cudaEvent_t begin{}, end{};
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }
    ~GpuTimer() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    void start() { CUDA_CHECK(cudaEventRecord(begin)); }
    double stop_ms() {
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));  // wait until the GPU reaches 'end'
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    }
};

// Call fn() (which returns elapsed milliseconds) once as a warm-up, then `reps`
// more times, and return the median. A first run slower than `slow_ms` is
// returned as is, so multi-second CPU runs are not repeated.
template <class F>
double median_ms(F fn, int reps = 5, double slow_ms = 2000.0) {
    const double first = fn();
    if (first > slow_ms) return first;
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(fn());
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Fill a vector with reproducible random numbers in [lo, hi).
inline void fill_random(std::vector<float>& v, float lo, float hi, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) x = dist(rng);
}

inline void print_csv_header() {
    std::printf("example,variant,size,time_ms,throughput,unit,error,check\n");
}

inline void print_row(const char* example, const char* variant, long long size,
                      double ms, double throughput, const char* unit,
                      double error, const char* check) {
    std::printf("%s,%s,%lld,%.4f,%.3f,%s,%.3e,%s\n", example, variant, size, ms,
                throughput, unit, error, check);
    std::fflush(stdout);
}

inline void print_device_info() {
    int dev = 0;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("# GPU: %s | compute capability %d.%d | %d SMs | %.1f GB\n",
                p.name, p.major, p.minor, p.multiProcessorCount,
                p.totalGlobalMem / 1e9);
}

// --- tiny command-line helpers ---------------------------------------------

inline bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// Value after an option such as "--radius 4"; `fallback` if it is absent.
inline const char* option_value(int argc, char** argv, const char* opt,
                                const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], opt) == 0) return argv[i + 1];
    return fallback;
}

// Positional numbers (problem sizes). Options that take a value are skipped,
// along with their value; "--cpu-only" takes no value.
inline std::vector<long long> sizes_from_args(int argc, char** argv,
                                              std::vector<long long> defaults) {
    std::vector<long long> sizes;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-' && argv[i][1] == '-') {
            if (std::strcmp(argv[i], "--cpu-only") != 0) ++i;  // skip its value
            continue;
        }
        sizes.push_back(std::atoll(argv[i]));
    }
    return sizes.empty() ? defaults : sizes;
}
