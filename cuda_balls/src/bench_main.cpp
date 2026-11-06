// Headless benchmark for any OS, including Linux and Google Colab: no window, no OpenGL.
// Times the same physics substep on one CPU core and on the GPU (see physicsBenchmark).
//
//   cuda_balls_bench                  default ball counts
//   cuda_balls_bench 1000 8000        your own counts
//   cuda_balls_bench --cpu-only       CPU timings only, no GPU needed
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "physics.h"

int main(int argc, char** argv) {
    std::vector<int> counts;
    bool cpuOnly = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpu-only") == 0) {
            cpuOnly = true;
            continue;
        }
        int n = std::atoi(argv[i]);
        if (n < 2 || n > 200000) {
            std::fprintf(stderr, "Usage: cuda_balls_bench [ball counts, 2-200000 ...] [--cpu-only]\n");
            return 1;
        }
        counts.push_back(n);
    }
    if (counts.empty()) counts = {100, 500, 1000, 2000, 5000, 10000, 20000};
    return physicsBenchmark(counts.data(), (int)counts.size(), cpuOnly);
}
