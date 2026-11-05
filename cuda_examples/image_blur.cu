// image_blur.cu - box blur of a grayscale image
//
// Every output pixel is the average of the (2r+1) x (2r+1) square of input pixels
// around it (r = radius). Edges are handled by clamping to the nearest pixel.
//
// CPU: two nested loops visit the pixels one at a time.
// GPU: a 2-D grid of 16 x 16 thread blocks, one thread per pixel.
// Both call the SAME blur_pixel() function: only the scheduling changes.
//
// Build: nvcc -O3 -std=c++17 -arch=native image_blur.cu -o image_blur
// Run:   ./image_blur                               (synthetic images, default sizes)
//        ./image_blur --radius 8 1024 4096          (your own radius and sizes)
//        ./image_blur --in photo.pgm --out blurred  (a real image; writes
//                                                    blurred_cpu.pgm and blurred_gpu.pgm)
#include "common.h"

#include <fstream>

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> px;  // row-major, one byte per pixel
};

// ---- PGM (portable graymap) input/output -------------------------------------
static bool read_pgm(const std::string& path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P5") return false;
    auto next_int = [&f]() {  // read an integer, skipping '#' comment lines
        int value = 0;
        while (f >> std::ws && f.peek() == '#') f.ignore(1 << 20, '\n');
        f >> value;
        return value;
    };
    img.w = next_int();
    img.h = next_int();
    const int maxval = next_int();
    f.get();  // the single whitespace byte before the pixel data
    if (img.w <= 0 || img.h <= 0 || maxval != 255) return false;
    img.px.resize(size_t(img.w) * img.h);
    f.read(reinterpret_cast<char*>(img.px.data()), img.px.size());
    return bool(f);
}

static bool write_pgm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << img.w << " " << img.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.px.data()), img.px.size());
    return bool(f);
}

// A synthetic test image: smooth gradients, rings and some pixel noise.
static Image make_test_image(int w, int h) {
    Image img;
    img.w = w;
    img.h = h;
    img.px.resize(size_t(w) * h);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> noise(-40, 40);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double dx = x - w / 2.0, dy = y - h / 2.0;
            const double ring = 0.5 + 0.5 * std::cos(std::sqrt(dx * dx + dy * dy) / (w / 40.0));
            int v = int(80.0 * x / w + 60.0 * y / h + 110.0 * ring) + noise(rng);
            img.px[size_t(y) * w + x] = (unsigned char)std::min(255, std::max(0, v));
        }
    return img;
}

// ---- the per-pixel work, shared by CPU and GPU ----------------------------------
__host__ __device__ inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline unsigned char blur_pixel(const unsigned char* in, int w, int h,
                                                    int x, int y, int r) {
    int sum = 0;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clamp_int(y + dy, 0, h - 1);
        for (int dx = -r; dx <= r; ++dx) {
            const int xx = clamp_int(x + dx, 0, w - 1);
            sum += in[yy * w + xx];
        }
    }
    const int area = (2 * r + 1) * (2 * r + 1);
    return (unsigned char)((sum + area / 2) / area);  // rounded average
}

// CPU: visit every pixel in turn.
void box_blur_cpu(const unsigned char* in, unsigned char* out, int w, int h, int r) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) out[y * w + x] = blur_pixel(in, w, h, x, y, r);
}

// GPU: each thread finds its own (x, y) and computes that one pixel.
__global__ void box_blur_gpu(const unsigned char* in, unsigned char* out, int w, int h,
                             int r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y * w + x] = blur_pixel(in, w, h, x, y, r);
}

static void run(const Image& input, int r, bool cpu_only, const char* out_prefix) {
    const int w = input.w, h = input.h;
    const size_t bytes = size_t(w) * h;
    const long long pixels = (long long)w * h;
    const double mpix = pixels / 1e6;

    Image out_cpu = input, out_gpu = input;

    // ---- CPU -------------------------------------------------------------
    const double cpu_ms = median_ms([&] {
        CpuTimer t;
        t.start();
        box_blur_cpu(input.px.data(), out_cpu.px.data(), w, h, r);
        return t.stop_ms();
    }, 3);
    print_row("image_blur", "cpu_sequential", pixels, cpu_ms, mpix / (cpu_ms / 1e3),
              "Mpixel/s", 0.0, "PASS");
    if (out_prefix) write_pgm(std::string(out_prefix) + "_cpu.pgm", out_cpu);
    if (cpu_only) return;

    // ---- GPU -------------------------------------------------------------
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, input.px.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    const double kernel_ms = median_ms([&] {
        GpuTimer t;
        t.start();
        box_blur_gpu<<<grid, block>>>(d_in, d_out, w, h, r);
        CUDA_CHECK_LAUNCH();
        return t.stop_ms();
    });

    const double total_ms = median_ms([&] {
        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(d_in, input.px.data(), bytes, cudaMemcpyHostToDevice));
        box_blur_gpu<<<grid, block>>>(d_in, d_out, w, h, r);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(out_gpu.px.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        return t.stop_ms();
    });

    // Integer arithmetic: the two images must be identical, pixel for pixel.
    int max_diff = 0;
    for (size_t i = 0; i < bytes; ++i)
        max_diff = std::max(max_diff, std::abs(int(out_gpu.px[i]) - int(out_cpu.px[i])));
    const char* check = max_diff == 0 ? "PASS" : "FAIL";

    print_row("image_blur", "gpu_kernel", pixels, kernel_ms, mpix / (kernel_ms / 1e3),
              "Mpixel/s", max_diff, check);
    print_row("image_blur", "gpu_incl_transfers", pixels, total_ms,
              mpix / (total_ms / 1e3), "Mpixel/s", max_diff, check);
    if (out_prefix) write_pgm(std::string(out_prefix) + "_gpu.pgm", out_gpu);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char** argv) {
    const bool cpu_only = has_flag(argc, argv, "--cpu-only");
    const int r = std::atoi(option_value(argc, argv, "--radius", "4"));
    const char* in_path = option_value(argc, argv, "--in", nullptr);
    const char* out_prefix = option_value(argc, argv, "--out", nullptr);

    if (!cpu_only) print_device_info();
    std::printf("# radius %d: each pixel averages %d neighbours\n", r,
                (2 * r + 1) * (2 * r + 1));
    print_csv_header();

    if (in_path) {  // one real image
        Image img;
        if (!read_pgm(in_path, img)) {
            std::fprintf(stderr, "Could not read %s (expected binary PGM, P5)\n", in_path);
            return EXIT_FAILURE;
        }
        run(img, r, cpu_only, out_prefix);
        return 0;
    }
    const auto sides = sizes_from_args(argc, argv, {256, 512, 1024, 2048, 4096});
    for (long long s : sides) run(make_test_image(int(s), int(s)), r, cpu_only, out_prefix);
    return 0;
}
