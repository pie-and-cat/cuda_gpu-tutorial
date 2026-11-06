// Ball physics, run entirely on the GPU.
//
// Balls are soft spheres: overlapping balls push apart with a spring and lose energy through a
// damper. Every ball's new state depends only on the previous state of all balls, so each one can
// be computed by its own GPU thread. Results go to a second buffer, which is then swapped in.
#include "physics.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr float kGravity = 9.81f;
constexpr float kContactStiffness = 4000.0f;  // N/m
constexpr float kContactDamping = 0.11f;      // damping ratio; about 0.7 restitution
constexpr float kFriction = 0.25f;            // fraction of the contact force resisting sliding
constexpr float kDensity = 1.0f;
constexpr float kSubstep = 1.0f / 960.0f;  // small enough that the stiff springs stay stable
constexpr float kAirDrag = 0.05f;
constexpr float kMaxSpeed = 40.0f;  // keeps a ball from tunnelling through a wall in one substep
constexpr int kThreadsPerBlock = 64;

// A failed CUDA call leaves the simulation state meaningless, so stop rather than carry on.
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err_));                                           \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Force on a body from a contact. `normal` points from the surface towards the body, `overlap` is
// the penetration depth, and `relVel` is the body's velocity relative to the surface.
__host__ __device__ inline float3 contactForce(float3 normal, float overlap, float3 relVel, float mass) {
    float normalSpeed = dot(relVel, normal);
    float damping = 2.0f * kContactDamping * sqrtf(kContactStiffness * mass);
    float pushing = kContactStiffness * overlap - damping * normalSpeed;
    if (pushing <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);  // contacts push, never pull

    float3 force = normal * pushing;

    // Simple friction: oppose sliding, limited so that it fades out as the ball stops sliding.
    float3 sliding = relVel - normal * normalSpeed;
    float slidingSpeed = sqrtf(dot(sliding, sliding));
    if (slidingSpeed > 1e-4f) {
        float limit = fminf(kFriction * pushing, mass * slidingSpeed / kSubstep);
        force = force - sliding * (limit / slidingSpeed);
    }
    return force;
}

// One substep for ball i. Positions are (x, y, z, radius) and velocities are (x, y, z, mass).
// __host__ __device__: the GPU kernel and the CPU comparison loop run exactly this code.
__host__ __device__ inline void stepBall(int i, const float4* pos, const float4* vel,
                                         float4* nextPos, float4* nextVel, int count,
                                         WorldBounds bounds) {
    float3 p = make_float3(pos[i].x, pos[i].y, pos[i].z);
    float3 v = make_float3(vel[i].x, vel[i].y, vel[i].z);
    float radius = pos[i].w;
    float mass = vel[i].w;

    float3 force = make_float3(0.0f, -kGravity * mass, 0.0f);
    force = force - v * (kAirDrag * mass);

    // Ball against every other ball. O(n^2), which is fine for this many balls.
    for (int j = 0; j < count; ++j) {
        if (j == i) continue;
        float3 delta = p - make_float3(pos[j].x, pos[j].y, pos[j].z);
        float reach = radius + pos[j].w;
        float distSq = dot(delta, delta);
        if (distSq >= reach * reach || distSq < 1e-10f) continue;

        float dist = sqrtf(distSq);
        float3 relVel = v - make_float3(vel[j].x, vel[j].y, vel[j].z);
        float reducedMass = mass * vel[j].w / (mass + vel[j].w);
        force = force + contactForce(delta * (1.0f / dist), reach - dist, relVel, reducedMass);
    }

    // Ball against the six faces of the box.
    const float3 normals[6] = {make_float3(0, 1, 0),  make_float3(0, -1, 0), make_float3(1, 0, 0),
                               make_float3(-1, 0, 0), make_float3(0, 0, 1),  make_float3(0, 0, -1)};
    const float gaps[6] = {p.y,
                           bounds.height - p.y,
                           p.x + bounds.halfWidth,
                           bounds.halfWidth - p.x,
                           p.z + bounds.halfWidth,
                           bounds.halfWidth - p.z};
    for (int f = 0; f < 6; ++f) {
        float overlap = radius - gaps[f];
        if (overlap > 0.0f) force = force + contactForce(normals[f], overlap, v, mass);
    }

    // Semi-implicit Euler: update the velocity first, then move using the new velocity.
    v = v + force * (kSubstep / mass);
    float speed = sqrtf(dot(v, v));
    if (speed > kMaxSpeed) v = v * (kMaxSpeed / speed);
    p = p + v * kSubstep;

    nextPos[i] = make_float4(p.x, p.y, p.z, radius);
    nextVel[i] = make_float4(v.x, v.y, v.z, mass);
}

// GPU: one thread per ball, all balls at once.
__global__ void stepKernel(const float4* __restrict__ pos, const float4* __restrict__ vel,
                           float4* __restrict__ nextPos, float4* __restrict__ nextVel, int count,
                           WorldBounds bounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    stepBall(i, pos, vel, nextPos, nextVel, count, bounds);
}

// CPU: one core, one ball after another. Used only by the benchmark.
void stepCpu(const float4* pos, const float4* vel, float4* nextPos, float4* nextVel, int count,
             WorldBounds bounds) {
    for (int i = 0; i < count; ++i) stepBall(i, pos, vel, nextPos, nextVel, count, bounds);
}

int gCount = 0;
WorldBounds gBounds = {};
float4* gPos = nullptr;
float4* gVel = nullptr;
float4* gNextPos = nullptr;
float4* gNextVel = nullptr;
float gTimeBank = 0.0f;
char gDeviceName[256] = "unknown";
std::vector<float4> gHostPos, gHostVel;
std::mt19937 gRng(12345);

float randomRange(std::mt19937& rng, float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}
float randomRange(float lo, float hi) { return randomRange(gRng, lo, hi); }

// Stack the balls on a loose grid high in the box so they never start overlapping.
void placeBalls(int count, WorldBounds bounds, std::mt19937& rng, std::vector<float4>& pos,
                std::vector<float4>& vel) {
    const float maxRadius = 0.6f;
    const float spacing = maxRadius * 2.2f;
    int perSide = (int)(2.0f * (bounds.halfWidth - maxRadius) / spacing);
    if (perSide < 1) perSide = 1;

    // Drop from 3 units up, or from lower down if that many layers wouldn't otherwise fit.
    int layers = (count - 1) / (perSide * perSide) + 1;
    float headroom = bounds.height - 2.0f * maxRadius - (layers - 1) * spacing;
    float dropHeight = fminf(3.0f, fmaxf(headroom, 0.0f));

    pos.resize(count);
    vel.resize(count);
    for (int i = 0; i < count; ++i) {
        int layer = i / (perSide * perSide);
        int row = (i / perSide) % perSide;
        int col = i % perSide;
        float radius = randomRange(rng, 0.3f, maxRadius);
        float x = (col - (perSide - 1) * 0.5f) * spacing + randomRange(rng, -0.05f, 0.05f);
        float z = (row - (perSide - 1) * 0.5f) * spacing + randomRange(rng, -0.05f, 0.05f);
        float y = maxRadius + dropHeight + layer * spacing;
        float mass = kDensity * (4.0f / 3.0f) * 3.14159265f * radius * radius * radius;
        pos[i] = make_float4(x, y, z, radius);
        vel[i] = make_float4(randomRange(rng, -2.0f, 2.0f), 0.0f, randomRange(rng, -2.0f, 2.0f), mass);
    }
}

double secondsNow() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Calls run(k) for k substeps and returns seconds per substep. Repeats until about `budget`
// seconds have been measured, so fast cases are averaged and slow ones are run only once.
template <class Run>
double secondsPerStep(Run run, double budget = 0.4) {
    double t0 = secondsNow();
    run(1);
    double one = secondsNow() - t0;
    int reps = (int)std::min(2000.0, std::max(1.0, budget / std::max(one, 1e-7)));
    if (reps == 1) return one;
    t0 = secondsNow();
    run(reps);
    return (secondsNow() - t0) / reps;
}

void readState() {
    CUDA_CHECK(cudaMemcpy(gHostPos.data(), gPos, gCount * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gHostVel.data(), gVel, gCount * sizeof(float4), cudaMemcpyDeviceToHost));
}

void writeState() {
    CUDA_CHECK(cudaMemcpy(gPos, gHostPos.data(), gCount * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gVel, gHostVel.data(), gCount * sizeof(float4), cudaMemcpyHostToDevice));
}

}  // namespace

bool physicsInit(int ballCount, WorldBounds bounds) {
    int devices = 0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "No CUDA GPU available: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        std::snprintf(gDeviceName, sizeof(gDeviceName), "%s", props.name);

    gCount = ballCount;
    gBounds = bounds;
    gHostPos.resize(gCount);
    gHostVel.resize(gCount);
    size_t bytes = gCount * sizeof(float4);
    if (cudaMalloc(&gPos, bytes) != cudaSuccess || cudaMalloc(&gVel, bytes) != cudaSuccess ||
        cudaMalloc(&gNextPos, bytes) != cudaSuccess || cudaMalloc(&gNextVel, bytes) != cudaSuccess) {
        std::fprintf(stderr, "Could not allocate GPU memory for %d balls\n", gCount);
        return false;
    }
    physicsReset();
    return true;
}

void physicsShutdown() {
    cudaFree(gPos);
    cudaFree(gVel);
    cudaFree(gNextPos);
    cudaFree(gNextVel);
    gPos = gVel = gNextPos = gNextVel = nullptr;
}

void physicsReset() {
    placeBalls(gCount, gBounds, gRng, gHostPos, gHostVel);
    writeState();
    gTimeBank = 0.0f;
}

void physicsStep(float seconds) {
    gTimeBank += seconds;
    int steps = (int)(gTimeBank / kSubstep);
    gTimeBank -= steps * kSubstep;

    int blocks = (gCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    for (int s = 0; s < steps; ++s) {
        stepKernel<<<blocks, kThreadsPerBlock>>>(gPos, gVel, gNextPos, gNextVel, gCount, gBounds);
        std::swap(gPos, gNextPos);
        std::swap(gVel, gNextVel);
    }
    CUDA_CHECK(cudaGetLastError());
}

void physicsKick() {
    readState();
    for (int i = 0; i < gCount; ++i) {
        gHostVel[i].x += randomRange(-4.0f, 4.0f);
        gHostVel[i].y += randomRange(8.0f, 13.0f);
        gHostVel[i].z += randomRange(-4.0f, 4.0f);
    }
    writeState();
}

void physicsRead(BallRender* out) {
    static_assert(sizeof(BallRender) == sizeof(float4), "BallRender must match the GPU layout");
    CUDA_CHECK(cudaMemcpy(out, gPos, gCount * sizeof(float4), cudaMemcpyDeviceToHost));
}

float physicsEnergy() {
    readState();
    float energy = 0.0f;
    for (int i = 0; i < gCount; ++i) {
        float3 v = make_float3(gHostVel[i].x, gHostVel[i].y, gHostVel[i].z);
        float mass = gHostVel[i].w;
        energy += 0.5f * mass * dot(v, v) + mass * kGravity * gHostPos[i].y;
    }
    return energy;
}

const char* physicsDeviceName() { return gDeviceName; }

// ------------------------------------------------------------------------------------------------
// Benchmark: the same substep on one CPU core and on the GPU.

WorldBounds physicsBoundsFor(int ballCount) {
    const float maxRadius = 0.6f;
    const float spacing = maxRadius * 2.2f;
    int perSide = std::max(1, (int)std::ceil(std::cbrt((double)ballCount)));
    int layers = (ballCount - 1) / (perSide * perSide) + 1;
    WorldBounds b;
    b.halfWidth = perSide * spacing * 0.5f + maxRadius + 0.01f;  // exactly perSide balls per row
    b.height = (layers - 1) * spacing + 2.0f * maxRadius + 3.0f + 1.0f;
    return b;
}

int physicsBenchmark(const int* counts, int numCounts, bool cpuOnly) {
    bool gpu = !cpuOnly;
    if (gpu) {
        int devices = 0;
        cudaError_t err = cudaGetDeviceCount(&devices);
        if (err != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "No CUDA GPU available (%s). Add --cpu-only to time the CPU alone.\n",
                         cudaGetErrorString(err));
            return 1;
        }
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
            std::snprintf(gDeviceName, sizeof(gDeviceName), "%s", props.name);
    }

    const double realTimeMs = kSubstep * 1000.0;  // a substep must finish this fast to keep up
    std::printf("One physics substep, timed on one CPU core%s.\n", gpu ? " and on the GPU" : "");
    if (gpu) std::printf("GPU: %s\n", gDeviceName);
    std::printf("Real time needs one substep every %.2f ms (%d substeps per simulated second).\n\n",
                realTimeMs, (int)std::lround(1.0 / kSubstep));
    std::printf("%8s %14s %13s %13s %9s %12s %12s\n", "balls", "pair checks", "CPU ms/step",
                "GPU ms/step", "speedup", "real time?", "CPU = GPU?");

    bool allMatch = true;
    static volatile float sink = 0.0f;  // keeps the compiler from skipping the timed CPU work
    for (int c = 0; c < numCounts; ++c) {
        const int n = counts[c];
        const WorldBounds bounds = physicsBoundsFor(n);
        std::mt19937 rng(777);
        std::vector<float4> pos, vel;
        placeBalls(n, bounds, rng, pos, vel);

        // CPU timing, ping-ponging between two buffers exactly like the GPU version.
        std::vector<float4> cpuPos[2] = {pos, pos}, cpuVel[2] = {vel, vel};
        int cur = 0;
        double cpuMs = 1000.0 * secondsPerStep([&](int k) {
            for (int s = 0; s < k; ++s) {
                stepCpu(cpuPos[cur].data(), cpuVel[cur].data(), cpuPos[1 - cur].data(),
                        cpuVel[1 - cur].data(), n, bounds);
                cur = 1 - cur;
            }
        });
        sink = cpuPos[cur][0].x;

        double gpuMs = 0.0, diff = 0.0;
        if (gpu) {
            // A crowded copy of the start state, so the check exercises ball-ball and floor contacts.
            std::vector<float4> crowd = pos;
            float minY = crowd[0].y;
            for (const float4& q : crowd) minY = fminf(minY, q.y);
            for (float4& q : crowd) {
                q.x *= 0.85f;
                q.z *= 0.85f;
                q.y = 0.95f * q.w + (q.y - minY) * 0.85f;
            }
            std::vector<float4> refPos(n), refVel(n), gpuVel(n);
            stepCpu(crowd.data(), vel.data(), refPos.data(), refVel.data(), n, bounds);

            const size_t bytes = n * sizeof(float4);
            float4 *dPos[2], *dVel[2];
            for (int b = 0; b < 2; ++b) {
                CUDA_CHECK(cudaMalloc(&dPos[b], bytes));
                CUDA_CHECK(cudaMalloc(&dVel[b], bytes));
            }
            const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;

            CUDA_CHECK(cudaMemcpy(dPos[0], crowd.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dVel[0], vel.data(), bytes, cudaMemcpyHostToDevice));
            stepKernel<<<blocks, kThreadsPerBlock>>>(dPos[0], dVel[0], dPos[1], dVel[1], n, bounds);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(gpuVel.data(), dVel[1], bytes, cudaMemcpyDeviceToHost));
            double maxDiff = 0.0, maxVel = 0.0;
            for (int i = 0; i < n; ++i) {
                maxDiff = std::max({maxDiff, (double)std::fabs(gpuVel[i].x - refVel[i].x),
                                    (double)std::fabs(gpuVel[i].y - refVel[i].y),
                                    (double)std::fabs(gpuVel[i].z - refVel[i].z)});
                maxVel = std::max({maxVel, (double)std::fabs(refVel[i].x),
                                   (double)std::fabs(refVel[i].y), (double)std::fabs(refVel[i].z)});
            }
            diff = maxDiff / (maxVel + 1.0);  // relative difference in the new velocities

            // GPU timing from the normal start state.
            CUDA_CHECK(cudaMemcpy(dPos[0], pos.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dVel[0], vel.data(), bytes, cudaMemcpyHostToDevice));
            int g = 0;
            gpuMs = 1000.0 * secondsPerStep([&](int k) {
                for (int s = 0; s < k; ++s) {
                    stepKernel<<<blocks, kThreadsPerBlock>>>(dPos[g], dVel[g], dPos[1 - g],
                                                             dVel[1 - g], n, bounds);
                    g = 1 - g;
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            });
            CUDA_CHECK(cudaGetLastError());
            for (int b = 0; b < 2; ++b) {
                CUDA_CHECK(cudaFree(dPos[b]));
                CUDA_CHECK(cudaFree(dVel[b]));
            }
        }

        const bool cpuKeeps = cpuMs <= realTimeMs, gpuKeeps = gpu && gpuMs <= realTimeMs;
        const char* keeps = cpuKeeps && gpuKeeps ? "both"
                            : gpuKeeps           ? "GPU only"
                            : cpuKeeps           ? "CPU only"
                                                 : "neither";
        char gpuCol[32] = "-", speedCol[32] = "-", matchCol[32] = "-";
        if (gpu) {
            std::snprintf(gpuCol, sizeof gpuCol, "%.3f", gpuMs);
            std::snprintf(speedCol, sizeof speedCol, "%.1fx", cpuMs / gpuMs);
            const bool match = diff < 1e-3;
            allMatch = allMatch && match;
            std::snprintf(matchCol, sizeof matchCol, "%s %.0e", match ? "yes" : "NO", diff);
        }
        std::printf("%8d %14lld %13.3f %13s %9s %12s %12s\n", n, (long long)n * (n - 1), cpuMs,
                    gpuCol, speedCol, gpu ? keeps : (cpuKeeps ? "yes" : "no"), matchCol);
        std::fflush(stdout);
    }
    if (gpu)
        std::printf("\n%s\n", allMatch ? "PASS: the CPU and GPU compute the same physics."
                                       : "FAIL: the CPU and GPU results differ.");
    return allMatch ? 0 : 1;
}
