// Ball physics, run entirely on the GPU.
//
// Balls are soft spheres: overlapping balls push apart with a spring and lose energy through a
// damper. Every ball's new state depends only on the previous state of all balls, so each one can
// be computed by its own GPU thread. Results go to a second buffer, which is then swapped in.
#include "physics.h"

#include <cuda_runtime.h>

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
__device__ inline float3 contactForce(float3 normal, float overlap, float3 relVel, float mass) {
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

// Positions are (x, y, z, radius) and velocities are (x, y, z, mass).
__global__ void stepKernel(const float4* __restrict__ pos, const float4* __restrict__ vel,
                           float4* __restrict__ nextPos, float4* __restrict__ nextVel, int count,
                           WorldBounds bounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

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

float randomRange(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(gRng);
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
    // Stack the balls on a loose grid high in the box so they never start overlapping.
    const float maxRadius = 0.6f;
    const float spacing = maxRadius * 2.2f;
    int perSide = (int)(2.0f * (gBounds.halfWidth - maxRadius) / spacing);
    if (perSide < 1) perSide = 1;

    // Drop from 3 units up, or from lower down if that many layers wouldn't otherwise fit.
    int layers = (gCount - 1) / (perSide * perSide) + 1;
    float headroom = gBounds.height - 2.0f * maxRadius - (layers - 1) * spacing;
    float dropHeight = fminf(3.0f, fmaxf(headroom, 0.0f));

    for (int i = 0; i < gCount; ++i) {
        int layer = i / (perSide * perSide);
        int row = (i / perSide) % perSide;
        int col = i % perSide;
        float radius = randomRange(0.3f, maxRadius);
        float x = (col - (perSide - 1) * 0.5f) * spacing + randomRange(-0.05f, 0.05f);
        float z = (row - (perSide - 1) * 0.5f) * spacing + randomRange(-0.05f, 0.05f);
        float y = maxRadius + dropHeight + layer * spacing;
        float mass = kDensity * (4.0f / 3.0f) * 3.14159265f * radius * radius * radius;
        gHostPos[i] = make_float4(x, y, z, radius);
        gHostVel[i] = make_float4(randomRange(-2.0f, 2.0f), 0.0f, randomRange(-2.0f, 2.0f), mass);
    }
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
