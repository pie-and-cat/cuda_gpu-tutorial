// Interface between the game (main.cpp, plain C++) and the CUDA simulation (physics.cu).
#pragma once

struct BallRender {
    float x, y, z, radius;
};

struct WorldBounds {
    float halfWidth;  // walls at x = +-halfWidth and z = +-halfWidth
    float height;     // floor at y = 0, ceiling at y = height
};

// Allocates GPU memory and places the balls. Returns false (and prints why) if CUDA is unavailable.
bool physicsInit(int ballCount, WorldBounds bounds);
void physicsShutdown();

// Puts the balls back at random starting positions.
void physicsReset();

// Advances the simulation by `seconds` of simulated time on the GPU.
void physicsStep(float seconds);

// Gives every ball an upward, randomly angled velocity kick.
void physicsKick();

// Copies current ball positions from the GPU into `out` (ballCount entries).
void physicsRead(BallRender* out);

// Total kinetic + potential energy, for the headless self-test.
float physicsEnergy();

// Name of the GPU that the simulation is running on.
const char* physicsDeviceName();
