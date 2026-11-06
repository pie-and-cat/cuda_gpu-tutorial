// CUDA Balls: a window, a camera and a renderer around the GPU simulation in physics.cu.
// Uses only Win32 and OpenGL 1.1, both of which ship with Windows.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "physics.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "opengl32.lib")

namespace {

const int kDefaultBalls = 24;
const int kMaxBalls = 500;  // as many as fit in the starting grid inside the box
const WorldBounds kBounds = {6.0f, 12.0f};
const float kPi = 3.14159265f;
const float kMoveSpeed = 7.0f;          // units per second
const float kLookSensitivity = 0.004f;  // radians per pixel
const float kMaxFrameTime = 1.0f / 20.0f;  // after a stall, slow down rather than spiral

struct Camera {
    float x = 0.0f, y = 4.5f, z = 15.0f;
    float yaw = 0.0f;  // radians; 0 looks along -z
    float pitch = -0.15f;
};

Camera gCamera;
bool gKeys[256] = {};
bool gDragging = false;
POINT gLastMouse = {};
int gWidth = 1280, gHeight = 720;
bool gKickRequested = false;
bool gResetRequested = false;

LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_KEYDOWN:
            if (wp < 256) gKeys[wp] = true;
            if (wp == VK_ESCAPE) DestroyWindow(hwnd);
            // Bit 30 is set on auto-repeat, so holding a key doesn't retrigger these.
            if (wp == VK_SPACE && !(lp & (1 << 30))) gKickRequested = true;
            if (wp == 'R' && !(lp & (1 << 30))) gResetRequested = true;
            return 0;
        case WM_KEYUP:
            if (wp < 256) gKeys[wp] = false;
            return 0;
        case WM_KILLFOCUS:
            // Key-up events go elsewhere after losing focus, so don't leave keys stuck down.
            std::memset(gKeys, 0, sizeof(gKeys));
            gDragging = false;
            ReleaseCapture();
            return 0;
        case WM_RBUTTONDOWN:
            gDragging = true;
            GetCursorPos(&gLastMouse);
            SetCapture(hwnd);
            return 0;
        case WM_RBUTTONUP:
            gDragging = false;
            ReleaseCapture();
            return 0;
        case WM_MOUSEMOVE:
            if (gDragging) {
                POINT now;
                GetCursorPos(&now);
                gCamera.yaw += (now.x - gLastMouse.x) * kLookSensitivity;
                gCamera.pitch -= (now.y - gLastMouse.y) * kLookSensitivity;
                gCamera.pitch = std::clamp(gCamera.pitch, -1.5f, 1.5f);
                gLastMouse = now;
            }
            return 0;
        case WM_SIZE:
            gWidth = std::max<int>(LOWORD(lp), 1);
            gHeight = std::max<int>(HIWORD(lp), 1);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

void updateCamera(float dt) {
    float sinYaw = std::sin(gCamera.yaw), cosYaw = std::cos(gCamera.yaw);
    float forward = (gKeys['W'] ? 1.0f : 0.0f) - (gKeys['S'] ? 1.0f : 0.0f);
    float strafe = (gKeys['D'] ? 1.0f : 0.0f) - (gKeys['A'] ? 1.0f : 0.0f);
    float lift = (gKeys['E'] ? 1.0f : 0.0f) - (gKeys['Q'] ? 1.0f : 0.0f);
    float step = kMoveSpeed * dt;

    // WASD moves along the ground plane in the direction the camera is facing.
    gCamera.x += (sinYaw * forward + cosYaw * strafe) * step;
    gCamera.z += (-cosYaw * forward + sinYaw * strafe) * step;
    gCamera.y += lift * step;
}

// Builds a lit unit sphere into a display list so that each ball is one glCallList.
GLuint buildSphereList() {
    const int stacks = 24, slices = 36;
    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);
    for (int i = 0; i < stacks; ++i) {
        float lat0 = kPi * ((float)i / stacks - 0.5f);
        float lat1 = kPi * ((float)(i + 1) / stacks - 0.5f);
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            float lon = 2.0f * kPi * j / slices;
            float cx = std::cos(lon), sz = std::sin(lon);
            float x1 = cx * std::cos(lat1), y1 = std::sin(lat1), z1 = sz * std::cos(lat1);
            float x0 = cx * std::cos(lat0), y0 = std::sin(lat0), z0 = sz * std::cos(lat0);
            glNormal3f(x0, y0, z0);
            glVertex3f(x0, y0, z0);
            glNormal3f(x1, y1, z1);
            glVertex3f(x1, y1, z1);
        }
        glEnd();
    }
    glEndList();
    return list;
}

void hueToRgb(float hue, float* rgb) {
    for (int c = 0; c < 3; ++c) {
        float t = std::fabs(std::fmod(hue * 6.0f + 4.0f - 2.0f * c + 6.0f, 6.0f) - 3.0f) - 1.0f;
        rgb[c] = 0.25f + 0.7f * std::clamp(t, 0.0f, 1.0f);
    }
}

void drawArena() {
    const float w = kBounds.halfWidth, h = kBounds.height;
    glDisable(GL_LIGHTING);

    // Checkerboard floor, which makes camera movement easy to see.
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 1.0f, 0.0f);
    int cells = (int)(w * 2.0f);
    for (int ix = 0; ix < cells; ++ix) {
        for (int iz = 0; iz < cells; ++iz) {
            float shade = ((ix + iz) & 1) ? 0.20f : 0.27f;
            glColor3f(shade, shade, shade + 0.03f);
            float x0 = -w + ix, z0 = -w + iz;
            glVertex3f(x0, 0.0f, z0);
            glVertex3f(x0, 0.0f, z0 + 1.0f);
            glVertex3f(x0 + 1.0f, 0.0f, z0 + 1.0f);
            glVertex3f(x0 + 1.0f, 0.0f, z0);
        }
    }
    glEnd();

    // Wireframe of the walls.
    glColor3f(0.55f, 0.6f, 0.7f);
    glBegin(GL_LINES);
    const float corners[4][2] = {{-w, -w}, {w, -w}, {w, w}, {-w, w}};
    for (int i = 0; i < 4; ++i) {
        const float* a = corners[i];
        const float* b = corners[(i + 1) % 4];
        glVertex3f(a[0], 0.0f, a[1]);
        glVertex3f(b[0], 0.0f, b[1]);
        glVertex3f(a[0], h, a[1]);
        glVertex3f(b[0], h, b[1]);
        glVertex3f(a[0], 0.0f, a[1]);
        glVertex3f(a[0], h, a[1]);
    }
    glEnd();
}

void render(const std::vector<BallRender>& balls, const std::vector<float>& colors, GLuint sphere) {
    glViewport(0, 0, gWidth, gHeight);
    glClearColor(0.07f, 0.08f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const float nearPlane = 0.1f, farPlane = 200.0f;
    float top = nearPlane * std::tan(55.0f * kPi / 360.0f);
    float right = top * (float)gWidth / gHeight;
    glFrustum(-right, right, -top, top, nearPlane, farPlane);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(-gCamera.pitch * 180.0f / kPi, 1.0f, 0.0f, 0.0f);
    glRotatef(gCamera.yaw * 180.0f / kPi, 0.0f, 1.0f, 0.0f);
    glTranslatef(-gCamera.x, -gCamera.y, -gCamera.z);

    // The light moves with the world because it is set after the camera transform.
    const GLfloat lightDir[4] = {0.4f, 1.0f, 0.5f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightDir);

    drawArena();

    glEnable(GL_LIGHTING);
    for (size_t i = 0; i < balls.size(); ++i) {
        glColor3fv(&colors[i * 3]);
        glPushMatrix();
        glTranslatef(balls[i].x, balls[i].y, balls[i].z);
        glScalef(balls[i].radius, balls[i].radius, balls[i].radius);
        glCallList(sphere);
        glPopMatrix();
    }
}

bool setupOpenGL(HDC dc) {
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    int format = ChoosePixelFormat(dc, &pfd);
    if (!format || !SetPixelFormat(dc, format, &pfd)) return false;
    HGLRC context = wglCreateContext(dc);
    if (!context || !wglMakeCurrent(dc, context)) return false;

    // Cap the frame rate to the display, so the GPU isn't flat out drawing a few spheres.
    typedef BOOL(WINAPI * SwapIntervalFn)(int);
    SwapIntervalFn setSwapInterval = (SwapIntervalFn)wglGetProcAddress("wglSwapIntervalEXT");
    if (setSwapInterval) setSwapInterval(1);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);  // the balls are scaled, which would otherwise skew their normals
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHT0);
    const GLfloat ambient[4] = {0.3f, 0.3f, 0.35f, 1.0f};
    const GLfloat specular[4] = {0.5f, 0.5f, 0.5f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 40.0f);
    return true;
}

double now() {
    static LARGE_INTEGER frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / frequency.QuadPart;
}

// Runs the simulation with no window and checks that it behaves physically.
int runSelfTest(int ballCount) {
    if (!physicsInit(ballCount, kBounds)) return 1;
    std::printf("Device: %s, balls: %d\n", physicsDeviceName(), ballCount);

    std::vector<BallRender> balls(ballCount);
    bool ok = true;
    float startEnergy = physicsEnergy();
    double startTime = now();
    const float simSeconds = 20.0f;
    for (int i = 0; i < (int)(simSeconds * 60.0f); ++i) {
        physicsStep(1.0f / 60.0f);
        physicsRead(balls.data());
        for (const BallRender& b : balls) {
            // Soft contacts may squash slightly into a surface, but never far past it.
            const float slack = 0.5f * b.radius;
            bool inside = std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z) &&
                          b.y > b.radius - slack && b.y < kBounds.height - b.radius + slack &&
                          std::fabs(b.x) < kBounds.halfWidth - b.radius + slack &&
                          std::fabs(b.z) < kBounds.halfWidth - b.radius + slack;
            if (ok && !inside) {
                std::printf("FAIL: ball escaped at frame %d (%f, %f, %f)\n", i, b.x, b.y, b.z);
                ok = false;
            }
        }
    }
    double elapsed = now() - startTime;
    float settledEnergy = physicsEnergy();

    float worstOverlap = 0.0f;
    for (int i = 0; i < ballCount; ++i) {
        for (int j = i + 1; j < ballCount; ++j) {
            float dx = balls[i].x - balls[j].x, dy = balls[i].y - balls[j].y,
                  dz = balls[i].z - balls[j].z;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            worstOverlap = std::max(worstOverlap, balls[i].radius + balls[j].radius - dist);
        }
    }

    std::printf("Energy: start %.2f J, after %.0f s %.2f J\n", startEnergy, simSeconds, settledEnergy);
    std::printf("Worst overlap after settling: %.4f\n", worstOverlap);
    std::printf("Simulated %.0f s in %.2f s of real time\n", simSeconds, elapsed);
    if (settledEnergy >= startEnergy) {
        std::printf("FAIL: energy should drop as the balls bounce and settle\n");
        ok = false;
    }
    if (worstOverlap > 0.05f) {
        std::printf("FAIL: balls are passing through each other\n");
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    physicsShutdown();
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int ballCount = kDefaultBalls;
    bool selfTest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) {
            selfTest = true;
        } else {
            int parsed = std::atoi(argv[i]);
            if (parsed <= 0 || parsed > kMaxBalls) {
                std::fprintf(stderr, "Usage: cuda_balls [ball count, 1-%d] [--selftest]\n", kMaxBalls);
                return 1;
            }
            ballCount = parsed;
        }
    }
    if (selfTest) return runSelfTest(ballCount);

    if (!physicsInit(ballCount, kBounds)) {
        MessageBoxA(nullptr, "Could not start CUDA. See the console for details.", "CUDA Balls",
                    MB_ICONERROR);
        return 1;
    }

    WNDCLASSA wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = windowProc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursorA(nullptr, (LPCSTR)IDC_ARROW);
    wc.lpszClassName = "CudaBallsWindow";
    RegisterClassA(&wc);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "CUDA Balls", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                              CW_USEDEFAULT, CW_USEDEFAULT, gWidth, gHeight, nullptr, nullptr,
                              wc.hInstance, nullptr);
    HDC dc = hwnd ? GetDC(hwnd) : nullptr;
    if (!dc || !setupOpenGL(dc)) {
        std::fprintf(stderr, "Could not create an OpenGL window\n");
        return 1;
    }
    GLuint sphere = buildSphereList();

    std::vector<BallRender> balls(ballCount);
    std::vector<float> colors(ballCount * 3);
    for (int i = 0; i < ballCount; ++i) hueToRgb((float)i / ballCount, &colors[i * 3]);

    std::printf("CUDA device: %s\n", physicsDeviceName());
    std::printf("WASD move | Q/E down/up | hold right mouse to look | Space kick | R reset | Esc quit\n");

    double lastFrame = now();
    double lastTitle = lastFrame;
    int framesSinceTitle = 0;
    bool running = true;
    while (running) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (!running) break;

        double current = now();
        float dt = std::min((float)(current - lastFrame), kMaxFrameTime);
        lastFrame = current;

        if (gResetRequested) physicsReset();
        if (gKickRequested) physicsKick();
        gResetRequested = gKickRequested = false;

        updateCamera(dt);
        physicsStep(dt);
        physicsRead(balls.data());
        render(balls, colors, sphere);
        SwapBuffers(dc);

        ++framesSinceTitle;
        if (current - lastTitle >= 0.5) {
            char title[256];
            std::snprintf(title, sizeof(title),
                          "CUDA Balls | %d balls on %s | %.0f FPS | camera (%.1f, %.1f, %.1f)",
                          ballCount, physicsDeviceName(), framesSinceTitle / (current - lastTitle),
                          gCamera.x, gCamera.y, gCamera.z);
            SetWindowTextA(hwnd, title);
            lastTitle = current;
            framesSinceTitle = 0;
        }
    }

    physicsShutdown();
    return 0;
}
