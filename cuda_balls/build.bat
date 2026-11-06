@echo off
rem Builds build\cuda_balls.exe. Uses an installed CUDA toolkit if CUDA_PATH is set,
rem otherwise the copy unpacked into tools\cuda.
setlocal
cd /d "%~dp0"

if not defined CUDA_PATH set "CUDA_PATH=%~dp0tools\cuda"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo Could not find nvcc.exe in "%CUDA_PATH%\bin". See README.md.
    exit /b 1
)
set "PATH=%CUDA_PATH%\bin;%PATH%"

where cl >nul 2>nul
if not errorlevel 1 goto compile
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
for /f "usebackq delims=" %%i in (`vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo Could not find Visual Studio with the C++ tools.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

:compile
if not exist build mkdir build
rem Ready-made machine code for each GPU generation CUDA 13 supports, so no driver has to
rem JIT-compile anything (a driver older than the toolkit can't JIT its PTX):
rem   sm_75 RTX 20 / GTX 16 / T4, sm_80 A100, sm_86 RTX 30, sm_89 RTX 40, sm_90 H100, sm_120 RTX 50.
rem The last entry also embeds PTX, so GPUs newer than these can run it with a new enough driver.
set "ARCHS=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"
nvcc -O2 -std=c++17 %ARCHS% -Xcompiler "/EHsc,/W3,/nologo" -I src src\main.cpp src\physics.cu -o build\cuda_balls.exe
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)
echo Built build\cuda_balls.exe
