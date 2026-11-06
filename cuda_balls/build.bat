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
rem Native code for RTX 40 series GPUs (sm_89). A driver older than this toolkit can't JIT-compile its PTX.
nvcc -O2 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc,/W3,/nologo" -I src src\main.cpp src\physics.cu -o build\cuda_balls.exe
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)
echo Built build\cuda_balls.exe
