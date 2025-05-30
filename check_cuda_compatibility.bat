@echo off
echo CUDA Compatibility Check
echo ========================

echo.
echo 1. Checking current compiler...
where gcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [FOUND] GCC compiler
    gcc --version | findstr "gcc"
    echo [INFO] You are using GCC/MinGW
) else (
    echo [NOT FOUND] GCC compiler
)

where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [FOUND] MSVC compiler
    cl 2>&1 | findstr "Microsoft"
    echo [INFO] You are using Visual Studio compiler
) else (
    echo [NOT FOUND] MSVC compiler
)

echo.
echo 2. Checking CUDA installation...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [FOUND] CUDA compiler
    nvcc --version | findstr "release"
) else (
    echo [NOT FOUND] CUDA compiler (nvcc)
)

where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [FOUND] NVIDIA driver
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo GPU info not available
    )
) else (
    echo [NOT FOUND] NVIDIA driver or GPU
)

echo.
echo 3. CUDA Compatibility Assessment:
echo ===================================

REM Check if using MinGW
where gcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    where cl >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [INCOMPATIBLE] You are using MinGW, which doesn't support CUDA
        echo.
        echo Recommendations:
        echo   Option 1: Install Visual Studio Community for CUDA support
        echo   Option 2: Use WSL2 with Linux for CUDA support  
        echo   Option 3: Continue with MinGW for CPU-only performance
        goto :end
    )
)

REM Check if have both MSVC and CUDA
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    where nvcc >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [COMPATIBLE] You can build with CUDA support!
        echo.
        echo To build with CUDA:
        echo   1. Open "Developer Command Prompt for VS"
        echo   2. Run: build_cuda.bat
        goto :end
    ) else (
        echo [PARTIAL] You have MSVC but need CUDA Toolkit
        echo   Download from: https://developer.nvidia.com/cuda-downloads
        goto :end
    )
)

echo [INCOMPLETE] Install Visual Studio and CUDA Toolkit for CUDA support

:end
echo.
pause 