# Building cnavier with CMake

This project now supports building with CMake on Windows, with OpenMP support for parallel processing.

## Prerequisites

- CMake 3.16 or higher
- A C compiler with OpenMP support:
  - **Visual Studio 2019/2022** (recommended for Windows)
  - **MinGW-w64** with GCC
  - **Clang** with OpenMP support

## Quick Build (Windows)

### Option 1: Using the build script
Simply run the provided batch script:
```cmd
build.bat
```
The script will automatically handle OpenMP detection issues and offer to build without OpenMP if needed.

### Option 2: Manual CMake build
```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Build Options

### Build Types
- **Release** (default): Optimized build (`-O3` for GCC, `/O2` for MSVC)
- **Debug**: Debug build with symbols (`-g -O0` for GCC, `/Od /Zi` for MSVC)

To build in Debug mode:
```cmd
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

### OpenMP Options
- **With OpenMP** (default): `cmake .. -DUSE_OPENMP=ON`
- **Without OpenMP**: `cmake .. -DUSE_OPENMP=OFF`

If OpenMP detection fails, the build script will offer to build without it automatically.

### Specifying Compiler
To use a specific compiler:
```cmd
# For Visual Studio 2022
cmake .. -G "Visual Studio 17 2022"

# For MinGW
cmake .. -G "MinGW Makefiles"

# For Ninja (if installed)
cmake .. -G "Ninja"
```

## OpenMP Support

The CMake configuration uses a robust OpenMP detection system:

1. **First attempt**: Uses CMake's `find_package(OpenMP)`
2. **Fallback**: Manual detection with compiler-specific flags
3. **Final fallback**: Builds without OpenMP if detection fails

### OpenMP Status Messages
- ✅ `OpenMP found: X.X` - Standard CMake detection succeeded
- ⚠️ `OpenMP manually enabled for MSVC/GCC/Clang` - Manual detection succeeded
- ❌ `OpenMP could not be enabled` - Building without parallel support

### Compiler-Specific Notes
- **MSVC**: Uses `/openmp` flag, usually works out of the box
- **GCC**: Uses `-fopenmp` flag, requires libgomp
- **Clang**: Uses `-fopenmp` flag, may need libomp installed

## Output

The built executable will be located at:
- `build/bin/cnavier.exe` (single-config generators)
- `build/bin/Release/cnavier.exe` (multi-config generators like Visual Studio)

## Cleaning

To clean the build:
```cmd
# Remove the entire build directory
rmdir /s build

# Or use CMake clean
cd build
cmake --build . --target clean
```

## Troubleshooting

### OpenMP not found
**Solution 1**: Build without OpenMP
```cmd
cmake .. -DUSE_OPENMP=OFF
```

**Solution 2**: Install OpenMP support
- **MSVC**: Install Visual Studio with C++ development tools
- **MinGW**: Use a recent MinGW-w64 distribution (8.0+)
- **Clang**: Install OpenMP library (`libomp`)

**Solution 3**: Use a different compiler
```cmd
# Try Visual Studio if available
cmake .. -G "Visual Studio 17 2022"
```

### CMake not found
- Install CMake from https://cmake.org/download/
- Make sure CMake is in your PATH

### Compiler not found
- Install Visual Studio with C++ development tools, or
- Install MinGW-w64, or
- Install Clang with OpenMP support

### Build succeeds but no parallel speedup
- Check if OpenMP was actually enabled in the build output
- Look for `OpenMP enabled: TRUE` in the configuration messages
- The code may need OpenMP pragmas added to utilize parallel processing 