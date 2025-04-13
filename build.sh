#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if CMakeLists.txt exists in the current directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in the current directory."
    echo "Please run this script from the project's root directory."
    exit 1
fi

# Define the build directory
BUILD_DIR="build"

# Check if the build directory exists
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create a fresh build directory
echo "Creating a new build directory..."
mkdir -p "$BUILD_DIR"

# Navigate to the build directory
cd "$BUILD_DIR"

# Run CMake to configure the project with verbose output
echo "Configuring the project with CMake..."
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..

# Build the project
echo "Building the project..."
cmake --build . -- VERBOSE=1

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "The executable is located at: $BUILD_DIR/bin/fluid_solver"
else
    echo "Build failed!"
    exit 1
fi
