#!/bin/bash

# Exit immediately if any command fails
set -e

# Check for glslc compiler
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc compiler not found. Please install the Vulkan SDK."
    exit 1
fi

# Ensure shader directory exists
SHADER_DIR="shaders"
if [ ! -d "$SHADER_DIR" ]; then
    echo "Error: Shader directory not found."
    exit 1
fi

# Compile shaders
echo "Compiling vertex shader..."
glslc -fshader-stage=vertex "$SHADER_DIR/vert.glsl" -o "$SHADER_DIR/vert.spv"

echo "Compiling fragment shader..."
glslc -fshader-stage=fragment "$SHADER_DIR/frag.glsl" -o "$SHADER_DIR/frag.spv"

echo "Shader compilation complete!"