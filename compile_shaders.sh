#!/bin/bash

# Define shader compiler path
# Use the explicit Vulkan SDK path
VULKAN_SDK="/Users/botond/VulkanSDK/1.4.309.0"
GLSLC="$VULKAN_SDK/macOS/bin/glslc"

if [ ! -f "$GLSLC" ]; then
    echo "Error: glslc compiler not found at $GLSLC"
    echo "Make sure the VULKAN_SDK path is correct"
    exit 1
fi

# Create output directory
mkdir -p shaders

# Compile poisson compute shader
$GLSLC -o shaders/poisson.spv src/vulkan/poisson_compute.comp
if [ $? -ne 0 ]; then
    echo "Failed to compile poisson_compute.comp"
    exit 1
fi

# Compile poisson SOR compute shader
$GLSLC -o shaders/poisson_sor.spv src/vulkan/poisson_sor_compute.comp
if [ $? -ne 0 ]; then
    echo "Failed to compile poisson_sor_compute.comp"
    exit 1
fi

echo "Shaders compiled successfully" 