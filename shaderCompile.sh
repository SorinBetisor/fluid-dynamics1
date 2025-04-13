#!/bin/bash

# Exit immediately if any command fails
set -e

glslc shaders/shader.vert -o shaders/vert.spv
glslc shaders/shader.frag -o shaders/frag.spv