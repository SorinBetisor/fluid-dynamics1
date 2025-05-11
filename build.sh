#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# build.sh — configure & compile project with Ninja + -O3, then build shaders
# -----------------------------------------------------------------------------

# 1) Locate directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SHADER_DIR="$SCRIPT_DIR/shaders"         # adjust if your shaders live elsewhere
SHADER_OUT_DIR="$BUILD_DIR/shaders"

# 2) Make & enter build directory
# rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 3) Configure with CMake + Ninja, Release + -O3
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
  -DTARGET_PLATFORM="${TARGET_PLATFORM:-Linux}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  "$SCRIPT_DIR"

# 4) Build everything
ninja

# 5) Compile GLSL → SPIR-V
echo "⏳ Compiling shaders..."
mkdir -p "$SHADER_OUT_DIR"

# file extensions to compile
# for ext in vert frag comp geom tesc tese; do
#   for shader in "$SHADER_DIR"/*."$ext".glsl; do
#     [ -e "$shader" ] || continue
#     base="$(basename "$shader" .glsl)"
#     out="$SHADER_OUT_DIR/${base}.spv"
#     echo " - $shader → $out"
#     glslangValidator -V "$shader" -o "$out"
#   done
# done
#
# echo "✅ Build and shader compilation complete."

