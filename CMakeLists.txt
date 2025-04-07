cmake_minimum_required(VERSION 3.10)

# Project name, version, and language
project(FluidDynamicsSolver VERSION 1.0
        DESCRIPTION "Fluid Dynamics PDE Solver with ML Integration"
        LANGUAGES C)

# Set the C standard 
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Include directories for header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Gather all C source files from the src/ directory
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.c)

# Ensure that there are source files
if (NOT SRC_FILES)
    message(FATAL_ERROR "No source files found in src/ directory!")
endif()

# Define the executable target
add_executable(fluid_solver ${SRC_FILES})

# Enable OpenMP if available and link dynamically
find_package(OpenMP)
if (OpenMP_FOUND)
    target_link_libraries(fluid_solver PRIVATE OpenMP::OpenMP_C)
endif()

# Add HDF5 support for data storage 
find_package(HDF5 REQUIRED COMPONENTS C)
if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(fluid_solver PRIVATE ${HDF5_LIBRARIES})
else()
    message(FATAL_ERROR "HDF5 not found!")
endif()

# Find and link CGLM library
find_package(cglm CONFIG REQUIRED)
target_link_libraries(fluid_solver PRIVATE cglm::cglm)

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Find GLFW3
find_package(glfw3 CONFIG REQUIRED)

# Find GLAD
find_package(glad CONFIG REQUIRED)

# Update target link libraries
target_link_libraries(fluid_solver PRIVATE 
    ${HDF5_LIBRARIES}
    cglm::cglm
    OpenGL::GL
    glfw
    glad::glad
)

# Optionally link the math library on UNIX systems
if (UNIX)
    target_link_libraries(fluid_solver PRIVATE m)
endif()

# Add compiler warnings based on the compiler used
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(fluid_solver PRIVATE -Wall -Wextra -Wpedantic)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(fluid_solver PRIVATE /W4)
    # Disable warnings about unsafe functions
    target_compile_definitions(fluid_solver PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Set output directories for runtime, archive, and library files
set_target_properties(fluid_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

