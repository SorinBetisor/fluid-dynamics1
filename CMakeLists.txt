cmake_minimum_required(VERSION 3.10)
project(cnavier C)

set(CMAKE_C_STANDARD 11)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES 
     "${CMAKE_SOURCE_DIR}/src/*.c"
)

# Initially disable GPU acceleration and OpenMP for Windows build
add_definitions(-DDISABLE_GPU -DDISABLE_VULKAN -DDISABLE_OPENMP)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# Add executable
add_executable(cnavier ${SOURCES})

# Windows doesn't have a separate math library
if(WIN32)
    target_link_libraries(cnavier)
else()
    target_link_libraries(cnavier m)
endif()

# Set output directory
set_target_properties(cnavier PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Find OpenMP if available
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(cnavier OpenMP::OpenMP_C)
    message(STATUS "OpenMP found and enabled")
    remove_definitions(-DDISABLE_OPENMP)
else()
    message(STATUS "OpenMP not found, disabled")
endif() 