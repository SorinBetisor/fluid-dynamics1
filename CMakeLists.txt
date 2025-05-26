cmake_minimum_required(VERSION 3.10)
project(cnavier C)

set(CMAKE_C_STANDARD 11)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES 
     "${CMAKE_SOURCE_DIR}/src/*.c"
)

# Initially disable GPU acceleration for Windows build, but try to enable OpenMP
add_definitions(-DDISABLE_GPU -DDISABLE_VULKAN)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# Find OpenMP first
find_package(OpenMP)

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

# Configure OpenMP
if(OpenMP_C_FOUND)
    target_link_libraries(cnavier OpenMP::OpenMP_C)
    message(STATUS "OpenMP found and enabled")
else()
    target_compile_definitions(cnavier PRIVATE -DDISABLE_OPENMP)
    message(STATUS "OpenMP not found, disabling")
endif()

# Copy config.txt to build directory
configure_file(${CMAKE_SOURCE_DIR}/config.txt ${CMAKE_BINARY_DIR}/config.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config.txt ${CMAKE_SOURCE_DIR}/Release/config.txt COPYONLY) 