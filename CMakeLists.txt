cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MyComplexCProject LANGUAGES C)

# C Standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Define the output executable name
set(EXECUTABLE_NAME test)

# Add include directories (for headers)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.c)

# Ensure there are source files to compile
if (NOT SRC_FILES)
    message(FATAL_ERROR "No source files found in src/ directory!")
endif()

# Add the executable
add_executable(${EXECUTABLE_NAME} ${SRC_FILES})

# Find SDL2 package
find_package(SDL2 REQUIRED)
if (SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${SDL2_LIBRARIES})
else()
    message(FATAL_ERROR "SDL2 not found!")
endif()

# Find Vulkan package
# Add Homebrew Vulkan paths for macOS
if(APPLE)
    set(VULKAN_INCLUDE_DIR "/opt/homebrew/include" CACHE PATH "Path to Vulkan include directory")
    set(VULKAN_LIBRARY "/opt/homebrew/lib/libvulkan.dylib" CACHE FILEPATH "Path to Vulkan library")
    set(ENV{VULKAN_SDK} "/opt/homebrew")
endif()

find_package(Vulkan REQUIRED)
if (Vulkan_FOUND)
    include_directories(${Vulkan_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${Vulkan_LIBRARIES})
else()
    message(FATAL_ERROR "Vulkan not found!")
endif()

# Compiler warnings
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -Wall -Wextra -Wpedantic)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${EXECUTABLE_NAME} PRIVATE /W4)
endif()

# Set build output directories
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

