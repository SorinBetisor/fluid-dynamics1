cmake_minimum_required(VERSION 3.15)
project(FluidDynamicsSolver VERSION 1.0
        DESCRIPTION "Fluid Dynamics PDE Solver with ML Integration"
        LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add HDF5 support for data storage
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Add source files
add_executable(fluid_solver 
    src/main.cpp
)

# Include directories
target_include_directories(fluid_solver PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(fluid_solver PRIVATE
    ${HDF5_LIBRARIES}
)

# Add compiler warnings
if(MSVC)
    target_compile_options(fluid_solver PRIVATE /W4)
else()
    target_compile_options(fluid_solver PRIVATE -Wall -Wextra -Wpedantic)
endif() 