cmake_minimum_required(VERSION 3.14)

project(CPPGame VERSION 1.0.0 LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON
    CACHE BOOL "Generate compile_commands.json for IDEs/LSPs" FORCE)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Select target platform: Linux (default) or Windows (cross-compile via MinGW)
set(TARGET_PLATFORM "Linux" CACHE STRING "Target platform to build for")
set_property(CACHE TARGET_PLATFORM PROPERTY STRINGS Linux Windows)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

# Where we’ll emit our .spv files:
set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
# Make sure it exists (at configure time):
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${SPIRV_OUTPUT_DIR}/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

if(TARGET_PLATFORM STREQUAL "Windows")
    message(STATUS "Configuring Windows cross-build via MinGW")
    # Ensure user provides a toolchain file or place one under toolchains/
    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/toolchains/mingw-w64.cmake"
            CACHE FILEPATH "Toolchain file for Windows cross-compilation")
    endif()
endif()

# FetchContent for external dependencies
include(FetchContent)

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG         release-3.2.10  # refs/tags/3.2.10
)

FetchContent_MakeAvailable(SDL3)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.9b                                  
)

FetchContent_GetProperties(imgui)

if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)

  add_library(ImGui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp   
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  )

  target_include_directories(ImGui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
  )

  target_link_libraries(ImGui
    PUBLIC
      SDL3::SDL3      # or SDL2::SDL2 if you’re on SDL2
      Vulkan::Vulkan
  )
endif()

FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git   
    GIT_TAG        v1.0.6                                             
)

set(TINYOBJLOADER_USE_DOUBLE OFF CACHE BOOL "Build with double precision" FORCE)

FetchContent_MakeAvailable(tinyobjloader)

FetchContent_Declare(
    fetch_vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        v1.3.296
)

FetchContent_MakeAvailable(fetch_vk_bootstrap)

include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.2.0
)

FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  fastgltf
  GIT_REPOSITORY https://github.com/spnda/fastgltf.git
  GIT_TAG        v0.8.0
)

FetchContent_MakeAvailable(fastgltf)

# Collect source files
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Define executable target
add_executable(${PROJECT_NAME} ${SOURCES})

add_dependencies(${PROJECT_NAME} Shaders)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Link external libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        fmt::fmt
        fastgltf
        ImGui
        glm::glm
        SDL3::SDL3
        tinyobjloader
        vk-bootstrap::vk-bootstrap
        Vulkan::Vulkan 
)

# Compile definitions for platform differentiation
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
)

# Set executable suffix for Windows builds
if(TARGET_PLATFORM STREQUAL "Windows")
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".exe")
endif()

# Installation rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

