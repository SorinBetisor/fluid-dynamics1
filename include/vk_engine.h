#pragma once
// -------------------------Incldues-------------------------
#include "glm/ext/vector_float4.hpp"
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vk_desriptors.h>
#include <vk_images.h>
#include <vk_pipelines.h>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

// -------------------------Constants------------------------
constexpr unsigned int FRAME_OVERLAP = 2;

// -------------------------Structs--------------------------
struct DeletionQueue {
  std::deque<std::function<void()>> deletors;

  void push_function(std::function<void()> &&function) {
    deletors.push_back(function);
  }

  void flush() {
    for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
      (*it)();
    };

    deletors.clear();
  }
};

struct FrameData {
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;

  VkSemaphore _swapchainSemaphore, _renderSemaphore;
  VkFence _renderFence;

  DeletionQueue _deletionQueue;
};

struct FluidSimPushConstants {
  glm::uvec2 gridDim; // Grid dimensions (width, height)
  float deltaTime;    // Time step Δt
  float density;   // Fluid density ρ (unused in this vorticity-streamfunction
                   // formulation)
  float viscosity; // Kinematic viscosity ν
  int numPressureIterations; // Iterations for Poisson solver (max_it from
                             // Algorithm 2)
  int numOverallIterations;  // Total simulation steps (likely for host control,
                             // unused in this single time-step shader)
  float omegaSOR;    // Relaxation factor for SOR in Poisson solver (ω_SOR from
                     // Algorithm 2)
  float lidVelocity; // Velocity of the top lid (U0)
  float h;           // Cell size (assuming Δx = Δy = h)
};

// -------------------------Class----------------------------
class VulkanEngine {
public:
  std::string _filePath{};
  bool _isInitialized{false};
  int _frameNumber{0};
  glm::uvec2 _fluidGridDimensions{256, 256}; // Example fluid grid size

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkPhysicalDevice _chosenGPU;
  VkDevice _device;

  VkQueue _graphicsQueue; // This queue can also be used for compute
  uint32_t _graphicsQueueFamily;

  DeletionQueue _mainDeletionQueue;

  VmaAllocator _allocator;

  VkFence _immFence;
  VkCommandBuffer _immCommandBuffer;
  VkCommandPool _immCommandPool;

  // Fluid Simulation Resources
  AllocatedBuffer _fluidVelocityBuffer;
  AllocatedBuffer _fluidVorticityBuffer;
  AllocatedBuffer _fluidPressureBuffer;
  AllocatedBuffer _fluidStreamFunctionBuffer;
  AllocatedBuffer _fluidTempScalarBuffer;
  AllocatedBuffer _fluidTempVecBuffer;

  VkPipeline _vorticityPipeline;
  VkPipeline _advectionPipeline;
  VkPipeline _poissonPipeline;
  VkPipeline _velocityPipeline;
  VkPipeline _pressurePipeline;

  VkDescriptorSetLayout _fluidSimDescriptorLayout;
  VkDescriptorSet _fluidSimDescriptorSet;
  VkPipeline _fluidSimPipeline;
  VkPipelineLayout _fluidSimPipelineLayout;
  FluidSimPushConstants _fluidSimConstants;

  FrameData _frames[FRAME_OVERLAP];

  FrameData &get_current_frame() {
    return _frames[_frameNumber % FRAME_OVERLAP];
  }

  static VulkanEngine &Get();

  DescriptorAllocator globalDescriptorAllocator;

  void init();

  void cleanup();

  void run_simulation_loop();
  void simulation_step();

  void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

private:
  void init_vulkan();

  void init_commands();

  void init_sync_structures();

  void init_descriptors();

  void init_pipelines();
  void init_fluid_simulation_resources();

  void dispatch_fluid_simulation(VkCommandBuffer cmd);

  VkPipeline create_compute_pipeline(const std::string &spvPath);

  AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                VmaMemoryUsage memoryUsage);

  void destroy_buffer(const AllocatedBuffer &buffer);
  template <typename T>
  std::vector<T> read_buffer_to_cpu(const AllocatedBuffer &gpuBuffer,
                                    size_t itemCount);
  template <typename T>
  void write_buffer_to_vtk(const AllocatedBuffer &gpuBuffer,
                           const std::string &dataName,
                           const std::string &baseFilename);
};
