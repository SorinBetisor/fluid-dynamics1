#include "execPath.h"
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <vulkan/vk_enum_string_helper.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <cassert>
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <sys/types.h>
#include <vk_engine.h>

#include "SDL3/SDL_init.h"

#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"

#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

constexpr bool bUseValidationLayers = true;

VulkanEngine *loadedEngine = nullptr;

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init() {
  assert(loadedEngine == nullptr);

  _filePath = getExecutableDir();

  loadedEngine = this;

  init_vulkan();

  init_commands();

  init_sync_structures();

  init_fluid_simulation_resources();

  _isInitialized = true;
}

// usage can be one of:
// VMA_MEMORY_USAGE_GPU_ONLY - intershader communication, cant read/write from
//
// CPU VMA_MEMORY_USAGE_CPU_ONLY - read/write from CPU, can be read from GPU,
// but costly
//
// VMA_MEMORY_USAGE_GPU_TO_CPU - for safe reading from the GPU
//
// VMA_MEMORY_USAGE_CPU_TO_GPU - faster access than CPU_ONLY, if resizeBar
// enabled can grow
AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize,
                                            VkBufferUsageFlags usage,
                                            VmaMemoryUsage memoryUsage) {
  VkBufferCreateInfo bufferInfo = {.sType =
                                       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.pNext = nullptr;
  bufferInfo.size = allocSize;
  bufferInfo.usage = usage;

  VmaAllocationCreateInfo vmaAllocInfo = {};
  vmaAllocInfo.usage = memoryUsage;
  vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
  AllocatedBuffer newBuffer;

  VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
                           &newBuffer.buffer, &newBuffer.allocation,
                           &newBuffer.info));

  return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buffer) {
  vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

// Helper function to read GPU buffer data to CPU
template <typename T>
std::vector<T>
VulkanEngine::read_buffer_to_cpu(const AllocatedBuffer &gpuBuffer,
                                 size_t itemCount) {
  size_t bufferSize = itemCount * sizeof(T);

  AllocatedBuffer stagingBuffer =
      create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_TO_CPU);

  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmd, gpuBuffer.buffer, stagingBuffer.buffer, 1,
                    &copyRegion);
  });

  std::vector<T> data(itemCount);
  if (stagingBuffer.info.pMappedData) {
    memcpy(data.data(), stagingBuffer.info.pMappedData, bufferSize);
  } else {
    // This case should ideally not happen with VMA_MEMORY_USAGE_GPU_TO_CPU
    // and VMA_ALLOCATION_CREATE_MAPPED_BIT.
    // If it does, explicit mapping might be required.
    void *pData;
    vmaMapMemory(_allocator, stagingBuffer.allocation, &pData);
    memcpy(data.data(), pData, bufferSize);
    vmaUnmapMemory(_allocator, stagingBuffer.allocation);
    fmt::println(stderr, "Warning: Staging buffer for reading GPU data was "
                         "manually mapped/unmapped.");
  }

  destroy_buffer(stagingBuffer);
  return data;
}

void VulkanEngine::init_vulkan() {
  vkb::InstanceBuilder builder;

  auto inst_ret = builder.set_app_name("CPPGame")
                      .request_validation_layers(bUseValidationLayers)
                      .use_default_debug_messenger()
                      .require_api_version(1, 3, 0)
                      .build();

  if (!inst_ret) {
    fmt::println("Error picking a device: {} ", inst_ret.error().message());
    abort();
  }

  vkb::Instance vkb_inst = inst_ret.value();

  _instance = vkb_inst.instance;
  _debug_messenger = vkb_inst.debug_messenger;

  VkPhysicalDeviceVulkan13Features features13 = {};
  features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features13.synchronization2 = true;
  features13.dynamicRendering = true;

  VkPhysicalDeviceVulkan12Features features12 = {};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features12.descriptorIndexing = true;
  features12.bufferDeviceAddress = true;

  vkb::PhysicalDeviceSelector selector{vkb_inst};
  auto physDevice_ret = selector.set_minimum_version(1, 3)
                            .set_required_features_13(features13)
                            .set_required_features_12(features12)
                            .require_present(false)
                            .select();

  if (!physDevice_ret) {
    fmt::println(stderr, "Failed to select Vulkan Physical Device: {}",
                 physDevice_ret.error().message());
    abort();
  }
  vkb::PhysicalDevice physicalDevice = physDevice_ret.value();

  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  vkb::Device vkbDevice = deviceBuilder.build().value();

  _device = vkbDevice.device;
  _chosenGPU = physicalDevice.physical_device;

  _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  _graphicsQueueFamily =
      vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  VmaAllocatorCreateInfo allocInfo = {};
  allocInfo.physicalDevice = _chosenGPU;
  allocInfo.device = _device;
  allocInfo.instance = _instance;
  allocInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocInfo, &_allocator);

  _mainDeletionQueue.push_function(
      [this]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_commands() {
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &_frames[i]._commandPool));
    VkCommandBufferAllocateInfo allocInfo =
        vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo,
                                      &_frames[i]._mainCommandBuffer));
    _mainDeletionQueue.push_function([this, i]() {
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
    });
  }

  VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                               &_immCommandPool));

  VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_immCommandPool, 1);

  VK_CHECK(
      vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

  _mainDeletionQueue.push_function(
      [this]() { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures() {
  VkFenceCreateInfo fenceCreateInfo =
      vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
  VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr,
                           &_frames[i]._renderFence))
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &_frames[i]._renderSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &_frames[i]._swapchainSemaphore));
    _mainDeletionQueue.push_function([this, i]() {
      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
    });
  };

  VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
  _mainDeletionQueue.push_function(
      [this]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::immediate_submit(
    std::function<void(VkCommandBuffer cmd)> &&function) {
  VK_CHECK(vkResetFences(_device, 1, &_immFence));
  VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

  VkCommandBuffer cmd = _immCommandBuffer;

  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  function(cmd);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

  VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 999999999999));
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    vkDeviceWaitIdle(_device);

    _mainDeletionQueue.flush();

    vkDestroyDevice(_device, nullptr);

    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);
  }

  loadedEngine = nullptr;
}

void VulkanEngine::init_fluid_simulation_resources() {
  uint32_t numCells = _fluidGridDimensions.x * _fluidGridDimensions.y;
  size_t velocityBufferSize = numCells * sizeof(glm::vec2);
  size_t densityBufferSize = numCells * sizeof(float);
  size_t pressureBufferSize = numCells * sizeof(float);
  size_t streamFuncBufferSize = numCells * sizeof(float);
  size_t tempBufferSize =
      numCells *
      sizeof(glm::vec2); // temporary storage (sized for the maximum possible
                         // size) cause we cant update the variables in place

  _fluidVelocityBuffer = create_buffer(velocityBufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidDensityBuffer = create_buffer(densityBufferSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidPressureBuffer = create_buffer(pressureBufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidStreamFunctionBuffer = create_buffer(
      streamFuncBufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidTempBuffer = create_buffer(tempBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VMA_MEMORY_USAGE_GPU_ONLY);

  std::vector<float> initialDensities(numCells, 0.0f);
  for (uint32_t y = _fluidGridDimensions.y / 4;
       y < 3 * _fluidGridDimensions.y / 4; ++y) {
    for (uint32_t x = _fluidGridDimensions.x / 4;
         x < 3 * _fluidGridDimensions.x / 4; ++x) {
      if (x > _fluidGridDimensions.x / 2 - 20 &&
          x < _fluidGridDimensions.x / 2 + 20 &&
          y > _fluidGridDimensions.y / 2 - 20 &&
          y < _fluidGridDimensions.y / 2 + 20)
        initialDensities[y * _fluidGridDimensions.x + x] = 1.0f;
    }
  }

  AllocatedBuffer stagingDensityBuffer =
      create_buffer(densityBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_CPU_ONLY);
  memcpy(stagingDensityBuffer.allocation->GetMappedData(),
         initialDensities.data(), densityBufferSize);
  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy copyRegion = {};
    copyRegion.dstOffset = 0;
    copyRegion.srcOffset = 0;
    copyRegion.size = densityBufferSize;
    vkCmdCopyBuffer(cmd, stagingDensityBuffer.buffer,
                    _fluidDensityBuffer.buffer, 1, &copyRegion);
  });
  destroy_buffer(stagingDensityBuffer);

  // Always initialize the global descriptor allocator's pool
  // as this function is called once during engine initialization.
  // The previous check (globalDescriptorAllocator.pool == VK_NULL_HANDLE)
  // could fail if the pool was uninitialized with a non-null garbage value.
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40}};
  globalDescriptorAllocator.init_pool(_device, 10, sizes);
  _mainDeletionQueue.push_function(
      [this]() { globalDescriptorAllocator.destroy_pool(_device); });

  DescriptorLayoutBuilder builder;
  builder.add_bindings(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  _fluidSimDescriptorLayout =
      builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

  _fluidSimDescriptorSet =
      globalDescriptorAllocator.allocate(_device, _fluidSimDescriptorLayout);

  VkDescriptorBufferInfo velocityBufferInfo{};
  velocityBufferInfo.buffer = _fluidVelocityBuffer.buffer;
  velocityBufferInfo.offset = 0;
  velocityBufferInfo.range = velocityBufferSize;

  VkDescriptorBufferInfo densityBufferInfo{};
  densityBufferInfo.buffer = _fluidDensityBuffer.buffer;
  densityBufferInfo.offset = 0;
  densityBufferInfo.range = densityBufferSize;

  VkDescriptorBufferInfo pressureBufferInfo{};
  pressureBufferInfo.buffer = _fluidPressureBuffer.buffer;
  pressureBufferInfo.offset = 0;
  pressureBufferInfo.range = pressureBufferSize;

  VkDescriptorBufferInfo streamFuncBufferInfo{};
  streamFuncBufferInfo.buffer = _fluidStreamFunctionBuffer.buffer;
  streamFuncBufferInfo.offset = 0;
  streamFuncBufferInfo.range = streamFuncBufferSize;

  VkDescriptorBufferInfo tempBufferInfo{};
  streamFuncBufferInfo.buffer = _fluidTempBuffer.buffer;
  streamFuncBufferInfo.offset = 0;
  streamFuncBufferInfo.range = tempBufferSize;

  VkWriteDescriptorSet writes[5];
  writes[0] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &velocityBufferInfo, 0);
  writes[1] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &densityBufferInfo, 1);
  writes[2] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &pressureBufferInfo, 2);
  writes[3] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &streamFuncBufferInfo, 3);
  writes[4] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &tempBufferInfo, 4);

  vkUpdateDescriptorSets(_device, 5, writes, 0, nullptr);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo =
      vkinit::pipeline_layout_create_info();
  pipelineLayoutInfo.pSetLayouts = &_fluidSimDescriptorLayout;
  pipelineLayoutInfo.setLayoutCount = 1;

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(FluidSimPushConstants);
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  pipelineLayoutInfo.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr,
                                  &_fluidSimPipelineLayout));

  VkShaderModule fluidComputeShader;
  std::string shaderPath = _filePath + "/shaders/navier.comp.spv";
  if (!vkutil::load_shader_module(shaderPath.c_str(), _device,
                                  &fluidComputeShader)) {
    fmt::println(stderr, "Error when building the fluid compute shader: {}",
                 shaderPath);
  } else {
    fmt::println("Fluid compute shader successfully loaded: {}", shaderPath);
  }

  VkPipelineShaderStageCreateInfo stageInfo{};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = fluidComputeShader;
  stageInfo.pName = "main";

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.layout = _fluidSimPipelineLayout;
  computePipelineCreateInfo.stage = stageInfo;

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &computePipelineCreateInfo, nullptr,
                                    &_fluidSimPipeline));

  vkDestroyShaderModule(_device, fluidComputeShader, nullptr);

  _mainDeletionQueue.push_function([this]() {
    vkDestroyPipelineLayout(_device, _fluidSimPipelineLayout, nullptr);
    vkDestroyPipeline(_device, _fluidSimPipeline, nullptr);
    vkDestroyDescriptorSetLayout(_device, _fluidSimDescriptorLayout, nullptr);
    destroy_buffer(_fluidVelocityBuffer);
    destroy_buffer(_fluidDensityBuffer);
    destroy_buffer(_fluidPressureBuffer);
    destroy_buffer(_fluidStreamFunctionBuffer);
    destroy_buffer(_fluidTempBuffer);
  });
}

void VulkanEngine::dispatch_fluid_simulation(VkCommandBuffer cmd) {
  VkMemoryBarrier2 memoryBarrier = {.sType =
                                        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memoryBarrier.dstAccessMask =
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

  VkDependencyInfo dependencyInfo = {.sType =
                                         VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.memoryBarrierCount = 1;
  dependencyInfo.pMemoryBarriers = &memoryBarrier;

  vkCmdPipelineBarrier2(cmd, &dependencyInfo);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _fluidSimPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _fluidSimPipelineLayout, 0, 1,
                          &_fluidSimDescriptorSet, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _fluidSimPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _fluidSimPipelineLayout, 0, 1,
                          &_fluidSimDescriptorSet, 0, nullptr);

  _fluidSimConstants.gridDim = _fluidGridDimensions; // e.g., {128, 128}
  _fluidSimConstants.deltaTime = 0.001f; // Adjust as needed for stability
  _fluidSimConstants.density =
      1.0f; // Nominal density, may not be directly used by shader
  _fluidSimConstants.viscosity =
      0.001f; // Kinematic viscosity (ν). For Re=U0*L/ν. If L=1, U0=1, Re=1000.
  _fluidSimConstants.numPressureIterations =
      50; // Iterations for Poisson solver for ψ
  _fluidSimConstants.numOverallIterations = 1; // Unused by this shader

  // For SOR method for Poisson equation ∇²ψ = -ω
  // Optimal omegaSOR is problem-dependent, typically between 1.0 (Gauss-Seidel)
  // and 2.0. For lid-driven cavity, values like 1.7-1.9 can be good. Start
  // with 1.5 or 1.7.
  _fluidSimConstants.omegaSOR = 1.7f;

  // Lid-driven cavity specific parameters
  _fluidSimConstants.lidVelocity = 1.0f; // Velocity U0 of the top lid
  if (_fluidGridDimensions.x > 1) {
    // Assuming domain is [0,1] x [0,1] or [0,width_units] x [0,height_units]
    // h = physical_length / (number_of_cells_in_that_direction -1)
    // For simplicity, if physical length of cavity is 1.0:
    _fluidSimConstants.h =
        1.0f / (float)(_fluidGridDimensions.x -
                       1); // Cell size, assuming square cells and Width=1.0
  } else {
    _fluidSimConstants.h = 0.1f; // Default placeholder if grid is tiny
  }

  vkCmdPushConstants(cmd, _fluidSimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(FluidSimPushConstants), &_fluidSimConstants);

  uint32_t groupSizeX = 16; // Must match local_size_x in shader
  uint32_t groupSizeY = 16; // Must match local_size_y in shader
  vkCmdDispatch(cmd, (_fluidGridDimensions.x + groupSizeX - 1) / groupSizeX,
                (_fluidGridDimensions.y + groupSizeY - 1) / groupSizeY, 1);
}

void VulkanEngine::simulation_step() {
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           1000000000));
  get_current_frame()._deletionQueue.flush();
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  dispatch_fluid_simulation(cmd);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit,
                          get_current_frame()._renderFence));

  // Wait for the simulation to complete before reading buffers
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           999999999999));
}

void VulkanEngine::run_simulation_loop() {
  bool bQuit = false;

  while (!bQuit) {
    simulation_step();
    // Read and print buffer contents
    uint32_t numCells = _fluidGridDimensions.x * _fluidGridDimensions.y;
    uint32_t printCount =
        std::min(numCells, 5u); // Print first 5 elements or less

    fmt::println("---- Iteration step {} ----", _frameNumber);

    // Velocity Buffer
    std::vector<glm::vec2> velocities =
        read_buffer_to_cpu<glm::vec2>(_fluidVelocityBuffer, numCells);

    // Density Buffer
    std::vector<float> densities =
        read_buffer_to_cpu<float>(_fluidDensityBuffer, numCells);

    // Pressure Buffer
    std::vector<float> pressures =
        read_buffer_to_cpu<float>(_fluidPressureBuffer, numCells);

    // Stream Function Buffer
    std::vector<float> streamFuncs =
        read_buffer_to_cpu<float>(_fluidStreamFunctionBuffer, numCells);
    fmt::println("---- End of Frame {} ----", _frameNumber);

    _frameNumber++;

    if (_frameNumber % 10 == 0) { // Example: output every 10 frames
      write_buffer_to_vtk<glm::vec2>(_fluidVelocityBuffer, "Velocity",
                                     "cavity_sim");
      write_buffer_to_vtk<float>(_fluidDensityBuffer, "Vorticity",
                                 "cavity_sim");
      write_buffer_to_vtk<float>(_fluidStreamFunctionBuffer, "StreamFunction",
                                 "cavity_sim");
      write_buffer_to_vtk<float>(_fluidPressureBuffer, "Pressure",
                                 "cavity_sim");
    }
  }
}

template <typename T>
void VulkanEngine::write_buffer_to_vtk(const AllocatedBuffer &gpuBuffer,
                                       const std::string &dataName,
                                       const std::string &baseFilename) {
  if (!_isInitialized) {
    std::cerr << "VulkanEngine not initialized, cannot write VTK for "
              << dataName << "." << std::endl;
    return;
  }

  uint32_t gridWidth = _fluidGridDimensions.x;
  uint32_t gridHeight = _fluidGridDimensions.y;
  size_t totalPoints = static_cast<size_t>(gridWidth) * gridHeight;

  if (totalPoints == 0) {
    std::cerr << "Error: Grid dimensions are zero (" << gridWidth << "x"
              << gridHeight << "), cannot write VTK for " << dataName
              << std::endl;
    return;
  }

  std::vector<T> cpuData = read_buffer_to_cpu<T>(gpuBuffer, totalPoints);

  if (cpuData.empty()) {
    std::cerr << "Error: Failed to read buffer to CPU for VTK export: "
              << dataName << std::endl;
    return;
  }
  if (cpuData.size() != totalPoints) {
    std::cerr << "Error: Read incorrect number of items from GPU buffer for "
              << dataName << ". Expected " << totalPoints << ", got "
              << cpuData.size() << std::endl;
    return;
  }

  // Ensure output directory exists
  const std::string outputDir =
      _filePath +
      "/output_vtk"; // Changed from "./output" to avoid potential conflicts
  try {
    if (!std::filesystem::exists(outputDir)) {
      std::filesystem::create_directories(outputDir);
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Error creating output directory " << outputDir << ": "
              << e.what() << std::endl;
    return;
  }

  std::string filename = outputDir + "/" + baseFilename + "_" + dataName + "_" +
                         std::to_string(_frameNumber) + ".vtk";
  std::ofstream vtkFile(filename);

  if (!vtkFile.is_open()) {
    std::cerr << "Error: Could not open VTK file for writing: " << filename
              << std::endl;
    return;
  }

  std::cout << "Writing VTK file: " << filename << std::endl;

  // VTK Headers
  vtkFile << "# vtk DataFile Version 2.0\n";
  vtkFile << dataName << " data from frame " << _frameNumber
          << " (grid: " << gridWidth << "x" << gridHeight
          << ")\n"; // More descriptive title
  vtkFile << "ASCII\n";
  vtkFile << "DATASET STRUCTURED_POINTS\n";
  vtkFile << "DIMENSIONS " << gridWidth << " " << gridHeight
          << " 1\n";            // nx ny nz
  vtkFile << "ORIGIN 0 0 0\n";  // Assuming origin is 0,0,0 for simplicity
  vtkFile << "SPACING 1 1 1\n"; // Assuming unit spacing for simplicity
  float spacing_h =
      _fluidSimConstants.h; // Get h from where it's stored/calculated
  if (spacing_h <= 0.0f)
    spacing_h = 1.0f; // Fallback if h is not set
  vtkFile << "ORIGIN 0 0 0\n";
  vtkFile << "SPACING " << std::fixed << std::setprecision(6) << spacing_h
          << " " << spacing_h << " " << spacing_h << "\n";

  vtkFile << "POINT_DATA " << totalPoints << "\n";

  // Type-specific header and data writing
  // VTK data order for STRUCTURED_POINTS: iterate x, then y, then z
  // For our 2D case (z=1): iterate i from 0 to gridWidth-1 (fastest), then j
  // from 0 to gridHeight-1 (slower) This corresponds to cpuData[j * gridWidth +
  // i] for row-major storage.

  if constexpr (std::is_same_v<T, float>) {
    vtkFile << "SCALARS " << dataName
            << " float 1\n"; // "1" is num_components, optional for scalars if 1
    vtkFile << "LOOKUP_TABLE default\n";
    vtkFile << std::fixed << std::setprecision(6);
    for (uint32_t j = 0; j < gridHeight; ++j) {  // y-coordinate (row)
      for (uint32_t i = 0; i < gridWidth; ++i) { // x-coordinate (column)
        size_t index = j * gridWidth + i;        // Row-major access
        vtkFile << (i == 0 ? "" : " ") << cpuData[index];
      }
      vtkFile << "\n";
    }
  } else if constexpr (std::is_same_v<T, glm::vec2>) {
    vtkFile << "VECTORS " << dataName
            << " float\n"; // VTK expects 3 components for VECTORS in
                           // structured_points. The data type (float) is
                           // specified here.
    vtkFile << std::fixed << std::setprecision(6);
    for (uint32_t j = 0; j < gridHeight; ++j) {  // y-coordinate (row)
      for (uint32_t i = 0; i < gridWidth; ++i) { // x-coordinate (column)
        size_t index = j * gridWidth + i;        // Row-major access
        vtkFile << cpuData[index].x << " " << cpuData[index].y
                << " 0.0\n"; // Add a Z=0 component
      }
    }
  } else {
    vtkFile.close();
    // std::filesystem::remove(filename);
    std::cerr << "Error: Unsupported data type for VTK export: "
              << typeid(T).name() << ". Only float and glm::vec2 are supported."
              << std::endl;
    // throw std::runtime_error("Unsupported data type for VTK export: " +
    // std::string(typeid(T).name()));
    return;
  }

  vtkFile.close();
  if (!vtkFile) { // Check for errors during close or writing
    std::cerr << "Error writing or closing VTK file: " << filename << std::endl;
  }
}

// Explicit template instantiations (optional, but can help catch compilation
// errors earlier and speed up linking) Place these at the end of
// VulkanEngine.cpp if you want them. Make sure glm::vec2 is defined before this
// point.
template void VulkanEngine::write_buffer_to_vtk<float>(const AllocatedBuffer &,
                                                       const std::string &,
                                                       const std::string &);
template void VulkanEngine::write_buffer_to_vtk<glm::vec2>(
    const AllocatedBuffer &, const std::string &, const std::string &);
