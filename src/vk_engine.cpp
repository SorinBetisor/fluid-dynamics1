#include "execPath.h"
#include "glm/ext/vector_float2.hpp"
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
#include <fmt/os.h>
#include <sys/types.h>
#include <vk_engine.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"

#include <vulkan/vulkan_core.h>

#include <chrono>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::duration<double, std::milli>;

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

  init_constants();

  _isInitialized = true;
}

void VulkanEngine::init_constants() {
  // for numerical stability:
  //  deltaTime <= min( h/lidVelocity, 0.25 * h^2 / viscosity )

  // Numerical and physical parameters, set up for a 1000 Re test
  _fluidSimConstants.gridDim = _fluidGridDimensions;
  _fluidSimConstants.deltaTime =
      0.001f; // just for safety this should set to <= .1 * the result you got
              // from the equation above
  _fluidSimConstants.viscosity = 0.001f;
  _fluidSimConstants.numJacobiIterations =
      1000; // the heigher this number the more (host controll) dispatches of
            // the Jacobi solver there will be
  _fluidSimConstants.omegaSOR =
      1.0f; // can be increased for faster convergance, 1 is Gauss-Seidel, which
            // is generaly stable
  _fluidSimConstants.h = 1.0f / static_cast<float>(_fluidGridDimensions.x);
  _fluidSimConstants.lidVelocity = _fluidSimConstants.h;

  // host controll
  _numOveralIterations = 100000;
  _saveInterval =
      100; // carefull with this as this is the time where ALL of the data kept
           // in the buffers is read to the cpu, which is slow
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

  VkPhysicalDeviceVulkan12Features features12 = {};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
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
  write_buffer_to_vtk<glm::vec2>(_fluidVelocityBuffer, "Velocity",
                                 "cavity_sim");
  write_buffer_to_vtk<float>(_fluidVorticityBuffer, "Vorticity", "cavity_sim");
  write_buffer_to_vtk<float>(_fluidStreamFunctionBuffer, "StreamFunction",
                             "cavity_sim");
  write_buffer_to_vtk<float>(_fluidPressureBuffer, "Pressure", "cavity_sim");

  fmt::println("Saved simulation state");

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
  size_t vorticityBufferSize = numCells * sizeof(float);
  size_t pressureBufferSize = numCells * sizeof(float);
  size_t streamFuncBufferSize = numCells * sizeof(float);
  size_t tempScalarBufferSize = numCells * sizeof(float);
  size_t tempVecBufferSize = numCells * sizeof(glm::vec2);

  _fluidVelocityBuffer = create_buffer(velocityBufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidVorticityBuffer = create_buffer(vorticityBufferSize,
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

  _fluidTempScalarBuffer = create_buffer(tempScalarBufferSize,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VMA_MEMORY_USAGE_GPU_ONLY);

  _fluidTempVecBuffer = create_buffer(tempVecBufferSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VMA_MEMORY_USAGE_GPU_ONLY);

  std::vector<float> initialScalars(numCells, 0.0f);
  std::vector<glm::vec2> initialVec2s(numCells, {0.0f, 0.0f});

  AllocatedBuffer stagingScalarBuffer =
      create_buffer(vorticityBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_CPU_ONLY);
  memcpy(stagingScalarBuffer.allocation->GetMappedData(), initialScalars.data(),
         tempScalarBufferSize);
  AllocatedBuffer stagingVecBuffer =
      create_buffer(tempVecBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_CPU_ONLY);
  memcpy(stagingVecBuffer.allocation->GetMappedData(), initialVec2s.data(),
         tempVecBufferSize);
  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy copyRegion = {};
    copyRegion.dstOffset = 0;
    copyRegion.srcOffset = 0;
    copyRegion.size = vorticityBufferSize;
    vkCmdCopyBuffer(cmd, stagingScalarBuffer.buffer,
                    _fluidVorticityBuffer.buffer, 1, &copyRegion);
    vkCmdCopyBuffer(cmd, stagingScalarBuffer.buffer,
                    _fluidPressureBuffer.buffer, 1, &copyRegion);
    vkCmdCopyBuffer(cmd, stagingScalarBuffer.buffer,
                    _fluidStreamFunctionBuffer.buffer, 1, &copyRegion);
    vkCmdCopyBuffer(cmd, stagingScalarBuffer.buffer,
                    _fluidTempScalarBuffer.buffer, 1, &copyRegion);

    copyRegion.size = tempScalarBufferSize;
    vkCmdCopyBuffer(cmd, stagingVecBuffer.buffer, _fluidVelocityBuffer.buffer,
                    1, &copyRegion);
    vkCmdCopyBuffer(cmd, stagingVecBuffer.buffer, _fluidTempVecBuffer.buffer, 1,
                    &copyRegion);
  });
  destroy_buffer(stagingScalarBuffer);
  destroy_buffer(stagingVecBuffer);

  // Always initialize the global descriptor allocator's pool
  // as this function is called once during engine initialization.
  // The previous check (globalDescriptorAllocator.pool == VK_NULL_HANDLE)
  // could fail if the pool was uninitialized with a non-null garbage value.
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 50}};
  globalDescriptorAllocator.init_pool(_device, 10, sizes);
  _mainDeletionQueue.push_function(
      [this]() { globalDescriptorAllocator.destroy_pool(_device); });

  DescriptorLayoutBuilder builder;
  builder.add_bindings(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  builder.add_bindings(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  _fluidSimDescriptorLayout =
      builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

  _fluidSimDescriptorSet =
      globalDescriptorAllocator.allocate(_device, _fluidSimDescriptorLayout);

  VkDescriptorBufferInfo velocityBufferInfo{};
  velocityBufferInfo.buffer = _fluidVelocityBuffer.buffer;
  velocityBufferInfo.offset = 0;
  velocityBufferInfo.range = velocityBufferSize;

  VkDescriptorBufferInfo vorticityBufferInfo{};
  vorticityBufferInfo.buffer = _fluidVorticityBuffer.buffer;
  vorticityBufferInfo.offset = 0;
  vorticityBufferInfo.range = vorticityBufferSize;

  VkDescriptorBufferInfo pressureBufferInfo{};
  pressureBufferInfo.buffer = _fluidPressureBuffer.buffer;
  pressureBufferInfo.offset = 0;
  pressureBufferInfo.range = pressureBufferSize;

  VkDescriptorBufferInfo streamFuncBufferInfo{};
  streamFuncBufferInfo.buffer = _fluidStreamFunctionBuffer.buffer;
  streamFuncBufferInfo.offset = 0;
  streamFuncBufferInfo.range = streamFuncBufferSize;

  VkDescriptorBufferInfo tempScalarBufferInfo{};
  tempScalarBufferInfo.buffer = _fluidTempScalarBuffer.buffer;
  tempScalarBufferInfo.offset = 0;
  tempScalarBufferInfo.range = tempScalarBufferSize;

  VkDescriptorBufferInfo tempVecBufferInfo{};
  tempVecBufferInfo.buffer = _fluidTempVecBuffer.buffer;
  tempVecBufferInfo.offset = 0;
  tempVecBufferInfo.range = tempVecBufferSize;

  VkWriteDescriptorSet writes[6];
  writes[0] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &velocityBufferInfo, 0);
  writes[1] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &vorticityBufferInfo, 1);
  writes[2] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &pressureBufferInfo, 2);
  writes[3] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &streamFuncBufferInfo, 3);
  writes[4] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &tempScalarBufferInfo, 4);
  writes[5] = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              _fluidSimDescriptorSet,
                                              &tempVecBufferInfo, 5);

  vkUpdateDescriptorSets(_device, 6, writes, 0, nullptr);

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

  _vorticityPipeline =
      create_compute_pipeline(_filePath + "/shaders/vorticity.comp.spv");
  _advectionPipeline =
      create_compute_pipeline(_filePath + "/shaders/advectDiffuse.comp.spv");
  _poissonPipeline =
      create_compute_pipeline(_filePath + "/shaders/poissonSOR.comp.spv");
  _velocityPipeline =
      create_compute_pipeline(_filePath + "/shaders/velocityFromPsi.comp.spv");
  // optional:
  _pressurePipeline =
      create_compute_pipeline(_filePath + "/shaders/pressureSOR.comp.spv");

  _mainDeletionQueue.push_function([this]() {
    vkDestroyPipeline(_device, _vorticityPipeline, nullptr);
    vkDestroyPipeline(_device, _advectionPipeline, nullptr);
    vkDestroyPipeline(_device, _poissonPipeline, nullptr);
    vkDestroyPipeline(_device, _velocityPipeline, nullptr);
    vkDestroyPipeline(_device, _pressurePipeline, nullptr);
    vkDestroyPipelineLayout(_device, _fluidSimPipelineLayout, nullptr);
    // vkDestroyPipeline(_device, _fluidSimPipeline, nullptr);
    vkDestroyDescriptorSetLayout(_device, _fluidSimDescriptorLayout, nullptr);
    destroy_buffer(_fluidVelocityBuffer);
    destroy_buffer(_fluidVorticityBuffer);
    destroy_buffer(_fluidPressureBuffer);
    destroy_buffer(_fluidStreamFunctionBuffer);
    destroy_buffer(_fluidTempVecBuffer);
    destroy_buffer(_fluidTempScalarBuffer);
  });
}

VkPipeline VulkanEngine::create_compute_pipeline(const std::string &spvPath) {
  // 1) load module
  VkShaderModule module;
  if (!vkutil::load_shader_module(spvPath.c_str(), _device, &module)) {
    throw std::runtime_error("failed loading " + spvPath);
  }

  // 2) stage info
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";

  // 3) pipeline create
  VkComputePipelineCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pci.layout = _fluidSimPipelineLayout; // reuse the one you built
  pci.stage = stage;

  VkPipeline p;
  VK_CHECK(
      vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pci, nullptr, &p));

  vkDestroyShaderModule(_device, module, nullptr);
  return p;
}

void VulkanEngine::dispatch_fluid_simulation(const VkCommandBuffer &cmd) {
  auto debug = [this]() { // pull the data to the CPU to check if there are
                          // nulls/inf, unused rn
    std::vector<float> Vorticity = read_buffer_to_cpu<float>(
        _fluidVorticityBuffer, _fluidGridDimensions.x * _fluidGridDimensions.y);
    std::vector<glm::vec2> Velocity = read_buffer_to_cpu<glm::vec2>(
        _fluidVelocityBuffer, _fluidGridDimensions.x * _fluidGridDimensions.y);
    std::vector<float> Pressure = read_buffer_to_cpu<float>(
        _fluidPressureBuffer, _fluidGridDimensions.x * _fluidGridDimensions.y);
    std::vector<float> StreamFunction = read_buffer_to_cpu<float>(
        _fluidStreamFunctionBuffer,
        _fluidGridDimensions.x * _fluidGridDimensions.y);

    bool hadNaN = false;
    bool hadInf = false;
    for (size_t i = 0; i < _fluidGridDimensions.x * _fluidGridDimensions.y;
         i++) {
      // Calculate grid position
      size_t x = i % _fluidGridDimensions.x;
      size_t y = i / _fluidGridDimensions.x;

      bool hasNaN = false;
      bool hasInf = false;
      std::string nanFields;
      std::string infFields;

      // Check Vorticity
      if (std::isnan(Vorticity[i])) {
        hasNaN = true;
        hadNaN = true;
        nanFields += "Vorticity ";
      } else if (std::isinf(Vorticity[i])) {
        hasInf = true;
        hadInf = true;
        infFields += "Vorticity ";
      }

      // Check Velocity
      if (std::isnan(Velocity[i].x) || std::isnan(Velocity[i].y)) {
        hasNaN = true;
        hadNaN = true;
        nanFields += "Velocity ";
      } else if (std::isinf(Velocity[i].x) || std::isinf(Velocity[i].y)) {
        hasInf = true;
        hadInf = true;
        infFields += "Velocity ";
      }

      // Check Pressure
      if (std::isnan(Pressure[i])) {
        hasNaN = true;
        nanFields += "Pressure ";
      } else if (std::isinf(Pressure[i])) {
        hasInf = true;
        infFields += "Pressure ";
      }

      // Check StreamFunction
      if (std::isnan(StreamFunction[i])) {
        hasNaN = true;
        hadNaN = true;
        nanFields += "StreamFunction ";
      } else if (std::isinf(StreamFunction[i])) {
        hasInf = true;
        hadInf = true;
        infFields += "StreamFunction ";
      }

      if (hasNaN) {
        fmt::println("NaN detected at grid position ({}, {}): {}", x, y,
                     nanFields);
      } else if (hasInf) {
        fmt::println("Inf detected at grid position ({}, {}): {}", x, y,
                     infFields);
      }
    }
    if (hadNaN || hadInf)
      throw std::runtime_error("NaN values detected in simulation");
  };

  // Helper to bind pipeline, descriptors, and push constants then dispatch
  auto run_compute_pass = [&](const VkPipeline &pipeline) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _fluidSimPipelineLayout, 0, 1,
                            &_fluidSimDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, _fluidSimPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(FluidSimPushConstants), &_fluidSimConstants);
    const uint32_t gx = (_fluidGridDimensions.x + 15) / 16;
    const uint32_t gy = (_fluidGridDimensions.y + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
  };

  VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  depInfo.memoryBarrierCount = 1;
  depInfo.pMemoryBarriers = &memBarrier;

  // helper for the barriers
  auto create_barrier = [&](const VkPipelineStageFlags2 &srcStage,
                            const VkAccessFlags2 &srcAccess,
                            const VkPipelineStageFlags2 &dstStage,
                            const VkAccessFlags2 &dstAccess) {
    memBarrier.srcStageMask = srcStage;
    memBarrier.srcAccessMask = srcAccess;
    memBarrier.dstStageMask = dstStage;
    memBarrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  };

  // Copy regions
  VkBufferCopy copyRegionScalar{
      0, 0, _fluidGridDimensions.x * _fluidGridDimensions.y * sizeof(float)};
  VkBufferCopy copyRegionVec{0, 0,
                             _fluidGridDimensions.x * _fluidGridDimensions.y *
                                 sizeof(glm::vec2)};

  // --- Simulation Cycle ---
  // Some of the simulation step outlines may be wrong, make sure to check the
  // shaders

  // 1. Advect Vorticity: ω_new = Advect(ω_old, u, dt)
  //    Inputs: _fluidVorticityBuffer (ω_old), _fluidVelocityBuffer (u)
  //    Output: _fluidTempScalarBuffer (ω_new_temp)
  //    Shader: _advectionPipeline
  run_compute_pass(_advectionPipeline);
  create_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
  vkCmdCopyBuffer(cmd, _fluidTempScalarBuffer.buffer,
                  _fluidVorticityBuffer.buffer, 1, &copyRegionScalar);
  create_barrier(
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  // fmt::println("Advection step completed.");
  // debug();

  // 2. Solve Poisson for Stream Function: ∇²ψ = -ω
  //    Inputs: _fluidVorticityBuffer (ω)
  //    Output: _fluidStreamFunctionBuffer (ψ) (via _fluidTempScalarBuffer)
  //    Shader: _poissonPipeline
  for (int i = 0; i < _fluidSimConstants.numJacobiIterations; ++i) {
    run_compute_pass(_poissonPipeline);
    create_barrier(
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyBuffer(cmd, _fluidTempScalarBuffer.buffer,
                    _fluidStreamFunctionBuffer.buffer, 1, &copyRegionScalar);
    if (i < _fluidSimConstants.numJacobiIterations - 1) {
      create_barrier(
          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }
  }
  create_barrier(
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  // fmt::println("Poisson step completed.");
  // debug();

  // 3. Calculate Velocity from Stream Function: u = ∇×ψ
  //    Inputs: _fluidStreamFunctionBuffer (ψ)
  //    Output: _fluidVelocityBuffer (u) (via _fluidTempVecBuffer)
  //    Shader: _velocityPipeline
  run_compute_pass(_velocityPipeline);
  create_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
  vkCmdCopyBuffer(cmd, _fluidTempVecBuffer.buffer, _fluidVelocityBuffer.buffer,
                  1, &copyRegionVec);
  create_barrier(
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  // fmt::println("Velocity calculation step completed.");
  // debug();

  // 4. Update Vorticity from Velocity (ω = ∇×u),
  // especially for boundaries
  //    Inputs: _fluidVelocityBuffer (u)
  //    Output: _fluidVorticityBuffer (ω) (via _fluidTempScalarBuffer)
  //    Shader: _vorticityPipeline
  //    This step ensures ω is consistent with u and applies boundary conditions
  //    for ω derived from u.
  run_compute_pass(_vorticityPipeline);
  create_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
  vkCmdCopyBuffer(cmd, _fluidTempScalarBuffer.buffer,
                  _fluidVorticityBuffer.buffer, 1, &copyRegionScalar);
  create_barrier(
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  // fmt::println("Vorticity step completed.");
  // debug();

  // // 5. Optional Pressure Pass (if needed for visualization or other physics)
  // //    Usually, for incompressible flow, pressure ensures ∇·u = 0.
  // //    In vorticity-streamfunction, this is implicitly handled if ψ is
  // solved
  // //    correctly. This pressure pass might be for deriving P from u,v (e.g.,
  // //    solving ∇²P = -∇·(u·∇u)).
  // if (_pressurePipeline != VK_NULL_HANDLE) {
  //   for (int i = 0; i < _fluidSimConstants.numPressureIterations; ++i) {
  //     run_compute_pass(_pressurePipeline);
  //     create_barrier(
  //         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //         VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
  //         VK_ACCESS_2_TRANSFER_READ_BIT);
  //     vkCmdCopyBuffer(cmd, _fluidTempScalarBuffer.buffer,
  //                     _fluidPressureBuffer.buffer, 1, &copyRegionScalar);
  //     if (i < _fluidSimConstants.numPressureIterations - 1) {
  //       create_barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
  //                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
  //                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                      VK_ACCESS_2_SHADER_READ_BIT);
  //     }
  //   }
  //   // No barrier needed after final copy if pressure isn't read by compute
  //   // immediately after
  // }
  // fmt::println("Pressure step completed.");
  // debug();
}

void VulkanEngine::simulation_step() {
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           1000000000));
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

  TimePoint t0 = Clock::now();

  while (!bQuit && _frameNumber < _numOveralIterations) {
    TimePoint iterStart = Clock::now();
    simulation_step();
    TimePoint iterEnd = Clock::now();
    Duration iterDur = iterEnd - iterStart;
    // This is the best we can do without pulling the data from the GPU (which
    // is very slow)
    fmt::println(" Iteration {}, took: {} ms", _frameNumber, iterDur.count());

    _frameNumber++;

    if (_frameNumber % _saveInterval == 0) {
      write_buffer_to_vtk<glm::vec2>(_fluidVelocityBuffer, "Velocity",
                                     "cavity_sim");
      write_buffer_to_vtk<float>(_fluidVorticityBuffer, "Vorticity",
                                 "cavity_sim");
      write_buffer_to_vtk<float>(_fluidStreamFunctionBuffer, "StreamFunction",
                                 "cavity_sim");
      write_buffer_to_vtk<float>(_fluidPressureBuffer, "Pressure",
                                 "cavity_sim");

      fmt::println("Saved simulation state");
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
