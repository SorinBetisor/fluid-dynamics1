#include "fluidSim.h"
#include "app.h"
#include "shader.h"
#include <assert.h>
#include <vulkan/vulkan.h> // for your Vulkan includes, if you have them consolidated

uint32_t findMemoryTypeSim(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  printf("failed to find suitable memory type!\n");
  abort();
}

void createFluidSim(VkDevice device, VkPhysicalDevice physicalDevice,
                    VkDescriptorPool descriptorPool, FluidSim *fluidSim) {
  // Descriptor set layout bindings for both buffer and image
  VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};

  VkDescriptorSetLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
      .pBindings = bindings};

  VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL,
                                       &fluidSim->descriptorSetLayout));

  VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(float) * 4 * 1024 * 1024, // Adjust size as needed
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VK_CHECK(
      vkCreateBuffer(device, &bufferInfo, NULL, &fluidSim->velocityBuffer));

  // Get memory requirements for velocity buffer
  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, fluidSim->velocityBuffer, &memReqs);

  // Allocate memory for velocity buffer
  VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memReqs.size,
      .memoryTypeIndex =
          findMemoryTypeSim(physicalDevice, memReqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL,
                            &fluidSim->velocityBufferMemory));
  VK_CHECK(vkBindBufferMemory(device, fluidSim->velocityBuffer,
                              fluidSim->velocityBufferMemory, 0));

  // Create simulation parameters buffer
  VkBufferCreateInfo uniformInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(float) * 2, // dt and viscosity
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VK_CHECK(
      vkCreateBuffer(device, &uniformInfo, NULL, &fluidSim->simParamsBuffer));

  // Get memory requirements for params buffer
  vkGetBufferMemoryRequirements(device, fluidSim->simParamsBuffer, &memReqs);

  // Allocate memory for params buffer
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex =
      findMemoryTypeSim(physicalDevice, memReqs.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL,
                            &fluidSim->simParamsBufferMemory));
  VK_CHECK(vkBindBufferMemory(device, fluidSim->simParamsBuffer,
                              fluidSim->simParamsBufferMemory, 0));

  // Pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &fluidSim->descriptorSetLayout};

  VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL,
                                  &fluidSim->pipelineLayout));

  // Compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = ({
                  Shaderfile shaderFile =
                      readShaderFile("shaders/compFluid.spv");
                  VkShaderModuleCreateInfo createInfo = {
                      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                      .codeSize = shaderFile.size,
                      .pCode = (const uint32_t *)shaderFile.code};
                  VkShaderModule shaderModule;
                  VK_CHECK(vkCreateShaderModule(device, &createInfo, NULL,
                                                &shaderModule));
                  free(shaderFile.code);
                  shaderModule;
                }),
                .pName = "main"},
      .layout = fluidSim->pipelineLayout};

  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                    NULL, &fluidSim->computePipeline));

  // Create descriptor set (assuming descriptorPool is already created)
  VkDescriptorSetAllocateInfo descriptorAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &fluidSim->descriptorSetLayout};

  VK_CHECK(vkAllocateDescriptorSets(device, &descriptorAllocInfo,
                                    &fluidSim->descriptorSet));

  // Update descriptor sets
  VkDescriptorBufferInfo velocityBufferInfo = {
      .buffer = fluidSim->velocityBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

  VkDescriptorBufferInfo simParamsBufferInfo = {
      .buffer = fluidSim->simParamsBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

  VkWriteDescriptorSet descriptorWrites[] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = fluidSim->descriptorSet,
       .dstBinding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .pBufferInfo = &velocityBufferInfo},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = fluidSim->descriptorSet,
       .dstBinding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .pBufferInfo = &simParamsBufferInfo}};

  vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, NULL);
}

void updateFluidSim(VkCommandBuffer cmdBuf, FluidSim *fluidSim) {
  VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

  vkBeginCommandBuffer(cmdBuf, &beginInfo);

  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                    fluidSim->computePipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                          fluidSim->pipelineLayout, 0, 1,
                          &fluidSim->descriptorSet, 0, NULL);

  vkCmdDispatch(cmdBuf, 32, 32, 1); // Adjust grid size as needed

  VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                             .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                             .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};

  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       NULL, 0, NULL);

  vkEndCommandBuffer(cmdBuf);
}

void destroyFluidSim(VkDevice device, FluidSim *fluidSim) {
  vkDestroyPipeline(device, fluidSim->computePipeline, NULL);
  vkDestroyPipelineLayout(device, fluidSim->pipelineLayout, NULL);
  vkDestroyDescriptorSetLayout(device, fluidSim->descriptorSetLayout, NULL);
  vkDestroyBuffer(device, fluidSim->particleBuffer, NULL);
  vkFreeMemory(device, fluidSim->particleBufferMemory, NULL);

  vkDestroyBuffer(device, fluidSim->velocityBuffer, NULL);
  vkFreeMemory(device, fluidSim->velocityBufferMemory, NULL);

  vkDestroyBuffer(device, fluidSim->simParamsBuffer, NULL);
  vkFreeMemory(device, fluidSim->simParamsBufferMemory, NULL);

  vkDestroyImage(device, fluidSim->velocityField, NULL);
  vkDestroyImageView(device, fluidSim->velocityFieldView, NULL);
  vkFreeMemory(device, fluidSim->velocityFieldMemory, NULL);
}
