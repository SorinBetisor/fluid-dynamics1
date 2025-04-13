#include "fluidSim.h"
#include "app.h"
#include "shader.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

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
  // Initialize fluid objects array
  fluidSim->objects = NULL; // Add this at the start of the function
  fluidSim->numObjects = 0; // Add this at the start of the function

  // Descriptor set layout bindings for both buffer and image
  VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2,
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

  // Create object buffer
  VkBufferCreateInfo objBufferCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(FluidObject) * MAX_OBJECTS,
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VK_CHECK(vkCreateBuffer(device, &objBufferCreateInfo, NULL,
                          &fluidSim->objectBuffer));

  // Allocate and bind memory for object buffer
  vkGetBufferMemoryRequirements(device, fluidSim->objectBuffer, &memReqs);

  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex =
      findMemoryTypeSim(physicalDevice, memReqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL,
                            &fluidSim->objectBufferMemory));
  VK_CHECK(vkBindBufferMemory(device, fluidSim->objectBuffer,
                              fluidSim->objectBufferMemory, 0));

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
                .module = VK_NULL_HANDLE,
                .pName = "main"},
      .layout = fluidSim->pipelineLayout};

  VkShaderModule shaderModule;
  {
    Shaderfile shaderFile = readShaderFile("shaders/compFluid.spv");
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shaderFile.size,
        .pCode = (const uint32_t *)shaderFile.code};
    VK_CHECK(vkCreateShaderModule(device, &createInfo, NULL, &shaderModule));
    free(shaderFile.code);
  }

  pipelineInfo.stage.module = shaderModule;

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

  VkDescriptorBufferInfo objectBufferInfo = {
      .buffer = fluidSim->objectBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

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
       .pBufferInfo = &simParamsBufferInfo},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = fluidSim->descriptorSet,
       .dstBinding = 2,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .pBufferInfo = &objectBufferInfo}};

  vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, NULL);
}

void setInitialConditions(VkDevice device, FluidSim *fluidSim,
                          InitialConditionType type, float strength) {
    size_t dataSize = fluidSim->gridWidth * fluidSim->gridHeight * 4 * sizeof(float);
    float *initialVelocities = (float *)malloc(dataSize);

    for (uint32_t y = 0; y < fluidSim->gridHeight; y++) {
        for (uint32_t x = 0; x < fluidSim->gridWidth; x++) {
            uint32_t idx = (y * fluidSim->gridWidth + x) * 4;
            float px = (float)x / fluidSim->gridWidth;
            float py = (float)y / fluidSim->gridHeight;

            switch (type) {
                case IC_UNIFORM_FLOW: {
                    // Stronger uniform flow with slight vertical variation
                    initialVelocities[idx] = strength * (1.0f + 0.1f * sinf(py * 10.0f)); // VelocityX
                    initialVelocities[idx + 1] = strength * 0.1f * sinf(px * 5.0f);       // VelocityY
                    break;
                }
                case IC_SOURCE_SINK: {
                    float centerX = 0.5f;
                    float centerY = 0.5f;
                    float dx = px - centerX;
                    float dy = py - centerY;
                    float r = sqrtf(dx * dx + dy * dy);
                    if (r < 0.01f) r = 0.01f;
                    
                    initialVelocities[idx] = strength * dx / r;     // VelocityX
                    initialVelocities[idx + 1] = strength * dy / r; // VelocityY
                    break;
                }
            }

            initialVelocities[idx + 2] = 0.0f; // Pressure (initially 0)
            initialVelocities[idx + 3] = 0.0f; // Reserved for future use
        }
    }

    // Copy to GPU memory
    void *data;
    vkMapMemory(device, fluidSim->velocityBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
    memcpy(data, initialVelocities, dataSize);
    vkUnmapMemory(device, fluidSim->velocityBufferMemory);

    free(initialVelocities);
}

float pointToLineDistance(float px, float py, float x1, float y1, float x2,
                          float y2) {
  float A = px - x1;
  float B = py - y1;
  float C = x2 - x1;
  float D = y2 - y1;

  float dot = A * C + B * D;
  float len_sq = C * C + D * D;
  float param = dot / len_sq;

  float xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + D;
  }

  float dx = px - xx;
  float dy = py - yy;
  return sqrt(dx * dx + dy * dy);
}

void updateFluidSim(VkCommandBuffer cmdBuf, FluidSim *fluidSim) {
  // Remove begin/end command buffer calls since this will be handled by the
  // caller
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
}

void destroyFluidSim(VkDevice device, FluidSim *fluidSim) {
  // Add this before other cleanup code
  if (fluidSim->objects) {
    // Free any airfoil points data
    for (int i = 0; i < fluidSim->numObjects; i++) {
      if (fluidSim->objects[i].points) {
        free(fluidSim->objects[i].points);
      }
    }
    free(fluidSim->objects);
    fluidSim->objects = NULL;
    fluidSim->numObjects = 0;
  }

  vkDestroyPipeline(device, fluidSim->computePipeline, NULL);
  vkDestroyPipelineLayout(device, fluidSim->pipelineLayout, NULL);
  vkDestroyDescriptorSetLayout(device, fluidSim->descriptorSetLayout, NULL);
  vkDestroyBuffer(device, fluidSim->particleBuffer, NULL);
  vkFreeMemory(device, fluidSim->particleBufferMemory, NULL);

  vkDestroyBuffer(device, fluidSim->velocityBuffer, NULL);
  vkFreeMemory(device, fluidSim->velocityBufferMemory, NULL);

  vkDestroyBuffer(device, fluidSim->simParamsBuffer, NULL);
  vkFreeMemory(device, fluidSim->simParamsBufferMemory, NULL);

  vkDestroyBuffer(device, fluidSim->objectBuffer, NULL);
  vkFreeMemory(device, fluidSim->objectBufferMemory, NULL);

  vkDestroyImage(device, fluidSim->velocityField, NULL);
  vkDestroyImageView(device, fluidSim->velocityFieldView, NULL);
  vkFreeMemory(device, fluidSim->velocityFieldMemory, NULL);
}

void initializeSimulationOutput(FluidSim *fluidSim, const char *filename) {
  fluidSim->gridWidth = 32; // Match with dispatch size
  fluidSim->gridHeight = 32;

  // Allocate host memory for velocity data
  size_t dataSize =
      fluidSim->gridWidth * fluidSim->gridHeight * 4 * sizeof(float);
  fluidSim->hostVelocityData = (float *)malloc(dataSize);

  // Open output file
  fluidSim->outputFile = fopen(filename, "w");
  if (!fluidSim->outputFile) {
    printf("Failed to open output file: %s\n", filename);
    abort();
  }

  // Write CSV header
  fprintf(fluidSim->outputFile, "Time,X,Y,VelocityX,VelocityY\n");
}

void saveSimulationFrame(VkDevice device, FluidSim *fluidSim, float time) {
  // Map memory to get velocity data
  void *data;
  vkMapMemory(device, fluidSim->velocityBufferMemory, 0, VK_WHOLE_SIZE, 0,
              &data);
  memcpy(fluidSim->hostVelocityData, data,
         fluidSim->gridWidth * fluidSim->gridHeight * 4 * sizeof(float));
  vkUnmapMemory(device, fluidSim->velocityBufferMemory);

  // Write to CSV
  for (uint32_t y = 0; y < fluidSim->gridHeight; y++) {
    for (uint32_t x = 0; x < fluidSim->gridWidth; x++) {
      uint32_t idx = (y * fluidSim->gridWidth + x) * 4;
      fprintf(fluidSim->outputFile, "%.4f,%d,%d,%.4f,%.4f\n", time, x, y,
              fluidSim->hostVelocityData[idx],    // VelocityX
              fluidSim->hostVelocityData[idx + 1] // VelocityY
      );
    }
  }
}

void closeSimulationOutput(FluidSim *fluidSim) {
  if (fluidSim->outputFile) {
    fclose(fluidSim->outputFile);
    fluidSim->outputFile = NULL;
  }

  if (fluidSim->hostVelocityData) {
    free(fluidSim->hostVelocityData);
    fluidSim->hostVelocityData = NULL;
  }
}

void loadAirfoilFromFile(const char *filename, float **points, int *numPoints) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("Failed to open airfoil file: %s\n", filename);
    abort();
  }

  // Count lines first
  int count = 0;
  char line[256];
  while (fgets(line, sizeof(line), file)) {
    if (line[0] != '#' && line[0] != '\n')
      count++;
  }

  // Allocate memory
  *points = (float *)malloc(count * 2 * sizeof(float));
  *numPoints = count;

  // Read points
  rewind(file);
  int i = 0;
  while (fgets(line, sizeof(line), file)) {
    if (line[0] != '#' && line[0] != '\n') {
      float x, y;
      if (sscanf(line, "%f %f", &x, &y) == 2) {
        (*points)[i * 2] = x;
        (*points)[i * 2 + 1] = y;
        i++;
      }
    }
  }

  fclose(file);
}

void addFluidObject(VkDevice device, FluidSim *fluidSim, ObjectType type,
                    float x, float y, float radius, const char *airfoilFile) {
  if (fluidSim->objects == NULL) {
    // First allocation
    fluidSim->objects = malloc(sizeof(FluidObject));
  } else {
    // Subsequent reallocations
    FluidObject *newObjects = realloc(
        fluidSim->objects, (fluidSim->numObjects + 1) * sizeof(FluidObject));
    if (newObjects == NULL) {
      printf("Failed to allocate memory for fluid object\n");
      return;
    }
    fluidSim->objects = newObjects;
  }

  FluidObject *obj = &fluidSim->objects[fluidSim->numObjects];
  obj->x = x;
  obj->y = y;
  obj->type = type;

  if (type == OBJ_CIRCLE) {
    obj->radius = radius;
    obj->points = NULL;
    obj->numPoints = 0;
  } else if (type == OBJ_AIRFOIL) {
    obj->radius = 0.0f;
    loadAirfoilFromFile(airfoilFile, &obj->points, &obj->numPoints);
  }

  fluidSim->numObjects++;

  // Update object buffer on GPU
  size_t bufferSize = sizeof(FluidObject) * fluidSim->numObjects;
  void *data;
  vkMapMemory(device, fluidSim->objectBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, fluidSim->objects, bufferSize);
  vkUnmapMemory(device, fluidSim->objectBufferMemory);
}
