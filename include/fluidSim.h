#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <vulkan/vulkan.h>

#include <stdint.h>

typedef struct FluidSim {
    VkPipeline computePipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkBuffer particleBuffer;
    VkDeviceMemory particleBufferMemory;
    VkImage velocityField;
    VkDeviceMemory velocityFieldMemory;
    VkImageView velocityFieldView;

    VkBuffer velocityBuffer;
    VkDeviceMemory velocityBufferMemory;
    VkBuffer simParamsBuffer;
    VkDeviceMemory simParamsBufferMemory;
} FluidSim;

// Initialize the fluid simulation
void createFluidSim(VkDevice device, VkPhysicalDevice physicalDevice,
                    VkDescriptorPool descriptorPool, FluidSim *fluidSim);

// Update simulation state
void updateFluidSim(VkCommandBuffer cmdBuf, FluidSim* fluidSim);

// Cleanup resources
void destroyFluidSim(VkDevice device, FluidSim* fluidSim);


#endif
