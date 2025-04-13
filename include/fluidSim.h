#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdint.h>

typedef enum InitialConditionType {
    IC_UNIFORM_FLOW,
    IC_SOURCE_SINK
} InitialConditionType;

typedef enum ObjectType {
    OBJ_CIRCLE,
    OBJ_AIRFOIL
} ObjectType;

typedef struct {
    float x, y;  // Center position
    float radius;  // For circle
    float *points; // For airfoil points
    int numPoints; // Number of points for airfoil
    ObjectType type;
} FluidObject;

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

  FILE *outputFile;
  uint32_t gridWidth;
  uint32_t gridHeight;
  float *hostVelocityData;
  FluidObject *objects;
  int numObjects;
  VkBuffer objectBuffer;
  VkDeviceMemory objectBufferMemory;
} FluidSim;

// Initialize the fluid simulation
void createFluidSim(VkDevice device, VkPhysicalDevice physicalDevice,
                    VkDescriptorPool descriptorPool, FluidSim *fluidSim);

// Update simulation state
void updateFluidSim(VkCommandBuffer cmdBuf, FluidSim *fluidSim);

// Cleanup resources
void destroyFluidSim(VkDevice device, FluidSim *fluidSim);

void initializeSimulationOutput(FluidSim* fluidSim, const char* filename);
void saveSimulationFrame(VkDevice device, FluidSim* fluidSim, float time);
void closeSimulationOutput(FluidSim* fluidSim);
void setInitialConditions(VkDevice device, FluidSim* fluidSim, 
                         InitialConditionType type, float strength);

void addFluidObject(VkDevice device, FluidSim *fluidSim, ObjectType type, 
                   float x, float y, float radius, const char *airfoilFile);
void loadAirfoilFromFile(const char *filename, float **points, int *numPoints);

float pointToLineDistance(float px, float py, float x1, float y1, float x2, float y2);

#endif
