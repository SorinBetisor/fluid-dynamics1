#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linearalg.h"
#include "poisson.h"
#include "vulkan_solver.h"

// Define these to avoid including problematic video codec headers
#define VK_VIDEO_H_ 1
#define VK_VIDEO_CODEC_H264STD_H_ 1
#define VK_VIDEO_CODEC_H264STD_DECODE_H_ 1
#define VK_VIDEO_CODEC_H264STD_ENCODE_H_ 1
#define VK_VIDEO_CODEC_H265STD_H_ 1 
#define VK_VIDEO_CODEC_H265STD_DECODE_H_ 1
#define VK_VIDEO_CODEC_H265STD_ENCODE_H_ 1
#define VK_VIDEO_CODECS_COMMON_H_ 1
#define VK_VIDEO_CODEC_AV1STD_H_ 1
#define VK_VIDEO_CODEC_AV1STD_DECODE_H_ 1
#define VK_VIDEO_CODEC_AV1STD_ENCODE_H_ 1

// Override the Vulkan video headers to use our local copies
#ifndef DISABLE_VULKAN
  // We need function prototypes
  // #define VK_NO_PROTOTYPES 1
  #define VK_ENABLE_BETA_EXTENSIONS 0
  
  #ifdef __APPLE__
    // Check for different possible paths on macOS
    #if __has_include("/Users/botond/VulkanSDK/1.4.309.0/macOS/include/vulkan/vulkan.h")
      // MoltenVK might be available but not directly include-able
      #define HAS_MOLTENVK 1
      // Define this before including vulkan.h
      #define VK_USE_PLATFORM_MACOS_MVK 1
      #include "/Users/botond/VulkanSDK/1.4.309.0/macOS/include/vulkan/vulkan.h"
    #elif __has_include("/Users/botond/VulkanSDK/1.4.309.0/macOS/Frameworks/vulkan.framework/Headers/vulkan.h")
      #define HAS_MOLTENVK 1
      #define VK_USE_PLATFORM_MACOS_MVK 1
      #include "/Users/botond/VulkanSDK/1.4.309.0/macOS/Frameworks/vulkan.framework/Headers/vulkan.h"
    #else
      #define DISABLE_VULKAN
      #warning "Vulkan headers not found, disabling Vulkan support"
    #endif
  #else
    // Linux/Windows
    #if __has_include(<vulkan/vulkan.h>)
      #include <vulkan/vulkan.h>
    #else
      #define DISABLE_VULKAN
      #warning "Vulkan headers not found, disabling Vulkan support"
    #endif
  #endif
#endif

// Forward declarations
int create_pipeline_layout(void);
int create_descriptor_sets(void);
int create_compute_pipelines(void);
int create_shader_module(const char* filename, VkShaderModule* shaderModule);
int transfer_matrix_to_gpu(mtrx matrix, VkBuffer buffer, VkDeviceMemory memory);
int transfer_gpu_to_matrix(VkBuffer buffer, VkDeviceMemory memory, mtrx matrix);
int transfer_grid_to_gpu(cell_properties** grid, int width, int height, VkBuffer buffer, VkDeviceMemory memory);
int create_command_buffer(void);
int dispatch_compute(VkPipeline pipeline, uint32_t workGroupsX, uint32_t workGroupsY);
int update_descriptor_sets(void);
void calculate_work_groups(uint32_t* workGroupsX, uint32_t* workGroupsY);
void swap_buffers(void);

#ifndef DISABLE_VULKAN

// Optional MoltenVK specific headers
#ifdef __APPLE__
  #if HAS_MOLTENVK
    /*
    #if __has_include("/Users/botond/VulkanSDK/1.4.309.0/macOS/include/MoltenVK/vk_mvk_moltenvk.h")
      #include "/Users/botond/VulkanSDK/1.4.309.0/macOS/include/MoltenVK/vk_mvk_moltenvk.h"
    #elif __has_include("/Users/botond/VulkanSDK/1.4.309.0/macOS/Frameworks/MoltenVK.framework/Headers/vk_mvk_moltenvk.h")
      #include "/Users/botond/VulkanSDK/1.4.309.0/macOS/Frameworks/MoltenVK.framework/Headers/vk_mvk_moltenvk.h"
    #endif
    */
  #endif
#endif

// Global variables for Vulkan
static VkInstance instance = VK_NULL_HANDLE;
static VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
static VkDevice device = VK_NULL_HANDLE;
static VkQueue computeQueue = VK_NULL_HANDLE;
static uint32_t queueFamilyIndex = 0;

// Shader resources
static VkShaderModule poissonShader = VK_NULL_HANDLE;
static VkShaderModule poissonSORShader = VK_NULL_HANDLE;

// Pipeline resources
static VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
static VkPipeline poissonPipeline = VK_NULL_HANDLE;
static VkPipeline poissonSORPipeline = VK_NULL_HANDLE;

// Command buffer resources
static VkCommandPool commandPool = VK_NULL_HANDLE;
static VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

// Descriptor set resources
static VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

// Buffer resources
static VkBuffer inputBuffer = VK_NULL_HANDLE;
static VkDeviceMemory inputMemory = VK_NULL_HANDLE;
static VkBuffer outputBuffer = VK_NULL_HANDLE;
static VkDeviceMemory outputMemory = VK_NULL_HANDLE;
static VkBuffer gridBuffer = VK_NULL_HANDLE;
static VkDeviceMemory gridMemory = VK_NULL_HANDLE;

// Add uniform buffer for shader constants
static VkBuffer uniformBuffer = VK_NULL_HANDLE;
static VkDeviceMemory uniformMemory = VK_NULL_HANDLE;

// Define structure for uniform buffer
typedef struct {
    float dx;
    float dy;
    int width;
    int height;
    float f_factor;  // dx^2 * dy^2
    float beta;      // SOR relaxation parameter (if needed)
    float padding[2]; // Padding to ensure 16-byte alignment
} PoissonConstants;

// Problem dimensions
static int width = 0;
static int height = 0;

// Check Vulkan result and print error message
static int check_vk_result(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        printf("Vulkan error during %s: %d\n", operation, result);
        return 0;
    }
    return 1;
}

// Find a suitable memory type for buffer allocation
uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    printf("Failed to find suitable memory type\n");
    return 0;
}

// Create a buffer and allocate memory for it
int create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                  VkMemoryPropertyFlags properties, VkBuffer* buffer, 
                  VkDeviceMemory* bufferMemory) {
    // Create buffer
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    if (!check_vk_result(vkCreateBuffer(device, &bufferInfo, NULL, buffer), 
                         "buffer creation")) {
        return 0;
    }
    
    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);
    
    // Allocate memory
    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties)
    };
    
    if (!check_vk_result(vkAllocateMemory(device, &allocInfo, NULL, bufferMemory), 
                         "memory allocation")) {
        vkDestroyBuffer(device, *buffer, NULL);
        return 0;
    }
    
    // Bind memory to buffer
    if (!check_vk_result(vkBindBufferMemory(device, *buffer, *bufferMemory, 0), 
                         "memory binding")) {
        vkDestroyBuffer(device, *buffer, NULL);
        vkFreeMemory(device, *bufferMemory, NULL);
        return 0;
    }
    
    return 1;
}

// Create uniform buffer for shader constants
int create_uniform_buffer() {
    // Size of uniform buffer with proper alignment
    VkDeviceSize bufferSize = sizeof(PoissonConstants);
    
    if (!create_buffer(bufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &uniformBuffer, &uniformMemory)) {
        fprintf(stderr, "Failed to create uniform buffer\n");
        return 0;
    }
    
    return 1;
}

// Update constants in uniform buffer
int update_uniform_buffer(double dx, double dy, double beta) {
    PoissonConstants constants = {
        .dx = (float)dx,
        .dy = (float)dy,
        .width = width,
        .height = height,
        .f_factor = (float)(dx * dx * dy * dy),
        .beta = (float)beta
    };
    
    void* mapped_memory;
    if (vkMapMemory(device, uniformMemory, 0, sizeof(constants), 0, &mapped_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map uniform buffer memory\n");
        return 0;
    }
    
    memcpy(mapped_memory, &constants, sizeof(constants));
    vkUnmapMemory(device, uniformMemory);
    
    return 1;
}

// Initialize Vulkan
int init_vulkan_solver(int nx, int ny) {
    width = nx;
    height = ny;
    
    // Create instance
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "CNavier Vulkan Solver",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    // Required extensions for Vulkan
    const char* extensions[] = {
#ifdef __APPLE__
        "VK_KHR_surface",                  // Basic surface extension
        "VK_MVK_macos_surface",            // macOS specific surface extension
        "VK_KHR_portability_enumeration",  // Required for MoltenVK
        "VK_KHR_get_physical_device_properties2"  // Required for MoltenVK
#else
        "VK_KHR_surface"                   // Basic surface extension for other platforms
#endif
    };

    // Create instance
    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]),
        .ppEnabledExtensionNames = extensions,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL
    };

    // Set the flag for MoltenVK portability
#ifdef __APPLE__
    createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
    printf("Creating Vulkan instance with %d extensions\n", createInfo.enabledExtensionCount);
    VkResult result = vkCreateInstance(&createInfo, NULL, &instance);
    if (!check_vk_result(result, "instance creation")) {
        printf("Instance creation failed, trying with fewer extensions\n");
        
        // Try again with just the essential extensions on macOS
#ifdef __APPLE__
        const char* minExtensions[] = {
            "VK_KHR_portability_enumeration"
        };
        
        createInfo.enabledExtensionCount = sizeof(minExtensions) / sizeof(minExtensions[0]);
        createInfo.ppEnabledExtensionNames = minExtensions;
        
        if (!check_vk_result(vkCreateInstance(&createInfo, NULL, &instance), 
                            "minimal instance creation")) {
            return 0;
        }
#else
        return 0;
#endif
    }
    
    // Configure MoltenVK if available
#ifdef __APPLE__
#if HAS_MOLTENVK
    /*
    if (__has_include(<MoltenVK/vk_mvk_moltenvk.h>)) {
        MVKConfiguration mvkConfig;
        size_t mvkConfigSize = sizeof(MVKConfiguration);
        vkGetMoltenVKConfigurationMVK(instance, &mvkConfig, &mvkConfigSize);
        mvkConfig.debugMode = true;  // Enable debugging
        vkSetMoltenVKConfigurationMVK(instance, &mvkConfig, &mvkConfigSize);
    }
    */
#endif
#endif
    
    // Select physical device (GPU)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    
    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    VkPhysicalDevice devices[deviceCount];
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    
    // Print device information
    printf("Found %d Vulkan capable device(s)\n", deviceCount);
    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        printf("Device %d: %s\n", i, deviceProperties.deviceName);
    }
    
    // Just use the first available device for simplicity
    physicalDevice = devices[0];
    
    // Find compute-capable queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    
    VkQueueFamilyProperties queueFamilies[queueFamilyCount];
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);
    
    int found = 0;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            found = 1;
            break;
        }
    }
    
    if (!found) {
        printf("Failed to find a compute queue family\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Create logical device with compute queue
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };
    
    // Required device extensions
    const char* deviceExtensions[] = {
#ifdef __APPLE__
        "VK_KHR_portability_subset"  // Required for MoltenVK
#endif
    };
    
    VkDeviceCreateInfo deviceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
#ifdef __APPLE__
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = deviceExtensions
#else
        .enabledExtensionCount = 0
#endif
    };
    
    if (!check_vk_result(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device), 
                         "logical device creation")) {
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Get queue handle
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    
    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndex
    };
    
    if (!check_vk_result(vkCreateCommandPool(device, &poolInfo, NULL, &commandPool), 
                         "command pool creation")) {
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Create buffers for computation
    size_t bufferSize = nx * ny * sizeof(float);
    
    if (!create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &inputBuffer, &inputMemory)) {
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    if (!create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &outputBuffer, &outputMemory)) {
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    if (!create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &gridBuffer, &gridMemory)) {
        vkDestroyBuffer(device, outputBuffer, NULL);
        vkFreeMemory(device, outputMemory, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Create uniform buffer
    if (!create_uniform_buffer()) {
        vkDestroyBuffer(device, gridBuffer, NULL);
        vkFreeMemory(device, gridMemory, NULL);
        vkDestroyBuffer(device, outputBuffer, NULL);
        vkFreeMemory(device, outputMemory, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Create pipeline layout and descriptor sets
    if (!create_pipeline_layout()) {
        vkDestroyBuffer(device, uniformBuffer, NULL);
        vkFreeMemory(device, uniformMemory, NULL);
        vkDestroyBuffer(device, gridBuffer, NULL);
        vkFreeMemory(device, gridMemory, NULL);
        vkDestroyBuffer(device, outputBuffer, NULL);
        vkFreeMemory(device, outputMemory, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    if (!create_descriptor_sets()) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyBuffer(device, uniformBuffer, NULL);
        vkFreeMemory(device, uniformMemory, NULL);
        vkDestroyBuffer(device, gridBuffer, NULL);
        vkFreeMemory(device, gridMemory, NULL);
        vkDestroyBuffer(device, outputBuffer, NULL);
        vkFreeMemory(device, outputMemory, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Create compute pipelines
    if (!create_compute_pipelines()) {
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyBuffer(device, uniformBuffer, NULL);
        vkFreeMemory(device, uniformMemory, NULL);
        vkDestroyBuffer(device, gridBuffer, NULL);
        vkFreeMemory(device, gridMemory, NULL);
        vkDestroyBuffer(device, outputBuffer, NULL);
        vkFreeMemory(device, outputMemory, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkFreeMemory(device, inputMemory, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    
    // Update descriptor sets with buffers
    update_descriptor_sets();
    
    printf("Vulkan initialization successful\n");
    return 1;
}

// Clean up Vulkan resources
void cleanup_vulkan_solver() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        
        // Clean up descriptor sets and layouts
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, NULL);
            descriptorPool = VK_NULL_HANDLE;
        }
        
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        
        // Clean up pipeline resources
        if (poissonPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, poissonPipeline, NULL);
            poissonPipeline = VK_NULL_HANDLE;
        }
        
        if (poissonSORPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, poissonSORPipeline, NULL);
            poissonSORPipeline = VK_NULL_HANDLE;
        }
        
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, NULL);
            pipelineLayout = VK_NULL_HANDLE;
        }
        
        // Clean up uniform buffer
        if (uniformBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, uniformBuffer, NULL);
            uniformBuffer = VK_NULL_HANDLE;
        }
        
        if (uniformMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, uniformMemory, NULL);
            uniformMemory = VK_NULL_HANDLE;
        }
        
        // Clean up data buffers
        if (gridBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, gridBuffer, NULL);
            gridBuffer = VK_NULL_HANDLE;
        }
        
        if (gridMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, gridMemory, NULL);
            gridMemory = VK_NULL_HANDLE;
        }
        
        if (outputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, outputBuffer, NULL);
            outputBuffer = VK_NULL_HANDLE;
        }
        
        if (outputMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, outputMemory, NULL);
            outputMemory = VK_NULL_HANDLE;
        }
        
        if (inputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, inputBuffer, NULL);
            inputBuffer = VK_NULL_HANDLE;
        }
        
        if (inputMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, inputMemory, NULL);
            inputMemory = VK_NULL_HANDLE;
        }
        
        // Clean up command resources
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, NULL);
            commandPool = VK_NULL_HANDLE;
        }
        
        // Clean up device
        vkDestroyDevice(device, NULL);
        device = VK_NULL_HANDLE;
    }
    
    // Clean up instance
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, NULL);
        instance = VK_NULL_HANDLE;
    }
    
    printf("Vulkan resources cleaned up\n");
}

// Vulkan-accelerated Poisson SOR solver with objects
mtrx poisson_SOR_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid) {
    // Check if Vulkan initialization was successful
    if (device == VK_NULL_HANDLE) {
        printf("Vulkan not initialized - falling back to CPU\n");
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    // Create result matrix
    mtrx u = initm(f.m, f.n);
    mtrx temp = initm(f.m, f.n); // For calculating error
    
    // Initialize result matrix with zeros
    zerosm(u);
    
    // Update uniform buffer with constants including beta
    if (!update_uniform_buffer(dx, dy, beta)) {
        printf("Failed to update uniform buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    // Transfer grid data to GPU
    if (!transfer_grid_to_gpu(grid, f.m, f.n, gridBuffer, gridMemory)) {
        printf("Failed to transfer grid data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    // Transfer initial data to GPU
    if (!transfer_matrix_to_gpu(u, inputBuffer, inputMemory)) {
        printf("Failed to transfer initial data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    if (!transfer_matrix_to_gpu(f, outputBuffer, outputMemory)) {
        printf("Failed to transfer RHS data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    // Update descriptor sets
    update_descriptor_sets();
    
    // Create command buffer
    if (!create_command_buffer()) {
        printf("Failed to create command buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
    }
    
    // Calculate work group counts
    uint32_t workGroupsX, workGroupsY;
    calculate_work_groups(&workGroupsX, &workGroupsY);
    
    // Main iteration loop
    for (int k = 0; k < itmax; k++) {
        // Dispatch compute shader
        if (!dispatch_compute(poissonSORPipeline, workGroupsX, workGroupsY)) {
            printf("Failed to dispatch compute shader - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
        }
        
        // After computation, transfer data back to check convergence
        if (!transfer_gpu_to_matrix(outputBuffer, outputMemory, temp)) {
            printf("Failed to transfer results from GPU - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
        }
        
        // Calculate error between iterations
        double e = error(u, temp);
        
        // Check for convergence
        if (e < tol) {
            printf("Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            // Copy result to output matrix
            mtrxcpy(u, temp);
            temp.M = freem(temp);
            return u;
        }
        
        // Update u for next iteration
        mtrxcpy(u, temp);
        
        // Transfer updated solution back to input buffer
        if (!transfer_matrix_to_gpu(u, inputBuffer, inputMemory)) {
            printf("Failed to transfer updated solution to GPU - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
        }
    }
    
    printf("Error: maximum number of iterations achieved for Poisson equation.\n");
    u.M = freem(u);
    temp.M = freem(temp);
    return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
}

// Vulkan-accelerated Poisson solver with objects
mtrx poisson_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid) {
    // Check if Vulkan initialization was successful
    if (device == VK_NULL_HANDLE) {
        printf("Vulkan not initialized - falling back to CPU\n");
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    // Create result matrix
    mtrx u = initm(f.m, f.n);
    mtrx temp = initm(f.m, f.n); // For calculating error
    
    // Initialize result matrix with zeros
    zerosm(u);
    
    // Update uniform buffer with constants
    if (!update_uniform_buffer(dx, dy, 0.0)) {
        printf("Failed to update uniform buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    // Transfer grid data to GPU
    if (!transfer_grid_to_gpu(grid, f.m, f.n, gridBuffer, gridMemory)) {
        printf("Failed to transfer grid data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    // Transfer initial data to GPU
    if (!transfer_matrix_to_gpu(u, inputBuffer, inputMemory)) {
        printf("Failed to transfer initial data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    if (!transfer_matrix_to_gpu(f, outputBuffer, outputMemory)) {
        printf("Failed to transfer RHS data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    // Update descriptor sets
    update_descriptor_sets();
    
    // Create command buffer
    if (!create_command_buffer()) {
        printf("Failed to create command buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson_with_object(f, dx, dy, itmax, tol, grid);
    }
    
    // Calculate work group counts
    uint32_t workGroupsX, workGroupsY;
    calculate_work_groups(&workGroupsX, &workGroupsY);
    
    // Main iteration loop
    for (int k = 0; k < itmax; k++) {
        // Dispatch compute shader
        if (!dispatch_compute(poissonPipeline, workGroupsX, workGroupsY)) {
            printf("Failed to dispatch compute shader - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_with_object(f, dx, dy, itmax, tol, grid);
        }
        
        // After computation, transfer data back to check convergence
        if (!transfer_gpu_to_matrix(outputBuffer, outputMemory, temp)) {
            printf("Failed to transfer results from GPU - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_with_object(f, dx, dy, itmax, tol, grid);
        }
        
        // Calculate error between iterations
        double e = error(u, temp);
        
        // Check for convergence
        if (e < tol) {
            printf("Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            // Copy result to output matrix
            mtrxcpy(u, temp);
            temp.M = freem(temp);
            return u;
        }
        
        // Update u for next iteration
        mtrxcpy(u, temp);
        
        // Transfer updated solution back to input buffer
        if (!transfer_matrix_to_gpu(u, inputBuffer, inputMemory)) {
            printf("Failed to transfer updated solution to GPU - falling back to CPU\n");
            u.M = freem(u);
            temp.M = freem(temp);
            return poisson_with_object(f, dx, dy, itmax, tol, grid);
        }
    }
    
    printf("Error: maximum number of iterations achieved for Poisson equation.\n");
    u.M = freem(u);
    temp.M = freem(temp);
    return poisson_with_object(f, dx, dy, itmax, tol, grid);
}

// Vulkan-accelerated Poisson SOR solver
mtrx poisson_SOR_vulkan(mtrx f, double dx, double dy, int itmax, double tol, double beta) {
    // For now, we'll fall back to the CPU implementation
    printf("Vulkan Poisson SOR solver not fully implemented yet - falling back to CPU\n");
    return poisson_SOR(f, dx, dy, itmax, tol, beta);
}

// Vulkan-accelerated Poisson solver
mtrx poisson_vulkan(mtrx f, double dx, double dy, int itmax, double tol) {
    // Check if Vulkan initialization was successful
    if (device == VK_NULL_HANDLE) {
        printf("Vulkan not initialized - falling back to CPU\n");
        return poisson(f, dx, dy, itmax, tol);
    }
    
    printf("Starting Vulkan Poisson solver (no relaxation) with grid size %dx%d\n", f.m, f.n);
    
    // Create result matrix
    mtrx u = initm(f.m, f.n);
    mtrx temp = initm(f.m, f.n); // For calculating error
    
    // Initialize result matrix with zeros
    zerosm(u);
    
    // Update uniform buffer with constants
    printf("Updating uniform buffer...\n");
    if (!update_uniform_buffer(dx, dy, 0.0)) {
        printf("Failed to update uniform buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson(f, dx, dy, itmax, tol);
    }
    
    // Transfer initial data to GPU
    printf("Transferring initial data to GPU...\n");
    if (!transfer_matrix_to_gpu(u, inputBuffer, inputMemory)) {
        printf("Failed to transfer initial data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson(f, dx, dy, itmax, tol);
    }
    
    if (!transfer_matrix_to_gpu(f, outputBuffer, outputMemory)) {
        printf("Failed to transfer RHS data to GPU - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson(f, dx, dy, itmax, tol);
    }
    
    // Update descriptor sets
    printf("Updating descriptor sets...\n");
    update_descriptor_sets();
    
    // Create command buffer
    printf("Creating command buffer...\n");
    if (!create_command_buffer()) {
        printf("Failed to create command buffer - falling back to CPU\n");
        u.M = freem(u);
        temp.M = freem(temp);
        return poisson(f, dx, dy, itmax, tol);
    }
    
    // Calculate work group counts
    uint32_t workGroupsX, workGroupsY;
    calculate_work_groups(&workGroupsX, &workGroupsY);
    // For now, we'll fall back to the CPU implementation
    printf("Vulkan Poisson solver not fully implemented yet - falling back to CPU\n");
    return poisson(f, dx, dy, itmax, tol);
}

// Create descriptor set layout
int create_pipeline_layout() {
    VkDescriptorSetLayoutBinding bindings[4] = {
        // Binding for solution buffer (p)
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Binding for right-hand side (f)
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Binding for residual
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Binding for constants
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings
    };
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor set layout\n");
        return 0;
    }
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout
    };
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create pipeline layout\n");
        return 0;
    }
    
    return 1;
}

// Create descriptor pool and allocate descriptor sets
int create_descriptor_sets() {
    VkDescriptorPoolSize poolSizes[2] = {
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 // p, f, residual
        },
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1 // constants
        }
    };
    
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 2,
        .pPoolSizes = poolSizes,
        .maxSets = 1
    };
    
    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
        return 0;
    }
    
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout
    };
    
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate descriptor set\n");
        return 0;
    }
    
    return 1;
}

// Create compute pipelines for Poisson solver
int create_compute_pipelines() {
    // Create Poisson solver shader module
    VkShaderModule poissonShaderModule;
    if (!create_shader_module("shaders/poisson.spv", &poissonShaderModule)) {
        fprintf(stderr, "Failed to create Poisson shader module\n");
        return 0;
    }
    
    // Create Poisson SOR shader module
    VkShaderModule poissonSORShaderModule;
    if (!create_shader_module("shaders/poisson_sor.spv", &poissonSORShaderModule)) {
        vkDestroyShaderModule(device, poissonShaderModule, NULL);
        fprintf(stderr, "Failed to create Poisson SOR shader module\n");
        return 0;
    }
    
    // Create Poisson pipeline
    VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = poissonShaderModule,
            .pName = "main"
        },
        .layout = pipelineLayout
    };
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &poissonPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, poissonShaderModule, NULL);
        vkDestroyShaderModule(device, poissonSORShaderModule, NULL);
        fprintf(stderr, "Failed to create Poisson compute pipeline\n");
        return 0;
    }
    
    // Create Poisson SOR pipeline
    pipelineInfo.stage.module = poissonSORShaderModule;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &poissonSORPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, poissonShaderModule, NULL);
        vkDestroyShaderModule(device, poissonSORShaderModule, NULL);
        fprintf(stderr, "Failed to create Poisson SOR compute pipeline\n");
        return 0;
    }
    
    // Clean up shader modules
    vkDestroyShaderModule(device, poissonShaderModule, NULL);
    vkDestroyShaderModule(device, poissonSORShaderModule, NULL);
    
    return 1;
}

// Load shader code from file
static char* read_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filename);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(file_size);
    if (!buffer) {
        fclose(file);
        fprintf(stderr, "Failed to allocate memory for shader code\n");
        return NULL;
    }
    
    size_t bytes_read = fread(buffer, 1, file_size, file);
    fclose(file);
    
    if (bytes_read != file_size) {
        free(buffer);
        fprintf(stderr, "Failed to read shader file\n");
        return NULL;
    }
    
    *size = file_size;
    return buffer;
}

// Create shader module from SPIR-V bytecode
int create_shader_module(const char* filename, VkShaderModule* shaderModule) {
    size_t code_size;
    char* code = read_file(filename, &code_size);
    if (!code) {
        return 0;
    }
    
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = (const uint32_t*)code
    };
    
    if (vkCreateShaderModule(device, &createInfo, NULL, shaderModule) != VK_SUCCESS) {
        free(code);
        fprintf(stderr, "Failed to create shader module\n");
        return 0;
    }
    
    free(code);
    return 1;
}

// Transfer matrix data to GPU buffer
int transfer_matrix_to_gpu(mtrx matrix, VkBuffer buffer, VkDeviceMemory memory) {
    void* mapped_memory;
    if (vkMapMemory(device, memory, 0, matrix.m * matrix.n * sizeof(float), 0, &mapped_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map buffer memory\n");
        return 0;
    }
    
    // Convert matrix to float array (row-major)
    float* data = (float*)mapped_memory;
    for (int i = 0; i < matrix.m; i++) {
        for (int j = 0; j < matrix.n; j++) {
            data[i * matrix.n + j] = (float)matrix.M[i][j];
        }
    }
    
    vkUnmapMemory(device, memory);
    return 1;
}

// Transfer GPU buffer data back to matrix
int transfer_gpu_to_matrix(VkBuffer buffer, VkDeviceMemory memory, mtrx matrix) {
    void* mapped_memory;
    if (vkMapMemory(device, memory, 0, matrix.m * matrix.n * sizeof(float), 0, &mapped_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map buffer memory\n");
        return 0;
    }
    
    // Convert float array back to matrix (row-major)
    float* data = (float*)mapped_memory;
    for (int i = 0; i < matrix.m; i++) {
        for (int j = 0; j < matrix.n; j++) {
            matrix.M[i][j] = (double)data[i * matrix.n + j];
        }
    }
    
    vkUnmapMemory(device, memory);
    return 1;
}

// Transfer cell properties to GPU buffer
int transfer_grid_to_gpu(cell_properties** grid, int width, int height, VkBuffer buffer, VkDeviceMemory memory) {
    void* mapped_memory;
    if (vkMapMemory(device, memory, 0, width * height * sizeof(float), 0, &mapped_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map grid buffer memory\n");
        return 0;
    }
    
    // Convert grid to float array (1.0 for solid, 0.0 for fluid)
    float* data = (float*)mapped_memory;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = grid[i][j].is_solid ? 1.0f : 0.0f;
        }
    }
    
    vkUnmapMemory(device, memory);
    return 1;
}

// Create a single-use command buffer
int create_command_buffer() {
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    
    return check_vk_result(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer),
                        "command buffer allocation");
}

// Record and submit a compute dispatch command
int dispatch_compute(VkPipeline pipeline, uint32_t workGroupsX, uint32_t workGroupsY) {
    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    
    if (!check_vk_result(vkBeginCommandBuffer(commandBuffer, &beginInfo),
                       "command buffer recording")) {
        return 0;
    }
    
    // Bind pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    
    // Bind descriptor sets
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
    
    // Dispatch compute work
    vkCmdDispatch(commandBuffer, workGroupsX, workGroupsY, 1);
    
    // End command buffer
    if (!check_vk_result(vkEndCommandBuffer(commandBuffer),
                       "command buffer recording")) {
        return 0;
    }
    
    // Submit command buffer
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };
    
    if (!check_vk_result(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE),
                       "compute queue submission")) {
        return 0;
    }
    
    // Wait for compute to finish
    if (!check_vk_result(vkQueueWaitIdle(computeQueue),
                       "compute queue wait")) {
        return 0;
    }
    
    return 1;
}

// Update descriptor sets with buffer bindings
int update_descriptor_sets() {
    VkDescriptorBufferInfo inputBufferInfo = {
        .buffer = inputBuffer,
        .offset = 0,
        .range = width * height * sizeof(float)
    };
    
    VkDescriptorBufferInfo outputBufferInfo = {
        .buffer = outputBuffer,
        .offset = 0,
        .range = width * height * sizeof(float)
    };
    
    VkDescriptorBufferInfo gridBufferInfo = {
        .buffer = gridBuffer,
        .offset = 0,
        .range = width * height * sizeof(float)
    };
    
    VkDescriptorBufferInfo uniformBufferInfo = {
        .buffer = uniformBuffer,
        .offset = 0,
        .range = sizeof(PoissonConstants)
    };
    
    VkWriteDescriptorSet descriptorWrites[4] = {
        // Input buffer (solution)
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &inputBufferInfo
        },
        // Output buffer (RHS)
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &outputBufferInfo
        },
        // Grid buffer
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &gridBufferInfo
        },
        // Uniform buffer
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &uniformBufferInfo
        }
    };
    
    vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, NULL);
    return 1;
}

// Calculate number of work groups needed
void calculate_work_groups(uint32_t* workGroupsX, uint32_t* workGroupsY) {
    const uint32_t workGroupSize = 16; // Must match shader local_size_x/y
    *workGroupsX = (width + workGroupSize - 1) / workGroupSize;
    *workGroupsY = (height + workGroupSize - 1) / workGroupSize;
}

// Swap input and output buffers
void swap_buffers() {
    VkBuffer tempBuffer = inputBuffer;
    inputBuffer = outputBuffer;
    outputBuffer = tempBuffer;
    
    VkDeviceMemory tempMemory = inputMemory;
    inputMemory = outputMemory;
    outputMemory = tempMemory;
}

#else // DISABLE_VULKAN

// Stubs for when Vulkan is disabled
int init_vulkan_solver(int nx, int ny) {
    printf("Vulkan support not available in this build\n");
    return 0;
}

void cleanup_vulkan_solver() {
    // No-op
}

mtrx poisson_vulkan(mtrx f, double dx, double dy, int itmax, double tol) {
    return poisson(f, dx, dy, itmax, tol);
}

mtrx poisson_SOR_vulkan(mtrx f, double dx, double dy, int itmax, double tol, double beta) {
    return poisson_SOR(f, dx, dy, itmax, tol, beta);
}

mtrx poisson_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid) {
    return poisson_with_object(f, dx, dy, itmax, tol, grid);
}

mtrx poisson_SOR_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid) {
    return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid);
}

#endif // DISABLE_VULKAN 