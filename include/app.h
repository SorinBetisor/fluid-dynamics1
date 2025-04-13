#ifndef APP_H
#define APP_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <cglm/cglm.h>

/* ----------------------------------------------------
 * Constants
 * ----------------------------------------------------*/
#define WIDTH 800
#define HEIGHT 600
#define NAME "TEST"

#define MAX_FRAMES_IN_FLIGHT 2

#define ATTRIBUTE_COUNT 2

extern const uint32_t validationLayerCount;
extern const char *validationLayers[];

extern const uint32_t deviceExtensionCount;
extern const char *deviceExtensions[];

#ifdef NDEBUG
static const bool enableValidationLayers = false;
#else
static const bool enableValidationLayers = true;
#endif

/* ----------------------------------------------------
 * Structs
 * ----------------------------------------------------*/

typedef struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    uint32_t formatCount;
    VkSurfaceFormatKHR *formats;
    uint32_t presentModeCount;
    VkPresentModeKHR *presentModes;
} SwapChainSupportDetails;

typedef struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    bool isGraphicsFamilySet;

    uint32_t presentationFamily;
    bool isPresentFamilySet;

    uint32_t transferFamily;
    bool isTransferFamilySet;
} QueueFamilyIndices;

typedef struct Shaderfile
{
    size_t size;
    char *code;
} Shaderfile;

typedef struct Vertex
{
    vec2 pos;
    vec3 color;
} Vertex;

typedef struct App
{
    SDL_Window *window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessanger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    QueueFamilyIndices QueueFamilyIndices;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentationQueue;
    VkQueue transferQueue;
    VkSwapchainKHR swapChain;
    uint32_t swapChainImageCount;
    VkImage *swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkImageView *swapchainImageView;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkFramebuffer *swapchainFramebuffers;
    VkCommandPool commandPool;
    VkCommandPool transferCommandPool;
    // VkCommandBuffer commandBuffer;
    VkCommandBuffer *commandBuffers;
    // VkSemaphore imageAvailableSemaphore;
    // VkSemaphore renderFinishedSemaphore;
    // VkFence inFlightFence;

    VkSemaphore *imageAvailableSemaphores;
    VkSemaphore *renderFinishedSemaphores;
    VkFence *inFlightFences;

    bool framebufferResized;

    uint32_t currentFrame;

    Vertex *vertices;
    uint32_t verticesCount;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    uint16_t *indices;
    uint32_t indicesCount;   
} App;

/* ----------------------------------------------------
 * Function Prototypes
 * ----------------------------------------------------*/

// From window.c
void initWindow(App *pApp);
void mainLoop(App *pApp);

// From vulkan.c
void initVulkan(App *pApp);
void cleanup(App *pApp);
void createInstance(App *pApp);
void setupDebugMessenger(App *pApp);
void createSurface(App *pApp);
void pickPhysicalDevice(App *pApp);
void createLogicalDevice(App *pApp);
void createSwapChain(App *pApp);
void cleanupSwapChain(App *pApp);
void recreateSwapChain(App *pApp);
void createRenderPass(App *pApp);
void createGraphicsPipeline(App *pApp);
void createFramebuffers(App *pApp);
void createCommandPool(App *pApp);
void createVertexBuffer(App *pApp);
void createCommandBuffers(App *pApp);
void createSyncObjects(App *pApp);
void drawFrame(App *pApp);
VkShaderModule createShaderModule(Shaderfile shader, App *pApp);
void createImageView(App *pApp);

// From debug.c
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks *pAllocator);
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData);

// From shader.c
Shaderfile readShaderFile(const char *filename);

// Additional helpers
bool checkValidationLayerSupport(void);
bool checkDeviceExtensionSupport(VkPhysicalDevice device);
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
void freeSwapChainSupportDetails(SwapChainSupportDetails *details);
VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkSurfaceFormatKHR *availableFormats, uint32_t formatCount);
VkPresentModeKHR chooseSwapPresentMode(VkPresentModeKHR *availablePresentModes, uint32_t presentModeCount);
VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR *capabilities, App *pApp);
uint32_t rateDeviceSuitability(VkPhysicalDevice device, VkSurfaceKHR surface);
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

#endif // APP_H
