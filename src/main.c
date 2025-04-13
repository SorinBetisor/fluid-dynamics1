#include "app.h"

const uint32_t validationLayerCount = 1;
const char* validationLayers[] = {
    "VK_LAYER_KHRONOS_validation"
};

const uint32_t deviceExtensionCount = 1;
const char* deviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

int main(void)
{
    App app = {0};

    initWindow(&app);
    initVulkan(&app);
    mainLoop(&app);
    cleanup(&app);

    return 0;
}
