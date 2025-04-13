#include "app.h"

VkVertexInputAttributeDescription *getAttributeDescriptions()
{

  // dynamically allocate the array
  VkVertexInputAttributeDescription *attributeDescriptions = malloc(sizeof(VkVertexInputAttributeDescription) * ATTRIBUTE_COUNT);

  if (attributeDescriptions == NULL)
  {
    // handle allocation failure here
    printf("Allocation of attribute descriptions failed\n");
    abort();
  }

  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);

  return attributeDescriptions;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, App *pApp)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(pApp->physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
  {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }

  printf("failed to find suitable memory type!\n");
  abort();
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer *buffer, VkDeviceMemory *bufferMemory, App *pApp)
{
  VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  if (vkCreateBuffer(pApp->device, &bufferInfo, NULL, buffer) != VK_SUCCESS)
  {
    printf("failed to create buffer!\n");
    abort();
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(pApp->device, *buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, pApp),
  };

  if (vkAllocateMemory(pApp->device, &allocInfo, NULL, bufferMemory) != VK_SUCCESS)
  {
    printf("failed to allocate buffer memory!\n");
    abort();
  }

  vkBindBufferMemory(pApp->device, *buffer, *bufferMemory, 0);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, App *pApp)
{
  VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandPool = pApp->transferCommandPool, // Use transfer command pool
      .commandBufferCount = 1,
  };

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(pApp->device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkBufferCopy copyRegion = {
      .srcOffset = 0,
      .dstOffset = 0,
      .size = size,
  };
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
  };
  vkQueueSubmit(pApp->transferQueue, 1, &submitInfo, VK_NULL_HANDLE); // Submit to transfer queue
  vkQueueWaitIdle(pApp->transferQueue);

  vkFreeCommandBuffers(pApp->device, pApp->transferCommandPool, 1, &commandBuffer); // Free using transfer command pool
}

void createVertexBuffer(App *pApp)
{
  VkDeviceSize bufferSize = sizeof(Vertex) * pApp->verticesCount;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory, pApp);

  void *data;
  vkMapMemory(pApp->device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, pApp->vertices, (size_t)bufferSize);
  vkUnmapMemory(pApp->device, stagingBufferMemory);

  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &pApp->vertexBuffer, &pApp->vertexBufferMemory, pApp);

  copyBuffer(stagingBuffer, pApp->vertexBuffer, bufferSize, pApp);
  vkDestroyBuffer(pApp->device, stagingBuffer, NULL);
  vkFreeMemory(pApp->device, stagingBufferMemory, NULL);
}

static VkVertexInputBindingDescription getBindingDescription()
{
  VkVertexInputBindingDescription bindingDescription = {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  return bindingDescription;
}

void populateVertecies(App *pApp) // WARN: This is a stub, replace later with proper Vertex generation
{
  pApp->verticesCount = 4;
  pApp->vertices = malloc(sizeof(Vertex) * pApp->verticesCount);
  if (pApp->vertices == NULL)
  {
    printf("Failed to allocate memory for vertices.\n");
    abort();
  }

  glm_vec2_copy((vec2){-0.5f, -0.5f}, pApp->vertices[0].pos);
  glm_vec3_copy((vec3){1.0f, 0.0f, 0.0f}, pApp->vertices[0].color);

  glm_vec2_copy((vec2){0.5f, -0.5f}, pApp->vertices[1].pos);
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, pApp->vertices[1].color);

  glm_vec2_copy((vec2){0.5f, 0.5f}, pApp->vertices[2].pos);
  glm_vec3_copy((vec3){0.0f, 0.0f, 0.1f}, pApp->vertices[2].color);

  glm_vec2_copy((vec2){-0.5f, 0.5f}, pApp->vertices[3].pos);
  glm_vec3_copy((vec3){1.0f, 1.0f, 1.0f}, pApp->vertices[3].color);

  // index data
  pApp->indicesCount = 6;
  pApp->indices = malloc(sizeof(uint16_t) * pApp->indicesCount);
  if (pApp->indices == NULL)
  {
    printf("Failed to allocate memory for indices\n");
    abort();
  }
  pApp->indices[0] = 0;
  pApp->indices[1] = 1;
  pApp->indices[2] = 2;
  pApp->indices[3] = 2;
  pApp->indices[4] = 3;
  pApp->indices[5] = 0;
}

void createIndexBuffer(App *pApp)
{
  VkDeviceSize bufferSize = sizeof(uint16_t) * pApp->indicesCount;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory, pApp);

  void *data;
  vkMapMemory(pApp->device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, pApp->indices, (size_t)bufferSize);
  vkUnmapMemory(pApp->device, stagingBufferMemory);

  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &pApp->indexBuffer, &pApp->indexBufferMemory, pApp);

  copyBuffer(stagingBuffer, pApp->indexBuffer, bufferSize, pApp);

  vkDestroyBuffer(pApp->device, stagingBuffer, NULL);
  vkFreeMemory(pApp->device, stagingBufferMemory, NULL);
  printf("Index buffer created\n");
}

void createSwapChain(App *pApp)
{
  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(pApp->physicalDevice, pApp->surface);

  VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats, swapChainSupport.formatCount);
  VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes, swapChainSupport.presentModeCount);
  VkExtent2D extent = chooseSwapExtent(&swapChainSupport.capabilities, pApp);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

  if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
  {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = pApp->surface,
      .minImageCount = imageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};

  QueueFamilyIndices indices = findQueueFamilies(pApp->physicalDevice, pApp->surface);
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily, indices.presentationFamily};

  if (indices.graphicsFamily != indices.presentationFamily)
  {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  else
  {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;  // Optional
    createInfo.pQueueFamilyIndices = NULL; // Optional
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(pApp->device, &createInfo, NULL, &pApp->swapChain) != VK_SUCCESS)
  {
    printf("failed to create swap chain!\n");
  }
  vkGetSwapchainImagesKHR(pApp->device, pApp->swapChain, &imageCount, NULL);
  pApp->swapChainImages = malloc(sizeof(VkImage) * imageCount);
  pApp->swapChainImageCount = imageCount;
  vkGetSwapchainImagesKHR(pApp->device, pApp->swapChain, &imageCount, pApp->swapChainImages);

  pApp->swapChainImageFormat = surfaceFormat.format;
  pApp->swapChainExtent = extent;

  freeSwapChainSupportDetails(&swapChainSupport);
}

void cleanupSwapChain(App *pApp)
{
  for (size_t i = 0; i < pApp->swapChainImageCount; i++)
  {
    vkDestroyFramebuffer(pApp->device, pApp->swapchainFramebuffers[i], NULL);
  }

  for (size_t i = 0; i < pApp->swapChainImageCount; i++)
  {
    vkDestroyImageView(pApp->device, pApp->swapchainImageView[i], NULL);
  }

  vkDestroySwapchainKHR(pApp->device, pApp->swapChain, NULL);
}

void recreateSwapChain(App *pApp)
{
  vkDeviceWaitIdle(pApp->device);

  cleanupSwapChain(pApp);

  createSwapChain(pApp);
  createImageView(pApp);
  createFramebuffers(pApp);
}

void initVulkan(App *pApp)
{
  createInstance(pApp);
  setupDebugMessenger(pApp);
  createSurface(pApp);
  pickPhysicalDevice(pApp);
  findQueueFamilies(pApp->physicalDevice, pApp->surface);
  createLogicalDevice(pApp);
  createSwapChain(pApp);
  createImageView(pApp);
  createRenderPass(pApp);
  populateVertecies(pApp);
  createGraphicsPipeline(pApp);
  createFramebuffers(pApp);
  createCommandPool(pApp);
  createVertexBuffer(pApp);
  createIndexBuffer(pApp);
  createCommandBuffers(pApp);
  createSyncObjects(pApp);
}

void createSyncObjects(App *pApp)
{

  pApp->imageAvailableSemaphores = malloc(sizeof(VkSemaphore) * MAX_FRAMES_IN_FLIGHT);
  if (!pApp->imageAvailableSemaphores)
  {
    fprintf(stderr, "Failed to allocate imageAvailableSemaphores.\n");
    abort();
  }

  pApp->renderFinishedSemaphores = malloc(sizeof(VkSemaphore) * MAX_FRAMES_IN_FLIGHT);
  if (!pApp->renderFinishedSemaphores)
  {
    fprintf(stderr, "Failed to allocate renderFinishedSemaphores.\n");
    abort();
  }

  pApp->inFlightFences = malloc(sizeof(VkFence) * MAX_FRAMES_IN_FLIGHT);
  if (!pApp->inFlightFences)
  {
    fprintf(stderr, "Failed to allocate inFlightFences.\n");
    abort();
  }

  pApp->currentFrame = 0;
  VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };

  VkFenceCreateInfo fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
  {
    if (vkCreateSemaphore(pApp->device, &semaphoreInfo, NULL, &pApp->imageAvailableSemaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(pApp->device, &semaphoreInfo, NULL, &pApp->renderFinishedSemaphores[i]) != VK_SUCCESS ||
        vkCreateFence(pApp->device, &fenceInfo, NULL, &pApp->inFlightFences[i]) != VK_SUCCESS)
    {
      printf("failed to create semaphores!\n");
      abort();
    }
  }
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, App *pApp)
{
  VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = 0,               // Optional
      .pInheritanceInfo = NULL, // Optional
  };

  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
  {
    printf("failed to begin recording command buffer!\n");
    abort();
  }

  VkRenderPassBeginInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = pApp->renderPass,
      .framebuffer = pApp->swapchainFramebuffers[imageIndex],
      .renderArea.offset = {0, 0},
      .renderArea.extent = pApp->swapChainExtent,
  };

  VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pApp->graphicsPipeline);

  VkBuffer vertexBuffers[] = {pApp->vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

  vkCmdBindIndexBuffer(commandBuffer, pApp->indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)pApp->swapChainExtent.width,
      .height = (float)pApp->swapChainExtent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  VkRect2D scissor = {
      .offset = {0, 0},
      .extent = pApp->swapChainExtent,
  };

  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  vkCmdDrawIndexed(commandBuffer, pApp->indicesCount, 1, 0, 0, 0);

  vkCmdEndRenderPass(commandBuffer);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
  {
    printf("failed to record command buffer!\n");
    abort();
  }
}

void createCommandBuffers(App *pApp)
{
  pApp->commandBuffers = malloc(sizeof(VkCommandBuffer) * MAX_FRAMES_IN_FLIGHT);
  if (!pApp->commandBuffers)
  {
    fprintf(stderr, "Failed to allocate commandBuffers.\n");
    abort();
  }
  VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pApp->commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
  };

  if (vkAllocateCommandBuffers(pApp->device, &allocInfo, pApp->commandBuffers) != VK_SUCCESS)
  {
    printf("failed to allocate command buffers!\n");
    abort();
  }
}

void createFramebuffers(App *pApp)
{
  pApp->swapchainFramebuffers = malloc(sizeof(VkFramebuffer) * pApp->swapChainImageCount);

  for (size_t i = 0; i < pApp->swapChainImageCount; i++)
  {
    VkImageView attachments[] = {
        pApp->swapchainImageView[i]};

    VkFramebufferCreateInfo framebufferInfo = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = pApp->renderPass,
        .attachmentCount = 1,
        .pAttachments = attachments,
        .width = pApp->swapChainExtent.width,
        .height = pApp->swapChainExtent.height,
        .layers = 1,
    };

    if (vkCreateFramebuffer(pApp->device, &framebufferInfo, NULL, &pApp->swapchainFramebuffers[i]) != VK_SUCCESS)
    {
      printf("failed to create framebuffer!\n");
      abort();
    }
  }
}

void createCommandPool(App *pApp)
{

  VkCommandPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pApp->QueueFamilyIndices.graphicsFamily,
  };
  if (vkCreateCommandPool(pApp->device, &poolInfo, NULL, &pApp->commandPool) != VK_SUCCESS)
  {
    printf("failed to create command pool!\n");
    abort();
  }
  VkCommandPoolCreateInfo poolInfoTransfer = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pApp->QueueFamilyIndices.transferFamily,
  }; // Added transfer
  if (vkCreateCommandPool(pApp->device, &poolInfoTransfer, NULL, &pApp->transferCommandPool) != VK_SUCCESS) // Added transfer
  {
    printf("failed to create transfer command pool!\n");
    abort();
  }
}

void createGraphicsPipeline(App *pApp)
{
  Shaderfile fragShaderCode = readShaderFile("shaders/frag.spv");
  Shaderfile vertShaderCode = readShaderFile("shaders/vert.spv");

  VkShaderModule vertShaderModule = createShaderModule(vertShaderCode, pApp);
  VkShaderModule fragShaderModule = createShaderModule(fragShaderCode, pApp);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertShaderModule,
      .pName = "main",
      .pSpecializationInfo = NULL,
  };
  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {

      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragShaderModule,
      .pName = "main",
      .pSpecializationInfo = NULL,
  };

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  VkVertexInputBindingDescription bindingDescription = getBindingDescription();
  VkVertexInputAttributeDescription *attributeDescriptions = getAttributeDescriptions();

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount = ATTRIBUTE_COUNT,
      .pVertexAttributeDescriptions = attributeDescriptions,
  };

  VkDynamicState dynamicStates[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR};
  uint32_t dynamicStatesSize = 2;

  VkPipelineDynamicStateCreateInfo dynamicState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = dynamicStatesSize,
      .pDynamicStates = dynamicStates,
  };

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  };

  VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)pApp->swapChainExtent.width,
      .height = (float)pApp->swapChainExtent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  VkRect2D scissor = {
      .offset = {0, 0},
      .extent = pApp->swapChainExtent,
  };

  VkPipelineViewportStateCreateInfo viewportState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  VkPipelineRasterizationStateCreateInfo rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .lineWidth = 1.0f,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.0f, // Optional
      .depthBiasClamp = 0.0f,          // Optional
      .depthBiasSlopeFactor = 0.0f,    // Optional
  };

  VkPipelineMultisampleStateCreateInfo multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .sampleShadingEnable = VK_FALSE,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .minSampleShading = 1.0f,          // Optional
      .pSampleMask = NULL,               // Optional
      .alphaToCoverageEnable = VK_FALSE, // Optional
      .alphaToOneEnable = VK_FALSE,      // Optional
  };

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
      .colorBlendOp = VK_BLEND_OP_ADD,             // Optional
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
      .alphaBlendOp = VK_BLEND_OP_ADD,             // Optional
  };

  VkPipelineColorBlendStateCreateInfo colorBlending = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY, // Optional
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
      .blendConstants[0] = 0.0f, // Optional
      .blendConstants[1] = 0.0f, // Optional
      .blendConstants[2] = 0.0f, // Optional
      .blendConstants[3] = 0.0f, // Optional
  };

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 0,         // Optional
      .pSetLayouts = NULL,         // Optional
      .pushConstantRangeCount = 0, // Optional
      .pPushConstantRanges = NULL, // Optional
  };

  if (vkCreatePipelineLayout(pApp->device, &pipelineLayoutInfo, NULL, &pApp->pipelineLayout) != VK_SUCCESS)
  {
    printf("failed to create pipeline layout!\n");
    abort();
  }

  VkGraphicsPipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = NULL, // Optional
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = pApp->pipelineLayout,
      .renderPass = pApp->renderPass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE, // Optional
      .basePipelineIndex = -1,              // Optional
  };

  if (vkCreateGraphicsPipelines(pApp->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pApp->graphicsPipeline) != VK_SUCCESS)
  {
    printf("failed to create graphics pipeline!\n");
    abort();
  }

  vkDestroyShaderModule(pApp->device, fragShaderModule, NULL);
  vkDestroyShaderModule(pApp->device, vertShaderModule, NULL);
  free(fragShaderCode.code);
  free(vertShaderCode.code);
  free(attributeDescriptions);
}

void drawFrame(App *pApp)
{
  vkWaitForFences(pApp->device, 1, &pApp->inFlightFences[pApp->currentFrame], VK_TRUE, UINT64_MAX);
  vkResetFences(pApp->device, 1, &pApp->inFlightFences[pApp->currentFrame]);

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(pApp->device, pApp->swapChain, UINT64_MAX, pApp->imageAvailableSemaphores[pApp->currentFrame], VK_NULL_HANDLE, &imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    recreateSwapChain(pApp);
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    printf("failed to acquire swap chain image!\n");
    abort();
  }

  vkResetCommandBuffer(pApp->commandBuffers[pApp->currentFrame], 0);
  recordCommandBuffer(pApp->commandBuffers[pApp->currentFrame], imageIndex, pApp);

  VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
  };

  VkSemaphore waitSemaphores[] = {pApp->imageAvailableSemaphores[pApp->currentFrame]};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &pApp->commandBuffers[pApp->currentFrame];

  VkSemaphore signalSemaphores[] = {pApp->renderFinishedSemaphores[pApp->currentFrame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(pApp->graphicsQueue, 1, &submitInfo, pApp->inFlightFences[pApp->currentFrame]) != VK_SUCCESS)
  {
    printf("failed to submit draw command buffer!\n");
    abort();
  }

  VkPresentInfoKHR presentInfo = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = signalSemaphores,
  };

  VkSwapchainKHR swapChains[] = {pApp->swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.pResults = NULL; // Optional

  result = vkQueuePresentKHR(pApp->presentationQueue, &presentInfo);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
  {
    pApp->framebufferResized = false;
    recreateSwapChain(pApp);
  }
  else if (result != VK_SUCCESS)
  {
    printf("failed to present swap chain image!\n");
    abort();
  }

  pApp->currentFrame = (pApp->currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

VkShaderModule createShaderModule(Shaderfile shader, App *pApp)
{
  VkShaderModuleCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = shader.size,
      .pCode = (uint32_t *)shader.code,
  };

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(pApp->device, &createInfo, NULL, &shaderModule) != VK_SUCCESS)
  {
    printf("Failed to create a shader module\n");
    abort();
  }
  return shaderModule;
}

void createRenderPass(App *pApp)
{
  VkAttachmentDescription colorAttachment = {
      .format = pApp->swapChainImageFormat,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VkAttachmentReference colorAttachmentRef = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };

  VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentRef,
  };

  VkSubpassDependency dependency = {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,

      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = 0,

      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };

  VkRenderPassCreateInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorAttachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency,
  };

  if (vkCreateRenderPass(pApp->device, &renderPassInfo, NULL, &pApp->renderPass) != VK_SUCCESS)
  {
    printf("failed to create render pass!\n");
    abort();
  }
}

void createImageView(App *pApp)
{
  pApp->swapchainImageView = malloc(sizeof(VkImageView) * pApp->swapChainImageCount);
  for (uint32_t i = 0; i < pApp->swapChainImageCount; i++)
  {
    VkImageViewCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = pApp->swapChainImages[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = pApp->swapChainImageFormat,
        .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1};
    if (vkCreateImageView(pApp->device, &createInfo, NULL, &pApp->swapchainImageView[i]) != VK_SUCCESS)
    {
      printf("failed to create image views!");
      abort();
    }
  }
}

void cleanup(App *pApp)
{
  cleanupSwapChain(pApp);
  vkDestroyBuffer(pApp->device, pApp->vertexBuffer, NULL);
  vkFreeMemory(pApp->device, pApp->vertexBufferMemory, NULL);
  vkDestroyBuffer(pApp->device, pApp->indexBuffer, NULL);
  vkFreeMemory(pApp->device, pApp->indexBufferMemory, NULL);
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
  {
    vkDestroySemaphore(pApp->device, pApp->imageAvailableSemaphores[i], NULL);
    vkDestroySemaphore(pApp->device, pApp->renderFinishedSemaphores[i], NULL);
    vkDestroyFence(pApp->device, pApp->inFlightFences[i], NULL);
  }
  free(pApp->vertices);
  free(pApp->imageAvailableSemaphores);
  free(pApp->renderFinishedSemaphores);
  free(pApp->inFlightFences);
  vkFreeCommandBuffers(pApp->device, pApp->commandPool, MAX_FRAMES_IN_FLIGHT, pApp->commandBuffers);
  free(pApp->commandBuffers);
  vkDestroyCommandPool(pApp->device, pApp->commandPool, NULL);
  vkDestroyCommandPool(pApp->device, pApp->transferCommandPool, NULL);
  free(pApp->swapchainFramebuffers);
  vkDestroyPipeline(pApp->device, pApp->graphicsPipeline, NULL);
  vkDestroyPipelineLayout(pApp->device, pApp->pipelineLayout, NULL);
  vkDestroyRenderPass(pApp->device, pApp->renderPass, NULL);
  free(pApp->swapchainImageView);
  if (pApp->swapChainImages)
  {
    free(pApp->swapChainImages);
  }
  if (enableValidationLayers)
  {
    DestroyDebugUtilsMessengerEXT(pApp->instance, pApp->debugMessanger, NULL);
  }
  vkDestroyDevice(pApp->device, NULL);
  vkDestroySurfaceKHR(pApp->instance, pApp->surface, NULL);
  vkDestroyInstance(pApp->instance, NULL);
  SDL_DestroyWindow(pApp->window);
  SDL_Quit();
  printf("Cleanup successful\n");
}

void createInstance(App *pApp)
{
  if (enableValidationLayers && !checkValidationLayerSupport())
  {
    printf("Validation Layers requested but not available\n");
  }

  VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = NAME,
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0};

  uint32_t SDLExtensionCount = 0;
  SDL_Vulkan_GetInstanceExtensions(pApp->window, &SDLExtensionCount, NULL);

  const char **SDLExtensions = malloc(SDLExtensionCount * sizeof(char *));
  if (!SDLExtensions)
  {
    fprintf(stderr, "Failed to allocate memory for SDLExtensions.\n");
    free(SDLExtensions);
    abort();
  }

  SDL_Vulkan_GetInstanceExtensions(pApp->window, &SDLExtensionCount, SDLExtensions);

  const char **extensions = malloc((SDLExtensionCount + 1) * sizeof(char *));
  if (!extensions)
  {
    fprintf(stderr, "Failed to allocate memory for extensions.\n");
    free(extensions);
    free(SDLExtensions);
    abort();
  }
  memcpy(extensions, SDLExtensions, SDLExtensionCount * sizeof(char *));
  if (enableValidationLayers)
  {
    extensions[SDLExtensionCount] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    SDLExtensionCount++;
  }

  VkInstanceCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = SDLExtensionCount,
      .ppEnabledExtensionNames = extensions,
      .enabledLayerCount = enableValidationLayers ? validationLayerCount : 0,
      .ppEnabledLayerNames = enableValidationLayers ? validationLayers : NULL};

  if (vkCreateInstance(&createInfo, NULL, &pApp->instance) != VK_SUCCESS)
  {
    printf("Failed to create Vulkan Instance\n");

    free(SDLExtensions);
    free(extensions);
    exit(-1);
  }

  free(SDLExtensions);
  free(extensions);

  printf("Vulkan Instance created successfully.\n");
}

void createSurface(App *pApp)
{
  if (SDL_Vulkan_CreateSurface(pApp->window, pApp->instance, &pApp->surface) != SDL_TRUE)
  {
    printf("failed to create window surface\n");
    abort();
  }
  printf("surface created\n");
}

bool checkValidationLayerSupport()
{
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, NULL);

  // Dynamically allocate memory for available layers
  VkLayerProperties *availableLayers = (VkLayerProperties *)malloc(layerCount * sizeof(VkLayerProperties));
  if (!availableLayers)
  {
    fprintf(stderr, "Failed to allocate memory for layer properties.\n");
    free(availableLayers);
    abort();
  }
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);
  for (uint32_t i = 0; i < validationLayerCount; i++)
  {
    bool layerFound = false;
    for (uint32_t j = 0; j < layerCount; j++)
    {
      if (strcmp(availableLayers[j].layerName, validationLayers[i]))
      {
        layerFound = true;
        printf("%s\n", availableLayers[j].layerName);
        break;
      }
    }
    if (!layerFound)
    {
      free(availableLayers);
      return false;
    }
  }
  free(availableLayers);
  return true;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);

  VkExtensionProperties *availableExtensions = malloc(sizeof(VkExtensionProperties) * extensionCount);
  if (!availableExtensions)
  {
    fprintf(stderr, "Failed to allocate memory for extension check.\n");
    free(availableExtensions);
    abort();
  }

  vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, availableExtensions);
  for (uint32_t i = 0; i < deviceExtensionCount; i++)
  {
    bool extensionfound = false;
    for (uint32_t j = 0; j < extensionCount; j++)
    {
      // printf("checking %s = %s \n", deviceExtensions[i], availableExtensions[j].extensionName);
      if (strcmp(deviceExtensions[i], availableExtensions[j].extensionName) == 0)
      {
        extensionfound = true;
        break;
      }
    }
    if (!extensionfound)
    {
      printf("Extension %s not found \n", deviceExtensions[i]);
      free(availableExtensions);
      return false;
    }
  }
  printf("All extensions found \n");
  free(availableExtensions);
  return true;
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
  SwapChainSupportDetails details = {0};

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &details.formatCount, NULL);

  if (details.formatCount != 0)
  {
    VkSurfaceFormatKHR *formats = malloc(sizeof(VkSurfaceFormatKHR) * details.formatCount);
    if (!formats)
    {
      fprintf(stderr, "Failed to allocate memory for Surface formats.\n");
      abort();
    }
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &details.formatCount, formats);
    details.formats = formats;
  }

  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &details.presentModeCount, NULL);

  if (details.presentModeCount != 0)
  {
    VkPresentModeKHR *presentMode = malloc(sizeof(VkPresentModeKHR) * details.presentModeCount);
    if (!presentMode)
    {
      fprintf(stderr, "Failed to allocate memory for Present modes.\n");
      free(details.formats);
      abort();
    }
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &details.presentModeCount, presentMode);
    details.presentModes = presentMode;
  }
  return details;
}

void freeSwapChainSupportDetails(SwapChainSupportDetails *details)
{
  free(details->formats);
  free(details->presentModes);
  details->formats = NULL;
  details->presentModes = NULL;
  details->formatCount = 0;
  details->presentModeCount = 0;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkSurfaceFormatKHR *availableFormats, uint32_t formatCount)
{
  for (uint32_t i = 0; i < formatCount; i++)
  {
    if (availableFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB && availableFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return availableFormats[i];
    }
  }
  return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(VkPresentModeKHR *availablePresentModes, uint32_t presentModeCount)
{
  for (uint32_t i = 0; i < presentModeCount; i++)
  {
    if (availablePresentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
    {
      return availablePresentModes[i];
    }
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR *capabilities, App *pApp)
{
  if (capabilities->currentExtent.width != UINT32_MAX)
  {
    return capabilities->currentExtent;
  }
  else
  {
    int width, height;
    SDL_Vulkan_GetDrawableSize(pApp->window, &width, &height);

    VkExtent2D actualExtent = {
        .width = (uint32_t)width,
        .height = (uint32_t)height};

    actualExtent.width = actualExtent.width < capabilities->minImageExtent.width
                             ? capabilities->minImageExtent.width
                             : (actualExtent.width > capabilities->maxImageExtent.width
                                    ? capabilities->maxImageExtent.width
                                    : actualExtent.width);

    actualExtent.height = actualExtent.height < capabilities->minImageExtent.height
                              ? capabilities->minImageExtent.height
                              : (actualExtent.height > capabilities->maxImageExtent.height
                                     ? capabilities->maxImageExtent.height
                                     : actualExtent.height);

    return actualExtent;
  }
}

uint32_t rateDeviceSuitability(VkPhysicalDevice device, VkSurfaceKHR surface)
{
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;

  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  uint32_t score = 0;

  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
  {
    score += 1000;
  }

  score += deviceProperties.limits.maxImageDimension2D;

  if (!deviceFeatures.geometryShader)
  {
    return 0;
  }

  // NOTE: To improve performance we could favour the queue families that have both graphics and present support.
  // We could check the returned indecies and check if they are the same increase the score.
  QueueFamilyIndices indices = findQueueFamilies(device, surface);
  if (!indices.isGraphicsFamilySet)
  {
    printf("Queue family not supported \n");
    return 0;
  }

  bool extensionSupported = checkDeviceExtensionSupport(device);

  if (!extensionSupported)
  {
    printf("No extension support provided \n");
    return 0;
  }

  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
  if (swapChainSupport.formatCount == 0 || swapChainSupport.presentModeCount == 0)
  {
    printf("Cant create swapchain\n");
    freeSwapChainSupportDetails(&swapChainSupport);
    return 0;
  }
  freeSwapChainSupportDetails(&swapChainSupport);

  return score;
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
{
  QueueFamilyIndices indices = {0};

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);

  VkQueueFamilyProperties *queueFamilies =
      malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
  if (!queueFamilies)
  {
    fprintf(stderr, "Failed to allocate memory for queueFamilies.\n");
    abort();
  }
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

  // We loop through all families to find any that support
  // both graphics and present. We do NOT break immediately,
  // so that we can check if there's a single queue that supports both.
  for (uint32_t i = 0; i < queueFamilyCount; i++)
  {
    // Check for graphics support
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      indices.graphicsFamily = i;
      indices.isGraphicsFamilySet = true;
    }
    // Added
    // Check for transfer support but not graphics support
    if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
    {
      indices.transferFamily = i;
      indices.isTransferFamilySet = true;
    }

    // Check for present support
    VkBool32 presentSupport = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
    if (presentSupport)
    {
      indices.presentationFamily = i;
      indices.isPresentFamilySet = true;
    }

    // If we have both, we can break early
    if (indices.isGraphicsFamilySet && indices.isPresentFamilySet && indices.isTransferFamilySet)
    {
      break;
    }
  }

  free(queueFamilies);
  return indices;
}

void createLogicalDevice(App *pApp)
{
  QueueFamilyIndices indices = pApp->QueueFamilyIndices;

  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(pApp->physicalDevice, &deviceFeatures);

  // Queue priority
  const float queuePriority = 1.0f;

  // Create queue info
  VkDeviceQueueCreateInfo queueCreateInfos[3];
  uint32_t queueCreateInfoCount = 0;

  if (indices.graphicsFamily == indices.presentationFamily && indices.graphicsFamily == indices.transferFamily)
  {
    // Single queue family handles both graphics, presentation, and transfer
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.graphicsFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos[queueCreateInfoCount++] = queueCreateInfo;
  }
  else if (indices.graphicsFamily == indices.presentationFamily)
  {
    // Separate queue families for transfer only
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.graphicsFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos[queueCreateInfoCount++] = queueCreateInfo;

    if (indices.graphicsFamily != indices.transferFamily)
    {
      VkDeviceQueueCreateInfo transferQueueCreateInfo = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = indices.transferFamily,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      };
      queueCreateInfos[queueCreateInfoCount++] = transferQueueCreateInfo;
    }
  }
  else
  {
    // Separate queue families for graphics and presentation and optionally transfer
    VkDeviceQueueCreateInfo graphicsQueueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.graphicsFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos[queueCreateInfoCount++] = graphicsQueueCreateInfo;

    VkDeviceQueueCreateInfo presentQueueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.presentationFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos[queueCreateInfoCount++] = presentQueueCreateInfo;

    if (indices.graphicsFamily != indices.transferFamily && indices.presentationFamily != indices.transferFamily)
    {
      VkDeviceQueueCreateInfo transferQueueCreateInfo = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = indices.transferFamily,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      };
      queueCreateInfos[queueCreateInfoCount++] = transferQueueCreateInfo;
    }
  }

  VkDeviceCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pQueueCreateInfos = queueCreateInfos,
      .queueCreateInfoCount = queueCreateInfoCount,
      .pEnabledFeatures = &deviceFeatures,
      .enabledExtensionCount = deviceExtensionCount,
      .ppEnabledExtensionNames = deviceExtensions};

  if (enableValidationLayers)
  {
    createInfo.enabledLayerCount = validationLayerCount;
    createInfo.ppEnabledLayerNames = validationLayers;
  }
  else
  {
    createInfo.enabledLayerCount = 0;
  }

  if (vkCreateDevice(pApp->physicalDevice, &createInfo, NULL, &pApp->device) != VK_SUCCESS)
  {
    printf("failed to create logical device! \n");
    abort();
  }
  printf("logical device created \n");

  // Retrieve graphics queue
  vkGetDeviceQueue(pApp->device, indices.graphicsFamily, 0, &pApp->graphicsQueue);

  // Retrieve presentation queue (reuse graphics queue if families are identical)
  if (indices.graphicsFamily == indices.presentationFamily)
  {
    pApp->presentationQueue = pApp->graphicsQueue; // Reuse
    printf("reused graphics queue \n");
  }
  else
  {
    printf("didnt reuse graphics queue \n");
    vkGetDeviceQueue(pApp->device, indices.presentationFamily, 0, &pApp->presentationQueue);
  }
  // Added
  vkGetDeviceQueue(pApp->device, indices.transferFamily, 0, &pApp->transferQueue);
}

void setupDebugMessenger(App *pApp)
{
  if (!enableValidationLayers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = debugCallback,
      .pUserData = NULL};
  if (CreateDebugUtilsMessengerEXT(pApp->instance, &createInfo, NULL, &pApp->debugMessanger) != VK_SUCCESS)
  {
    printf("Failed to set up debug messenger");
  }
}

void pickPhysicalDevice(App *pApp)
{
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(pApp->instance, &deviceCount, NULL);

  if (deviceCount == 0)
  {
    printf("Failed to find a gpu with Vulkan support");
    abort();
  }
  VkPhysicalDevice devices[deviceCount];
  vkEnumeratePhysicalDevices(pApp->instance, &deviceCount, devices);

  VkPhysicalDevice device;
  uint32_t deviceScore = 0;
  for (uint32_t i = 0; i < deviceCount; i++)
  {
    uint32_t score = rateDeviceSuitability(devices[i], pApp->surface);
    if (score > deviceScore)
    {
      deviceScore = score;
      device = devices[i];
    }
  }

  if (device == NULL)
  {
    printf("failed to find a suitable GPU");
  }
  pApp->physicalDevice = device;
  printf("Gpu selected. \n");

  QueueFamilyIndices indices = findQueueFamilies(device, pApp->surface);
  pApp->QueueFamilyIndices = indices;
}
