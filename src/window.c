#include "app.h"
#include "commandBuffer.h"

void initWindow(App *pApp) {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Failed to initialize SDL2: %s\n", SDL_GetError());
    abort();
  }

  pApp->window =
      SDL_CreateWindow(NAME, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       WIDTH, HEIGHT, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

  if (!pApp->window) {
    fprintf(stderr, "Failed to create SDL window: %s\n", SDL_GetError());
    SDL_Quit();
    abort();
  }

  // Store the pointer to 'pApp' in SDL's per-window user data
  SDL_SetWindowData(pApp->window, "APP_PTR", pApp);
}

void mainLoop(App *pApp) {
  // Initialize fluid simulation
  FluidSim fluidSim;
  createFluidSim(pApp->device, pApp->physicalDevice, pApp->descriptorPool,
                 &fluidSim);
  initializeSimulationOutput(&fluidSim, "fluid_simulation.csv");

  // Set initial conditions:
  addFluidObject(pApp->device, &fluidSim, OBJ_CIRCLE, 0.5f, 0.5f, 0.1f, NULL);

  // Or add an airfoil
//   addFluidObject(pApp->device, &fluidSim, OBJ_AIRFOIL, 0.5f, 0.5f, 0.0f,
//                  "naca2412.dat");

  // For wind tunnel:
  setInitialConditions(pApp->device, &fluidSim, IC_UNIFORM_FLOW, 1.0f);
  // Or for source/sink:
  // setInitialConditions(pApp->device, &fluidSim, IC_SOURCE_SINK, 2.0f);

  // Simulation parameters
  float currentTime = 0.0f;
  const float timeStep = 0.016f; // ~60 FPS

  int running = 1;
  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = 0;
      } else if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
          // Retrieve 'App*' from the SDL window:
          SDL_Window *sdlWin = SDL_GetWindowFromID(event.window.windowID);
          if (sdlWin) {
            App *theApp = (App *)SDL_GetWindowData(sdlWin, "APP_PTR");
            if (theApp) {
              theApp->framebufferResized = true;
            }
          }

          // (Optional) Print out the new size:
          printf("Window resized to %dx%d\n",
                 event.window.data1,  // new width
                 event.window.data2); // new height
        }
      }
    }

      // Update simulation
      VkCommandBuffer cmdBuf =
          beginSingleTimeCommands(pApp->device, pApp->commandPool);
      updateFluidSim(cmdBuf, &fluidSim);
      endSingleTimeCommands(pApp->device, pApp->graphicsQueue,
                            pApp->commandPool, cmdBuf);

      // Save simulation data
      saveSimulationFrame(pApp->device, &fluidSim, currentTime);
      currentTime += timeStep;

      // For simplicity, always draw after handling an event:
      drawFrame(pApp);

    // Delay to prevent high CPU usage
    // SDL_Delay(16); // ~60 FPS
  }

  vkDeviceWaitIdle(pApp->device);
  closeSimulationOutput(&fluidSim);
}