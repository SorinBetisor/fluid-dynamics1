#include "fluid.h"
#include "fluidRenderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "object.h"
#include <GLFW/glfw3.h>

// Callback for keyboard input
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main(void)
{
    // Initialize the fluid simulation.
    char buffer[250];
    FILE *conf = fopen("./windTunelConf", "r");

    // Default configuration parameters.
    int gridSize = 256;
    float diffusion = 0.0001f;
    float viscosity = 0.00018f;
    int pressureIterations = 20;
    float dt = 0.1f;
    int numIterations = 1000;
    float fluidDen = 100.0f;
    float fluidVel = 5.0f;

    // If the configuration file exists, try to read parameters.
    if (conf)
    {
        while (fgets(buffer, sizeof(buffer), conf))
        {
            if (sscanf(buffer, "gridSize %d", &gridSize) == 1)
                continue;
            if (sscanf(buffer, "diffusion %f", &diffusion) == 1)
                continue;
            if (sscanf(buffer, "viscosity %f", &viscosity) == 1)
                continue;
            if (sscanf(buffer, "pressureIterations %d", &pressureIterations) == 1)
                continue;
            if (sscanf(buffer, "dt %f", &dt) == 1)
                continue;
            if (sscanf(buffer, "numIterations %d", &numIterations) == 1)
                continue;
            if (sscanf(buffer, "fluidVel %f", &fluidDen) == 1)
                continue;
            if (sscanf(buffer, "fluidDen %f", &fluidVel) == 1)
                continue;
        }
        fclose(conf);
    }

    // Print the loaded configuration.
    printf("Configuration:\n");
    printf("  gridSize: %d\n", gridSize);
    printf("  diffusion: %f\n", diffusion);
    printf("  viscosity: %f\n", viscosity);
    printf("  pressureIterations: %d\n", pressureIterations);
    printf("  dt: %f\n", dt);
    printf("  numIterations: %d\n", numIterations);
    printf("  fluid density: %f\n", fluidDen);
    printf("  fluid velocity: %f\n", fluidVel);

    Fluid fluid;
    fluid_init(&fluid, gridSize, diffusion, viscosity, pressureIterations, dt);

    Object *airfoil = NULL;
    // Try to load the airfoil, but continue if it fails
    airfoil = object_load("airfoil.dat", fluid.gridSize);
    if (!airfoil) {
        printf("No airfoil object loaded, continuing without obstacle\n");
        
        // Create a default airfoil if loading failed
        airfoil = (Object*)malloc(sizeof(Object));
        if (airfoil) {
            airfoil->name = strdup("Default Airfoil");
            airfoil->count = 0;
            airfoil->x = NULL;
            airfoil->y = NULL;
            airfoil->mask = (int*)calloc(fluid.gridSize * fluid.gridSize, sizeof(int));
            
            // Create a simple airfoil shape directly in the mask
            int centerX = fluid.gridSize / 4;
            int centerY = fluid.gridSize / 2;
            int width = fluid.gridSize / 6;
            int height = fluid.gridSize / 12;
            
            for (int j = 0; j < fluid.gridSize; j++) {
                for (int i = 0; i < fluid.gridSize; i++) {
                    // Simple elliptical airfoil shape
                    float dx = (float)(i - centerX) / width;
                    float dy = (float)(j - centerY) / height;
                    if (dx*dx + 4*dy*dy < 1.0f) {
                        airfoil->mask[i + j * fluid.gridSize] = 1;
                    }
                }
            }
            
            printf("Created default airfoil\n");
        }
    } else {
        printf("Loaded airfoil from file: %s\n", airfoil->name);
        
        // Ensure the airfoil has a sufficient number of mask points
        int maskCount = 0;
        for (int i = 0; i < fluid.gridSize * fluid.gridSize; i++) {
            if (airfoil->mask[i]) maskCount++;
        }
        printf("Airfoil mask has %d active points\n", maskCount);
        
        // If too few points, enlarge the airfoil
        if (maskCount < 100) {
            printf("Enlarging airfoil for better visibility\n");
            int* expandedMask = (int*)calloc(fluid.gridSize * fluid.gridSize, sizeof(int));
            
            // Dilate the mask to make the airfoil more visible
            for (int j = 1; j < fluid.gridSize - 1; j++) {
                for (int i = 1; i < fluid.gridSize - 1; i++) {
                    if (airfoil->mask[i + j * fluid.gridSize] ||
                        airfoil->mask[(i-1) + j * fluid.gridSize] ||
                        airfoil->mask[(i+1) + j * fluid.gridSize] ||
                        airfoil->mask[i + (j-1) * fluid.gridSize] ||
                        airfoil->mask[i + (j+1) * fluid.gridSize]) {
                        expandedMask[i + j * fluid.gridSize] = 1;
                    }
                }
            }
            
            // Replace with expanded mask
            free(airfoil->mask);
            airfoil->mask = expandedMask;
        }
    }
    
    // Initialize the OpenGL renderer
    if (!fluid_renderer_init(800, 800)) {
        fprintf(stderr, "Failed to initialize renderer\n");
        // Clean up.
        object_free(airfoil);
        fluid_free(&fluid);
        return 1;
    }
    
    // Set the keyboard callback
    glfwSetKeyCallback(glfwGetCurrentContext(), key_callback);
    
    // Used to calculate frame rate
    double lastTime = glfwGetTime();
    int frameCount = 0;
    
    // Main simulation loop
    int iter = 0;
    while (iter < numIterations) {
        // Calculate frame rate
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            printf("FPS: %d, Iteration: %d\n", frameCount, iter);
            frameCount = 0;
            lastTime = currentTime;
        }
        
        // Add taps influence: for cells within the tap region, add velocity and density.
        for (int i = 0; i < gridSize; i++) {
            // Add a constant velocity impulse to push fluid rightwards.
            fluid_add_velocity(&fluid, 1, i, fluidVel, 0.0f);

            // Also add density so the object is visible in the background.
            fluid_add_density(&fluid, 1, i, fluidDen);
        }

        // Step the fluid simulation.
        fluid_step(&fluid);

        // Apply object boundaries
        if (airfoil) {
            object_apply_ib(&fluid, airfoil);
        }

        // Render the current simulation state
        fluid_renderer_draw_frame(&fluid);
        
        // Exit if window is closed
        if (glfwWindowShouldClose(glfwGetCurrentContext())) {
            break;
        }
        
        iter++;
    }

    // Clean up resources
    fluid_renderer_cleanup();
    object_free(airfoil);
    fluid_free(&fluid);
    return 0;
}
