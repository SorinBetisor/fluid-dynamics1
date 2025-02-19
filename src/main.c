#include "fluid.h"
#include "fluidRenderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "object.h"

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
    if (conf) {
        while (fgets(buffer, sizeof(buffer), conf)) {
            if (sscanf(buffer, "gridSize %d", &gridSize) == 1) continue;
            if (sscanf(buffer, "diffusion %f", &diffusion) == 1) continue;
            if (sscanf(buffer, "viscosity %f", &viscosity) == 1) continue;
            if (sscanf(buffer, "pressureIterations %d", &pressureIterations) == 1) continue;
            if (sscanf(buffer, "dt %f", &dt) == 1) continue;
            if (sscanf(buffer, "numIterations %d", &numIterations) == 1) continue;
            if (sscanf(buffer, "fluidVel %f", &fluidDen) == 1) continue;
            if (sscanf(buffer, "fluidDen %f", &fluidVel) == 1) continue;
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

    Object *airfoil = object_load("airfoil.dat", fluid.gridSize);
    
    char filename[256];

    for (int iter = 0; iter < numIterations; iter++)
    {
        // Add taps influence: for cells within the tap region, add velocity and density.
        printf("step %i", iter);
        for (int i = 0; i < gridSize; i++)
        {
            // Add a constant velocity impulse to push fluid rightwards.
            fluid_add_velocity(&fluid, 1, i, fluidVel, 0.0f);

            // Also add density so the object is visible in the background.
            fluid_add_density(&fluid, 1, i, fluidDen);
        }
        printf(".");

        // Step the fluid simulation.
        fluid_step(&fluid);

        object_apply(&fluid, airfoil);

        // Create a filename that includes the iteration number.
        sprintf(filename, "frame_%03d.ppm", iter);
        // Render the current fluid density and overlay particles.
        fluid_renderer_draw(&fluid, filename);
        printf("done\n");
    }

    // Clean up.
    object_free(airfoil);
    fluid_free(&fluid);
    return 0;
}
