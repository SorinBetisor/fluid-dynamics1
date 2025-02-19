#include "fluid.h"
#include "fluidRenderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main(void) {
    // Initialize the fluid simulation.
    Fluid fluid;
    int gridSize = 512;
    fluid_init(&fluid, gridSize, 0.0001f, 0.0001f, 20, 0.1f);

    //Define a source, where the fluid gets added
    int tapCenterX = gridSize / 4;
    int tapCenterY = gridSize / 2;
    int tapRadius  = 1;

    int numIterations = 300;
    char filename[256];

    for (int iter = 0; iter < numIterations; iter++) {
        // Add taps influence: for cells within the tap region, add velocity and density.
        printf("step %i", iter);
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                int dx = i - tapCenterX;
                int dy = j - tapCenterY;
                if (dx * dx + dy * dy < tapRadius * tapRadius) {
                    // Add a constant velocity impulse to push fluid rightwards.
                    fluid_add_velocity(&fluid, i, j, 1.0f, 0.0f);
                    // Also add density so the object is visible in the background.
                    fluid_add_density(&fluid, i, j, 100.0f);
                }
            }
        }
        
        printf(".");

        // Step the fluid simulation.
        fluid_step(&fluid);
        
        // Create a filename that includes the iteration number.
        sprintf(filename, "frame_%03d.ppm", iter);
        // Render the current fluid density and overlay particles.
        fluid_renderer_draw(&fluid, filename);
        printf("done\n");
    }

    // Clean up.
    fluid_free(&fluid);
    return 0;
}
