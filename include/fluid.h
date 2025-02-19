#ifndef FLUID_H
#define FLUID_H

#include "fluidSim.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the fluid simulation with grid size and physical parameters.
void fluid_init(Fluid *fluid, int gridSize, float diffusion, float viscosity, int pressureIterations, float dt);

// Free the fluid simulation’s allocated memory.
void fluid_free(Fluid *fluid);

// Step the simulation forward one time step.
void fluid_step(Fluid *fluid);

// Add a specified amount of density at grid cell (x,y).
void fluid_add_density(Fluid *fluid, int x, int y, float amount);

// Add velocity (amountX,amountY) at grid cell (x,y).
void fluid_add_velocity(Fluid *fluid, int x, int y, float amountX, float amountY);

#ifdef __cplusplus
}
#endif

#endif // FLUID_H
