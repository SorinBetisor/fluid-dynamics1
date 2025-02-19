#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cglm/cglm.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int gridSize;

        float diffusion;
        float viscosity;

        int pressureIterations;
        float dt;

        int size; // gridSize * gridSize, with index mapping: index = i + j * gridSize.
        
        float *density;  // current density
        float *density0; // previous density 
        
        float *u;        // x–component of velocity
        float *v;        // y–component of velocity

        float *u0;       // temporary velocity buffers
        float *v0;
    } Fluid;

    typedef struct
    {
        vec2 position; // (x,y) in grid coordinates
        vec2 velocity;
    } FluidParticle;

#ifdef __cplusplus
}
#endif

#endif // FLUID_SIM_H
