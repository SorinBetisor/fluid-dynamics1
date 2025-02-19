#ifndef FLUID_RENDERER_H
#define FLUID_RENDERER_H

#include "fluidSim.h"

#ifdef __cplusplus
extern "C" {
#endif

// Render the current density field as a grayscale PPM image to the current directory.
void fluid_renderer_draw(Fluid *fluid, const char *filename);

#ifdef __cplusplus
}
#endif

#endif // FLUID_RENDERER_H
