#ifndef FLUID_RENDERER_H
#define FLUID_RENDERER_H

#include "fluidSim.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the OpenGL renderer
int fluid_renderer_init(int width, int height);

// Cleanup the OpenGL renderer resources
void fluid_renderer_cleanup(void);

// Draw a frame with the current fluid state
void fluid_renderer_draw_frame(Fluid *fluid);

// The original PPM writer function (kept for compatibility)
void fluid_renderer_draw(Fluid *fluid, const char *filename);

#ifdef __cplusplus
}
#endif

#endif // FLUID_RENDERER_H
