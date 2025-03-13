#ifndef OBJECT_H
#define OBJECT_H

#include "fluid.h"  // so we can refer to Fluid and the IX() macro

// Structure to represent an obstacle read from a file.
// The airfoil is stored as a polygon (its outline) and an associated grid mask.
typedef struct {
    int count;      // Number of points in the polygon
    float *x;       // X coordinates of the polygon vertices (mapped to grid coordinates)
    float *y;       // Y coordinates of the polygon vertices (mapped to grid coordinates)
    char *name;     // Name of the airfoil (from the first line of the .dat file)
    int *mask;      // A grid mask array (size = gridSize*gridSize). 1 = obstacle, 0 = fluid.
} Object;

// Reads an object from a Selig .dat file.
// The file is expected to have a header line (name) and then one coordinate pair per line.
// The coordinates are assumed to be normalized (for example, chord from 0 to 1).
// The function maps these coordinates onto the simulation grid (of size gridSize).
Object* object_load(const char *filename, int gridSize);

// Free all allocated memory in the object.
void object_free(Object *obj);

// After the fluid simulation step, call this function to apply the obstacle boundary:
// For grid cells that fall inside the object, it zeroes the velocity 
// so that the obstacle remains “solid.”
void object_apply(Fluid *fluid, Object *obj);

void object_apply_ib(Fluid *fluid, Object *obj);

#endif // OBJECT_H
