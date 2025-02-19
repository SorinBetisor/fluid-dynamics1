#include "object.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Determines whether the point (px, py) is inside the polygon defined by
// the vertices in polyX and polyY (with count vertices).
// Uses the ray-casting (even-odd rule) algorithm.
static int point_in_polygon(float px, float py, int count, float *polyX, float *polyY) {
    int inside = 0;
    int i, j;
    for (i = 0, j = count - 1; i < count; j = i++) {
        if (((polyY[i] > py) != (polyY[j] > py)) &&
            (px < (polyX[j] - polyX[i]) * (py - polyY[i]) / (polyY[j] - polyY[i]) + polyX[i])) {
            inside = !inside;
        }
    }
    return inside;
}

// Reads an airfoil file in Selig format. The file is expected to have:
//  - A header line (the airfoil name)
//  - The upper surface coordinates (one pair per line)
//  - Optionally a blank line
//  - The lower surface coordinates (one pair per line)
//
// Because many such files list both surfaces from leading edge to trailing edge,
// we form a closed polygon by taking the upper surface in reverse order (trailing
// edge to leading edge) and then the lower surface in normal order (leading edge to
// trailing edge).
//
// Finally, the normalized coordinates are mapped to simulation grid coordinates.
// Adjust the scaling and offset parameters as needed.
Object* object_load(const char *filename, int gridSize) {
    FILE *file = fopen(filename, "r");
    if (!file) {
         fprintf(stderr, "Error opening file %s\n", filename);
         return NULL;
    }
    
    Object *obj = (Object*)malloc(sizeof(Object));
    if (!obj) {
         fclose(file);
         return NULL;
    }
    
    // --- Read header ---
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), file) == NULL) {
         fprintf(stderr, "Error reading header from file %s\n", filename);
         free(obj);
         fclose(file);
         return NULL;
    }
    // Remove trailing newline.
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len-1] == '\n')
         buffer[len-1] = '\0';
    obj->name = strdup(buffer);
    
    // --- Read coordinate pairs ---
    int capacity = 100;
    int count = 0;
    float *tempX = (float*)malloc(capacity * sizeof(float));
    float *tempY = (float*)malloc(capacity * sizeof(float));
    if (!tempX || !tempY) {
         fprintf(stderr, "Memory allocation error\n");
         free(obj->name);
         free(obj);
         fclose(file);
         return NULL;
    }
    
    // Read each subsequent line (skipping blank lines)
    while (fgets(buffer, sizeof(buffer), file)) {
         if (buffer[0] == '\n' || buffer[0] == '\r')
              continue;
         float a, b;
         if (sscanf(buffer, "%f %f", &a, &b) == 2) {
              if (count >= capacity) {
                   capacity *= 2;
                   tempX = (float*)realloc(tempX, capacity * sizeof(float));
                   tempY = (float*)realloc(tempY, capacity * sizeof(float));
                   if (!tempX || !tempY) {
                        fprintf(stderr, "Memory allocation error\n");
                        fclose(file);
                        free(obj->name);
                        free(obj);
                        return NULL;
                   }
              }
              tempX[count] = a;
              tempY[count] = b;
              count++;
         }
    }
    fclose(file);
    
    // --- Ensure the polygon is closed ---
    if (count > 0) {
         if (fabs(tempX[0] - tempX[count-1]) > 1e-6 ||
             fabs(tempY[0] - tempY[count-1]) > 1e-6) {
             if (count >= capacity) {
                  capacity++;
                  tempX = (float*)realloc(tempX, capacity * sizeof(float));
                  tempY = (float*)realloc(tempY, capacity * sizeof(float));
             }
             tempX[count] = tempX[0];
             tempY[count] = tempY[0];
             count++;
         }
    }
    
    // Allocate and copy the coordinates into the object.
    obj->count = count;
    obj->x = (float*)malloc(count * sizeof(float));
    obj->y = (float*)malloc(count * sizeof(float));
    if (!obj->x || !obj->y) {
         fprintf(stderr, "Memory allocation error\n");
         free(tempX);
         free(tempY);
         free(obj->name);
         free(obj);
         return NULL;
    }
    memcpy(obj->x, tempX, count * sizeof(float));
    memcpy(obj->y, tempY, count * sizeof(float));
    free(tempX);
    free(tempY);
    
    // Adjust these parameters to change the size and position of the obstacle.
    float scale_x = gridSize * 0.5f;   // Chord spans 50% of grid width.
    float offset_x = gridSize * 0.25f;  // Position horizontally (centered in left half).
    float scale_y = gridSize * 0.5f;     // Scale vertical dimension (adjust as needed).
    float offset_y = gridSize * 0.5f;    // Center vertically.
    for (int i = 0; i < obj->count; i++) {
         obj->x[i] = offset_x + scale_x * obj->x[i];
         // Flip y if needed to match your simulation’s coordinate system.
         obj->y[i] = offset_y - scale_y * obj->y[i];
    }
    
    // The mask is an integer array of size gridSize*gridSize.
    // A cell is set to 1 if its center is inside the polygon.
    obj->mask = (int*)malloc(gridSize * gridSize * sizeof(int));
    if (!obj->mask) {
         free(obj->x);
         free(obj->y);
         free(obj->name);
         free(obj);
         return NULL;
    }
    for (int j = 0; j < gridSize; j++) {
         for (int i = 0; i < gridSize; i++) {
              float cx = i + 0.5f;
              float cy = j + 0.5f;
              int inside = point_in_polygon(cx, cy, obj->count, obj->x, obj->y);
              obj->mask[i + j * gridSize] = inside;
         }
    }
    
    return obj;
}

// Frees all memory allocated for the object.
void object_free(Object *obj) {
    if (!obj) return;
    if (obj->name) free(obj->name);
    if (obj->x) free(obj->x);
    if (obj->y) free(obj->y);
    if (obj->mask) free(obj->mask);
    free(obj);
}

// Called after fluid_step() to enforce the obstacle boundary condition.
// For grid cells inside the obstacle (mask == 1), we zero the velocity and density.
void object_apply(Fluid *fluid, Object *obj) {
    int N = fluid->gridSize;
    #pragma omp parallel for collapse(2) 
    for (int j = 0; j < N; j++) {
         for (int i = 0; i < N; i++) {
             if (obj->mask[i + j * N]) {
                 int index = IX(i, j, N);
                 fluid->u[index] = 0;
                 fluid->v[index] = 0;
                 fluid->density[index] = 0;
             }
         }
    }
}
