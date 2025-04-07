#include "object.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Determines whether the point (px, py) is inside the polygon defined by
// the vertices in polyX and polyY (with count vertices).
// Uses the ray-casting (even-odd rule) algorithm.
static int point_in_polygon(float px, float py, int count, float *polyX, float *polyY)
{
     int inside = 0;
     int i, j;
     for (i = 0, j = count - 1; i < count; j = i++)
     {
          if (((polyY[i] > py) != (polyY[j] > py)) &&
              (px < (polyX[j] - polyX[i]) * (py - polyY[i]) / (polyY[j] - polyY[i]) + polyX[i]))
          {
               inside = !inside;
          }
     }
     return inside;
}

#include "object.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//---------------------------------------------------------------------
// Helper: ib_delta
//
// A simple one-dimensional delta function with support [-1,1].
// Here we use a hat function: delta(r) = 1 - |r| for |r|<1, and 0 otherwise.
static float ib_delta(float r)
{
     r = fabsf(r);
     return (r < 1.0f) ? (1.0f - r) : 0.0f;
}

//---------------------------------------------------------------------
// object_apply_ib
//
// Applies an immersed-boundary penalty force to enforce a no-slip condition
// along the obstacle boundary. For each boundary marker (each vertex in the
// object's polygon), the function interpolates the local fluid velocity, computes
// a force proportional to the discrepancy from the desired zero velocity, and then
// spreads that force onto the Eulerian fluid grid.
//
// The penalty parameter 'alpha' controls how strongly the no-slip condition is enforced.
void object_apply_ib(Fluid *fluid, Object *obj)
{
     int N = fluid->gridSize;
     float dt = fluid->dt;
     // Penalty stiffness (tune as needed; higher values enforce the condition more strongly)
     float alpha = 100.0f;

     // Create temporary force arrays for the x and y momentum contributions.
     float *Fx = (float *)calloc(fluid->size, sizeof(float));
     float *Fy = (float *)calloc(fluid->size, sizeof(float));
     if (!Fx || !Fy)
     {
          fprintf(stderr, "Error allocating force arrays.\n");
          free(Fx);
          free(Fy);
          return;
     }

     // Loop over each boundary marker (vertex in the obstacle polygon)
     for (int m = 0; m < obj->count; m++)
     {
          // Marker location in grid coordinates (assumed to be the mapped values)
          float X = obj->x[m];
          float Y = obj->y[m];

          // Interpolate the fluid velocity at the marker location using a simple bilinear scheme.
          // The grid cell centers are assumed to be at (i+0.5, j+0.5).
          float u_interp = 0.0f;
          float v_interp = 0.0f;
          // Determine a local support region (here we use 2 cells in each direction)
          int i_min = (int)floorf(X - 1);
          int i_max = (int)ceilf(X + 1);
          int j_min = (int)floorf(Y - 1);
          int j_max = (int)ceilf(Y + 1);
          for (int i = i_min; i <= i_max; i++)
          {
               for (int j = j_min; j <= j_max; j++)
               {
                    // Check that the indices are within the grid.
                    if (i < 0 || i >= N || j < 0 || j >= N)
                         continue;
                    // Compute the distance from the marker to the cell center.
                    float dx = X - (i + 0.5f);
                    float dy = Y - (j + 0.5f);
                    // Compute the weight using the product of one-dimensional delta kernels.
                    float weight = ib_delta(dx) * ib_delta(dy);
                    int index = IX(i, j, N);
                    u_interp += fluid->u[index] * weight;
                    v_interp += fluid->v[index] * weight;
               }
          }

          // The desired velocity at the obstacle boundary is zero (no-slip condition).
          // Compute the penalty force at the marker.
          float fu = alpha * (0.0f - u_interp);
          float fv = alpha * (0.0f - v_interp);

          // Spread the marker force back onto the Eulerian grid.
          for (int i = i_min; i <= i_max; i++)
          {
               for (int j = j_min; j <= j_max; j++)
               {
                    if (i < 0 || i >= N || j < 0 || j >= N)
                         continue;
                    float dx = X - (i + 0.5f);
                    float dy = Y - (j + 0.5f);
                    float weight = ib_delta(dx) * ib_delta(dy);
                    int index = IX(i, j, N);
                    Fx[index] += fu * weight;
                    Fy[index] += fv * weight;
               }
          }
     }

     // Update the fluid velocities explicitly using the computed force field.
     // (In a more advanced implementation this force could be integrated into the solver.)
     for (int i = 0; i < fluid->size; i++)
     {
          fluid->u[i] += dt * Fx[i];
          fluid->v[i] += dt * Fy[i];
     }

     free(Fx);
     free(Fy);
     // Finally, zero out the velocity in any grid cell whose center is inside the object.
     // The object's mask was pre-computed during object_load.
     for (int j = 0; j < N; j++)
     {
          for (int i = 0; i < N; i++)
          {
               int idx = IX(i, j, N);
               if (obj->mask[idx])
               {
                    fluid->u[idx] = 0.0f;
                    fluid->v[idx] = 0.0f;
               }
          }
     }
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
Object *object_load(const char *filename, int gridSize)
{
     FILE *file = fopen(filename, "r");
     if (!file)
     {
          fprintf(stderr, "Error opening file %s\n", filename);
          return NULL;
     }

     Object *obj = (Object *)malloc(sizeof(Object));
     if (!obj)
     {
          fclose(file);
          return NULL;
     }

     // --- Read header ---
     char buffer[256];
     if (fgets(buffer, sizeof(buffer), file) == NULL)
     {
          fprintf(stderr, "Error reading header from file %s\n", filename);
          free(obj);
          fclose(file);
          return NULL;
     }
     // Remove trailing newline.
     size_t len = strlen(buffer);
     if (len > 0 && buffer[len - 1] == '\n')
          buffer[len - 1] = '\0';
     obj->name = strdup(buffer);

     // --- Read coordinate pairs ---
     int capacity = 100;
     int count = 0;
     float *tempX = (float *)malloc(capacity * sizeof(float));
     float *tempY = (float *)malloc(capacity * sizeof(float));
     if (!tempX || !tempY)
     {
          fprintf(stderr, "Memory allocation error\n");
          free(obj->name);
          free(obj);
          fclose(file);
          return NULL;
     }

     // Read each subsequent line (skipping blank lines)
     while (fgets(buffer, sizeof(buffer), file))
     {
          if (buffer[0] == '\n' || buffer[0] == '\r')
               continue;
          float a, b;
          if (sscanf(buffer, "%f %f", &a, &b) == 2)
          {
               if (count >= capacity)
               {
                    capacity *= 2;
                    tempX = (float *)realloc(tempX, capacity * sizeof(float));
                    tempY = (float *)realloc(tempY, capacity * sizeof(float));
                    if (!tempX || !tempY)
                    {
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
     if (count > 0)
     {
          if (fabs(tempX[0] - tempX[count - 1]) > 1e-6 ||
              fabs(tempY[0] - tempY[count - 1]) > 1e-6)
          {
               if (count >= capacity)
               {
                    capacity++;
                    tempX = (float *)realloc(tempX, capacity * sizeof(float));
                    tempY = (float *)realloc(tempY, capacity * sizeof(float));
               }
               tempX[count] = tempX[0];
               tempY[count] = tempY[0];
               count++;
          }
     }

     // Allocate and copy the coordinates into the object.
     obj->count = count;
     obj->x = (float *)malloc(count * sizeof(float));
     obj->y = (float *)malloc(count * sizeof(float));
     if (!obj->x || !obj->y)
     {
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
     float offset_x = gridSize * 0.25f; // Position horizontally (centered in left half).
     float scale_y = gridSize * 0.5f;   // Scale vertical dimension (adjust as needed).
     float offset_y = gridSize * 0.5f;  // Center vertically.
     for (int i = 0; i < obj->count; i++)
     {
          obj->x[i] = offset_x + scale_x * obj->x[i];
          // Flip y if needed to match your simulation's coordinate system.
          obj->y[i] = offset_y - scale_y * obj->y[i];
     }

     // The mask is an integer array of size gridSize*gridSize.
     // A cell is set to 1 if its center is inside the polygon.
     obj->mask = (int *)malloc(gridSize * gridSize * sizeof(int));
     if (!obj->mask)
     {
          free(obj->x);
          free(obj->y);
          free(obj->name);
          free(obj);
          return NULL;
     }
     for (int j = 0; j < gridSize; j++)
     {
          for (int i = 0; i < gridSize; i++)
          {
               float cx = i + 0.5f;
               float cy = j + 0.5f;
               int inside = point_in_polygon(cx, cy, obj->count, obj->x, obj->y);
               obj->mask[i + j * gridSize] = inside;
          }
     }

     return obj;
}

// Frees all memory allocated for the object.
void object_free(Object *obj)
{
     if (!obj)
          return;
     if (obj->name)
          free(obj->name);
     if (obj->x)
          free(obj->x);
     if (obj->y)
          free(obj->y);
     if (obj->mask)
          free(obj->mask);
     free(obj);
}

// Called after fluid_step() to enforce the obstacle boundary condition.
// For grid cells inside the obstacle (mask == 1), we zero the velocity and density.
void object_apply(Fluid *fluid, Object *obj)
{
     int N = fluid->gridSize;
     int i, j;
     
#pragma omp parallel for private(i, j)
     for (j = 0; j < N; j++)
     {
          for (i = 0; i < N; i++)
          {
               if (obj->mask[i + j * N])
               {
                    int index = IX(i, j, N);
                    fluid->u[index] = 0;
                    fluid->v[index] = 0;
                    fluid->density[index] = 0;
               }
          }
     }
}
