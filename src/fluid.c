/*
 * This file implements a 2D fluid simulation based on the Navier–Stokes equations.
 * The simulation uses the "Stable Fluids" method (Jos Stam, 1999), which is widely
 * used for real-time fluid simulation in graphics.
 *
 * The simulation models three key physical processes:
 *   1. Diffusion: This represents how quantities (e.g., velocity and density) spread
 *      over time due to random molecular motion. In the simulation, diffusion is
 *      computed implicitly via an iterative (Gauss–Seidel) solver.
 *
 *   2. Advection: This is the transport of quantities by the fluid's own velocity.
 *      It uses a semi-Lagrangian method, where each grid cell "traces" backward
 *      along the velocity field to sample the advected quantity.
 *
 *   3. Projection: This step enforces the incompressibility of the fluid by
 *      removing divergence from the velocity field. It is equivalent to solving
 *      a Poisson equation for the pressure and then subtracting the gradient
 *      to obtain a divergence-free field.
 *
 * Typical parameter values (simulation units – note that these may be scaled):
 *
 *   For water:
 *     - Diffusion: 0.0001 or lower (water has very low diffusivity)
 *     - Viscosity: 0.001 (water's kinematic viscosity is roughly 1e-6 m^2/s in reality,
 *                  but simulation units are often scaled for visual effect)
 *
 *   For air:
 *     - Diffusion: 0.00001 or lower (air has even lower diffusivity for many scalar fields)
 *     - Viscosity: 0.000018 (air's kinematic viscosity is around 1.5e-5 m^2/s in real terms)
 */

#include "fluid.h"


// Internal helper function prototypes.
// Each helper function represents one of the core physical steps (diffusion, advection,
// projection) or sets the boundary conditions.
static void set_bnd(int b, float *x, Fluid *fluid);
static void diffuse(int b, float *x, float *x0, float diff, float dt, Fluid *fluid);
static void advect(int b, float *d, float *d0, float *u, float *v, float dt, Fluid *fluid);
static void project(float *u, float *v, float *u0, float *v0, Fluid *fluid);

// Initialize the fluid struct with grid size, physical parameters (diffusion, viscosity),
// number of iterations for the pressure solver (affecting the projection step), and timestep.
void fluid_init(Fluid *fluid, int gridSize, float diffusion, float viscosity, int pressureIterations, float dt) {
    fluid->gridSize = gridSize;
    fluid->diffusion = diffusion;
    fluid->viscosity = viscosity;
    fluid->pressureIterations = pressureIterations;
    fluid->dt = dt;
    fluid->size = gridSize * gridSize;

    // Allocate memory for the density and velocity fields.
    fluid->density  = (float*)calloc(fluid->size, sizeof(float));
    fluid->density0 = (float*)calloc(fluid->size, sizeof(float));
    fluid->u        = (float*)calloc(fluid->size, sizeof(float));
    fluid->v        = (float*)calloc(fluid->size, sizeof(float));
    fluid->u0       = (float*)calloc(fluid->size, sizeof(float));
    fluid->v0       = (float*)calloc(fluid->size, sizeof(float));
}

// Free the memory allocated for the fluid fields.
void fluid_free(Fluid *fluid) {
    free(fluid->density);
    free(fluid->density0);
    free(fluid->u);
    free(fluid->v);
    free(fluid->u0);
    free(fluid->v0);
}

// Add a density value at a particular grid cell.
// This simulates adding a scalar quantity (like smoke density or dye concentration).
void fluid_add_density(Fluid *fluid, int x, int y, float amount) {
    int N = fluid->gridSize;
    int index = IX(x, y, N);
    fluid->density[index] += amount;
}

// Add velocity components to a cell.
// This is used to simulate an external force or injection of momentum.
void fluid_add_velocity(Fluid *fluid, int x, int y, float amountX, float amountY) {
    int N = fluid->gridSize;
    int index = IX(x, y, N);
    fluid->u[index] += amountX;
    fluid->v[index] += amountY;
}

// Perform one timestep of the fluid simulation.
// The sequence of operations corresponds to the physical evolution: velocity diffusion,
// advection, projection (to enforce incompressibility), and the similar process for the density.
void fluid_step(Fluid *fluid) {
    int N = fluid->gridSize;
    float dt = fluid->dt;

    // Save the current velocity fields into temporary buffers.
    for (int i = 0; i < fluid->size; i++) {
        fluid->u0[i] = fluid->u[i];
        fluid->v0[i] = fluid->v[i];
    }

    // Step 1: Diffuse the velocity fields.
    // Diffusion represents the spreading (or smoothing) of momentum due to viscosity.
    diffuse(1, fluid->u, fluid->u0, fluid->viscosity, dt, fluid);
    diffuse(2, fluid->v, fluid->v0, fluid->viscosity, dt, fluid);
    
    // Step 2: Project the velocity field to make it divergence-free.
    // This enforces the incompressibility condition (constant density flow).
    project(fluid->u, fluid->v, fluid->u0, fluid->v0, fluid);

    // Step 3: Advect (move) the velocity fields along themselves.
    // Advection transports the velocities based on the current velocity field.
    for (int i = 0; i < fluid->size; i++) {
        fluid->u0[i] = fluid->u[i];
        fluid->v0[i] = fluid->v[i];
    }
    advect(1, fluid->u, fluid->u0, fluid->u0, fluid->v0, dt, fluid);
    advect(2, fluid->v, fluid->v0, fluid->u0, fluid->v0, dt, fluid);
    
    // // Re-project the velocity field after advection.
    // project(fluid->u, fluid->v, fluid->u0, fluid->v0, fluid);

    // Step 4: Diffuse and advect the density field.
    // This moves the scalar quantity with the fluid flow.
    for (int i = 0; i < fluid->size; i++) {
        fluid->density0[i] = fluid->density[i];
    }
    diffuse(0, fluid->density, fluid->density0, fluid->diffusion, dt, fluid);
    advect(0, fluid->density, fluid->density0, fluid->u, fluid->v, dt, fluid);
}

// set_bnd sets the boundary conditions on the simulation grid.
// Depending on the value of b, the function treats the boundaries differently:
//   b == 1: horizontal velocity component (u)
//   b == 2: vertical velocity component (v)
//   b == 0: scalar fields like density.
//
// The boundary condition here is a simple "reflection" condition
// (mirroring the value or inverting the sign for velocity components)
// to simulate a wall.
static void set_bnd(int b, float *x, Fluid *fluid) {
    int N = fluid->gridSize;
    for (int i = 1; i < N - 1; i++) {
        // For left and right boundaries.
    //     x[IX(0, i, N)]       = (b == 1) ? -x[IX(1, i, N)] : x[IX(1, i, N)];
    //     x[IX(N - 1, i, N)]   = (b == 1) ? -x[IX(N - 2, i, N)] : x[IX(N - 2, i, N)];
        // For top and bottom boundaries.
        x[IX(i, 0, N)]       = (b == 2) ? -x[IX(i, 1, N)] : x[IX(i, 1, N)];
        x[IX(i, N - 1, N)]   = (b == 2) ? -x[IX(i, N - 2, N)] : x[IX(i, N - 2, N)];
    }
    // Set the four corner values as the average of the two neighboring edges.
    x[IX(0, 0, N)]             = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    x[IX(0, N - 1, N)]         = 0.5f * (x[IX(1, N - 1, N)] + x[IX(0, N - 2, N)]);
    x[IX(N - 1, 0, N)]         = 0.5f * (x[IX(N - 2, 0, N)] + x[IX(N - 1, 1, N)]);
    x[IX(N - 1, N - 1, N)]     = 0.5f * (x[IX(N - 2, N - 1, N)] + x[IX(N - 1, N - 2, N)]);
}

// diffuse simulates the diffusion process on the field x.
// The diffusion equation is solved implicitly (using Gauss–Seidel iterations)
// to ensure stability even for relatively large timesteps.
// The parameter 'a' encapsulates dt, diffusive constant, and grid spacing.
static void diffuse(int b, float *x, float *x0, float diff, float dt, Fluid *fluid) {
    int N = fluid->gridSize;
    float a = dt * diff * (N - 2) * (N - 2);
    // Iterative solver: run a fixed number of iterations to approximate the solution.
    for (int k = 0; k < fluid->pressureIterations; k++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                x[IX(i, j, N)] = (x0[IX(i, j, N)] +
                    a * (x[IX(i - 1, j, N)] +
                         x[IX(i + 1, j, N)] +
                         x[IX(i, j - 1, N)] +
                         x[IX(i, j + 1, N)])) / (1 + 4 * a);
            }
        }
        // After each iteration, enforce the boundary conditions.
        set_bnd(b, x, fluid);
    }
}

// advect transports the quantity in d0 (density, u, or v) along the velocity field (u, v).
// The semi-Lagrangian method works by tracing the flow backward in time to find
// where a fluid element originated and then sampling from d0 using bilinear interpolation.
static void advect(int b, float *d, float *d0, float *u, float *v, float dt, Fluid *fluid) {
    int N = fluid->gridSize;
    float dt0 = dt * (N - 2);
    int i, j;
    
    #pragma omp parallel for private(i, j)
    for (i = 1; i < N - 1; i++) {
        for (j = 1; j < N - 1; j++) {
            // Compute the previous position of the fluid element at (i,j).
            float x = i - dt0 * u[IX(i, j, N)];
            float y = j - dt0 * v[IX(i, j, N)];
            // Clamp the positions to lie within the grid boundaries.
            if (x < 0.5f) x = 0.5f;
            if (x > N - 1.5f) x = N - 1.5f;
            if (y < 0.5f) y = 0.5f;
            if (y > N - 1.5f) y = N - 1.5f;
            int i0 = (int)x;
            int i1 = i0 + 1;
            int j0 = (int)y;
            int j1 = j0 + 1;
            // Compute interpolation weights.
            float s1 = x - i0;
            float s0 = 1 - s1;
            float t1 = y - j0;
            float t0 = 1 - t1;
            // Perform bilinear interpolation to compute the advected value.
            d[IX(i, j, N)] =
                s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
        }
    }
    // Reset the boundaries after advection.
    set_bnd(b, d, fluid);
}

// project enforces incompressibility by adjusting the velocity field so that it has zero divergence.
// It computes a pressure-like field (p) by solving a Poisson equation using iterative relaxation,
// and then subtracts the gradient of the pressure from the velocity field.
static void project(float *u, float *v, float *u0, float *v0, Fluid *fluid) {
    int N = fluid->gridSize;
    // Allocate temporary arrays for pressure (p) and divergence (div).
    float *p = (float*)calloc(fluid->size, sizeof(float));
    float *div = (float*)calloc(fluid->size, sizeof(float));

    // Compute the divergence of the velocity field.
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            div[IX(i, j, N)] = -0.5f * (
                u[IX(i + 1, j, N)] - u[IX(i - 1, j, N)] +
                v[IX(i, j + 1, N)] - v[IX(i, j - 1, N)]
            ) / N;
            p[IX(i, j, N)] = 0;
        }
    }
    // Set boundary conditions for divergence and pressure.
    set_bnd(0, div, fluid);
    set_bnd(0, p, fluid);

    // Iteratively solve for the pressure p.
    for (int k = 0; k < fluid->pressureIterations; k++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                p[IX(i, j, N)] = (div[IX(i, j, N)] +
                    p[IX(i - 1, j, N)] + p[IX(i + 1, j, N)] +
                    p[IX(i, j - 1, N)] + p[IX(i, j + 1, N)]
                ) / 4;
            }
        }
        set_bnd(0, p, fluid);
    }

    // Subtract the gradient of the pressure from the velocity field to obtain a divergence-free field.
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            u[IX(i, j, N)] -= 0.5f * N * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]);
            v[IX(i, j, N)] -= 0.5f * N * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]);
        }
    }
    // Reinforce boundary conditions for the velocity fields.
    set_bnd(1, u, fluid);
    set_bnd(2, v, fluid);

    // Free the temporary arrays.
    free(p);
    free(div);
}

