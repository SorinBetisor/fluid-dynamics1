#ifndef VULKAN_SOLVER_H
#define VULKAN_SOLVER_H

#include "linearalg.h"
#include "poisson.h"

// Initialize Vulkan for GPU computing
int init_vulkan_solver(int nx, int ny);

// Clean up Vulkan resources
void cleanup_vulkan_solver();

// Vulkan accelerated Poisson solvers
mtrx poisson_vulkan(mtrx f, double dx, double dy, int itmax, double tol);
mtrx poisson_SOR_vulkan(mtrx f, double dx, double dy, int itmax, double tol, double beta);
mtrx poisson_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid);
mtrx poisson_SOR_vulkan_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid);

#endif // VULKAN_SOLVER_H 