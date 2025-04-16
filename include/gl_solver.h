// OpenGL GPU accelerated solver

#ifndef GL_SOLVER_H_INCLUDED
#define GL_SOLVER_H_INCLUDED

#include "linearalg.h"
#include "poisson.h"

// Initialize OpenGL for GPU computing
int init_gl_solver(int nx, int ny);

// Clean up OpenGL resources
void cleanup_gl_solver();

// GPU accelerated Poisson solver using shaders
mtrx poisson_gpu(mtrx f, double dx, double dy, int itmax, double tol);

// GPU accelerated Poisson SOR solver using shaders
mtrx poisson_SOR_gpu(mtrx f, double dx, double dy, int itmax, double tol, double beta);

// GPU accelerated Poisson solver with objects
mtrx poisson_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid);

// GPU accelerated Poisson SOR solver with objects
mtrx poisson_SOR_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid);

#endif // GL_SOLVER_H_INCLUDED 