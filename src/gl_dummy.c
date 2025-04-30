#include "gl_solver.h"
#include "poisson.h"

int init_gl_solver(int nx, int ny) { 
    return 0; 
}

void cleanup_gl_solver() {}

mtrx poisson_gpu(mtrx f, double dx, double dy, int itmax, double tol) { 
    return poisson(f, dx, dy, itmax, tol); 
}

mtrx poisson_SOR_gpu(mtrx f, double dx, double dy, int itmax, double tol, double beta) { 
    return poisson_SOR(f, dx, dy, itmax, tol, beta); 
}

mtrx poisson_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid) { 
    return poisson_with_object(f, dx, dy, itmax, tol, grid); 
}

mtrx poisson_SOR_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid) { 
    return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid); 
} 