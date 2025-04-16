// Poisson solver library

#ifndef POISSON_H_INCLUDED
#define POISSON_H_INCLUDED

#include "linearalg.h"

#define PI 3.14159265359

// We'll include the struct directly in main.c and poisson.c
typedef struct {
    int is_solid;  // 1 if cell is solid, 0 if fluid
} cell_properties;

double error(mtrx u1, mtrx u2);
mtrx poisson(mtrx f, double dx, double dy, int itmax, double tol);
mtrx poisson_SOR(mtrx f, double dx, double dy, int itmax, double tol, double beta);

// Add new function prototypes for solvers that handle objects
mtrx poisson_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid);
mtrx poisson_SOR_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid);

#endif // POISSON_H_INCLUDED