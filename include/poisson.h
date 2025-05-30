// Poisson solver library

#ifndef POISSON_H_INCLUDED
#define POISSON_H_INCLUDED

#include "linearalg.h"
#include <stdio.h>

#define PI 3.14159265359

double error(mtrx u1, mtrx u2);
mtrx poisson(mtrx f, double dx, double dy, int itmax, double tol);
mtrx poisson_SOR(mtrx f, double dx, double dy, int itmax, double tol, double beta);

// New versions with logging support
mtrx poisson_log(mtrx f, double dx, double dy, int itmax, double tol, FILE *log_file);
mtrx poisson_SOR_log(mtrx f, double dx, double dy, int itmax, double tol, double beta, FILE *log_file);

#endif // POISSON_H_INCLUDED