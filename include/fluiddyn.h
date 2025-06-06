// Fluid dynamics library

#ifndef FLUIDDYN_H_INCLUDED
#define FLUIDDYN_H_INCLUDED

#include "linearalg.h"

void euler(mtrx w, mtrx dwdx, mtrx dwdy, mtrx d2wdx2, mtrx d2wdy2, mtrx u, mtrx v, double Re, double dt); // Euler time-advancement
mtrx continuity(mtrx dudx, mtrx dvdy);                                                                    // computes continuity equation
mtrx vorticity(mtrx dvdx, mtrx dudy);                                                                     // computes vorticity
mtrx pressure(void);                                                                                      // computes pressure

// OpenMP configuration
void set_fluiddyn_openmp_config(int enabled);

#endif // FLUIDDYN_H_INCLUDED