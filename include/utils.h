// Utilities library

#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "linearalg.h"

double randdouble(double min, double max); // generates random double number between min and max
void printvtk(mtrx A, char *title, const char *output_dir);        // prints matrix A to a vtk file in specified directory

#endif