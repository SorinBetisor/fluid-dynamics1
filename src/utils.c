#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
#include "linearalg.h"
#include "utils.h"

double randdouble(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void printvtk(mtrx A, char *title, const char *output_dir)
{
    int i, j;
    char c[320];
    static int count = 0;
    char name[256];
    char full_path[512];
    FILE *pf;

    if (A.M == NULL)
    {
        printf("\n** Error: Aborting program **\n");
        exit(1);
    }
    if ((A.m < 1) || (A.n < 1))
    {
        printf("\n** Error: Invalid parameter **\n");
        exit(1);
    }

    // Create the full output directory path
    snprintf(full_path, sizeof(full_path), "./output/%s", output_dir);
    
    // Create the directory if it doesn't exist (recursive creation)
    char temp_path[512];
    strcpy(temp_path, "./output");
    mkdir(temp_path, 0755);  // Create output directory first
    mkdir(full_path, 0755);  // Then create the subdirectory
    
    snprintf(name, sizeof(name), "%s/%s-1-%d.vtk", full_path, title, count);

    if ((pf = fopen(name, "a")) == NULL)
    {
        printf("\nError while opening file: %s\n", name);
        exit(1);
    }

    printf("%s\n", name);

    fprintf(pf, "# vtk DataFile Version 2.0\n"); // vtk file headers
    fprintf(pf, "test\n");
    fprintf(pf, "ASCII\n");
    fprintf(pf, "DATASET STRUCTURED_POINTS\n");
    fprintf(pf, "DIMENSIONS %d %d 1\n", A.m, A.n);
    fprintf(pf, "ORIGIN 0 0 0\n");
    fprintf(pf, "SPACING 1 1 1\n");
    fprintf(pf, "POINT_DATA %d\n", A.m * A.n);
    fprintf(pf, "SCALARS values float\n");
    fprintf(pf, "LOOKUP_TABLE default");

    for (i = 0; i < A.m; i++)
    {
        fprintf(pf, "\n");
        for (j = 0; j < A.n; j++)
        {
            if ((j == 0))
            {
                sprintf(c, "%.6lf", A.M[i][j]);
                fprintf(pf, "%s", c);
            }
            else
            {
                sprintf(c, " %.6lf", A.M[i][j]);
                fprintf(pf, "%s", c);
            }
        }
    }
    fclose(pf);
    count++;
}