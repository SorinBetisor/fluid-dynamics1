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

int create_directory(const char *path) {
    struct stat st = {0};
    
    if (stat(path, &st) == -1) {
        #ifdef _WIN32
        if (_mkdir(path) != 0) {
        #else
        if (mkdir(path, 0755) != 0) {
        #endif
            return -1; // Failed to create directory
        }
    }
    return 0; // Directory exists or was created successfully
}

void printvtk(mtrx A, char *title, const char *output_dir)
{
    int i, j;
    char c[320];
    static int count = 0;
    char name[512];
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

    // Create the output directory if it doesn't exist
    if (create_directory(output_dir) != 0) {
        printf("Warning: Could not create output directory %s\n", output_dir);
    }

    // Create the full file path
    snprintf(name, sizeof(name), "%s/%s-1-%d.vtk", output_dir, title, count);

    if ((pf = fopen(name, "a")) == NULL)
    {
        printf("\nError while opening file %s\n", name);
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