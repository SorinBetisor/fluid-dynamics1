#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include "linearalg.h"
#include "poisson.h"
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

// Global OpenMP configuration (matches linearalg.c pattern)
static int g_openmp_enabled = 0;

// Function to set OpenMP configuration
void set_poisson_openmp_config(int enabled) {
#ifdef OPENMP_ENABLED
    g_openmp_enabled = enabled;
#else
    g_openmp_enabled = 0; // Force disable if not compiled with OpenMP
#endif
}

// Helper function for logging (matches main.c pattern)
static void poisson_log_message(FILE *log_file, const char *format, ...) {
    if (log_file != NULL) {
        va_list args;
        va_start(args, format);
        vfprintf(log_file, format, args);
        va_end(args);
        fflush(log_file);
    }
}

double error(mtrx u1, mtrx u2)
{
    double e = 0;
    int i, j;

    if (g_openmp_enabled) {
#ifdef OPENMP_ENABLED
        #pragma omp parallel for reduction(+:e) collapse(2) schedule(static)
#endif
        for (i = 0; i < u1.m; i++)
        {
            for (j = 0; j < u1.n; j++)
            {
                e += sqrt(pow(u2.M[i][j] - u1.M[i][j], 2));
            }
        }
    } else {
        for (i = 0; i < u1.m; i++)
        {
            for (j = 0; j < u1.n; j++)
            {
                e += sqrt(pow(u2.M[i][j] - u1.M[i][j], 2));
            }
        }
    }
    return e;
}

mtrx poisson(mtrx f, double dx, double dy, int itmax, double tol)
{
    int i, j, k, nx, ny;
    double e;
    //double u1max, u2max;
    mtrx u, u0;
    u = initm(f.m, f.n);
    u0 = initm(f.m, f.n);
    nx = f.m;
    ny = f.n;

    for (k = 0; k < itmax; k++)
    {
        mtrxcpy(u0, u);
        if (g_openmp_enabled) {
#ifdef OPENMP_ENABLED
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 128 && ny > 128)
#endif
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy));
                }
            }
        } else {
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy));
                }
            }
        }
        e = error(u, u0);
        if (e < tol)
        {
            printf("Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            u0.M = freem(u0);
            return u;
        }
    }
    printf("Error: maximum number of iterations achieved for Poisson equation.\n");

    u.M = freem(u);
    u0.M = freem(u0);
    exit(1);
}

mtrx poisson_SOR(mtrx f, double dx, double dy, int itmax, double tol, double beta)
{
    int i, j, k, nx, ny;
    double e;
    //double u1max, u2max;
    mtrx u, u0;
    u = initm(f.m, f.n);
    u0 = initm(f.m, f.n);
    nx = f.m;
    ny = f.n;

    for (k = 0; k < itmax; k++)
    {
        mtrxcpy(u0, u);
        if (g_openmp_enabled) {
#ifdef OPENMP_ENABLED
            // Red-black coloring for SOR parallelization
            // Red points: (i+j) is even
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 128 && ny > 128)
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    if ((i + j) % 2 == 0) { // Red points
                        u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                    }
                }
            }
            // Black points: (i+j) is odd
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 128 && ny > 128)
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    if ((i + j) % 2 == 1) { // Black points
                        u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                    }
                }
            }
#endif
        } else {
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                }
            }
        }
        e = error(u, u0);
        if (e < tol)
        {
            printf("Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            u0.M = freem(u0);
            return u;
        }
    }
    printf("Error: maximum number of iterations achieved for Poisson equation.\n");

    u.M = freem(u);
    u0.M = freem(u0);
    exit(1);
}

// New versions with logging support
mtrx poisson_log(mtrx f, double dx, double dy, int itmax, double tol, FILE *log_file)
{
    int i, j, k, nx, ny;
    double e;
    mtrx u, u0;
    u = initm(f.m, f.n);
    u0 = initm(f.m, f.n);
    nx = f.m;
    ny = f.n;

    for (k = 0; k < itmax; k++)
    {
        mtrxcpy(u0, u);
        if (g_openmp_enabled) {
#ifdef OPENMP_ENABLED
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 64 && ny > 64)
#endif
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy));
                }
            }
        } else {
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy));
                }
            }
        }
        e = error(u, u0);
        if (e < tol)
        {
            poisson_log_message(log_file, "Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            u0.M = freem(u0);
            return u;
        }
    }
    poisson_log_message(log_file, "Error: maximum number of iterations achieved for Poisson equation.\n");

    u.M = freem(u);
    u0.M = freem(u0);
    exit(1);
}

mtrx poisson_SOR_log(mtrx f, double dx, double dy, int itmax, double tol, double beta, FILE *log_file)
{
    int i, j, k, nx, ny;
    double e;
    mtrx u, u0;
    u = initm(f.m, f.n);
    u0 = initm(f.m, f.n);
    nx = f.m;
    ny = f.n;

    for (k = 0; k < itmax; k++)
    {
        mtrxcpy(u0, u);
        if (g_openmp_enabled) {
#ifdef OPENMP_ENABLED
            // Red-black coloring for SOR parallelization
            // Red points: (i+j) is even
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 128 && ny > 128)
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    if ((i + j) % 2 == 0) { // Red points
                        u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                    }
                }
            }
            // Black points: (i+j) is odd
            #pragma omp parallel for collapse(2) schedule(static) if(nx > 128 && ny > 128)
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    if ((i + j) % 2 == 1) { // Black points
                        u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                    }
                }
            }
#endif
        } else {
            for (i = 1; i < nx - 1; i++)
            {
                for (j = 1; j < ny - 1; j++)
                {
                    u.M[i][j] = beta * (dy * dy * (u.M[i + 1][j] + u.M[i - 1][j]) + dx * dx * (u.M[i][j + 1] + u.M[i][j - 1]) - dx * dx * dy * dy * f.M[i][j]) / (2 * (dx * dx + dy * dy)) + (1 - beta) * u0.M[i][j];
                }
            }
        }
        e = error(u, u0);
        if (e < tol)
        {
            poisson_log_message(log_file, "Poisson equation solved with %d iterations - root-sum-of-squares error: %E\n", k, e);
            u0.M = freem(u0);
            return u;
        }
    }
    poisson_log_message(log_file, "Error: maximum number of iterations achieved for Poisson equation.\n");

    u.M = freem(u);
    u0.M = freem(u0);
    exit(1);
}