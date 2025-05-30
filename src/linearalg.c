#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "linearalg.h"
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

// Global variable to control OpenMP usage
static int g_openmp_enabled = 0;

/**
 * @brief Set OpenMP configuration for linear algebra operations
 * 
 * @param enabled 1 to enable OpenMP, 0 to disable
 */
void set_openmp_config(int enabled) {
#ifdef OPENMP_ENABLED
    g_openmp_enabled = enabled;
#else
    g_openmp_enabled = 0; // Force disable if not compiled with OpenMP
#endif
}

void zerosm(mtrx A)
{
    int i, j;
#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = 0;
            }
        }
    } else {
#endif
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = 0;
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
}

double **allocm(int m, int n)
{
    int i;
    double **A;

    if ((m < 1) || (n < 1))
    {
        printf("** Error: invalid parameter **\n");
        exit(1);
    }

    A = (double **)malloc(m * sizeof(double *));

    if (A == NULL)
    {
        printf("** Error: insufficient memory **");
        exit(1);
    }

    for (i = 0; i < m; i++)
    {
        A[i] = (double *)malloc(n * sizeof(double));
        if (A[i] == NULL)
        {
            printf("** Error: insufficient memory **");
            exit(1);
        }
    }
    return (A);
}

double **freem(mtrx A)
{
    int i;
    if (A.M == NULL)
        return (NULL);
    if ((A.m < 1) || (A.n < 1))
    {
        printf("** Error: invalid parameter **\n");
        exit(1);
    }
    for (i = 0; i < A.m; i++)
        free(A.M[i]);
    free(A.M);
    //printf("Liberated successfully\n");
    return (NULL);
}

double **readm(char *filename, int *m, int *n)
{
    int i, j;
    FILE *f;
    double **A;
    f = fopen(filename, "r"); // opens file
    fscanf(f, "%d", m);       // read row size of matrix
    fscanf(f, "%d", n);       // read col size of matrix
    A = allocm(*m, *n);       // allocate memory for matrix
    for (i = 0; i < *m; i++)
    {
        for (j = 0; j < *n; j++)
        {
            fscanf(f, "%lf", &A[i][j]);
        }
    }
    fclose(f);
    return A;
}

void printm(mtrx A)
{
    int i, j;
    printf("\n");
    for (i = 0; i < A.m; i++)
    {
        printf("[");
        for (j = 0; j < A.n; j++)
        {
            if (j == A.n - 1)
            {
                printf(" %.4lf ", A.M[i][j]);
            }
            else
            {
                printf(" %.4lf", A.M[i][j]);
            }
        }
        if (i == A.n - 1)
        {
            printf("]");
        }
        else
        {
            printf("]");
        }
        printf("\n");
    }
}

void zerosv(vec v)
{
    int i;
#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < v.n; i++)
        {
            v.v[i] = 0;
        }
    } else {
#endif
        for (i = 0; i < v.n; i++)
        {
            v.v[i] = 0;
        }
#ifdef OPENMP_ENABLED
    }
#endif
}

double *allocv(int n)
{
    double *v;
    if (n < 1)
    {
        printf("** Error: invalid parameter **\n");
        exit(1);
    }
    v = (double *)malloc(n * sizeof(double));
    if (v == NULL)
    {
        printf("** Error: insufficient memory **");
        exit(1);
    }
    return (v);
}

double *freev(vec v)
{
    if (v.v == NULL)
        return (NULL);
    if (v.n < 1)
    {
        printf("** Error: invalid parameter **\n");
        exit(1);
    }
    free(v.v);
    return (NULL);
}

double *readv(char *filename, int *n)
{
    int i;
    FILE *f;
    double *v;
    f = fopen(filename, "r"); // opens file
    fscanf(f, "%d", n);       // read size of vector
    v = allocv(*n);           // allocate memory for matrix
    for (i = 0; i < *n; i++)
    {
        fscanf(f, "%lf", &v[i]);
    }
    fclose(f);
    return v;
}

void printv(vec v)
{
    int i;
    printf("\n[ ");
    for (i = 0; i < v.n; i++)
    {
        if ((i == 0) || (i == (v.n - 1)))
        {
            printf("%.4lf", v.v[i]);
        }
        else
        {
            printf(" %.4lf ", v.v[i]);
        }
    }
    printf(" ]\n");
}

mtrx mtrxmul(mtrx A, mtrx B)
{
    mtrx C;
    int i, j, k;

    if (A.n != B.m)
    {
        printf("** Error: the first matrix number of columns must be equal to the second matrix number of rows **\n");
        printf("Columns of first matrix: %d\n", A.n);
        printf("Rows of second matrix: %d\n", B.m);
        exit(1);
    }

    C = initm(A.m, B.n);

#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) schedule(static) if(A.m > 64 && B.n > 64)
        for (i = 0; i < C.m; i++)
        {
            for (j = 0; j < C.n; j++)
            {
                double sum = 0.0;  // Use local variable to improve cache performance
                for (k = 0; k < A.n; k++)
                {
                    sum += A.M[i][k] * B.M[k][j];
                }
                C.M[i][j] = sum;
            }
        }
    } else {
#endif
        // Sequential version
        for (i = 0; i < C.m; i++)
        {
            for (j = 0; j < C.n; j++)
            {
                double sum = 0.0;  // Use local variable to improve cache performance
                for (k = 0; k < A.n; k++)
                {
                    sum += A.M[i][k] * B.M[k][j];
                }
                C.M[i][j] = sum;
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
    return C;
}

vec gaussian(mtrx A, vec b)
{
    int i, j, m, n, k;

    // Augmented matrix
    double **a;
    a = allocm(b.n, b.n + 1);
    for (i = 0; i < b.n; i++)
    {
        for (j = 0; j < b.n; j++)
        {
            a[i][j] = A.M[i][j];
        }
    }
    for (i = 0; i < b.n; i++)
    {
        a[i][b.n] = b.v[i];
    }
    m = b.n;
    n = b.n + 1;

    vec x;
    x.v = allocv(n - 1);
    x.n = n - 1;

    for (i = 0; i < m - 1; i++)
    {
        // Partial Pivoting
        for (k = i + 1; k < m; k++)
        {
            // If diagonal element(absolute value) is smaller than any of the terms below it
            if (fabs(a[i][i]) < fabs(a[k][i]))
            {
                // Swap the rows
                for (j = 0; j < n; j++)
                {
                    double temp;
                    temp = a[i][j];
                    a[i][j] = a[k][j];
                    a[k][j] = temp;
                }
            }
        }
        // Begin Gauss Elimination
        for (k = i + 1; k < m; k++)
        {
            double term = a[k][i] / a[i][i];
            for (j = 0; j < n; j++)
            {
                a[k][j] = a[k][j] - term * a[i][j];
            }
        }
    }
    // Begin Back-substitution
    for (i = m - 1; i >= 0; i--)
    {
        x.v[i] = a[i][n - 1];
        for (j = i + 1; j < n - 1; j++)
        {
            x.v[i] = x.v[i] - a[i][j] * x.v[j];
        }
        x.v[i] = x.v[i] / a[i][i];
    }
    return x;
}

mtrx kronecker(mtrx A, mtrx B)
{
    int i, j;
    int n = A.n * B.n;
    mtrx C;
    C.M = allocm(n, n);
    C.m = n;
    C.n = n;

    // Note: Removed OpenMP due to potential issues with very large matrices
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C.M[i][j] = A.M[i / A.n][j / A.n] * B.M[i % B.n][j % B.n];
        }
    }
    return C;
}

mtrx reshape(mtrx A, int m, int n)
{
    mtrx B;
    int i, j, k, l;

    if ((A.m * A.n) != (m * n))
    {
        printf("** Error: the reshaped matrix must have the same number of elements **\n");
        printf("Number of elements of input matrix: %d\n", A.m * A.n);
        printf("Number of elements of output matrix: %d\n", m * n);
        exit(1);
    }

    B = initm(m, n);

    // Sequential reshape due to data dependencies - cannot parallelize effectively
    k = 0;
    l = 0;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            B.M[i][j] = A.M[k][l];
            if (l < (A.n - 1))
            {
                l++;
            }
            else
            {
                k++;
                l = 0;
            }
        }
    }
    return B;
}

mtrx eye(int n)
{
    int i;
    mtrx A;
    A.M = allocm(n, n);
    A.m = n;
    A.n = n;
    zerosm(A);

#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < n; i++)
        {
            A.M[i][i] = 1;
        }
    } else {
#endif
        for (i = 0; i < n; i++)
        {
            A.M[i][i] = 1;
        }
#ifdef OPENMP_ENABLED
    }
#endif
    return A;
}

mtrx initm(int m, int n)
{
    mtrx A;
    A.M = allocm(m, n);
    A.m = m;
    A.n = n;
    zerosm(A);
    return A;
}

void invsig(mtrx A)
{
    int i, j;
#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = -A.M[i][j];
            }
        }
    } else {
#endif
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = -A.M[i][j];
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
}

double maxel(mtrx A)
{
    int i, j;
    double max_element = -DBL_MAX;

#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) reduction(max:max_element) schedule(static)
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                if (A.M[i][j] > max_element)
                {
                    max_element = A.M[i][j];
                }
            }
        }
    } else {
#endif
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                if (A.M[i][j] > max_element)
                {
                    max_element = A.M[i][j];
                }
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
    return max_element;
}

double minel(mtrx A)
{
    int i, j;
    double min_element = DBL_MAX;

#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) reduction(min:min_element) schedule(static)
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                if (A.M[i][j] < min_element)
                {
                    min_element = A.M[i][j];
                }
            }
        }
    } else {
#endif
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                if (A.M[i][j] < min_element)
                {
                    min_element = A.M[i][j];
                }
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
    return min_element;
}

void mtrxcpy(mtrx A, mtrx B)
{
    int i, j;

#ifdef OPENMP_ENABLED
    if (g_openmp_enabled) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = B.M[i][j];
            }
        }
    } else {
#endif
        for (i = 0; i < A.m; i++)
        {
            for (j = 0; j < A.n; j++)
            {
                A.M[i][j] = B.M[i][j];
            }
        }
#ifdef OPENMP_ENABLED
    }
#endif
}