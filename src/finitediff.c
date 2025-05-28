/**
 * @file finitediff.c
 * @brief Finite difference matrix generation for spatial derivatives
 * 
 * This module provides functions to generate finite difference matrices for
 * computing first and second derivatives with various orders of accuracy.
 * The matrices are designed for use with periodic or Dirichlet boundary conditions
 * and support 2nd, 4th, and 6th order accurate schemes.
 * 
 * The finite difference matrices can be used with Kronecker products to create
 * 2D derivative operators for solving partial differential equations on
 * structured grids.
 * 
 * Supported schemes:
 * - 2nd order: Standard central differences with forward/backward at boundaries
 * - 4th order: Higher accuracy central differences with boundary treatments
 * - 6th order: Very high accuracy for smooth solutions
 * 
 * @author Fluid Dynamics Simulation Team
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include "linearalg.h"
#include "finitediff.h"

/**
 * @brief Generate finite difference matrix for first derivative
 * 
 * Creates an n×n matrix that approximates the first derivative operator
 * using finite differences of specified order. The matrix uses central
 * differences in the interior and forward/backward differences at boundaries.
 * 
 * Accuracy and stencil information:
 * - 2nd order: 3-point stencil [-1, 0, 1]/(2*dx) in interior
 * - 4th order: 5-point stencil [1, -8, 0, 8, -1]/(12*dx) in interior  
 * - 6th order: 7-point stencil [-1, 9, -45, 0, 45, -9, 1]/(60*dx) in interior
 * 
 * Boundary treatment uses forward/backward differences of appropriate order
 * to maintain overall accuracy while handling domain boundaries.
 * 
 * @param n Number of grid points (matrix dimension)
 * @param o Order of accuracy (2, 4, or 6)
 * @param dx Grid spacing
 * @return Matrix representing the first derivative operator
 * 
 * @note The returned matrix must be freed using freem() when no longer needed
 * @warning Only orders 2, 4, and 6 are supported; other values cause program exit
 */
mtrx Diff1(int n, int o, double dx)
{
    int i;
    mtrx D;
    D = initm(n, n);

    if (o == 2) // Second order accurate scheme
    {
        // Forward difference at left boundary: f'(0) ≈ (-f₀ + f₁)/dx
        D.M[0][0] = -1. / dx;
        D.M[0][1] = 1. / dx;
        
        // Central differences in interior: f'(i) ≈ (-f_{i-1} + f_{i+1})/(2*dx)
        for (i = 1; i < (n - 1); i++)
        {
            D.M[i][i - 1] = -0.5 / dx;
            D.M[i][i] = 0. / dx;
            D.M[i][i + 1] = 0.5 / dx;
        }
        
        // Backward difference at right boundary: f'(n-1) ≈ (-f_{n-2} + f_{n-1})/dx
        D.M[n - 1][n - 1] = D.M[0][1];
        D.M[n - 1][n - 2] = D.M[0][0];
        return D;
    }
    else if (o == 4) // Fourth order accurate scheme
    {
        // Forward difference at left boundary (2nd order)
        D.M[0][0] = (double)-1 / dx;
        D.M[0][1] = (double)1 / dx;
        
        // Central difference at second point (2nd order)
        D.M[1][0] = (double)-0.5 / dx;
        D.M[1][1] = (double)0 / dx;
        D.M[1][2] = (double)0.5 / dx;
        
        // Fourth-order central differences in interior
        // f'(i) ≈ (f_{i-2} - 8*f_{i-1} + 8*f_{i+1} - f_{i+2})/(12*dx)
        for (i = 2; i < (n - 2); i++)
        {
            D.M[i][i - 2] = (double)1 / 12 / dx;
            D.M[i][i - 1] = (double)-2 / 3 / dx;
            D.M[i][i] = 0;
            D.M[i][i + 1] = (double)2 / 3 / dx;
            D.M[i][i + 2] = (double)-1 / 12 / dx;
        }
        
        // Boundary treatments (mirror interior schemes)
        D.M[n - 1][n - 1] = D.M[0][1];
        D.M[n - 1][n - 2] = D.M[0][0];
        D.M[n - 2][n - 1] = D.M[1][2];
        D.M[n - 2][n - 2] = D.M[1][1];
        D.M[n - 2][n - 3] = D.M[1][0];
        return D;
    }
    else if (o == 6) // Sixth order accurate scheme
    {
        // Boundary treatments using lower-order schemes
        D.M[0][0] = (double)-1 / dx;        // 1st order forward
        D.M[0][1] = (double)1 / dx;
        
        D.M[1][0] = (double)-0.5 / dx;      // 2nd order central
        D.M[1][1] = (double)0 / dx;
        D.M[1][2] = (double)0.5 / dx;
        
        D.M[2][0] = (double)1 / 12 / dx;    // 4th order central
        D.M[2][1] = (double)-2 / 3 / dx;
        D.M[2][2] = (double)0 / dx;
        D.M[2][3] = (double)2 / 3 / dx;
        D.M[2][4] = (double)-1 / 12 / dx;
        
        // Sixth-order central differences in interior
        // f'(i) ≈ (-f_{i-3} + 9*f_{i-2} - 45*f_{i-1} + 45*f_{i+1} - 9*f_{i+2} + f_{i+3})/(60*dx)
        for (i = 3; i < (n - 3); i++)
        {
            D.M[i][i - 3] = (double)-1 / 60 / dx;
            D.M[i][i - 2] = (double)3 / 20 / dx;
            D.M[i][i - 1] = (double)-3 / 4 / dx;
            D.M[i][i] = (double)0 / dx;
            D.M[i][i + 1] = (double)3 / 4 / dx;
            D.M[i][i + 2] = (double)-3 / 20 / dx;
            D.M[i][i + 3] = (double)1 / 60 / dx;
        }
        
        // Mirror boundary treatments for right side
        D.M[n - 1][n - 1] = D.M[0][1];
        D.M[n - 1][n - 2] = D.M[0][0];
        D.M[n - 2][n - 1] = D.M[1][2];
        D.M[n - 2][n - 2] = D.M[1][1];
        D.M[n - 2][n - 3] = D.M[1][0];
        D.M[n - 3][n - 1] = D.M[2][4];
        D.M[n - 3][n - 2] = D.M[2][3];
        D.M[n - 3][n - 3] = D.M[2][2];
        D.M[n - 3][n - 4] = D.M[2][1];
        D.M[n - 3][n - 5] = D.M[2][0];
        return D;
    }
    else
    {
        printf("** Error: valid orders are 2, 4 or 6 **\n");
        exit(1);
    }
}

/**
 * @brief Generate finite difference matrix for second derivative
 * 
 * Creates an n×n matrix that approximates the second derivative operator
 * using finite differences of specified order. The matrix uses central
 * differences in the interior and forward/backward differences at boundaries.
 * 
 * Accuracy and stencil information:
 * - 2nd order: 3-point stencil [1, -2, 1]/dx² in interior
 * - 4th order: 5-point stencil [-1, 16, -30, 16, -1]/(12*dx²) in interior
 * - 6th order: 7-point stencil [1, -6, 15, -20, 15, -6, 1]/(90*dx²) in interior
 * 
 * The second derivative operator is essential for diffusion terms in the
 * Navier-Stokes equations and for solving Poisson equations.
 * 
 * @param n Number of grid points (matrix dimension)
 * @param o Order of accuracy (2, 4, or 6)
 * @param dx Grid spacing
 * @return Matrix representing the second derivative operator
 * 
 * @note The returned matrix must be freed using freem() when no longer needed
 * @warning Only orders 2, 4, and 6 are supported; other values cause program exit
 */
mtrx Diff2(int n, int o, double dx)
{
    int i;
    mtrx D = initm(n, n);

    if (o == 2) // Second order accurate scheme
    {
        // Forward scheme at left boundary (2nd order)
        // f''(0) ≈ (2*f₀ - 5*f₁ + 4*f₂ - f₃)/dx²
        D.M[0][0] = 2. / (dx * dx);
        D.M[0][1] = -5. / (dx * dx);
        D.M[0][2] = 4. / (dx * dx);
        D.M[0][3] = -1. / (dx * dx);
        
        // Central differences in interior: f''(i) ≈ (f_{i-1} - 2*f_i + f_{i+1})/dx²
        for (i = 1; i < (n - 1); i++)
        {
            D.M[i][i - 1] = 1. / (dx * dx);
            D.M[i][i] = -2. / (dx * dx);
            D.M[i][i + 1] = 1. / (dx * dx);
        }
        
        // Backward scheme at right boundary (mirror of forward)
        D.M[n - 1][n - 1] = D.M[0][0];
        D.M[n - 1][n - 2] = D.M[0][1];
        D.M[n - 1][n - 3] = D.M[0][2];
        D.M[n - 1][n - 4] = D.M[0][3];
        return D;
    }
    else if (o == 4) // Fourth order accurate scheme
    {
        // Forward scheme at left boundary (2nd order)
        D.M[0][0] = (double)2 / (dx * dx);
        D.M[0][1] = (double)-5 / (dx * dx);
        D.M[0][2] = (double)4 / (dx * dx);
        D.M[0][3] = (double)-1 / (dx * dx);
        
        // Central scheme at second point (2nd order)
        D.M[1][0] = (double)1 / (dx * dx);
        D.M[1][1] = (double)-2 / (dx * dx);
        D.M[1][2] = (double)1 / (dx * dx);
        
        // Fourth-order central differences in interior
        // f''(i) ≈ (-f_{i-2} + 16*f_{i-1} - 30*f_i + 16*f_{i+1} - f_{i+2})/(12*dx²)
        for (i = 2; i < (n - 2); i++)
        {
            D.M[i][i - 2] = (double)-1 / 12 / (dx * dx);
            D.M[i][i - 1] = (double)4 / 3 / (dx * dx);
            D.M[i][i] = (double)-5 / 2 / (dx * dx);
            D.M[i][i + 1] = (double)4 / 3 / (dx * dx);
            D.M[i][i + 2] = (double)-1 / 12 / (dx * dx);
        }
        
        // Mirror boundary treatments for right side
        D.M[n - 1][n - 1] = D.M[0][0];
        D.M[n - 1][n - 2] = D.M[0][1];
        D.M[n - 1][n - 3] = D.M[0][2];
        D.M[n - 1][n - 4] = D.M[0][3];
        D.M[n - 2][n - 1] = D.M[1][0];
        D.M[n - 2][n - 2] = D.M[1][1];
        D.M[n - 2][n - 3] = D.M[1][2];
        return D;
    }
    else if (o == 6) // Sixth order accurate scheme
    {
        // Boundary treatments using lower-order schemes
        D.M[0][0] = (double)2 / (dx * dx);      // Forward (2nd order)
        D.M[0][1] = (double)-5 / (dx * dx);
        D.M[0][2] = (double)4 / (dx * dx);
        D.M[0][3] = (double)-1 / (dx * dx);
        
        D.M[1][0] = (double)1 / (dx * dx);      // Central (2nd order)
        D.M[1][1] = (double)-2 / (dx * dx);
        D.M[1][2] = (double)1 / (dx * dx);
        
        D.M[2][0] = (double)-1 / 12 / (dx * dx); // Central (4th order)
        D.M[2][1] = (double)4 / 3 / (dx * dx);
        D.M[2][2] = (double)-5 / 2 / (dx * dx);
        D.M[2][3] = (double)4 / 3 / (dx * dx);
        D.M[2][4] = (double)-1 / 12 / (dx * dx);
        
        // Sixth-order central differences in interior
        // f''(i) ≈ (f_{i-3} - 6*f_{i-2} + 15*f_{i-1} - 20*f_i + 15*f_{i+1} - 6*f_{i+2} + f_{i+3})/(90*dx²)
        for (i = 3; i < (n - 3); i++)
        {
            D.M[i][i - 3] = (double)1 / 90 / (dx * dx);
            D.M[i][i - 2] = (double)-3 / 20 / (dx * dx);
            D.M[i][i - 1] = (double)3 / 2 / (dx * dx);
            D.M[i][i] = (double)-49 / 18 / (dx * dx);
            D.M[i][i + 1] = (double)3 / 2 / (dx * dx);
            D.M[i][i + 2] = (double)-3 / 20 / (dx * dx);
            D.M[i][i + 3] = (double)1 / 90 / (dx * dx);
        }
        
        // Mirror boundary treatments for right side
        D.M[n - 1][n - 1] = D.M[0][0];
        D.M[n - 1][n - 2] = D.M[0][1];
        D.M[n - 1][n - 3] = D.M[0][2];
        D.M[n - 1][n - 4] = D.M[0][3];
        D.M[n - 2][n - 1] = D.M[1][0];
        D.M[n - 2][n - 2] = D.M[1][1];
        D.M[n - 2][n - 3] = D.M[1][2];
        D.M[n - 3][n - 1] = D.M[2][0];
        D.M[n - 3][n - 2] = D.M[2][1];
        D.M[n - 3][n - 3] = D.M[2][2];
        D.M[n - 3][n - 4] = D.M[2][3];
        D.M[n - 3][n - 5] = D.M[2][4];
        return D;
    }
    else
    {
        printf("** Error: valid orders are 2, 4 or 6 **\n");
        exit(1);
    }
}