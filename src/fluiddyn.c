/**
 * @file fluiddyn.c
 * @brief Core fluid dynamics functions for Navier-Stokes simulation
 * 
 * This module implements the fundamental fluid dynamics operations for solving
 * the incompressible Navier-Stokes equations using the vorticity-stream function
 * formulation. The implementation includes time advancement schemes and
 * diagnostic functions for flow analysis.
 * 
 * The vorticity-stream function approach transforms the Navier-Stokes equations
 * into a more computationally efficient form by eliminating the pressure term
 * and ensuring automatic satisfaction of the continuity equation.
 * 
 * Key equations implemented:
 * - Vorticity transport equation: ∂ω/∂t + u·∇ω = (1/Re)∇²ω
 * - Stream function-vorticity relation: ∇²ψ = -ω
 * - Velocity-stream function relation: u = ∂ψ/∂y, v = -∂ψ/∂x
 * 
 * @author Fluid Dynamics Simulation Team
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include "fluiddyn.h"
#include "linearalg.h"

/**
 * @brief Advance vorticity field using explicit Euler time integration
 * 
 * Implements the vorticity transport equation using forward Euler time stepping:
 * ω^{n+1} = ω^n + dt * [-(u·∇ω) + (1/Re)∇²ω]
 * 
 * The vorticity transport equation describes how vorticity (curl of velocity)
 * evolves due to:
 * 1. Convection: -(u·∇ω) - vorticity is transported by the flow
 * 2. Diffusion: (1/Re)∇²ω - viscous diffusion smooths vorticity
 * 
 * This explicit scheme is simple but requires small time steps for stability.
 * The CFL condition must be satisfied: dt ≤ min(dx²Re/4, dx/|u|max)
 * 
 * @param w Vorticity field (modified in-place)
 * @param dwdx Vorticity gradient in x-direction
 * @param dwdy Vorticity gradient in y-direction  
 * @param d2wdx2 Second derivative of vorticity in x-direction
 * @param d2wdy2 Second derivative of vorticity in y-direction
 * @param u Velocity component in x-direction
 * @param v Velocity component in y-direction
 * @param Re Reynolds number (controls viscous effects)
 * @param dt Time step size
 * 
 * @note All input matrices must have the same dimensions
 * @note The function modifies the vorticity field w in-place
 * @warning Stability requires appropriate time step selection based on CFL condition
 */
void euler(mtrx w, mtrx dwdx, mtrx dwdy, mtrx d2wdx2, mtrx d2wdy2, mtrx u, mtrx v, double Re, double dt)
{
    int i, j;

    // Apply vorticity transport equation at each grid point
    for (i = 0; i < w.m; i++)
    {
        for (j = 0; j < w.n; j++)
        {
            // Vorticity transport: ∂ω/∂t = -(u·∇ω) + (1/Re)∇²ω
            // Convection term: -(u*∂ω/∂x + v*∂ω/∂y)
            // Diffusion term: (1/Re)*(∂²ω/∂x² + ∂²ω/∂y²)
            w.M[i][j] = (-u.M[i][j] * dwdx.M[i][j] - v.M[i][j] * dwdy.M[i][j] + (1. / Re) * (d2wdx2.M[i][j] + d2wdy2.M[i][j])) * dt + w.M[i][j];
        }
    }
}

/**
 * @brief Compute continuity equation residual for flow diagnostics
 * 
 * Calculates the divergence of velocity field: ∇·u = ∂u/∂x + ∂v/∂y
 * 
 * For incompressible flow, the continuity equation requires ∇·u = 0.
 * This function computes the residual to check how well this constraint
 * is satisfied numerically. Large values indicate:
 * - Insufficient Poisson solver convergence
 * - Numerical errors in derivative calculations
 * - Boundary condition inconsistencies
 * 
 * The maximum absolute value of the continuity residual is typically
 * used as a convergence criterion and solution quality indicator.
 * 
 * @param dudx Velocity gradient ∂u/∂x
 * @param dvdy Velocity gradient ∂v/∂y
 * @return Matrix containing continuity residual at each grid point
 * 
 * @note The returned matrix must be freed using freem() when no longer needed
 * @note For well-converged solutions, residual should be near machine precision
 */
mtrx continuity(mtrx dudx, mtrx dvdy)
{
    int i, j;
    mtrx temp;
    temp = initm(dudx.m, dudx.n);

    // Compute divergence: ∇·u = ∂u/∂x + ∂v/∂y
    for (i = 0; i < temp.m; i++)
    {
        for (j = 0; j < temp.n; j++)
        {
            temp.M[i][j] = dudx.M[i][j] + dvdy.M[i][j];
        }
    }
    return temp;
}

/**
 * @brief Compute vorticity from velocity gradients
 * 
 * Calculates the vorticity field using the definition: ω = ∇×u = ∂v/∂x - ∂u/∂y
 * 
 * Vorticity represents the local rotation rate of fluid elements and is a
 * fundamental quantity in fluid dynamics. It provides insight into:
 * - Flow structures (vortices, shear layers)
 * - Energy dissipation mechanisms
 * - Turbulence characteristics
 * - Boundary layer behavior
 * 
 * In 2D flows, vorticity is a scalar quantity pointing perpendicular to
 * the flow plane. Positive values indicate counterclockwise rotation,
 * negative values indicate clockwise rotation.
 * 
 * @param dudy Velocity gradient ∂u/∂y
 * @param dvdx Velocity gradient ∂v/∂x
 * @return Matrix containing vorticity field ω = ∂v/∂x - ∂u/∂y
 * 
 * @note The returned matrix must be freed using freem() when no longer needed
 * @note Vorticity is often used for flow visualization and analysis
 */
mtrx vorticity(mtrx dudy, mtrx dvdx)
{
    int i, j;
    mtrx temp;
    temp = initm(dudy.m, dudy.n);

    // Compute vorticity: ω = ∇×u = ∂v/∂x - ∂u/∂y
    for (i = 0; i < temp.m; i++)
    {
        for (j = 0; j < temp.n; j++)
        {
            temp.M[i][j] = dvdx.M[i][j] - dudy.M[i][j];
        }
    }
    return temp;
}