#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef DISABLE_OPENMP
#include <omp.h>  // Include OpenMP header
#endif
#include "linearalg.h"
#include "finitediff.h"
#include "utils.h"
#include "poisson.h"
#include "fluiddyn.h"
#include "gl_solver.h"  // Include GPU solver header
#include "vulkan_solver.h" // Include Vulkan solver header
#include "config.h" // Include configuration header

// Timing functions for when OpenMP is not available
#ifdef DISABLE_OPENMP
double omp_get_wtime(void) {
    static time_t start_time = 0;
    if (start_time == 0) {
        start_time = time(NULL);
    }
    return (double)(time(NULL) - start_time);
}

int omp_get_max_threads(void) {
    return 1;
}

void omp_set_num_threads(int num_threads) {
    // Do nothing
}
#endif

int main(int argc, char *argv[])
{
    // srand(time(NULL));
    int i, j, t;
    
    // Parse configuration file (use default if not specified)
    const char* configFile = "config.txt";
    if (argc > 1) {
        configFile = argv[1];
    }
    
    Config config = parseConfigFile(configFile);

    // Physical parameters
    double Re = config.Re;      // Reynolds number
    int Lx = config.Lx;         // length
    int Ly = config.Ly;         // width

    // Numerical parameters
    int nx = config.nx;                                                            // grid resolution in x direction
    int ny = config.ny;                                                            // grid resolution in y direction
    double dt = config.dt;                                                         // time step
    double tf = config.tf;                                                         // final time
    double max_co = config.max_co;                                                 // max Courant number
    int order = config.order;                                                      // finite difference order for spatial derivatives
    int poisson_max_it = config.poisson_max_it;                                    // Poisson equation max number of iterations
    double poisson_tol = config.poisson_tol;                                       // Poisson equation criterion for convergence
    int output_interval = config.output_interval;                                  // Output interval for .vtk files
    int poisson_type = config.poisson_type;                                        // 1 - no relaxation | 2 - successive overrelaxation
    double beta = 0.5 * (2 / (1 + sin(PI / (nx + 1))) + 2 / (1 + sin(PI / (ny + 1)))); // SOR poisson parameter
    
    // Acceleration options
    int use_omp = config.use_omp;              // Use OpenMP for parallelization
    int use_gpu = config.use_gpu;              // Use GPU for Poisson solver
    int use_vulkan = config.use_vulkan;        // Use Vulkan for Poisson solver
    
#ifdef DISABLE_OPENMP
    use_omp = 0;  // Force disable OpenMP if not available
    printf("OpenMP support not available in this build\n");
#else
    int num_threads = omp_get_max_threads();  // Get max number of available threads
    if (use_omp) {
        printf("Using OpenMP with %d threads\n", num_threads);
        omp_set_num_threads(num_threads);
    } else {
        printf("OpenMP disabled\n");
    }
#endif

    // Check if we should use Vulkan on Apple Silicon
#ifdef __APPLE__
#if defined(__arm64__) || defined(__aarch64__)
    // On Apple Silicon, prefer Vulkan over OpenGL
    use_vulkan = 1;
    use_gpu = 0;
    printf("Apple Silicon detected, preferring Vulkan/Metal acceleration\n");
#endif
#endif

#ifdef DISABLE_GPU
    use_gpu = 0;  // Force disable GPU if not available
    printf("OpenGL acceleration not available in this build\n");
#else
    // Initialize GPU solver if requested
    if (use_gpu && !use_vulkan) {
        printf("Initializing OpenGL solver\n");
        if (!init_gl_solver(nx, ny)) {
            printf("Failed to initialize OpenGL solver, checking Vulkan...\n");
            use_gpu = 0;
            use_vulkan = 1;
        }
    } else if (!use_vulkan) {
        printf("OpenGL acceleration disabled\n");
    }
#endif

#ifdef DISABLE_VULKAN
    use_vulkan = 0;  // Force disable Vulkan if not available
    printf("Vulkan acceleration not available in this build\n");
#else
    printf("Initializing Vulkan solver...\n");
    // Initialize Vulkan solver
    if (!init_vulkan_solver(nx, ny)) {
        printf("Vulkan initialization failed, falling back to CPU solver\n");
    } else {
        printf("Vulkan initialization succeeded, using GPU acceleration\n");
    }
#endif

    // Object parameters
    double center_x = config.center_x;  // Center of object in x direction
    double center_y = config.center_y;  // Center of object in y direction
    double radius = config.radius;      // Radius of circular object

    printf("Poisson SOR parameter: %lf\n", beta);
    printf("Adding circular object at (%.2f, %.2f) with radius %.2f\n", center_x, center_y, radius);

    // Boundary conditions (Dirichlet)
    double ui = 0.; // internal field for u
    double vi = 0.; // internal field for v

    double u1 = 0.; // right boundary condition
    double u2 = 0.; // left boundary condition
    double u3 = 0.; // bottom boundary condition
    double u4 = 1.; // top boundary condition

    double v1 = 0.;
    double v2 = 0.;
    double v3 = 0.;
    double v4 = 0.;

    // Computes cell sizes
    double dx = (double)Lx / nx;
    double dy = (double)Ly / ny;

    // Create grid to track solid cells
    cell_properties **grid = (cell_properties **)malloc(nx * sizeof(cell_properties *));
    for (i = 0; i < nx; i++) {
        grid[i] = (cell_properties *)malloc(ny * sizeof(cell_properties));
        for (j = 0; j < ny; j++) {
            grid[i][j].is_solid = 0;  // Initialize all cells as fluid
        }
    }

    // Define circular object
    #pragma omp parallel for private(j) if(use_omp)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            double x = i * dx;
            double y = j * dy;
            double distance = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
            if (distance < radius) {
                grid[i][j].is_solid = 1;  // Mark cell as solid
            }
        }
    }

    // Generates derivatives operators
    mtrx d_x = Diff1(nx, order, dx);
    mtrx d_y = Diff1(ny, order, dy);
    mtrx d_x2 = Diff2(nx, order, dx);
    mtrx d_y2 = Diff2(ny, order, dy);

    mtrx Ix = eye(nx);              // identity matrix
    mtrx Iy = eye(ny);              // identity matrix
    mtrx DX = kronecker(Ix, d_x);   // kronecker product for x first derivative
    mtrx DY = kronecker(d_y, Iy);   // kronecker product for y first derivative
    mtrx DX2 = kronecker(Ix, d_x2); // kronecker product for x second derivative
    mtrx DY2 = kronecker(d_y2, Iy); // kronecker product for y second derivative

    d_x.M = freem(d_x);
    d_y.M = freem(d_y);
    d_x2.M = freem(d_x2);
    d_y2.M = freem(d_y2);
    Ix.M = freem(Ix);
    Iy.M = freem(Iy);

    // Maximum number of iterations
    int it_max = (int)((tf / dt) - 1);

    // Courant numbers
    double r1 = u1 * dt / (dx);
    double r2 = u1 * dt / (dy);

    if ((r1 > max_co) || (r2 > max_co))
    {
        printf("Unstable Solution!\n");
        printf("r1: %lf\n", r1);
        printf("r2: %lf\n", r2);
        exit(1);
    }

    // Variables
    // Initialize velocities
    mtrx u = initm(nx, ny);   // x-velocity
    mtrx v = initm(nx, ny);   // y-velocity
    mtrx w = initm(nx, ny);   // vorticity
    mtrx psi = initm(nx, ny); // stream-function
    
    // Additional matrix to visualize the solid object
    mtrx obj = initm(nx, ny); // object marker (1 for solid, 0 for fluid)

    // Fill object marker matrix
    #pragma omp parallel for private(j) if(use_omp)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            obj.M[i][j] = grid[i][j].is_solid;
        }
    }

    // Derivatives
    // mtrx p;   // pressure
    mtrx dwdx;             // vorticity x-derivative
    mtrx dwdy;             // vorticity y-derivative
    mtrx d2wdx2;           // vorticity x-derivative (2nd)
    mtrx d2wdy2;           // vorticity y-derivative (2nd)
    mtrx dpsidx;           // stream-function x-derivative
    mtrx dpsidy;           // stream-function y-derivative
    mtrx dudx;             // x-velocity x-derivative
    mtrx dudy;             // x-velocity y-derivative
    mtrx dvdx;             // y-velocity x-derivative
    mtrx dvdy;             // y-velocity y-derivative
    mtrx check_continuity; // continuity equation

    // Auxiliary variables
    mtrx w0;
    mtrx dwdx0;
    mtrx dwdy0;
    mtrx d2wdx20;
    mtrx d2wdy20;
    mtrx psi0;
    mtrx dpsidx0;
    mtrx dpsidy0;
    mtrx u0;
    mtrx v0;
    mtrx dudx0;
    mtrx dudy0;
    mtrx dvdx0;
    mtrx dvdy0;

    // Initial condition
    #pragma omp parallel for private(j) if(use_omp)
    for (i = 1; i < nx - 1; i++)
    {
        for (j = 1; j < ny - 1; j++)
        {
            u.M[i][j] = ui;
            v.M[i][j] = vi;
        }
    }

    // Timing variables
    double start_time, end_time, total_time = 0.0;
    double poisson_time = 0.0;
    
    // Main time loop
    for (t = 0; t <= it_max; t++)
    {
        start_time = omp_get_wtime();
        
        // Initialize variables

        // Boundary conditions
        #pragma omp parallel for if(use_omp)
        for (j = 0; j < ny; j++)
        {
            v.M[0][j] = v3;
            v.M[nx - 1][j] = v4;
            u.M[0][j] = u3;
            u.M[nx - 1][j] = u4;
        }
        
        #pragma omp parallel for if(use_omp)
        for (i = 0; i < nx; i++)
        {
            v.M[i][0] = v1;
            v.M[i][ny - 1] = v2;
            u.M[i][0] = u1;
            u.M[i][ny - 1] = u2;
        }

        // Apply no-slip condition on the object boundary
        #pragma omp parallel for private(j) if(use_omp)
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                if (grid[i][j].is_solid) {
                    u.M[i][j] = 0.0;
                    v.M[i][j] = 0.0;
                }
            }
        }

        u0 = reshape(u, nx * ny, 1);
        v0 = reshape(v, nx * ny, 1);
        dudy0 = mtrxmul(DY, u0);
        dvdx0 = mtrxmul(DX, v0);

        dudy = reshape(dudy0, nx, ny);
        dvdx = reshape(dvdx0, nx, ny);

        u0.M = freem(u0);
        v0.M = freem(v0);
        dudy0.M = freem(dudy0);
        dvdx0.M = freem(dvdx0);

        #pragma omp parallel for if(use_omp)
        for (j = 0; j < ny; j++)
        {
            w.M[0][j] = dvdx.M[0][j] - dudy.M[0][j];
            w.M[nx - 1][j] = dvdx.M[nx - 1][j] - dudy.M[nx - 1][j];
        }
        
        #pragma omp parallel for if(use_omp)
        for (i = 0; i < nx; i++)
        {
            w.M[i][0] = dvdx.M[i][0] - dudy.M[i][0];
            w.M[i][ny - 1] = dvdx.M[i][ny - 1] - dudy.M[i][ny - 1];
        }

        // Computes derivatives
        w0 = reshape(w, nx * ny, 1);
        dwdx0 = mtrxmul(DX, w0);
        dwdy0 = mtrxmul(DY, w0);

        dwdx = reshape(dwdx0, nx, ny);
        dwdy = reshape(dwdy0, nx, ny);

        dwdx0.M = freem(dwdx0);
        dwdy0.M = freem(dwdy0);

        d2wdx20 = mtrxmul(DX2, w0);
        d2wdy20 = mtrxmul(DY2, w0);

        d2wdx2 = reshape(d2wdx20, nx, ny);
        d2wdy2 = reshape(d2wdy20, nx, ny);

        d2wdx20.M = freem(d2wdx20);
        d2wdy20.M = freem(d2wdy20);
        w0.M = freem(w0);

        // Time - advancement(Euler) with OpenMP parallelization
        #pragma omp parallel if(use_omp)
        {
            euler_parallel(w, dwdx, dwdy, d2wdx2, d2wdy2, u, v, Re, dt);
        }

        // Set vorticity inside solid to avoid updating it
        #pragma omp parallel for private(j) if(use_omp)
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                if (grid[i][j].is_solid) {
                    w.M[i][j] = 0.0;
                }
            }
        }

        // Solves poisson equation for stream function
        psi.M = freem(psi);
        invsig(w);
        
        double poisson_start = omp_get_wtime();
        
        if (poisson_type == 1)
        {
            if (use_gpu) {
                psi = poisson_gpu_with_object(w, dx, dy, poisson_max_it, poisson_tol, grid);
            } else if (use_vulkan) {
                psi = poisson_vulkan_with_object(w, dx, dy, poisson_max_it, poisson_tol, grid);
            } else {
                psi = poisson_with_object(w, dx, dy, poisson_max_it, poisson_tol, grid);
            }
        }
        else if (poisson_type == 2)
        {
            if (use_gpu) {
                psi = poisson_SOR_gpu_with_object(w, dx, dy, poisson_max_it, poisson_tol, beta, grid);
            } else if (use_vulkan) {
                psi = poisson_SOR_vulkan_with_object(w, dx, dy, poisson_max_it, poisson_tol, beta, grid);
            } else {
                psi = poisson_SOR_with_object(w, dx, dy, poisson_max_it, poisson_tol, beta, grid);
            }
        }
        else
        {
            printf("Error - invalid option for Poisson solver\n");
            exit(1);
        }
        
        double poisson_end = omp_get_wtime();
        poisson_time += (poisson_end - poisson_start);

        invsig(w);

        // Computes velocities
        psi0 = reshape(psi, nx * ny, 1);
        dpsidx0 = mtrxmul(DX, psi0);
        dpsidy0 = mtrxmul(DY, psi0);

        dpsidx = reshape(dpsidx0, nx, ny);
        dpsidy = reshape(dpsidy0, nx, ny);

        psi0.M = freem(psi0);
        dpsidx0.M = freem(dpsidx0);
        dpsidy0.M = freem(dpsidy0);

        u.M = freem(u);
        v.M = freem(v);
        u = initm(nx, ny);
        v = initm(nx, ny);
        mtrxcpy(u, dpsidy);
        invsig(dpsidx);
        mtrxcpy(v, dpsidx);

        // Apply no-slip condition on the object boundary again
        #pragma omp parallel for private(j) if(use_omp)
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                if (grid[i][j].is_solid) {
                    u.M[i][j] = 0.0;
                    v.M[i][j] = 0.0;
                }
            }
        }

        // Checks continuity equation
        u0 = reshape(u, nx * ny, 1);
        v0 = reshape(v, nx * ny, 1);
        dudx0 = mtrxmul(DX, u0);
        dvdy0 = mtrxmul(DY, v0);

        dudx = reshape(dudx0, nx, ny);
        dvdy = reshape(dvdy0, nx, ny);
        check_continuity = continuity(dudx, dvdy);
        
        end_time = omp_get_wtime();
        total_time += (end_time - start_time);
        
        printf("Iteration: %d | ", t);
        printf("Time: %lf | ", (double)t * dt);
        printf("Progress: %.2lf%% | ", (double)100 * t / it_max);
        printf("Step time: %.3fs\n", end_time - start_time);
        printf("Continuity max: %E | ", maxel(check_continuity));
        printf("Continuity min: %E\n", minel(check_continuity));

        u0.M = freem(u0);
        v0.M = freem(v0);
        dudx0.M = freem(dudx0);
        dvdy0.M = freem(dvdy0);

        if (t % output_interval == 0)
        {
            // printvtk(psi, "stream-function");
            printvtk(w, "vorticity");
            printvtk(u, "x-velocity");
            printvtk(v, "y-velocity");
            printvtk(obj, "object");
            // printvtk(p, "pressure");
        }

        // Free memory
        // freem(p);
        dwdx.M = freem(dwdx);
        dwdy.M = freem(dwdy);
        d2wdx2.M = freem(d2wdx2);
        d2wdy2.M = freem(d2wdy2);
        dpsidx.M = freem(dpsidx);
        dpsidy.M = freem(dpsidy);
        dudx.M = freem(dudx);
        dudy.M = freem(dudy);
        dvdx.M = freem(dvdx);
        dvdy.M = freem(dvdy);
        check_continuity.M = freem(check_continuity);
    }
    
    // Print performance statistics
    printf("\nPerformance Summary:\n");
    printf("Total simulation time: %.3f seconds\n", total_time);
    printf("Time spent in Poisson solver: %.3f seconds (%.1f%%)\n", 
           poisson_time, 100.0 * poisson_time / total_time);
    printf("Average time per iteration: %.3f seconds\n", total_time / (it_max + 1));
    
    // Clean up GPU resources if used
    if (use_gpu) {
        cleanup_gl_solver();
    } else if (use_vulkan) {
        cleanup_vulkan_solver();
    }
    
    // Free memory
    u.M = freem(u);
    v.M = freem(v);
    w.M = freem(w);
    psi.M = freem(psi);
    obj.M = freem(obj);
    DX.M = freem(DX);
    DY.M = freem(DY);
    DX2.M = freem(DX2);
    DY2.M = freem(DY2);
    
    // Free grid memory
    for (i = 0; i < nx; i++) {
        free(grid[i]);
    }
    free(grid);

    printf("Simulation complete!\n");

    return 0;
}
