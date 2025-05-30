#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include "linearalg.h"
#include "finitediff.h"
#include "utils.h"
#include "poisson.h"
#include "fluiddyn.h"
#include "config.h"
// #include <omp.h>

// Helper function for logging to file only
void log_message(FILE *log_file, const char *format, ...) {
    // Only log to file if it's available
    if (log_file != NULL) {
        va_list args;
        va_start(args, format);
        vfprintf(log_file, format, args);
        va_end(args);
        fflush(log_file);
    }
}

int main(int argc, char *argv[])
{
    // srand(time(NULL));
    int i, j, t;
    Config config;
    char output_dir[256] = "./output"; // Default output directory

    double start_time = clock();

    // Parse command line arguments
    if (argc == 1) {
        // No arguments - use default configuration
        printf("No configuration file specified. Using default values.\n");
        config = load_default_config();
    } else if (argc == 2) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[1], "--help-config") == 0) {
            printf("Complete list of configuration parameters:\n\n");
            printf("Physical Parameters:\n");
            printf("  Re           - Reynolds number (default: 1000.0)\n");
            printf("  Lx           - Domain length (default: 1)\n");
            printf("  Ly           - Domain width (default: 1)\n");
            printf("\nNumerical Parameters:\n");
            printf("  nx           - Grid points in x direction (default: 64)\n");
            printf("  ny           - Grid points in y direction (default: 64)\n");
            printf("  dt           - Time step (default: 0.005)\n");
            printf("  tf           - Final time (default: 20.0)\n");
            printf("  max_co       - Maximum Courant number (default: 1.0)\n");
            printf("  order        - Finite difference order (default: 6)\n");
            printf("  poisson_max_it - Poisson max iterations (default: 10000)\n");
            printf("  poisson_tol  - Poisson tolerance (default: 1E-3)\n");
            printf("  output_interval - Output interval for VTK files (default: 10)\n");
            printf("  poisson_type - Poisson solver type: 1=no relaxation, 2=SOR (default: 2)\n");
            printf("\nPerformance Parameters:\n");
            printf("  openmp_enabled - Enable OpenMP parallelization: 0=disabled, 1=enabled (default: 1 if compiled with OpenMP)\n");
            printf("\nBoundary Conditions:\n");
            printf("  ui           - Internal u field (default: 0.0)\n");
            printf("  vi           - Internal v field (default: 0.0)\n");
            printf("  u1           - Right boundary u (default: 0.0)\n");
            printf("  u2           - Left boundary u (default: 0.0)\n");
            printf("  u3           - Bottom boundary u (default: 0.0)\n");
            printf("  u4           - Top boundary u (default: 1.0)\n");
            printf("  v1           - Right boundary v (default: 0.0)\n");
            printf("  v2           - Left boundary v (default: 0.0)\n");
            printf("  v3           - Bottom boundary v (default: 0.0)\n");
            printf("  v4           - Top boundary v (default: 0.0)\n");
            return 0;
        } else {
            // Load configuration from file
            config = load_config_from_file(argv[1]);
        }
    } else if (argc == 3) {
        // Load configuration from file and set output directory
        config = load_config_from_file(argv[1]);
        snprintf(output_dir, sizeof(output_dir), "./output/%s", argv[2]);
        printf("Using output directory: %s\n", output_dir);
    } else {
        printf("Error: Too many arguments.\n");
        print_usage(argv[0]);
        return 1;
    }

    // Print the configuration being used
    print_config(&config);

    // Print OpenMP status information
    print_openmp_status(&config);

    // Extract configuration values for easier access
    double Re = config.Re;
    int Lx = config.Lx;
    int Ly = config.Ly;
    int nx = config.nx;
    int ny = config.ny;
    double dt = config.dt;
    double tf = config.tf;
    double max_co = config.max_co;
    int order = config.order;
    int poisson_max_it = config.poisson_max_it;
    double poisson_tol = config.poisson_tol;
    int output_interval = config.output_interval;
    int poisson_type = config.poisson_type;
    
    // Configure OpenMP for linear algebra operations
    set_openmp_config(config.openmp_enabled);
    
    // Boundary conditions
    double ui = config.ui;
    double vi = config.vi;
    double u1 = config.u1;
    double u2 = config.u2;
    double u3 = config.u3;
    double u4 = config.u4;
    double v1 = config.v1;
    double v2 = config.v2;
    double v3 = config.v3;
    double v4 = config.v4;

    // Calculate SOR parameter (depends on grid size)
    double beta = 0.5 * (2 / (1 + sin(PI / (nx + 1))) + 2 / (1 + sin(PI / (ny + 1))));
    printf("Poisson SOR parameter: %lf\n", beta);

    // Computes cell sizes
    double dx = (double)Lx / nx;
    double dy = (double)Ly / ny;

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
    for (i = 1; i < nx - 1; i++)
    {
        for (j = 1; j < ny - 1; j++)
        {
            u.M[i][j] = ui;
            v.M[i][j] = vi;
        }
    }

    // Timing variables
    double iteration_start_time, iteration_end_time, iteration_time;
    double total_elapsed_time, avg_iteration_time, estimated_remaining_time;
    double simulation_start_time = clock(); // More descriptive name for overall simulation timing

    // Setup logging
    char log_dir[512] = "./output/logs";
    char log_filename[1024];  // Increased from 512 to 1024 to accommodate log_dir + run_name + ".txt"
    char run_name[256] = "default";
    FILE *log_file = NULL;
    
    // Extract run name from output directory
    if (strstr(output_dir, "./output/") != NULL) {
        // If output_dir is like "./output/run_name", extract "run_name"
        const char *run_start = output_dir + strlen("./output/");
        if (strlen(run_start) > 0) {
            strncpy(run_name, run_start, sizeof(run_name) - 1);
            run_name[sizeof(run_name) - 1] = '\0';
        }
    }
    
    // Create log filename
    snprintf(log_filename, sizeof(log_filename), "%s/%s.txt", log_dir, run_name);
    
    // Create logs directory (platform-independent approach)
    #ifdef _WIN32
        system("if not exist output\\logs mkdir output\\logs");
    #else
        system("mkdir -p output/logs");
    #endif
    
    // Open log file
    log_file = fopen(log_filename, "w");
    if (log_file == NULL) {
        printf("Warning: Could not create log file %s. Logging to console instead.\n", log_filename);
    } else {
        printf("Logging simulation progress to: %s\n", log_filename);
        
        // Write header to log file
        fprintf(log_file, "=== Fluid Dynamics Simulation Log ===\n");
        fprintf(log_file, "Run name: %s\n", run_name);
        fprintf(log_file, "Output directory: %s\n", output_dir);
        fprintf(log_file, "Reynolds number: %.2f\n", Re);
        fprintf(log_file, "Grid size: %dx%d\n", nx, ny);
        fprintf(log_file, "Time step: %.6f\n", dt);
        fprintf(log_file, "Final time: %.2f\n", tf);
        fprintf(log_file, "Max iterations: %d\n", it_max + 1);
        fprintf(log_file, "OpenMP enabled: %s\n", config.openmp_enabled ? "Yes" : "No");
        fprintf(log_file, "=====================================\n\n");
        fflush(log_file);
    }

    // Main time loop
    for (t = 0; t <= it_max; t++)
    {
        iteration_start_time = clock(); // Start timing this iteration
        
        // Initialize variables

        // Boundary conditions
        for (j = 0; j < ny; j++)
        {
            v.M[0][j] = v3;
            v.M[nx - 1][j] = v4;
            u.M[0][j] = u3;
            u.M[nx - 1][j] = u4;
        }
        for (i = 0; i < nx; i++)
        {
            v.M[i][0] = v1;
            v.M[i][ny - 1] = v2;
            u.M[i][0] = u1;
            u.M[i][ny - 1] = u2;
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

        for (j = 0; j < ny; j++)
        {
            w.M[0][j] = dvdx.M[0][j] - dudy.M[0][j];
            w.M[nx - 1][j] = dvdx.M[nx - 1][j] - dudy.M[nx - 1][j];
        }
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

        // Time - advancement(Euler)
        euler(w, dwdx, dwdy, d2wdx2, d2wdy2, u, v, Re, dt);

        // Solves poisson equation for stream function
        psi.M = freem(psi);
        invsig(w);
        if (poisson_type == 1)
        {
            psi = poisson_log(w, dx, dy, poisson_max_it, poisson_tol, log_file);
        }
        else if (poisson_type == 2)
        {
            psi = poisson_SOR_log(w, dx, dy, poisson_max_it, poisson_tol, beta, log_file);
        }
        else
        {
            printf("Error - invalid option for Poisson solver\n");
            exit(1);
        }

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

        // Checks continuity equation

        u0 = reshape(u, nx * ny, 1);
        v0 = reshape(v, nx * ny, 1);

        dudx0 = mtrxmul(DX, u0);
        dvdy0 = mtrxmul(DY, v0);

        dudx = reshape(dudx0, nx, ny);
        dvdy = reshape(dvdy0, nx, ny);
        check_continuity = continuity(dudx, dvdy);
        
        iteration_end_time = clock(); // End timing this iteration
        iteration_time = (double)(iteration_end_time - iteration_start_time) / CLOCKS_PER_SEC;
        total_elapsed_time = (double)(iteration_end_time - simulation_start_time) / CLOCKS_PER_SEC;
        avg_iteration_time = total_elapsed_time / (t + 1);
        estimated_remaining_time = avg_iteration_time * (it_max - t);
        
        log_message(log_file, "Iteration: %d | ", t);
        log_message(log_file, "Time: %lf | ", (double)t * dt);
        log_message(log_file, "Progress: %.2lf%% | ", (double)100 * t / it_max);
        log_message(log_file, "Iter time: %.3f s\n", iteration_time);
        log_message(log_file, "Continuity max: %E | ", maxel(check_continuity));
        log_message(log_file, "Continuity min: %E | ", minel(check_continuity));
        log_message(log_file, "Elapsed: %.1f s | ", total_elapsed_time);
        if (t > 0) {
            log_message(log_file, "Est. remaining: %.1f s\n", estimated_remaining_time);
        } else {
            log_message(log_file, "Est. remaining: -- s\n");
        }

        u0.M = freem(u0);
        v0.M = freem(v0);
        dudx0.M = freem(dudx0);
        dvdy0.M = freem(dvdy0);

        // Computes pressure
        //	dudx = np.reshape(DX @ np.reshape(u,(nx*ny,1)),(nx,ny))
        //	dudy = np.reshape(DY @ np.reshape(u,(nx*ny,1)),(nx,ny))
        //	dvdx = np.reshape(DX @ np.reshape(v,(nx*ny,1)),(nx,ny))
        //	dvdy = np.reshape(DY @ np.reshape(v,(nx*ny,1)),(nx,ny))
        //	f = dudx**2+dvdy**2+2*dudy*dvdx
        //	p = fft_poisson(-f,dx)

        if (t % output_interval == 0)
        {
            printvtk(psi, "stream-function", output_dir);
            printvtk(w, "vorticity", output_dir);
            printvtk(u, "x-velocity", output_dir);
            printvtk(v, "y-velocity", output_dir);
            // printvtk(p, "pressure", output_dir);
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
    // Free memory
    u.M = freem(u);
    v.M = freem(v);
    w.M = freem(w);
    psi.M = freem(psi);
    DX.M = freem(DX);
    DY.M = freem(DY);
    DX2.M = freem(DX2);
    DY2.M = freem(DY2);

    printf("Simulation complete!\n");
    double end_time = clock();
    double total_simulation_time = (double)(end_time - simulation_start_time) / CLOCKS_PER_SEC;
    double setup_time = (double)(simulation_start_time - start_time) / CLOCKS_PER_SEC;
    
    log_message(log_file, "\n=== Timing Summary ===\n");
    log_message(log_file, "Setup time: %.4f seconds\n", setup_time);
    log_message(log_file, "Simulation time: %.2f seconds\n", total_simulation_time);
    log_message(log_file, "Total program time: %.2f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    log_message(log_file, "Average iteration time: %.4f seconds\n", total_simulation_time / (it_max + 1));
    log_message(log_file, "Iterations completed: %d\n", it_max + 1);
    log_message(log_file, "======================\n");
    
    // Close log file
    if (log_file != NULL) {
        fclose(log_file);
        printf("Log saved to: %s\n", log_filename);
    }
    return 0;
}
