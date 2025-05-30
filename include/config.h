#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    // Physical parameters
    double Re;          // Reynolds number
    int Lx;             // length
    int Ly;             // width
    
    // Numerical parameters
    int nx;             // number of points in x direction
    int ny;             // number of points in y direction
    double dt;          // time step
    double tf;          // final time
    double max_co;      // max Courant number
    int order;          // finite difference order for spatial derivatives
    int poisson_max_it; // Poisson equation max number of iterations
    double poisson_tol; // Poisson equation criterion for convergence
    int output_interval; // Output interval for .vtk files
    int poisson_type;   // 1 - no relaxation | 2 - successive overrelaxation
    
    // Performance parameters
    int openmp_enabled; // 0 - disabled, 1 - enabled
    
    // Boundary conditions (Dirichlet)
    double ui;          // internal field for u
    double vi;          // internal field for v
    double u1;          // right boundary condition
    double u2;          // left boundary condition
    double u3;          // bottom boundary condition
    double u4;          // top boundary condition
    double v1;          // boundary condition for v
    double v2;          // boundary condition for v
    double v3;          // boundary condition for v
    double v4;          // boundary condition for v
} Config;

// Function declarations
Config load_default_config(void);
Config load_config_from_file(const char* filename);
void print_config(const Config* config);
void print_usage(const char* program_name);
void print_openmp_status(const Config* config);

#endif // CONFIG_H 