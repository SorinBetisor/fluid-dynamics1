#ifndef CONFIG_H
#define CONFIG_H

// Configuration structure for simulation parameters
typedef struct {
    // Physical parameters
    double Re;              // Reynolds number
    int Lx;                 // Length 
    int Ly;                 // Width
    
    // Numerical parameters
    int nx;                 // Grid resolution in x direction 
    int ny;                 // Grid resolution in y direction
    double dt;              // Time step
    double tf;              // Final time
    double max_co;          // Max Courant number
    int order;              // Finite difference order
    int poisson_max_it;     // Poisson equation max iterations
    double poisson_tol;     // Poisson equation tolerance
    int output_interval;    // Output interval for VTK files
    int poisson_type;       // Poisson solver type
    
    // Acceleration options
    int use_omp;            // Use OpenMP
    int use_gpu;            // Use GPU
    int use_vulkan;         // Use Vulkan
    
    // Object parameters
    double center_x;        // Object center x
    double center_y;        // Object center y
    double radius;          // Object radius
} Config;

// Function to parse a configuration file
Config parseConfigFile(const char* filename);

#endif // CONFIG_H 