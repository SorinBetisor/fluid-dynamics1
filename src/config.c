/**
 * @file config.c
 * @brief Configuration management system for fluid dynamics simulation
 * 
 * This module provides functionality to load and manage simulation parameters
 * from configuration files or use default values. It supports reading key-value
 * pairs from text files with comments and provides comprehensive parameter
 * validation and display.
 * 
 * The configuration system allows users to modify simulation parameters without
 * recompiling the code, making it easy to run different scenarios and experiments.
 * 
 * @author Fluid Dynamics Simulation Team
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "utils.h"
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#ifndef PI
#define PI 3.14159265359
#endif

/**
 * @brief Load default configuration parameters
 * 
 * Creates a configuration structure with default values suitable for
 * a standard lid-driven cavity flow simulation. These values represent
 * a well-tested configuration that provides stable numerical results.
 * 
 * Default configuration:
 * - Reynolds number: 1000 (moderate turbulence)
 * - Grid: 64x64 points
 * - Time step: 0.005 (stable for given grid and Re)
 * - Simulation time: 20.0 time units
 * - Lid-driven cavity boundary conditions
 * 
 * @return Config structure with default parameters
 */
Config load_default_config(void) {
    Config config;
    
    // Physical parameters (default values from original main.c)
    config.Re = 1000.0;     // Reynolds number - controls flow regime
    config.Lx = 1;          // Domain length (non-dimensional)
    config.Ly = 1;          // Domain width (non-dimensional)
    
    // Numerical parameters
    config.nx = 64;         // Grid points in x-direction
    config.ny = 64;         // Grid points in y-direction
    config.dt = 0.005;      // Time step size
    config.tf = 20.0;       // Final simulation time
    config.max_co = 1.0;    // Maximum allowable Courant number
    config.order = 6;       // Finite difference order (2, 4, or 6)
    config.poisson_max_it = 10000;  // Maximum Poisson solver iterations
    config.poisson_tol = 1E-3;      // Poisson solver convergence tolerance
    config.output_interval = 10;    // VTK output frequency
    config.poisson_type = 2;        // Poisson solver type (1=basic, 2=SOR)
    
    // Performance parameters
#ifdef OPENMP_ENABLED
    config.openmp_enabled = 1;      // Enable OpenMP by default if available
#else
    config.openmp_enabled = 0;      // Disable OpenMP if not compiled with it
#endif
    
    // Boundary conditions (Dirichlet) - Lid-driven cavity setup
    config.ui = 0.0;        // Initial internal u-velocity
    config.vi = 0.0;        // Initial internal v-velocity
    config.u1 = 0.0;        // Right boundary u-velocity
    config.u2 = 0.0;        // Left boundary u-velocity
    config.u3 = 0.0;        // Bottom boundary u-velocity
    config.u4 = 1.0;        // Top boundary u-velocity (moving lid)
    config.v1 = 0.0;        // Right boundary v-velocity
    config.v2 = 0.0;        // Left boundary v-velocity
    config.v3 = 0.0;        // Bottom boundary v-velocity
    config.v4 = 0.0;        // Top boundary v-velocity
    
    return config;
}

/**
 * @brief Load configuration from a text file
 * 
 * Reads simulation parameters from a configuration file in key=value format.
 * The function starts with default values and overwrites them with values
 * found in the file. This ensures that missing parameters don't cause errors.
 * 
 * File format:
 * - Lines starting with # or ; are treated as comments
 * - Empty lines are ignored
 * - Parameters are specified as: parameter_name = value
 * - Whitespace around = is ignored
 * - Unknown parameters generate warnings but don't stop execution
 * 
 * @param filename Path to the configuration file
 * @return Config structure with loaded parameters (defaults used for missing values)
 */
Config load_config_from_file(const char* filename) {
    Config config = load_default_config(); // Start with defaults
    FILE* file;
    char line[256];
    char key[64];
    char value[64];
    
    // Attempt to open the configuration file 2
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open configuration file '%s'\n", filename);
        printf("Using default configuration.\n");
        return config;
    }
    
    printf("Loading configuration from: %s\n", filename);
    
    // Parse file line by line
    while (fgets(line, sizeof(line), file)) {
        // Skip empty lines and comments
        if (line[0] == '\n' || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Parse key=value pairs
        if (sscanf(line, "%63[^=]=%63s", key, value) == 2) {
            // Remove whitespace from key
            char* key_trimmed = key;
            while (*key_trimmed == ' ' || *key_trimmed == '\t') key_trimmed++;
            char* key_end = key_trimmed + strlen(key_trimmed) - 1;
            while (key_end > key_trimmed && (*key_end == ' ' || *key_end == '\t' || *key_end == '\n' || *key_end == '\r')) {
                *key_end = '\0';
                key_end--;
            }
            
            // Remove whitespace and newlines from value
            char* value_trimmed = value;
            while (*value_trimmed == ' ' || *value_trimmed == '\t') value_trimmed++;
            char* value_end = value_trimmed + strlen(value_trimmed) - 1;
            while (value_end > value_trimmed && (*value_end == ' ' || *value_end == '\t' || *value_end == '\n' || *value_end == '\r')) {
                *value_end = '\0';
                value_end--;
            }
            
            // Physical parameters
            if (strcmp(key_trimmed, "Re") == 0) {
                config.Re = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "Lx") == 0) {
                config.Lx = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "Ly") == 0) {
                config.Ly = atoi(value_trimmed);
            }
            // Numerical parameters
            else if (strcmp(key_trimmed, "nx") == 0) {
                config.nx = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "ny") == 0) {
                config.ny = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "dt") == 0) {
                config.dt = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "tf") == 0) {
                config.tf = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "max_co") == 0) {
                config.max_co = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "order") == 0) {
                config.order = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "poisson_max_it") == 0) {
                config.poisson_max_it = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "poisson_tol") == 0) {
                config.poisson_tol = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "output_interval") == 0) {
                config.output_interval = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "poisson_type") == 0) {
                config.poisson_type = atoi(value_trimmed);
            }
            // Performance parameters
            else if (strcmp(key_trimmed, "openmp_enabled") == 0) {
                config.openmp_enabled = atoi(value_trimmed);
            }
            // Boundary conditions
            else if (strcmp(key_trimmed, "ui") == 0) {
                config.ui = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "vi") == 0) {
                config.vi = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "u1") == 0) {
                config.u1 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "u2") == 0) {
                config.u2 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "u3") == 0) {
                config.u3 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "u4") == 0) {
                config.u4 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "v1") == 0) {
                config.v1 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "v2") == 0) {
                config.v2 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "v3") == 0) {
                config.v3 = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "v4") == 0) {
                config.v4 = atof(value_trimmed);
            } else {
                printf("Warning: Unknown configuration parameter '%s'\n", key_trimmed);
            }
        }
    }
    
    fclose(file);
    printf("Configuration loaded successfully.\n");
    return config;
}

/**
 * @brief Print current configuration to console
 * 
 * Displays all configuration parameters in a formatted, human-readable way.
 * Also calculates and displays derived parameters such as grid spacing,
 * SOR parameter, and maximum iterations for user reference.
 * 
 * This function is useful for:
 * - Verifying loaded configuration
 * - Debugging simulation setup
 * - Recording simulation parameters in output logs
 * 
 * @param config Pointer to configuration structure to display
 */
void print_config(const Config* config) {
    printf("\n=== Simulation Configuration ===\n");
    printf("Physical Parameters:\n");
    printf("  Reynolds number (Re): %.2f\n", config->Re);
    printf("  Domain length (Lx): %d\n", config->Lx);
    printf("  Domain width (Ly): %d\n", config->Ly);
    
    printf("\nNumerical Parameters:\n");
    printf("  Grid points x (nx): %d\n", config->nx);
    printf("  Grid points y (ny): %d\n", config->ny);
    printf("  Time step (dt): %.6f\n", config->dt);
    printf("  Final time (tf): %.2f\n", config->tf);
    printf("  Max Courant number: %.2f\n", config->max_co);
    printf("  Finite difference order: %d\n", config->order);
    printf("  Poisson max iterations: %d\n", config->poisson_max_it);
    printf("  Poisson tolerance: %.2E\n", config->poisson_tol);
    printf("  Output interval: %d\n", config->output_interval);
    printf("  Poisson solver type: %d\n", config->poisson_type);
    
    printf("\nPerformance Parameters:\n");
    printf("  OpenMP enabled (config): %s\n", config->openmp_enabled ? "Yes" : "No");
#ifdef OPENMP_ENABLED
    printf("  OpenMP compiled support: Yes\n");
#ifdef _OPENMP
    printf("  OpenMP runtime version: %d\n", _OPENMP);
#endif
#else
    printf("  OpenMP compiled support: No\n");
#endif
    
    printf("\nBoundary Conditions:\n");
    printf("  Internal u field (ui): %.2f\n", config->ui);
    printf("  Internal v field (vi): %.2f\n", config->vi);
    printf("  u boundaries (u1,u2,u3,u4): %.2f, %.2f, %.2f, %.2f\n", 
           config->u1, config->u2, config->u3, config->u4);
    printf("  v boundaries (v1,v2,v3,v4): %.2f, %.2f, %.2f, %.2f\n", 
           config->v1, config->v2, config->v3, config->v4);
    
    // Calculate derived parameters for user information
    double dx = (double)config->Lx / config->nx;
    double dy = (double)config->Ly / config->ny;
    double beta = 0.5 * (2 / (1 + sin(PI / (config->nx + 1))) + 2 / (1 + sin(PI / (config->ny + 1))));
    int it_max = (int)((config->tf / config->dt) - 1);
    
    printf("\nDerived Parameters:\n");
    printf("  Grid spacing dx: %.6f\n", dx);
    printf("  Grid spacing dy: %.6f\n", dy);
    printf("  SOR parameter (beta): %.6f\n", beta);
    printf("  Maximum iterations: %d\n", it_max);
    printf("================================\n\n");
}

/**
 * @brief Print usage information for the program
 * 
 * Displays comprehensive help information including:
 * - Command line syntax
 * - Available options
 * - Configuration file format
 * - Example configuration entries
 * - References to additional help
 * 
 * @param program_name Name of the executable (typically argv[0])
 */
void print_usage(const char* program_name) {
    printf("Usage: %s [config_file] [output_folder]\n", program_name);
    printf("\nOptions:\n");
    printf("  config_file    Path to configuration file (optional)\n");
    printf("                 If not provided, default values will be used\n");
    printf("  output_folder  Name of output subfolder (optional)\n");
    printf("                 Files will be saved to ./output/[output_folder]/\n");
    printf("                 If not provided, files will be saved to ./output/\n");
    printf("\nExamples:\n");
    printf("  %s                           # Use defaults, output to ./output/\n", program_name);
    printf("  %s config.txt               # Use config.txt, output to ./output/\n", program_name);
    printf("  %s config.txt run1          # Use config.txt, output to ./output/run1/\n", program_name);
    printf("\nConfiguration file format:\n");
    printf("  # Comments start with # or ;\n");
    printf("  parameter_name = value\n");
    printf("\nExample configuration file:\n");
    printf("  # Physical parameters\n");
    printf("  Re = 1000.0\n");
    printf("  Lx = 1\n");
    printf("  Ly = 1\n");
    printf("  \n");
    printf("  # Numerical parameters\n");
    printf("  nx = 64\n");
    printf("  ny = 64\n");
    printf("  dt = 0.005\n");
    printf("  tf = 20.0\n");
    printf("  \n");
    printf("  # Boundary conditions\n");
    printf("  u4 = 1.0  # top boundary velocity\n");
    printf("\nFor a complete list of parameters, run with --help-config\n");
}

/**
 * @brief Print OpenMP status information at startup
 * 
 * Displays whether OpenMP is available at compile time, enabled in config,
 * and provides thread count information if available.
 * 
 * @param config Pointer to configuration structure
 */
void print_openmp_status(const Config* config) {
    printf("\n=== OpenMP Status ===\n");
    
#ifdef OPENMP_ENABLED
    printf("OpenMP Support: AVAILABLE\n");
    printf("Configuration Setting: %s\n", config->openmp_enabled ? "ENABLED" : "DISABLED");
    
    if (config->openmp_enabled) {
#ifdef _OPENMP
        int max_threads = omp_get_max_threads();
        printf("Maximum Threads: %d\n", max_threads);
        printf("OpenMP Version: %d\n", _OPENMP);
        printf("Status: OpenMP will be used for parallel computations\n");
#else
        printf("Status: OpenMP enabled but runtime not detected\n");
#endif
    } else {
        printf("Status: OpenMP disabled by configuration\n");
    }
#else
    printf("OpenMP Support: NOT AVAILABLE\n");
    printf("Configuration Setting: %s (ignored - not compiled with OpenMP)\n", 
           config->openmp_enabled ? "ENABLED" : "DISABLED");
    printf("Status: Sequential execution only\n");
#endif
    
    printf("====================\n\n");
} 