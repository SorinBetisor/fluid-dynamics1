#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "utils.h"

#ifndef PI
#define PI 3.14159265359
#endif

Config load_default_config(void) {
    Config config;
    
    // Physical parameters (default values from original main.c)
    config.Re = 1000.0;
    config.Lx = 1;
    config.Ly = 1;
    
    // Numerical parameters
    config.nx = 64;
    config.ny = 64;
    config.dt = 0.005;
    config.tf = 20.0;
    config.max_co = 1.0;
    config.order = 6;
    config.poisson_max_it = 10000;
    config.poisson_tol = 1E-3;
    config.output_interval = 10;
    config.poisson_type = 2;
    
    // Output settings
    strcpy(config.output_dir, "./output");
    
    // Boundary conditions (Dirichlet)
    config.ui = 0.0;
    config.vi = 0.0;
    config.u1 = 0.0;  // right boundary condition
    config.u2 = 0.0;  // left boundary condition
    config.u3 = 0.0;  // bottom boundary condition
    config.u4 = 1.0;  // top boundary condition
    config.v1 = 0.0;
    config.v2 = 0.0;
    config.v3 = 0.0;
    config.v4 = 0.0;
    
    return config;
}

Config load_config_from_file(const char* filename) {
    Config config = load_default_config(); // Start with defaults
    FILE* file;
    char line[256];
    char key[64];
    char value[64];
    
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open configuration file '%s'\n", filename);
        printf("Using default configuration.\n");
        return config;
    }
    
    printf("Loading configuration from: %s\n", filename);
    
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
            // Output settings
            else if (strcmp(key_trimmed, "output_dir") == 0) {
                strncpy(config.output_dir, value_trimmed, sizeof(config.output_dir) - 1);
                config.output_dir[sizeof(config.output_dir) - 1] = '\0';
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
    printf("  Output directory: %s\n", config->output_dir);
    
    printf("\nBoundary Conditions:\n");
    printf("  Internal u field (ui): %.2f\n", config->ui);
    printf("  Internal v field (vi): %.2f\n", config->vi);
    printf("  u boundaries (u1,u2,u3,u4): %.2f, %.2f, %.2f, %.2f\n", 
           config->u1, config->u2, config->u3, config->u4);
    printf("  v boundaries (v1,v2,v3,v4): %.2f, %.2f, %.2f, %.2f\n", 
           config->v1, config->v2, config->v3, config->v4);
    
    // Calculate derived parameters
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

void print_usage(const char* program_name) {
    printf("Usage: %s [config_file] [output_subfolder]\n", program_name);
    printf("\nOptions:\n");
    printf("  config_file       Path to configuration file (optional)\n");
    printf("                    If not provided, default values will be used\n");
    printf("  output_subfolder  Name of subfolder inside ./output/ for VTK files (optional)\n");
    printf("                    If not provided, files will be saved to ./output/\n");
    printf("\nExamples:\n");
    printf("  %s                           # Use defaults, save to ./output/\n", program_name);
    printf("  %s config.txt               # Use config.txt, save to ./output/\n", program_name);
    printf("  %s config.txt run1          # Use config.txt, save to ./output/run1/\n", program_name);
    printf("  %s config_high_re.txt turbulent  # High Re config, save to ./output/turbulent/\n", program_name);
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

void set_output_directory(Config* config, const char* subfolder) {
    if (subfolder == NULL || strlen(subfolder) == 0) {
        strcpy(config->output_dir, "./output");
        return;
    }
    
    // Create the output path: ./output/subfolder
    snprintf(config->output_dir, sizeof(config->output_dir), "./output/%s", subfolder);
    
    // Remove any trailing slashes
    int len = strlen(config->output_dir);
    while (len > 0 && (config->output_dir[len-1] == '/' || config->output_dir[len-1] == '\\')) {
        config->output_dir[len-1] = '\0';
        len--;
    }
} 