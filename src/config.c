#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

// Trim whitespace from a string
static char* trim(char* str) {
    char* end;
    
    // Trim leading space
    while(*str == ' ' || *str == '\t') str++;
    
    if(*str == 0)  // All spaces?
        return str;
    
    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end--;
    
    // Write new null terminator
    *(end + 1) = 0;
    
    return str;
}

// Parse the configuration file
Config parseConfigFile(const char* filename) {
    FILE* file;
    char line[256];
    char key[128], value_str[128];
    double value;
    Config config;
    
    // Set default values
    config.Re = 1000.0;
    config.Lx = 1;
    config.Ly = 1;
    config.nx = 128;
    config.ny = 128;
    config.dt = 0.1;
    config.tf = 15.0;
    config.max_co = 1.0;
    config.order = 6;
    config.poisson_max_it = 10000;
    config.poisson_tol = 0.001;
    config.output_interval = 10;
    config.poisson_type = 2;
    config.use_omp = 1;
    config.use_gpu = 1;
    config.use_vulkan = 0;
    config.center_x = 0.5;
    config.center_y = 0.5;
    config.radius = 0.1;
    
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Config file %s not found. Using default values.\n", filename);
        return config;
    }
    
    printf("Reading configuration from %s\n", filename);
    
    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        // Skip whitespace
        char* trimmed_line = trim(line);
        if (strlen(trimmed_line) == 0) {
            continue;
        }
        
        // Find the equals sign
        char* equals = strchr(trimmed_line, '=');
        if (equals == NULL) {
            continue; // No equals sign, skip this line
        }
        
        // Split key and value
        *equals = '\0'; // Replace equals with null terminator
        strcpy(key, trim(trimmed_line)); // Get key
        
        // Get value (everything after the equals sign)
        char* value_part = trim(equals + 1);
        
        // Remove any trailing comments
        char* comment = strchr(value_part, '#');
        if (comment != NULL) {
            *comment = '\0'; // Remove comment
            value_part = trim(value_part); // Trim again after removing comment
        }
        
        // Convert to double/int
        value = atof(value_part);
        
        // Set the appropriate config value
        if (strcmp(key, "Re") == 0) config.Re = value;
        else if (strcmp(key, "Lx") == 0) config.Lx = (int)value;
        else if (strcmp(key, "Ly") == 0) config.Ly = (int)value;
        else if (strcmp(key, "nx") == 0) config.nx = (int)value;
        else if (strcmp(key, "ny") == 0) config.ny = (int)value;
        else if (strcmp(key, "dt") == 0) config.dt = value;
        else if (strcmp(key, "tf") == 0) config.tf = value;
        else if (strcmp(key, "max_co") == 0) config.max_co = value;
        else if (strcmp(key, "order") == 0) config.order = (int)value;
        else if (strcmp(key, "poisson_max_it") == 0) config.poisson_max_it = (int)value;
        else if (strcmp(key, "poisson_tol") == 0) config.poisson_tol = value;
        else if (strcmp(key, "output_interval") == 0) config.output_interval = (int)value;
        else if (strcmp(key, "poisson_type") == 0) config.poisson_type = (int)value;
        else if (strcmp(key, "use_omp") == 0) config.use_omp = (int)value;
        else if (strcmp(key, "use_gpu") == 0) config.use_gpu = (int)value;
        else if (strcmp(key, "use_vulkan") == 0) config.use_vulkan = (int)value;
        else if (strcmp(key, "center_x") == 0) config.center_x = value;
        else if (strcmp(key, "center_y") == 0) config.center_y = value;
        else if (strcmp(key, "radius") == 0) config.radius = value;
        else {
            printf("Unknown config parameter: %s\n", key);
        }
    }
    
    fclose(file);
    
    // Print loaded configuration
    printf("Configuration loaded:\n");
    printf("Reynolds number: %.1f\n", config.Re);
    printf("Grid: %d x %d\n", config.nx, config.ny);
    printf("Time step: %.3f, Final time: %.1f\n", config.dt, config.tf);
    printf("Object at (%.2f, %.2f) with radius %.2f\n", config.center_x, config.center_y, config.radius);
    
    return config;
} 