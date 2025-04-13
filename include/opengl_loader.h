#ifndef OPENGL_LOADER_H
#define OPENGL_LOADER_H

#include <stdio.h>
#include <stdlib.h>

// Silence macOS OpenGL deprecation warnings
#define GL_SILENCE_DEPRECATION

// Use OpenGL directly from the system on Mac
#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/gl3.h>
    #include <GLFW/glfw3.h>
    
    // Define Apple-specific versions of functions
    #define glGenVertexArrays glGenVertexArraysAPPLE
    #define glBindVertexArray glBindVertexArrayAPPLE
    #define glDeleteVertexArrays glDeleteVertexArraysAPPLE
    
    // Simple initialization function
    static int initialize_glad(void) {
        // No initialization needed on Mac
        return 1;
    }
#else
    // For other platforms, conditionally use GLAD if available
    #ifdef HAVE_GLAD
        #include <glad/glad.h>
        #include <GLFW/glfw3.h>
        
        // Function to initialize GLAD
        static int initialize_glad(void) {
            // Initialize GLAD
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                fprintf(stderr, "Failed to initialize GLAD\n");
                return 0;
            }
            return 1;
        }
    #else
        // Fallback for systems without GLAD
        #include <GLFW/glfw3.h>
        
        static int initialize_glad(void) {
            fprintf(stderr, "Warning: GLAD not available, OpenGL functionality may be limited\n");
            return 1;
        }
    #endif
#endif

#endif // OPENGL_LOADER_H 