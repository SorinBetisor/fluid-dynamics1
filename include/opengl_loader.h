#ifndef OPENGL_LOADER_H
#define OPENGL_LOADER_H

#include <stdio.h>
#include <stdlib.h>

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

#endif // OPENGL_LOADER_H 