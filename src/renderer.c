#include "fluidRenderer.h"
#include "opengl_loader.h"
#include "vertexShaderSource.h"
#include "fragmentShaderSource.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <GLFW/glfw3.h>

// Global variables
static GLFWwindow* window = NULL;
static unsigned int shaderProgram = 0;
static unsigned int VAO = 0, VBO = 0, EBO = 0;
static unsigned int texture = 0;
static int windowWidth = 800, windowHeight = 800;

// Compile a shader
unsigned int compileShader(GLenum type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::COMPILATION_FAILED\n%s\n", infoLog);
        return 0;
    }
    return shader;
}

// Initialize the OpenGL renderer
int fluid_renderer_init(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 0;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create a window
    window = glfwCreateWindow(windowWidth, windowHeight, "Fluid Simulation", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 0;
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize GLAD
    if (!initialize_glad()) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    }
    
    // Compile vertex shader
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    if (!vertexShader) return 0;
    
    // Compile fragment shader
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    if (!fragmentShader) {
        glDeleteShader(vertexShader);
        return 0;
    }
    
    // Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }
    
    // Delete shaders as they're linked into our program now
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Set up vertex data (a square to display our texture)
    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Create a texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    return 1;
}

// Cleanup the OpenGL renderer resources
void fluid_renderer_cleanup(void) {
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    if (texture) glDeleteTextures(1, &texture);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    
    if (window) {
        glfwDestroyWindow(window);
        window = NULL;
    }
    glfwTerminate();
}

// Draw a frame with the current fluid state
void fluid_renderer_draw_frame(Fluid *fluid) {
    if (!window || glfwWindowShouldClose(window)) return;
    
    int N = fluid->gridSize;
    
    // Process events
    glfwPollEvents();
    
    // Clear the screen
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Find maximum density for normalization
    float maxDensity = 0.0f;
    for (int i = 0; i < fluid->size; i++) {
        if (fluid->density[i] > maxDensity) {
            maxDensity = fluid->density[i];
        }
    }
    if (maxDensity <= 0.0f) maxDensity = 1.0f;
    
    // Create a normalized density texture
    unsigned char* textureData = (unsigned char*)malloc(N * N * 3 * sizeof(unsigned char));
    if (!textureData) {
        fprintf(stderr, "Failed to allocate texture memory\n");
        return;
    }
    
    // Fill texture with density data
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int idx = (i + j * N) * 3;
            float normDensity = fminf(fluid->density[i + j * N] / maxDensity, 1.0f);
            textureData[idx] = (unsigned char)(normDensity * 255);     // Red
            textureData[idx + 1] = (unsigned char)(normDensity * 255); // Green
            textureData[idx + 2] = (unsigned char)(normDensity * 255); // Blue
        }
    }
    
    // Draw airfoil boundary in bright red
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (fluid->u[IX(i, j, N)] == 0 && fluid->v[IX(i, j, N)] == 0 && fluid->density[IX(i, j, N)] == 0) {
                // This is likely the airfoil as we set velocity and density to 0 in object_apply
                int idx = (i + j * N) * 3;
                textureData[idx] = 255;     // Red (bright red for airfoil)
                textureData[idx + 1] = 50;  // Green
                textureData[idx + 2] = 50;  // Blue
            }
        }
    }
    
    // Update the texture with the RGB data
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, N, N, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData);
    
    // Use the shader program
    glUseProgram(shaderProgram);
    
    // Render the quad
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    
    // Swap buffers
    glfwSwapBuffers(window);
    
    free(textureData);
}

// The original PPM writer function (kept for compatibility)
void fluid_renderer_draw(Fluid *fluid, const char *filename) {
    int N = fluid->gridSize;
    // Allocate an image buffer (RGB for each pixel).
    unsigned char *image = (unsigned char *)malloc(N * N * 3);
    if (!image) {
        perror("Failed to allocate image buffer");
        return;
    }

    // Determine the maximum density for normalization (avoid division by zero).
    float maxDensity = 0.0f;
    for (int i = 0; i < fluid->size; i++) {
        if (fluid->density[i] > maxDensity)
            maxDensity = fluid->density[i];
    }
    if (maxDensity <= 0.0f)
        maxDensity = 1.0f;

    // Fill the background: map the density to a grayscale value.
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int index = i + j * N;
            float normDensity = fminf(fluid->density[index] / maxDensity, 1.0f);
            unsigned char c = (unsigned char)(normDensity * 255);
            image[3 * index + 0] = c;
            image[3 * index + 1] = c;
            image[3 * index + 2] = c;
        }
    }

    // Write the image buffer to a PPM file.
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for writing");
        free(image);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", N, N);
    fwrite(image, sizeof(unsigned char), N * N * 3, fp);
    fclose(fp);
    free(image);
}
