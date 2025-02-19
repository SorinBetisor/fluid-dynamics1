#include "fluidRenderer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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

    // Define the maximum particle speed for color mapping.
    const float maxParticleSpeed = 2.0f;  // adjust as needed


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
