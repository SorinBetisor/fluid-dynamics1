#include "shader.h"

Shaderfile readShaderFile(const char* filename)
{
    FILE* pfile = fopen(filename, "rb");
    Shaderfile shader = {0};

    if (!pfile) {
        printf("failed to read the file %s \n", filename);
        abort();
    }
    fseek(pfile, 0L, SEEK_END);
    shader.size = ftell(pfile);
    fseek(pfile, 0L, SEEK_SET);

    shader.code = malloc(sizeof(char) * (shader.size));
    fread(shader.code, shader.size, sizeof(char), pfile);

    fclose(pfile);
    return shader;
}
