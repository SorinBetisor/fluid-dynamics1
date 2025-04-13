#ifndef FRAGMENT_SHADER_SOURCE_H
#define FRAGMENT_SHADER_SOURCE_H

// Fragment shader source for RGB texture
const char* fragmentShaderSource = 
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D densityMap;\n"
    "void main()\n"
    "{\n"
    "   // Sample texture and add color adjustment for visibility\n"
    "   vec3 color = texture(densityMap, TexCoord).rgb;\n"
    "   \n"
    "   // Enhance contrast\n"
    "   color = pow(color, vec3(0.8));\n"
    "   \n"
    "   // Output the final color\n"
    "   FragColor = vec4(color, 1.0);\n"
    "}\0";

#endif // FRAGMENT_SHADER_SOURCE_H 