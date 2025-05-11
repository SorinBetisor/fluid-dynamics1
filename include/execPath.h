#pragma once

#include <string>

// Returns the absolute path of the directory containing the running executable.
// Throws std::runtime_error on failure.
std::string getExecutableDir();
