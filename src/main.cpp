#include <iostream>
// #include "solver/fluid_solver.hpp"
// #include "io/data_writer.hpp"

// int main(int argc, char** argv) {
//     try {
//         std::cout << "Initializing Fluid Dynamics Solver..." << std::endl;

//         // Create solver instance
//         FluidSolver solver;
        
//         // Initialize simulation parameters
//         solver.initialize();

//         // Main simulation loop
//         for (int step = 0; step < solver.getMaxTimeSteps(); ++step) {
//             solver.solve();
            
//             if (step % solver.getOutputFrequency() == 0) {
//                 DataWriter::writeSnapshot(solver.getState(), step);
//             }
//         }

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// } 

int main() {
    std::cout << "Hello from Fluid Dynamics Solver!" << std::endl;
    return 0;
} 