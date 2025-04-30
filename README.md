# cnavier

Explicit incompressible Navier-Stokes solver written in C

## Overview

This code solves the incompressible Navier-Stokes equations using a vorticity-streamfunction formulation with an explicit finite difference method. It simulates fluid flow in a 2D cavity with a potential circular obstacle, demonstrating fundamental fluid dynamics principles.

### Key Features

- Lid-driven cavity flow simulation
- Optional circular obstacle with customizable position and size
- Configurable Reynolds number and grid resolution
- Parallelization with OpenMP
- Optional GPU acceleration support (OpenGL/Vulkan)
- VTK output format for easy visualization with ParaView

## Building the Code

### On Windows

```
mkdir build_win
cd build_win
cmake ..
cmake --build . --config Release
```

### On Linux/macOS

```
mkdir build
cd build
cmake ..
make
```

## Running the Simulation

The simulation can be run with default parameters using:

```
./cnavier
```

Or with a specific configuration file:

```
./cnavier configs/stable.txt
```

## Configuration System

The project includes a configuration system that allows changing simulation parameters without recompiling the code. Sample configurations are provided in the `configs` directory:

- `stable.txt`: Very stable parameters for guaranteed convergence
- `no_obstacle.txt`: Configuration without any obstacle (pure lid-driven cavity flow)
- `high_res.txt`: High-resolution simulation for better visualization
- `off_center.txt`: Simulation with an off-center obstacle

See [README_CONFIG.md](README_CONFIG.md) for detailed information about the configuration system.

## Main Parameters

- **Reynolds Number (`Re`)**: Controls the flow regime from laminar to turbulent
- **Grid Resolution (`nx`, `ny`)**: Higher values provide more accurate results but slower simulation
- **Time Step (`dt`)**: Smaller values provide better stability
- **Object Position/Size**: Control the position and size of the circular obstacle

## Visualization

Output files are generated in VTK format in the `output` directory. These can be visualized using ParaView:

1. Open ParaView
2. File > Open > Select the output files
3. Click "Apply" in the Properties panel
4. Choose appropriate visualization options

## Physics

The simulation solves for:

- Vorticity (rotation of the fluid)
- Stream function (flow patterns)
- Velocity components (x and y directions)

The lid-driven cavity problem simulates fluid in a square cavity with the top wall moving at a constant velocity, creating a characteristic circulating flow pattern.

## Troubleshooting

If the simulation fails with "Maximum number of iterations achieved for Poisson equation", try:

1. Decreasing the Reynolds number
2. Decreasing the time step
3. Increasing the Poisson tolerance
4. Using one of the stable configurations provided
