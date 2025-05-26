# Configuration System for Fluid Dynamics Simulation

This project now includes a configuration system that allows you to change simulation parameters without recompiling the code.

## How to Use

1. Edit the parameters in `config.txt` or use one of the provided configuration files.
2. Run the simulation with:

   ```
   .\Release\cnavier.exe [path_to_config_file] [output_directory_name]
   ```

   **Usage Examples:**

   - `.\Release\cnavier.exe` - Uses default config.txt and saves to `./output/default/`
   - `.\Release\cnavier.exe .\configs\stable.txt` - Uses stable.txt config and saves to `./output/default/`
   - `.\Release\cnavier.exe .\configs\stable.txt my_simulation` - Uses stable.txt config and saves to `./output/my_simulation/`
3. The output VTK files will be generated in the `output/[specified_directory_name]/` directory.
4. Open the VTK files in ParaView to visualize the simulation results.

## Organizing Multiple Simulation Runs

The output directory parameter is particularly useful for organizing different simulation runs:

```bash
# Parameter study with different Reynolds numbers
.\Release\cnavier.exe .\configs\re100.txt re100_run
.\Release\cnavier.exe .\configs\re500.txt re500_run
.\Release\cnavier.exe .\configs\re1000.txt re1000_run

# Different configurations
.\Release\cnavier.exe .\configs\stable.txt stable_test
.\Release\cnavier.exe .\configs\high_res.txt high_resolution
.\Release\cnavier.exe .\configs\no_obstacle.txt cavity_flow
```

Each simulation will create its own subdirectory under `./output/` with all the VTK files organized separately.

## Configuration Parameters

The configuration file includes the following parameters:

### Physical Parameters

- `Re`: Reynolds number - higher values create more turbulent flow (100-1000)
- `Lx`, `Ly`: Domain dimensions

### Numerical Parameters

- `nx`, `ny`: Grid resolution (higher values provide more detail but slower simulation)
- `dt`: Time step (smaller values are more stable but slower)
- `tf`: Final simulation time
- `poisson_max_it`: Maximum iterations for Poisson equation solver
- `poisson_tol`: Tolerance for Poisson equation convergence (higher values converge faster)
- `output_interval`: How often to save output files
- `poisson_type`: Solver type (1 = standard, 2 = SOR for faster convergence)

### Object Parameters

- `center_x`, `center_y`: Center position of the circular obstacle
- `radius`: Radius of the circular obstacle (use 0 for no obstacle)

## Sample Configurations

We provide several sample configurations in the `configs` directory:

1. `stable.txt`: Very stable parameters for guaranteed convergence (32x32 grid, low Reynolds number)
2. `no_obstacle.txt`: Configuration without any obstacle (pure lid-driven cavity flow)
3. `high_res.txt`: High-resolution simulation for better visualization (128x128 grid)
4. `off_center.txt`: Simulation with an off-center obstacle

## Troubleshooting

If you see the error "Maximum number of iterations achieved for Poisson equation", try these solutions:

1. Decrease the Reynolds number (`Re`)
2. Decrease the time step (`dt`)
3. Increase the Poisson tolerance (`poisson_tol`)
4. Increase the maximum Poisson iterations (`poisson_max_it`)
5. Reduce the grid resolution (`nx`, `ny`)

## Visualizing Results in ParaView

1. Open ParaView
2. Click "File > Open" and navigate to the `output` directory
3. Select the VTK files for the variables you want to visualize
4. Click "Apply" in the Properties panel
5. Choose appropriate visualization options:
   - Surface with color: Shows scalar fields like pressure or vorticity
   - Glyph with arrows: Shows vector fields like velocity
   - Streamlines: Shows flow patterns
