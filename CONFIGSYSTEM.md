# Configuration System Documentation

## Overview

The fluid dynamics simulation now supports a flexible configuration system that allows you to:
- Run simulations with different parameters without recompiling
- Organize simulation outputs into custom directories
- Use configuration files for reproducible simulations
- Easily switch between different simulation scenarios

## Command Line Usage

### Basic Syntax
```bash
cnavier [config_file] [output_subfolder]
```

### Arguments

1. **config_file** (optional): Path to a configuration file containing simulation parameters
2. **output_subfolder** (optional): Name of the subfolder inside `./output/` where VTK files will be saved

### Usage Examples

```bash
# Use default configuration, save to ./output/
./cnavier

# Use custom config, save to ./output/
./cnavier my_config.txt

# Use custom config, save to ./output/my_run/
./cnavier my_config.txt my_run

# High Reynolds number simulation in organized folder
./cnavier config_high_re.txt turbulent_flow

# Quick test with default config in test folder
./cnavier "" test_folder
```

### Help Commands

```bash
# Show usage information and examples
./cnavier --help

# Show complete list of configuration parameters
./cnavier --help-config
```

## Configuration File Format

Configuration files use a simple `key = value` format with support for comments.

### File Structure
```ini
# Comments start with # or ;
# Blank lines are ignored

# Physical Parameters
Re = 1000.0
Lx = 1
Ly = 1

# Numerical Parameters  
nx = 64
ny = 64
dt = 0.005
tf = 20.0

# Boundary Conditions
u4 = 1.0  # Top boundary velocity (lid-driven cavity)
```

### Complete Parameter List

#### Physical Parameters
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `Re` | Reynolds number (controls flow regime) | 1000.0 | `Re = 5000.0` |
| `Lx` | Domain length | 1 | `Lx = 2` |
| `Ly` | Domain width | 1 | `Ly = 1` |

#### Numerical Parameters
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `nx` | Grid points in x direction | 64 | `nx = 128` |
| `ny` | Grid points in y direction | 64 | `ny = 128` |
| `dt` | Time step size | 0.005 | `dt = 0.001` |
| `tf` | Final simulation time | 20.0 | `tf = 10.0` |
| `max_co` | Maximum Courant number | 1.0 | `max_co = 0.5` |
| `order` | Finite difference order | 6 | `order = 4` |
| `poisson_max_it` | Poisson solver max iterations | 10000 | `poisson_max_it = 15000` |
| `poisson_tol` | Poisson solver tolerance | 1E-3 | `poisson_tol = 5E-4` |
| `poisson_type` | Poisson solver type (1=no relaxation, 2=SOR) | 2 | `poisson_type = 1` |
| `output_interval` | VTK output frequency | 10 | `output_interval = 5` |

#### Boundary Conditions
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `ui` | Internal u velocity field | 0.0 | `ui = 0.1` |
| `vi` | Internal v velocity field | 0.0 | `vi = 0.0` |
| `u1` | Right boundary u velocity | 0.0 | `u1 = 0.0` |
| `u2` | Left boundary u velocity | 0.0 | `u2 = 0.0` |
| `u3` | Bottom boundary u velocity | 0.0 | `u3 = 0.0` |
| `u4` | Top boundary u velocity | 1.0 | `u4 = 2.0` |
| `v1` | Right boundary v velocity | 0.0 | `v1 = 0.0` |
| `v2` | Left boundary v velocity | 0.0 | `v2 = 0.0` |
| `v3` | Bottom boundary v velocity | 0.0 | `v3 = 0.0` |
| `v4` | Top boundary v velocity | 0.0 | `v4 = 0.0` |

## Output Directory System

### How It Works

The simulation creates VTK files for visualization. The output directory system allows you to organize these files:

1. **Default behavior**: Files saved to `./output/`
2. **Custom subfolder**: Files saved to `./output/your_subfolder/`
3. **Automatic creation**: Directories are created automatically if they don't exist

### Directory Structure Example
```
project_root/
├── output/
│   ├── default_run/
│   │   ├── vorticity-1-0.vtk
│   │   ├── vorticity-1-1.vtk
│   │   └── ...
│   ├── high_re_test/
│   │   ├── vorticity-1-0.vtk
│   │   └── ...
│   └── parameter_study/
│       ├── vorticity-1-0.vtk
│       └── ...
```

## Practical Examples

### Example 1: Default Lid-Driven Cavity
```bash
# Run with all default parameters
./cnavier
```
- Uses Re=1000, 64x64 grid, saves to `./output/`
- Good for quick testing

### Example 2: High Reynolds Number Study
Create `high_re.txt`:
```ini
# High Reynolds number configuration
Re = 5000.0
nx = 128
ny = 128
dt = 0.001
tf = 10.0
max_co = 0.5
output_interval = 5
```

Run:
```bash
./cnavier high_re.txt turbulent_flow
```
- Higher resolution for accuracy
- Smaller time step for stability
- Results saved to `./output/turbulent_flow/`

### Example 3: Parameter Study
Create multiple configs and run:
```bash
./cnavier re1000.txt re1000_run
./cnavier re2000.txt re2000_run  
./cnavier re5000.txt re5000_run
```
- Organized results for comparison
- Easy to identify different runs

### Example 4: Quick Test with Modified Parameters
Create `quick_test.txt`:
```ini
# Quick test - shorter simulation
tf = 5.0
output_interval = 2
nx = 32
ny = 32
```

Run:
```bash
./cnavier quick_test.txt debug
```
- Faster execution for debugging
- Lower resolution for speed

## Configuration Best Practices

### 1. Stability Considerations
- **High Reynolds numbers**: Use smaller `dt` and higher resolution (`nx`, `ny`)
- **Courant number**: Keep `max_co ≤ 1.0` for stability
- **Grid resolution**: Higher Re typically needs finer grids

### 2. Performance Optimization
- **Time step**: Larger `dt` = faster simulation (but check stability)
- **Grid size**: Smaller grids = faster computation
- **Output frequency**: Higher `output_interval` = less I/O overhead

### 3. File Organization
- Use descriptive subfolder names: `re5000_128x128`, `parameter_study_1`
- Include key parameters in folder names for easy identification
- Separate different studies into different folders

### 4. Reproducibility
- Save configuration files with descriptive names
- Include comments in config files explaining the purpose
- Document parameter choices and expected outcomes

## Troubleshooting

### Common Issues

1. **"Unstable Solution!" error**
   - Reduce `dt` (time step)
   - Reduce `max_co` (Courant number limit)
   - Check Reynolds number vs grid resolution

2. **Slow convergence**
   - Increase `poisson_max_it`
   - Decrease `poisson_tol`
   - Try different `poisson_type`

3. **File creation errors**
   - Check write permissions in output directory
   - Ensure sufficient disk space
   - Verify path separators (use `/` or `\` as appropriate)

### Performance Tips

1. **For quick tests**: Use smaller grids (32x32) and shorter times
2. **For production runs**: Use appropriate resolution for your Reynolds number
3. **For parameter studies**: Automate with scripts to run multiple configurations

## Integration with Visualization

The VTK files can be opened with:
- **ParaView**: Professional visualization software
- **VisIt**: Scientific visualization tool  
- **VTK-based tools**: Custom visualization applications

Example ParaView workflow:
1. Open ParaView
2. File → Open → Navigate to `./output/your_subfolder/`
3. Select `vorticity-1-*.vtk` files
4. Apply → Play animation to see time evolution

## Advanced Usage

### Batch Processing
Create a script to run multiple configurations:
```bash
#!/bin/bash
for re in 1000 2000 5000; do
    ./cnavier config_re${re}.txt re${re}_study
done
```

### Configuration Templates
Create base configurations and modify specific parameters:
```ini
# base_config.txt - template
Re = 1000.0
nx = 64
ny = 64
# ... other parameters

# high_res_config.txt - inherits from base, overrides resolution
nx = 128  
ny = 128
```

This configuration system provides the flexibility to explore different flow regimes, conduct parameter studies, and organize simulation results efficiently without requiring code recompilation. 