# Configuration System Documentation

This document explains how to use the command line arguments and configuration system for the fluid dynamics simulation.

## Command Line Usage

```bash
./cnavier [config_file] [output_folder]
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `config_file` | String | No | Path to configuration file. If not provided, default values are used |
| `output_folder` | String | No | Name of output subfolder. Files will be saved to `./output/[output_folder]/` |

### Examples

```bash
# Use default configuration, save to ./output/
./cnavier

# Use custom config file, save to ./output/
./cnavier config_default.txt

# Use custom config file, save to ./output/run1/
./cnavier config_default.txt run1

# Use custom config file, save to ./output/high_reynolds/
./cnavier config_high_re.txt high_reynolds

# Get help information
./cnavier --help
./cnavier --help-config
```

## Output Directory Structure

```
output/
├── here.txt                    # Reference file
├── run1/                       # Custom run folder
│   ├── vorticity-1-0.vtk
│   ├── vorticity-1-1.vtk
│   └── ...
├── baseline/                   # Another custom run
│   ├── vorticity-1-0.vtk
│   └── ...
└── vorticity-1-0.vtk          # Default location (no subfolder specified)
```

## Configuration File Format

Configuration files use a simple key-value format:

```ini
# Comments start with # or ;
# Empty lines are ignored

# Physical parameters
Re = 1000.0
Lx = 1
Ly = 1

# Numerical parameters
nx = 64
ny = 64
dt = 0.005
tf = 20.0
```

### Complete Parameter Reference

#### Physical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Re` | double | 1000.0 | Reynolds number - controls flow regime |
| `Lx` | int | 1 | Domain length (non-dimensional) |
| `Ly` | int | 1 | Domain width (non-dimensional) |

#### Numerical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nx` | int | 64 | Grid points in x direction |
| `ny` | int | 64 | Grid points in y direction |
| `dt` | double | 0.005 | Time step size |
| `tf` | double | 20.0 | Final simulation time |
| `max_co` | double | 1.0 | Maximum allowable Courant number |
| `order` | int | 6 | Finite difference order (2, 4, or 6) |
| `poisson_max_it` | int | 10000 | Maximum Poisson solver iterations |
| `poisson_tol` | double | 1E-3 | Poisson solver convergence tolerance |
| `output_interval` | int | 10 | VTK output frequency (every N iterations) |
| `poisson_type` | int | 2 | Poisson solver type: 1=basic, 2=SOR |

#### Boundary Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ui` | double | 0.0 | Initial internal u-velocity |
| `vi` | double | 0.0 | Initial internal v-velocity |
| `u1` | double | 0.0 | Right boundary u-velocity |
| `u2` | double | 0.0 | Left boundary u-velocity |
| `u3` | double | 0.0 | Bottom boundary u-velocity |
| `u4` | double | 1.0 | Top boundary u-velocity (moving lid) |
| `v1` | double | 0.0 | Right boundary v-velocity |
| `v2` | double | 0.0 | Left boundary v-velocity |
| `v3` | double | 0.0 | Bottom boundary v-velocity |
| `v4` | double | 0.0 | Top boundary v-velocity |

## Example Configuration Files

### Default Lid-Driven Cavity

```ini
# config_default.txt
# Standard lid-driven cavity flow at Re=1000

# Physical parameters
Re = 1000.0
Lx = 1
Ly = 1

# Numerical parameters
nx = 64
ny = 64
dt = 0.005
tf = 20.0
max_co = 1.0
order = 6
poisson_max_it = 10000
poisson_tol = 1E-3
output_interval = 10
poisson_type = 2

# Boundary conditions (lid-driven cavity)
ui = 0.0
vi = 0.0
u1 = 0.0  # right wall
u2 = 0.0  # left wall  
u3 = 0.0  # bottom wall
u4 = 1.0  # top wall (moving lid)
v1 = 0.0  # right wall
v2 = 0.0  # left wall
v3 = 0.0  # bottom wall
v4 = 0.0  # top wall
```

### High Reynolds Number Study

```ini
# config_high_re.txt
# High Reynolds number flow for turbulence study

# Physical parameters
Re = 5000.0
Lx = 1
Ly = 1

# Numerical parameters (finer grid for stability)
nx = 128
ny = 128
dt = 0.001
tf = 50.0
max_co = 0.5
order = 6
poisson_max_it = 15000
poisson_tol = 1E-4
output_interval = 50
poisson_type = 2

# Boundary conditions
ui = 0.0
vi = 0.0
u1 = 0.0
u2 = 0.0
u3 = 0.0
u4 = 1.0
v1 = 0.0
v2 = 0.0
v3 = 0.0
v4 = 0.0
```

## Usage Workflows

### Running Parameter Studies

```bash
# Compare different Reynolds numbers
./cnavier config_re100.txt re100_study
./cnavier config_re1000.txt re1000_study  
./cnavier config_re5000.txt re5000_study

# Compare different grid resolutions
./cnavier config_64x64.txt grid_64
./cnavier config_128x128.txt grid_128
./cnavier config_256x256.txt grid_256
```

### Organizing Results

Each run creates its own subfolder, making it easy to:
- Compare results from different configurations
- Archive successful runs
- Organize parameter studies
- Prevent overwriting previous results

### Error Handling

- **Missing config file**: Program continues with default values and shows warning
- **Invalid parameters**: Program shows warning but continues with default for that parameter
- **Missing output directory**: Automatically created
- **Invalid Courant number**: Program exits with stability warning

## Tips

1. **Start with defaults**: Use `./cnavier` first to verify the simulation works
2. **Small changes**: Modify one parameter at a time when studying effects
3. **Stability**: Lower `dt` or `max_co` if simulation becomes unstable
4. **Output frequency**: Adjust `output_interval` based on simulation length
5. **Naming**: Use descriptive output folder names like `re1000_fine_grid`

## Getting Help

```bash
# Show usage information
./cnavier --help

# Show all available parameters
./cnavier --help-config
``` 