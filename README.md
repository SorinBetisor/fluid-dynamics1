# Fluid Dynamics Neural Network Accelerator

This project implements a hybrid fluid dynamics simulation system combining traditional PDE solvers with neural network acceleration.

## Prerequisites

### Windows Setup

1. **Install Visual Studio Build Tools**

   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select:
     - MSVC C++ build tools
     - Windows SDK
     - C++ CMake tools for Windows
     - C++ ATL for latest build tools
     - C++ MFC for latest build tools

2. **Install CUDA Toolkit** (for GPU acceleration)

   - Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Add CUDA to system PATH

3. **Install Python Dependencies**

   - Download [Python](https://www.python.org/downloads/) (3.8 or later)
   - Install required packages:
     ```bash
     pip install torch torchvision tensorflow numpy h5py mpi4py
     ```

4. **Install OpenFOAM**
   - Download and install [OpenFOAM for Windows](https://openfoam.org/download/windows/)
   - Add OpenFOAM bin directory to system PATH

### macOS Setup

1. **Install Xcode Command Line Tools**

   ```bash
   xcode-select --install
   ```

2. **Install Homebrew**

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install Required Dependencies**

   ```bash
   brew install cmake
   brew install open-mpi
   brew install hdf5
   brew install openfoam
   ```

4. **Install Python Dependencies**
   ```bash
   pip3 install torch torchvision tensorflow numpy h5py mpi4py
   ```

### Linux setup (Arch in particular)

1. **Install build utilities**
   ```bash
   sudo pacman -S base-devel cmake 
   ```
2. **Install necessery python packages to the Virtual env**
   ```bash
   python -m venv pyenv
   source myenv/bin/activate

   pip install --upgrade pip
   pip install torch torchvision tensorflow numpy h5py mpi4py
   ```

3. **Install openFOAM**
   ```bash
   yay -S openfoam-com
   ```

## Project Structure

```
project/
├── src/
│   ├── pde_solver/      # C++ implementation of PDE solver
│   ├── neural_net/      # Python NN implementation
│   └── utils/           # Shared utilities
├── data/                # Training data storage
├── tests/               # Unit tests
└── docs/               # Documentation
```

## Building the Project

### Windows

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### macOS

```bash
mkdir build
cd build
cmake ..
make
```

### linux
```bash
./build.sh
```

## Running Tests

```bash
cd build
ctest
```

## Project Phases

### Phase 1: Baseline Simulation & Data Generation

- Implementation of fluid dynamics PDE solver
- Data collection and storage
- Timeline: 2-3 weeks

### Phase 2: Offline Neural Network Training

- PINN or CNN/RNN surrogate model implementation
- Model training and validation
- Timeline: 3-4 weeks

### Phase 3: In-Situ Integration

- Live data transfer implementation
- Performance optimization
- Timeline: 4-5 weeks

### Phase 4: Hybrid Simulation & Adaptive Refinement

- Integration of NN predictions with PDE solver
- Performance benchmarking
- Timeline: 3-4 weeks

## Tools & Technologies

- **Languages**: C++17 or later, Python 3.8+
- **Libraries**: OpenFOAM/MFEM/deal.II, PyTorch, TensorFlow
- **Parallel Computing**: MPI, OpenMP, CUDA
- **Profiling**: VTune, NVIDIA Nsight, perf

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
