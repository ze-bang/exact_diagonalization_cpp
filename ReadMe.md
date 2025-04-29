# Exact Diagonalization C++ Library

A high-performance C++ library for exact diagonalization of quantum spin systems, featuring symmetry-based optimizations and numerical linked cluster expansion (NLCE) capabilities.

## Features

- Multiple diagonalization algorithms: Lanczos, full matrix, Krylov-Schur, Davidson, etc.
- Symmetry-based optimizations for reduced computational requirements
- Thermal property calculations including energy, specific heat, and entropy
- NLCE workflow for thermodynamic properties of extended systems
- Parallel execution support for improved performance

## Prerequisites

- C++ compiler with C++17 support (GCC 8+, Clang 9+)
- CMake 3.10+
- BLAS/LAPACK libraries
- Python 3.6+ (for NLCE and utility scripts)
- Python dependencies:
    - NumPy
    - Matplotlib
    - tqdm
    - SciPy
    - networkx

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/exact_diagonalization_cpp.git
cd exact_diagonalization_cpp

# Create build directory
mkdir build && cd build


```

## Symmetrized Exact Diagonalization

The symmetrized ED approach exploits the symmetries of quantum systems to significantly reduce computational requirements:

1. The code identifies automorphisms (symmetries) of the input Hamiltonian
2. Generates symmetrized basis states that transform according to irreducible representations
3. Block-diagonalizes the Hamiltonian into smaller independent matrices
4. Diagonalizes each block separately, improving performance and memory usage

To use symmetrized ED:

```bash
./build/ED input_directory --symmetrized --method=FULL --output=results
```

The symmetrized basis is generated automatically the first time, then stored for reuse.

## Numerical Linked Cluster Expansion (NLCE)

The NLCE workflow consists of:

1. Cluster generation for the lattice type
2. Hamiltonian preparation for each cluster
3. ED calculations for each cluster
4. Summation with proper weights to get thermodynamic properties

```bash
python3 util/nlce.py --max_order=4 --base_dir=./results --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo
```

Advanced resummation techniques like Euler resummation can improve convergence for critical systems.

## Example Use Cases

### 1. Basic ED for a small system

```bash
./build/ED path/to/hamiltonian --method=LANCZOS --eigenvalues=10 --output=output_dir
```

### 2. Symmetrized ED for larger systems

```bash
./build/ED path/to/hamiltonian --symmetrized --method=FULL --output=symmetrized_results
```

### 3. Thermodynamic properties calculation

```bash
./build/ED path/to/hamiltonian --method=FULL --thermo --temp-min=0.001 --temp-max=20.0 --temp-bins=100
```

### 4. NLCE calculation with parallel execution

```bash
python3 util/nlce.py --max_order=4 --base_dir=./pyrochlore_results \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --h=0.0 --field_dir=0 0 1 \
    --parallel --num_cores=4 --thermo --euler_resum
```

### 5. Parameter fitting using NLCE results

```bash
python3 util/nlc_fit.py --exp_data=measurements.txt --max_order=3 \
    --temp_min=0.1 --temp_max=10.0 --h=0.0 --field_dir=0 0 1
```

### 6. Convergence analysis with increasing NLCE order

```bash
python3 util/nlc_convergence.py --max_order=6 --base_dir=./convergence_study \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --method=FULL --thermo
```

## Command Line Arguments

### ED Executable

- `--method`: Diagonalization method (LANCZOS, FULL, KRYLOV_SCHUR, etc.)
- `--eigenvalues`: Number of eigenvalues to compute
- `--symmetrized`: Use symmetry analysis to reduce computation
- `--thermo`: Calculate thermodynamic properties
- `--temp-min/max/bins`: Temperature range and resolution
- `--output`: Output directory

### NLCE Script

- `--max_order`: Maximum cluster order to include
- `--base_dir`: Base directory for all output
- `--Jxx/Jyy/Jzz`: Heisenberg coupling parameters
- `--h`: Magnetic field strength
- `--field_dir`: Magnetic field direction
- `--euler_resum`: Use Euler resummation technique
- `--parallel`: Enable parallel processing

## References

- Documentation and tutorials are available in the `docs` folder
- API reference is available through Doxygen-generated HTML
- Examples are provided in the `examples` directory