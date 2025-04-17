# exact_diagonalization_cpp

This project implements exact diagonalization techniques for quantum systems, focusing on the Lanczos algorithm and related methods for solving eigenvalue problems in large Hilbert spaces.

## Files Overview

### `lanczos.cpp`
This file contains the implementation of the Lanczos algorithm and its variants for solving eigenvalue problems. The Lanczos algorithm is a powerful iterative method for finding a few eigenvalues and eigenvectors of large sparse matrices. Key features include:

- **Lanczos Algorithm**: Iteratively constructs an orthonormal basis for the Krylov subspace and reduces the matrix to tridiagonal form.
- **Shift-Invert Lanczos**: Targets eigenvalues near a specific shift value for better convergence.
- **Chebyshev Filtered Lanczos**: Enhances convergence by applying a Chebyshev polynomial filter to the starting vector.
- **Thermodynamic Calculations**: Computes thermodynamic quantities like energy, entropy, and specific heat from the eigenvalue spectrum.
- **Finite Temperature Lanczos Method (FTLM)**: Estimates thermal averages and dynamical correlation functions at finite temperatures.
- **Full Diagonalization**: Provides a fallback method for small systems using LAPACK routines.

The file also includes utility functions for generating random vectors, refining eigenvectors, and handling degenerate subspaces. It supports saving and loading basis vectors and eigenvalues to/from disk for large-scale computations.

### `construct_ham.h`
This header file is intended to define the structure and functions for constructing the Hamiltonian matrix of the quantum system. It provides:

- **Hamiltonian Construction**: Functions to build the Hamiltonian matrix based on the system's parameters, such as interaction terms and external fields.
- **Operator Definitions**: Definitions for one-body and two-body operators used in the Hamiltonian.
- **File I/O**: Functions to load and save Hamiltonian data from/to files.

This file serves as the interface for creating and manipulating the Hamiltonian, which is then used by the Lanczos algorithm in `lanczos.cpp`.

## Usage
1. Construct the Hamiltonian using the functions defined in `construct_ham.h`.
2. Use the Lanczos algorithm in `lanczos.cpp` to compute eigenvalues and eigenvectors.
3. Analyze the results to extract physical properties, such as thermodynamic quantities or dynamical correlation functions.


## Dependencies
1. BLAS/LAPACK
2. CuBLAS (if you want to use lanczos_cuda.cu)
3. Eigen3
4. OpenMP
5. MPI
6. ezARPACK-ng / arpack-ng