# Exact Diagonalization C++

A C++ library for exact diagonalization of spin systems with comprehensive symmetry analysis capabilities. This project provides efficient tools for studying quantum Hamiltonians with a focus on symmetry exploitation to reduce computational complexity.

## Key Features

### Advanced Symmetry Analysis
- **Automatic symmetry detection** - Identifies all automorphisms of a given Hamiltonian
- **Symmetry visualization** - Creates graphical representations of symmetry structures
- **Minimal generator finding** - Reduces symmetry groups to their minimal generating sets
- **Clique analysis** - Identifies maximal sets of compatible symmetries that can be simultaneously exploited
- **Symmetrized basis construction** - Generates symmetry-adapted basis states to block-diagonalize Hamiltonians

### Diagonalization Methods
- Support for multiple diagonalization algorithms:
    - Full diagonalization for smaller systems
    - Various Lanczos algorithm and their variants
    - Various CG algorithms and their variants
    - Support for symmetric, asymmetric, and complex eigenproblems
