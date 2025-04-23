# Exact Diagonalization C++

A C++ library for exact diagonalization of quantum many-body systems with comprehensive symmetry analysis capabilities. This project provides efficient tools for studying quantum Hamiltonians with a focus on symmetry exploitation to reduce computational complexity.

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
    - Lanczos algorithm for sparse Hamiltonians (finding extremal eigenvalues)
    - Integration with ARPACK for high-performance sparse matrix eigensolvers
    - Support for symmetric, asymmetric, and complex eigenproblems

### Visualization Tools
- Hamiltonian structure visualization with GraphViz integration
- Symmetry group visualization and analysis
- Interactive exploration of symmetry relationships

### Input/Output
- Flexible file format support for Hamiltonian specification
- Symmetry-block output for further analysis
- Support for standard quantum many-body model formats

## Performance Optimization

This library dramatically reduces computational requirements for exact diagonalization by:
1. Leveraging symmetries to block-diagonalize Hamiltonians
2. Employing sparse matrix techniques for memory efficiency
3. Utilizing high-performance ARPACK routines for eigenvalue calculations
4. Providing parallelized implementations for multi-core systems

## Applications

Ideal for studying:
- Quantum spin systems and magnetic materials
- Strongly correlated electronic systems
- Topological phases of matter
- Quantum circuits and algorithms
- Non-equilibrium quantum dynamics