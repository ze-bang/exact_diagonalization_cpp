// lanczos_gpu.cuh - GPU-accelerated Lanczos algorithm
#ifndef LANCZOS_GPU_CUH
#define LANCZOS_GPU_CUH

#include "gpu_hamiltonian.cuh"
#include "gpu_vector.cuh"
#include <vector>
#include <string>
#include <functional>

namespace gpu {

/**
 * GPU-accelerated Lanczos algorithm for finding lowest eigenvalues
 * 
 * @param H GPU Hamiltonian operator
 * @param max_iter Maximum number of Lanczos iterations
 * @param num_eigs Number of eigenvalues to compute
 * @param tol Convergence tolerance
 * @param eigenvalues Output eigenvalues
 * @param dir Directory to save eigenvectors (if needed)
 * @param compute_eigenvectors Whether to compute and save eigenvectors
 */
void lanczos_gpu(
    GPUHamiltonianOperator& H,
    int max_iter,
    int num_eigs,
    double tol,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_eigenvectors = false
);

/**
 * GPU-accelerated Lanczos for fixed Sz sector
 * 
 * Uses the fixed Sz operator for efficient calculation
 */
void lanczos_fixed_sz_gpu(
    GPUFixedSzOperator& H,
    int max_iter,
    int num_eigs,
    double tol,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_eigenvectors = false
);

/**
 * Solve tridiagonal eigenvalue problem on CPU
 * (Same as CPU version, called from GPU code)
 */
int solve_tridiagonal_eigenvalue_problem(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    int num_eigs,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors,
    bool compute_eigenvectors = false
);

/**
 * Transform Lanczos eigenvectors to full space
 * 
 * Constructs full eigenvectors from Lanczos basis
 */
void transform_lanczos_eigenvectors(
    const std::vector<std::vector<double>>& lanczos_eigenvecs,
    const std::vector<GPUVector*>& lanczos_basis,
    const std::vector<double>& eigenvalues,
    const std::string& dir,
    int num_eigs
);

} // namespace gpu

#endif // LANCZOS_GPU_CUH
