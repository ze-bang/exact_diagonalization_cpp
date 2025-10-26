// lanczos_gpu.cu - GPU Lanczos algorithm implementation
#include "lanczos_gpu.cuh"
#include "blas_lapack_wrapper.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace gpu {

int solve_tridiagonal_eigenvalue_problem(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    int num_eigs,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors,
    bool compute_eigenvectors
) {
    int m = alpha.size();
    num_eigs = std::min(num_eigs, m);
    
    // Prepare arrays for LAPACK
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    int info;
    
    if (compute_eigenvectors) {
        // Compute eigenvalues and eigenvectors
        std::vector<double> evecs(m * m);
        
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                             diag.data(), offdiag.data(), 
                             evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error " << info << std::endl;
            return info;
        }
        
        // Extract eigenvalues and eigenvectors
        eigenvalues.resize(num_eigs);
        eigenvectors.resize(num_eigs);
        
        for (int i = 0; i < num_eigs; i++) {
            eigenvalues[i] = diag[i];
            eigenvectors[i].resize(m);
            
            // Copy column i of eigenvector matrix
            for (int j = 0; j < m; j++) {
                eigenvectors[i][j] = evecs[j + i * m];
            }
        }
    } else {
        // Compute only eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m,
                             diag.data(), offdiag.data(),
                             nullptr, m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error " << info << std::endl;
            return info;
        }
        
        eigenvalues.resize(num_eigs);
        for (int i = 0; i < num_eigs; i++) {
            eigenvalues[i] = diag[i];
        }
    }
    
    return 0;
}

void transform_lanczos_eigenvectors(
    const std::vector<std::vector<double>>& lanczos_eigenvecs,
    const std::vector<GPUVector*>& lanczos_basis,
    const std::vector<double>& eigenvalues,
    const std::string& dir,
    int num_eigs
) {
    if (dir.empty()) return;
    
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd = "mkdir -p " + evec_dir;
    system(cmd.c_str());
    
    size_t N = lanczos_basis[0]->size();
    int m = lanczos_basis.size();
    
    std::cout << "Transforming " << num_eigs << " Lanczos eigenvectors to full space..." << std::endl;
    
    for (int i = 0; i < num_eigs; i++) {
        if (i % 10 == 0) {
            std::cout << "  Eigenvector " << i + 1 << "/" << num_eigs << std::endl;
        }
        
        // Create result vector on GPU
        GPUVector full_evec(N, lanczos_basis[0]->get_cublas_handle());
        full_evec.zero();
        
        // Linear combination: |ψ_i⟩ = Σ_j c_ij |v_j⟩
        for (int j = 0; j < m; j++) {
            double coef = lanczos_eigenvecs[i][j];
            if (std::abs(coef) > 1e-14) {
                full_evec.axpy(std::complex<double>(coef, 0.0), *lanczos_basis[j]);
            }
        }
        
        // Normalize
        full_evec.normalize();
        
        // Download and save to file
        std::vector<std::complex<double>> host_vec;
        full_evec.download(host_vec);
        
        std::string filename = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<const char*>(host_vec.data()),
                         N * sizeof(std::complex<double>));
            outfile.close();
        }
    }
    
    // Save eigenvalues
    std::string eval_file = evec_dir + "/eigenvalues.dat";
    std::ofstream eval_outfile(eval_file, std::ios::binary);
    if (eval_outfile) {
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<const char*>(eigenvalues.data()),
                          n_evals * sizeof(double));
        eval_outfile.close();
    }
    
    std::cout << "Eigenvectors saved to " << evec_dir << std::endl;
}

void lanczos_gpu(
    GPUHamiltonianOperator& H,
    int max_iter,
    int num_eigs,
    double tol,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_eigenvectors
) {
    size_t N = H.get_dimension();
    cublasHandle_t cublas_handle = H.get_cublas_handle();
    
    std::cout << "=== GPU Lanczos Algorithm ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "Number of eigenvalues: " << num_eigs << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // Initialize random starting vector
    GPUVector v(N, cublas_handle);
    v.randomize(12345);
    v.normalize();
    
    GPUVector v_old(N, cublas_handle);
    v_old.zero();
    
    GPUVector Hv(N, cublas_handle);
    
    // Lanczos coefficients
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    // Store Lanczos basis vectors (always store for reorthogonalization)
    std::vector<GPUVector*> lanczos_basis;
    lanczos_basis.push_back(v.clone().release());
    
    GPUTimer timer;
    timer.start();
    
    // Parameters for reorthogonalization
    const int full_reorth_freq = 10;  // Full reorthogonalization every N iterations
    
    // Lanczos iteration
    for (int iter = 0; iter < max_iter; iter++) {
        // Apply Hamiltonian: Hv = H * v
        H.apply(v, Hv);
        
        // alpha[iter] = <v|H|v>
        std::complex<double> alpha_complex = v.dot(Hv);
        alpha.push_back(alpha_complex.real());
        
        // Compute v_new = H*v - alpha*v - beta*v_old
        GPUVector v_new(N, cublas_handle);
        v_new.copy_from(Hv);
        v_new.axpy(std::complex<double>(-alpha[iter], 0.0), v);
        if (iter > 0) {
            v_new.axpy(std::complex<double>(-beta[iter], 0.0), v_old);
        }
        
        // Full reorthogonalization (periodically for numerical stability)
        if ((iter + 1) % full_reorth_freq == 0 || iter < 2) {
            // Orthogonalize against all previous Lanczos vectors
            for (size_t j = 0; j < lanczos_basis.size(); j++) {
                std::complex<double> overlap = lanczos_basis[j]->dot(v_new);
                v_new.axpy(-overlap, *lanczos_basis[j]);
            }
        }
        
        // beta[iter+1] = ||v_new||
        double beta_new = v_new.norm();
        beta.push_back(beta_new);
        
        // Check for convergence or breakdown
        if (beta_new < tol) {
            std::cout << "Lanczos breakdown at iteration " << iter + 1 << std::endl;
            break;
        }
        
        // Normalize: v_new = v_new / beta[iter+1]
        v_new.scale(std::complex<double>(1.0 / beta_new, 0.0));
        
        // Update vectors for next iteration
        v_old.copy_from(v);
        v.copy_from(v_new);
        
        // Always store basis vector for reorthogonalization
        lanczos_basis.push_back(v.clone().release());
        
        // Periodically check convergence
        if ((iter + 1) % 10 == 0 || iter == max_iter - 1) {
            std::vector<double> current_eigenvalues;
            std::vector<std::vector<double>> dummy_eigenvecs;
            
            solve_tridiagonal_eigenvalue_problem(
                alpha, beta, num_eigs, current_eigenvalues, dummy_eigenvecs, false
            );
            
            std::cout << "Iteration " << std::setw(4) << iter + 1 << ": ";
            std::cout << "Lowest eigenvalues: ";
            for (int i = 0; i < std::min(5, (int)current_eigenvalues.size()); i++) {
                std::cout << std::setw(12) << std::setprecision(8) << current_eigenvalues[i] << " ";
            }
            // Debug: print alpha and beta norms
            if ((iter + 1) % 100 == 0) {
                double alpha_max = 0, beta_max = 0;
                for (const auto& a : alpha) alpha_max = std::max(alpha_max, std::abs(a));
                for (const auto& b : beta) beta_max = std::max(beta_max, std::abs(b));
                std::cout << " [α_max=" << alpha_max << ", β_max=" << beta_max << "]";
            }
            std::cout << std::endl;
            
            // Check convergence of lowest eigenvalues
            if (iter > 20) {
                bool converged = true;
                if (iter >= 10 && !eigenvalues.empty()) {
                    // Compare with previous iteration (stored eigenvalues)
                    for (int i = 0; i < std::min(num_eigs, (int)eigenvalues.size()); i++) {
                        if (i < (int)current_eigenvalues.size() && std::abs(current_eigenvalues[i] - eigenvalues[i]) > tol) {
                            converged = false;
                            break;
                        }
                    }
                }
                eigenvalues = current_eigenvalues;
                
                if (converged && iter > 50) {
                    std::cout << "Lanczos converged at iteration " << iter + 1 << std::endl;
                    break;
                }
            } else {
                eigenvalues = current_eigenvalues;
            }
        }
    }
    
    float elapsed_ms = timer.stop();
    std::cout << "Lanczos GPU time: " << elapsed_ms / 1000.0 << " seconds" << std::endl;
    
    // Final solution of tridiagonal problem
    std::vector<std::vector<double>> lanczos_eigenvecs;
    solve_tridiagonal_eigenvalue_problem(
        alpha, beta, num_eigs, eigenvalues, lanczos_eigenvecs, compute_eigenvectors
    );
    
    std::cout << "\nFinal eigenvalues:" << std::endl;
    for (int i = 0; i < std::min(num_eigs, (int)eigenvalues.size()); i++) {
        std::cout << "  λ_" << i << " = " << std::setprecision(12) << eigenvalues[i] << std::endl;
    }
    
    // Transform and save eigenvectors if requested
    if (compute_eigenvectors && !dir.empty()) {
        transform_lanczos_eigenvectors(
            lanczos_eigenvecs, lanczos_basis, eigenvalues, dir, num_eigs
        );
    }
    
    // Clean up basis vectors
    for (auto* vec : lanczos_basis) {
        delete vec;
    }
}

void lanczos_fixed_sz_gpu(
    GPUFixedSzOperator& H,
    int max_iter,
    int num_eigs,
    double tol,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_eigenvectors
) {
    size_t N = H.get_fixed_sz_dimension();
    cublasHandle_t cublas_handle = H.get_cublas_handle();
    
    std::cout << "=== GPU Lanczos Algorithm (Fixed Sz Sector) ===" << std::endl;
    std::cout << "Fixed Sz dimension: " << N << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "Number of eigenvalues: " << num_eigs << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // Initialize random starting vector
    GPUVector v(N, cublas_handle);
    v.randomize(12345);
    v.normalize();
    
    GPUVector v_old(N, cublas_handle);
    v_old.zero();
    
    GPUVector Hv(N, cublas_handle);
    
    // Lanczos coefficients
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    // Store Lanczos basis vectors (always store for reorthogonalization)
    std::vector<GPUVector*> lanczos_basis;
    lanczos_basis.push_back(v.clone().release());
    
    GPUTimer timer;
    timer.start();
    
    // Parameters for reorthogonalization
    const int full_reorth_freq = 10;  // Full reorthogonalization every N iterations
    
    // Lanczos iteration
    for (int iter = 0; iter < max_iter; iter++) {
        // Apply Hamiltonian: Hv = H * v
        H.apply(v, Hv);
        
        // alpha[iter] = <v|H|v>
        std::complex<double> alpha_complex = v.dot(Hv);
        alpha.push_back(alpha_complex.real());
        
        // Compute v_new = H*v - alpha*v - beta*v_old
        GPUVector v_new(N, cublas_handle);
        v_new.copy_from(Hv);
        v_new.axpy(std::complex<double>(-alpha[iter], 0.0), v);
        if (iter > 0) {
            v_new.axpy(std::complex<double>(-beta[iter], 0.0), v_old);
        }
        
        // Full reorthogonalization (periodically for numerical stability)
        if ((iter + 1) % full_reorth_freq == 0 || iter < 2) {
            // Orthogonalize against all previous Lanczos vectors
            for (size_t j = 0; j < lanczos_basis.size(); j++) {
                std::complex<double> overlap = lanczos_basis[j]->dot(v_new);
                v_new.axpy(-overlap, *lanczos_basis[j]);
            }
        }
        
        // beta[iter+1] = ||v_new||
        double beta_new = v_new.norm();
        beta.push_back(beta_new);
        
        // Check for convergence or breakdown
        if (beta_new < tol) {
            std::cout << "Lanczos breakdown at iteration " << iter + 1 << std::endl;
            break;
        }
        
        // Normalize: v_new = v_new / beta[iter+1]
        v_new.scale(std::complex<double>(1.0 / beta_new, 0.0));
        
        // Update vectors for next iteration
        v_old.copy_from(v);
        v.copy_from(v_new);
        
        // Always store basis vector for reorthogonalization
        lanczos_basis.push_back(v.clone().release());
        
        // Periodically check convergence
        if ((iter + 1) % 10 == 0 || iter == max_iter - 1) {
            std::vector<double> current_eigenvalues;
            std::vector<std::vector<double>> dummy_eigenvecs;
            
            solve_tridiagonal_eigenvalue_problem(
                alpha, beta, num_eigs, current_eigenvalues, dummy_eigenvecs, false
            );
            
            std::cout << "Iteration " << std::setw(4) << iter + 1 << ": ";
            std::cout << "Lowest eigenvalues: ";
            for (int i = 0; i < std::min(5, (int)current_eigenvalues.size()); i++) {
                std::cout << std::setw(12) << std::setprecision(8) << current_eigenvalues[i] << " ";
            }
            std::cout << std::endl;
            
            // Check convergence
            if (iter > 20) {
                bool converged = true;
                if (iter >= 10 && !eigenvalues.empty()) {
                    for (int i = 0; i < std::min(num_eigs, (int)eigenvalues.size()); i++) {
                        if (std::abs(current_eigenvalues[i] - eigenvalues[i]) > tol) {
                            converged = false;
                            break;
                        }
                    }
                }
                eigenvalues = current_eigenvalues;
                
                if (converged && iter > 50) {
                    std::cout << "Lanczos converged at iteration " << iter + 1 << std::endl;
                    break;
                }
            } else {
                eigenvalues = current_eigenvalues;
            }
        }
    }
    
    float elapsed_ms = timer.stop();
    std::cout << "Lanczos GPU time: " << elapsed_ms / 1000.0 << " seconds" << std::endl;
    
    // Final solution of tridiagonal problem
    std::vector<std::vector<double>> lanczos_eigenvecs;
    solve_tridiagonal_eigenvalue_problem(
        alpha, beta, num_eigs, eigenvalues, lanczos_eigenvecs, compute_eigenvectors
    );
    
    std::cout << "\nFinal eigenvalues:" << std::endl;
    for (int i = 0; i < std::min(num_eigs, (int)eigenvalues.size()); i++) {
        std::cout << "  λ_" << i << " = " << std::setprecision(12) << eigenvalues[i] << std::endl;
    }
    
    // Transform and save eigenvectors if requested
    if (compute_eigenvectors && !dir.empty()) {
        transform_lanczos_eigenvectors(
            lanczos_eigenvecs, lanczos_basis, eigenvalues, dir, num_eigs
        );
    }
    
    // Clean up basis vectors
    for (auto* vec : lanczos_basis) {
        delete vec;
    }
}

} // namespace gpu
