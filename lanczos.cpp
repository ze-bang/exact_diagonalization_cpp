#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include "construct_ham.h"
#include <iomanip>
#include <algorithm>
#include <ezarpack/arpack_solver.hpp>
#include <ezarpack/storages/eigen.hpp>
#include <ezarpack/version.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseQR>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseCholesky>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist
                                    ) {
    ComplexVector result(N);
    
    // Generate a random vector
    result = generateRandomVector(N, gen, dist);
    
    // Orthogonalize against all provided vectors using Gram-Schmidt
    for (const auto& v : vectors) {
        // Calculate projection: <v, result>
        Complex projection;
        cblas_zdotc_sub(N, v.data(), 1, result.data(), 1, &projection);
        
        // Subtract projection: result -= projection * v
        Complex neg_projection = -projection;
        cblas_zaxpy(N, &neg_projection, v.data(), 1, result.data(), 1);
    }
    
    // Check if the resulting vector has sufficient magnitude
    double norm = cblas_dznrm2(N, result.data(), 1);
        // Normalize
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, result.data(), 1);
    return result;
    
    // If all attempts failed, throw an exception
}

// Helper function to refine a single eigenvector with CG
void refine_eigenvector_with_cg(std::function<void(const Complex*, Complex*, int)> H,
                               ComplexVector& v, double& lambda, int N, double tol) {
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v.data(), 1);
    
    ComplexVector r(N), p(N), Hp(N), Hv(N);
    
    // Apply H to v: Hv = H*v
    H(v.data(), Hv.data(), N);
    
    // Initial residual: r = Hv - λv
    std::copy(Hv.begin(), Hv.end(), r.begin());
    Complex neg_lambda(-lambda, 0.0);
    cblas_zaxpy(N, &neg_lambda, v.data(), 1, r.data(), 1);
    
    // Initial search direction
    std::copy(r.begin(), r.end(), p.begin());
    
    // CG iteration
    const int max_cg_iter = 50;
    const double cg_tol = tol * 0.1;
    double res_norm = cblas_dznrm2(N, r.data(), 1);
    
    for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol; iter++) {
        // Apply (H - λI) to p
        H(p.data(), Hp.data(), N);
        cblas_zaxpy(N, &neg_lambda, p.data(), 1, Hp.data(), 1);
        
        // α = (r·r) / (p·(H-λI)p)
        Complex r_dot_r, p_dot_Hp;
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
        cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &p_dot_Hp);
        
        Complex alpha = r_dot_r / p_dot_Hp;
        
        // v = v + α*p
        cblas_zaxpy(N, &alpha, p.data(), 1, v.data(), 1);
        
        // Store old r·r
        Complex r_dot_r_old = r_dot_r;
        
        // r = r - α*(H-λI)p
        Complex neg_alpha = -alpha;
        cblas_zaxpy(N, &neg_alpha, Hp.data(), 1, r.data(), 1);
        
        // Check convergence
        res_norm = cblas_dznrm2(N, r.data(), 1);
        
        // β = (r_new·r_new) / (r_old·r_old)
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
        Complex beta = r_dot_r / r_dot_r_old;
        
        // p = r + β*p
        for (int j = 0; j < N; j++) {
            p[j] = r[j] + beta * p[j];
        }
    }
    
    // Normalize final eigenvector
    norm = cblas_dznrm2(N, v.data(), 1);
    scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v.data(), 1);
}

// Helper function to refine a set of degenerate eigenvectors
void refine_degenerate_eigenvectors(std::function<void(const Complex*, Complex*, int)> H,
                                  std::vector<ComplexVector>& vectors, double lambda, int N, double tol) {
    const int block_size = vectors.size();
    
    // Make sure the initial set is orthogonal
    for (int i = 0; i < block_size; i++) {
        // Normalize first
        double norm = cblas_dznrm2(N, vectors[i].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, vectors[i].data(), 1);
        
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++) {
            Complex overlap;
            cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
            
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
        }
        
        // Renormalize
        norm = cblas_dznrm2(N, vectors[i].data(), 1);
        if (norm > tol) {
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, vectors[i].data(), 1);
        }
    }
    
    // Now optimize the entire degenerate subspace together
    // Here we use subspace iteration approach rather than CG
    
    // Workspace for matrix-vector operations
    std::vector<ComplexVector> HV(block_size, ComplexVector(N));
    std::vector<ComplexVector> Y(block_size, ComplexVector(N));
    
    for (int iter = 0; iter < 20; iter++) {  // Fixed number of iterations
        // Apply H to each vector
        for (int i = 0; i < block_size; i++) {
            H(vectors[i].data(), HV[i].data(), N);
        }
        
        // Compute the projection matrix <v_i|H|v_j>
        std::vector<std::vector<Complex>> projection(block_size, std::vector<Complex>(block_size));
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                Complex proj;
                cblas_zdotc_sub(N, vectors[i].data(), 1, HV[j].data(), 1, &proj);
                projection[i][j] = proj;
            }
        }
        
        // Diagonalize the projection matrix
        std::vector<double> evals(block_size);
        std::vector<Complex> evecs(block_size * block_size);
        
        // This is a small matrix, so we can use zheev directly
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                evecs[j*block_size + i] = projection[i][j];
            }
        }
        
        int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', block_size,
                               reinterpret_cast<lapack_complex_double*>(evecs.data()),
                               block_size, evals.data());
        
        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed in refine_degenerate_eigenvectors" << std::endl;
            return;
        }
        
        // Compute new vectors Y = V * evecs
        for (int i = 0; i < block_size; i++) {
            std::fill(Y[i].begin(), Y[i].end(), Complex(0.0, 0.0));
            
            for (int j = 0; j < block_size; j++) {
                Complex coef = evecs[i*block_size + j];
                for (int k = 0; k < N; k++) {
                    Y[i][k] += coef * vectors[j][k];
                }
            }
        }
        
        // Replace old vectors with the new ones
        vectors = Y;
        
        // Re-orthogonalize for stability
        for (int i = 0; i < block_size; i++) {
            // Orthogonalize against previous vectors
            for (int j = 0; j < i; j++) {
                Complex overlap;
                cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, vectors[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, vectors[i].data(), 1);
        }
    }
}


// Lanczos algorithm implementation
// H: Function representing the Hamiltonian operator (H|v⟩)
// N: Dimension of the Hilbert space
// max_iter: Maximum number of Lanczos iterations
// tol: Tolerance for convergence and detecting invariant subspaces
// eigenvalues: Output vector to store the eigenvalues
// eigenvectors: Output matrix to store the eigenvectors (optional)
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize random starting vector using Eigen
    Eigen::VectorXcd v_current = Eigen::VectorXcd::Random(N);
    v_current.normalize();
    
    // Convert to std::vector for compatibility with H function
    ComplexVector v_current_std(N);
    for (int i = 0; i < N; i++) {
        v_current_std[i] = v_current(i);
    }
    
    // Initialize Lanczos vectors and coefficients
    std::vector<Eigen::VectorXcd> basis_vectors;
    basis_vectors.push_back(v_current);
    
    Eigen::VectorXcd v_prev = Eigen::VectorXcd::Zero(N);
    Eigen::VectorXcd v_next(N);
    Eigen::VectorXcd w(N);
    ComplexVector w_std(N), v_next_std(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current_std.data(), w_std.data(), N);
        
        // Convert to Eigen for easier manipulation
        for (int i = 0; i < N; i++) {
            w(i) = w_std[i];
        }
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            w -= beta[j] * v_prev;
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product = v_current.dot(w);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        w -= alpha[j] * v_current;
        
        // Full reorthogonalization (twice for numerical stability)
        for (int iter = 0; iter < 2; iter++) {
            for (int k = 0; k <= j; k++) {
                Complex overlap = basis_vectors[k].dot(w);
                w -= overlap * basis_vectors[k];
            }
        }
        
        // beta_{j+1} = ||w||
        double norm = w.norm();
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector orthogonal to basis
            v_next = Eigen::VectorXcd::Random(N);
            
            // Orthogonalize against all basis vectors
            for (int iter = 0; iter < 2; iter++) {
                for (const auto& basis_vec : basis_vectors) {
                    Complex overlap = basis_vec.dot(v_next);
                    v_next -= overlap * basis_vec;
                }
            }
            
            // Update the norm
            norm = v_next.norm();
            
            // If still too small, we've reached an invariant subspace
            if (norm < tol) {
                break;
            }
        } else {
            v_next = w;
        }
        
        beta.push_back(norm);
        
        // Normalize v_next
        v_next.normalize();
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
        }
        
        // Convert to std::vector for next iteration
        for (int i = 0; i < N; i++) {
            v_next_std[i] = v_next(i);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
        v_current_std = v_next_std;
    }
    
    // Construct tridiagonal matrix using Eigen
    int m = alpha.size();
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    
    // Fill diagonal
    for (int i = 0; i < m; i++) {
        T(i, i) = alpha[i];
    }
    
    // Fill off-diagonals
    for (int i = 0; i < m - 1; i++) {
        T(i, i+1) = beta[i+1];
        T(i+1, i) = beta[i+1];
    }
    
    // Compute eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(T);
    
    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Eigendecomposition failed!" << std::endl;
        return;
    }
    
    // Get eigenvalues
    Eigen::VectorXd evals = eigenSolver.eigenvalues();
    eigenvalues.resize(m);
    for (int i = 0; i < m; i++) {
        eigenvalues[i] = evals(i);
    }
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        // Get Lanczos eigenvectors
        Eigen::MatrixXd z = eigenSolver.eigenvectors();
        
        // Identify clusters of degenerate eigenvalues
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < m; i++) {
            bool added_to_cluster = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(evals(i) - evals(cluster[0])) < degen_tol) {
                    cluster.push_back(i);
                    added_to_cluster = true;
                    break;
                }
            }
            if (!added_to_cluster) {
                degen_clusters.push_back({i});
            }
        }
        
        // Transform to original basis and handle degeneracy
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N));
        
        // Process each cluster separately
        for (const auto& cluster : degen_clusters) {
            if (cluster.size() == 1) {
                // Non-degenerate case - standard treatment
                int idx = cluster[0];
                Eigen::VectorXcd evec = Eigen::VectorXcd::Zero(N);
                
                for (int k = 0; k < m; k++) {
                    evec += z(k, idx) * basis_vectors[k];
                }
                
                // Normalize and store
                evec.normalize();
                for (int i = 0; i < N; i++) {
                    (*eigenvectors)[idx][i] = evec(i);
                }
            } else {
                // Degenerate case - special handling
                int subspace_dim = cluster.size();
                std::vector<Eigen::VectorXcd> subspace_vectors(subspace_dim, Eigen::VectorXcd(N));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (int k = 0; k < m; k++) {
                        subspace_vectors[c] += z(k, idx) * basis_vectors[k];
                    }
                }
                
                // Re-orthogonalize within degenerate subspace using QR
                for (int c = 0; c < subspace_dim; c++) {
                    // Normalize current vector
                    subspace_vectors[c].normalize();
                    
                    // Orthogonalize against previous vectors
                    for (int prev = 0; prev < c; prev++) {
                        Complex overlap = subspace_vectors[prev].dot(subspace_vectors[c]);
                        subspace_vectors[c] -= overlap * subspace_vectors[prev];
                    }
                    
                    // Renormalize if necessary
                    double norm = subspace_vectors[c].norm();
                    if (norm > tol) {
                        subspace_vectors[c].normalize();
                    }
                    
                    // Store vector
                    int idx = cluster[c];
                    for (int i = 0; i < N; i++) {
                        (*eigenvectors)[idx][i] = subspace_vectors[c](i);
                    }
                }
            }
        }
        
        // Final verification of orthogonality
        for (int i = 0; i < m; i++) {
            // Convert to Eigen for normalization
            Eigen::VectorXcd evec = Eigen::VectorXcd::Zero(N);
            for (int j = 0; j < N; j++) {
                evec(j) = (*eigenvectors)[i][j];
            }
            
            // Normalize
            evec.normalize();
            
            // Convert back to std::vector
            for (int j = 0; j < N; j++) {
                (*eigenvectors)[i][j] = evec(j);
            }
        }
    }
}

void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Block size for handling degenerate eigenvalues
    const int block_size = 4;  // Can adjust based on expected degeneracy
    
    // Initialize random starting vectors using Eigen
    std::vector<Eigen::VectorXcd> block_eigen(block_size, Eigen::VectorXcd(N));
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    
    // Generate orthonormal set of starting vectors
    block_eigen[0] = Eigen::VectorXcd::Random(N);
    block_eigen[0].normalize();
    
    // Convert to std::vector for compatibility with H function
    for (int i = 0; i < N; i++) {
        block_vectors[0][i] = block_eigen[0](i);
    }
    
    for (int i = 1; i < block_size; i++) {
        // Generate random vector
        block_eigen[i] = Eigen::VectorXcd::Random(N);
        
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++) {
            Complex projection = block_eigen[j].adjoint() * block_eigen[i];
            block_eigen[i] -= projection * block_eigen[j];
        }
        
        // Normalize
        block_eigen[i].normalize();
        
        // Convert to std::vector
        for (int j = 0; j < N; j++) {
            block_vectors[i][j] = block_eigen[i](j);
        }
    }
    
    // Initialize Lanczos vectors and coefficients for block Lanczos
    std::vector<Eigen::VectorXcd> basis_eigen;
    std::vector<ComplexVector> basis_vectors;
    
    for (int i = 0; i < block_size; i++) {
        basis_eigen.push_back(block_eigen[i]);
        basis_vectors.push_back(block_vectors[i]);
    }
    
    std::vector<Eigen::VectorXcd> prev_block_eigen = block_eigen;
    std::vector<Eigen::VectorXcd> curr_block_eigen = block_eigen;
    std::vector<Eigen::VectorXcd> next_block_eigen(block_size, Eigen::VectorXcd(N));
    std::vector<Eigen::VectorXcd> work_block_eigen(block_size, Eigen::VectorXcd(N));
    
    std::vector<ComplexVector> prev_block = block_vectors;
    std::vector<ComplexVector> curr_block = block_vectors;
    std::vector<ComplexVector> next_block(block_size, ComplexVector(N));
    std::vector<ComplexVector> work_block(block_size, ComplexVector(N));
    
    // Block tridiagonal matrix elements
    std::vector<Eigen::MatrixXcd> alpha;  // Diagonal blocks
    std::vector<Eigen::MatrixXcd> beta;   // Off-diagonal blocks
    
    // First empty beta block
    beta.push_back(Eigen::MatrixXcd::Zero(block_size, block_size));
    
    // Number of Lanczos steps (each processes a block)
    int num_steps = max_iter / block_size;
    num_steps = std::min(N / block_size, num_steps);
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        // Current alpha block
        Eigen::MatrixXcd curr_alpha = Eigen::MatrixXcd::Zero(block_size, block_size);
        
        // Apply H to each vector in the current block
        for (int b = 0; b < block_size; b++) {
            H(curr_block[b].data(), work_block[b].data(), N);
            
            // Convert to Eigen
            for (int i = 0; i < N; i++) {
                work_block_eigen[b](i) = work_block[b][i];
            }
        }
        
        // Subtract beta_j * prev_block
        if (j > 0) {
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    work_block_eigen[i] -= beta[j](i,k) * prev_block_eigen[k];
                }
            }
        }
        
        // Compute alpha_j block
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                curr_alpha(i,k) = curr_block_eigen[k].adjoint() * work_block_eigen[i];
                
                // Subtract from work vector
                work_block_eigen[i] -= curr_alpha(i,k) * curr_block_eigen[k];
            }
        }
        
        alpha.push_back(curr_alpha);
        
        // Full reorthogonalization against all previous basis vectors
        for (int b = 0; b < block_size; b++) {
            for (int iter = 0; iter < 2; iter++) {  // Do twice for numerical stability
                for (size_t k = 0; k < basis_eigen.size(); k++) {
                    Complex overlap = basis_eigen[k].adjoint() * work_block_eigen[b];
                    work_block_eigen[b] -= overlap * basis_eigen[k];
                }
            }
            
            // Update std::vector version
            for (int i = 0; i < N; i++) {
                work_block[b][i] = work_block_eigen[b](i);
            }
        }
        
        // QR factorization of the work block to get next orthonormal block
        Eigen::MatrixXcd next_beta = Eigen::MatrixXcd::Zero(block_size, block_size);
        
        for (int i = 0; i < block_size; i++) {
            // Compute the norm of the work vector
            double norm = work_block_eigen[i].norm();
            
            // If nearly zero, generate a new orthogonal vector
            if (norm < tol) {
                next_block_eigen[i] = Eigen::VectorXcd::Random(N);
                
                // Orthogonalize against all basis vectors
                for (size_t k = 0; k < basis_eigen.size(); k++) {
                    Complex overlap = basis_eigen[k].adjoint() * next_block_eigen[i];
                    next_block_eigen[i] -= overlap * basis_eigen[k];
                }
                
                norm = next_block_eigen[i].norm();
            } else {
                // Copy work to next
                next_block_eigen[i] = work_block_eigen[i];
            }
            
            // Set the diagonal beta element
            next_beta(i,i) = norm;
            
            // Normalize
            next_block_eigen[i] /= norm;
            
            // Convert to std::vector
            for (int idx = 0; idx < N; idx++) {
                next_block[i][idx] = next_block_eigen[i](idx);
            }
            
            // Orthogonalize remaining work vectors against this one
            for (int j = i + 1; j < block_size; j++) {
                Complex overlap = next_block_eigen[i].adjoint() * work_block_eigen[j];
                next_beta(j,i) = overlap;  // Off-diagonal beta element
                work_block_eigen[j] -= overlap * next_block_eigen[i];
            }
        }
        
        beta.push_back(next_beta);
        
        // Store the new basis vectors
        if (j < num_steps - 1) {
            for (int i = 0; i < block_size; i++) {
                basis_eigen.push_back(next_block_eigen[i]);
                basis_vectors.push_back(next_block[i]);
            }
        }
        
        // Update for next iteration
        prev_block_eigen = curr_block_eigen;
        curr_block_eigen = next_block_eigen;
        prev_block = curr_block;
        curr_block = next_block;
    }
    
    // Convert block tridiagonal matrix to regular format for solving
    int total_dim = basis_vectors.size();
    Eigen::MatrixXcd block_matrix = Eigen::MatrixXcd::Zero(total_dim, total_dim);
    
    // Fill diagonal blocks (alpha)
    for (size_t j = 0; j < alpha.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = j * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix(row, col) = alpha[j](r,c);
                }
            }
        }
    }
    
    // Fill off-diagonal blocks (beta)
    for (size_t j = 1; j < beta.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = (j-1) * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix(row, col) = beta[j](r,c);
                    block_matrix(col, row) = std::conj(beta[j](r,c));
                }
            }
        }
    }
    
    // Diagonalize the block tridiagonal matrix using Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(block_matrix);
    
    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Eigendecomposition failed!" << std::endl;
        return;
    }
    
    // Store eigenvalues
    eigenvalues.resize(total_dim);
    Eigen::VectorXd evals = eigenSolver.eigenvalues();
    for (int i = 0; i < total_dim; i++) {
        eigenvalues[i] = evals(i);
    }
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors) {
        eigenvectors->clear();
        eigenvectors->resize(total_dim, ComplexVector(N, Complex(0.0, 0.0)));
        
        Eigen::MatrixXcd evecs = eigenSolver.eigenvectors();
        
        // Group eigenvalues into degenerate clusters
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < total_dim; i++) {
            bool added_to_cluster = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(evals(i) - evals(cluster[0])) < degen_tol) {
                    cluster.push_back(i);
                    added_to_cluster = true;
                    break;
                }
            }
            if (!added_to_cluster) {
                degen_clusters.push_back({i});
            }
        }
        
        // Process each cluster separately
        for (const auto& cluster : degen_clusters) {
            if (cluster.size() == 1) {
                // Non-degenerate case
                int idx = cluster[0];
                Eigen::VectorXcd eigenvec = Eigen::VectorXcd::Zero(N);
                
                for (size_t k = 0; k < basis_eigen.size(); k++) {
                    eigenvec += evecs(k, idx) * basis_eigen[k];
                }
                
                // Convert to std::vector
                for (int i = 0; i < N; i++) {
                    (*eigenvectors)[idx][i] = eigenvec(i);
                }
            } else {
                // Degenerate case
                int subspace_dim = cluster.size();
                std::vector<Eigen::VectorXcd> subspace_vectors(subspace_dim, Eigen::VectorXcd::Zero(N));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (size_t k = 0; k < basis_eigen.size(); k++) {
                        subspace_vectors[c] += evecs(k, idx) * basis_eigen[k];
                    }
                }
                
                // Orthogonalize within degenerate subspace
                for (int c = 0; c < subspace_dim; c++) {
                    for (int prev = 0; prev < c; prev++) {
                        Complex overlap = subspace_vectors[prev].adjoint() * subspace_vectors[c];
                        subspace_vectors[c] -= overlap * subspace_vectors[prev];
                    }
                    
                    // Normalize
                    double norm = subspace_vectors[c].norm();
                    if (norm > tol) {
                        subspace_vectors[c] /= norm;
                    }
                }
                
                // Store the orthogonalized eigenvectors
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (int i = 0; i < N; i++) {
                        (*eigenvectors)[idx][i] = subspace_vectors[c](i);
                    }
                }
            }
        }
        
        // Final verification of orthogonality
        for (int i = 0; i < total_dim; i++) {
            // Convert to Eigen for normalization
            Eigen::VectorXcd eigenvec = Eigen::VectorXcd::Zero(N);
            for (int j = 0; j < N; j++) {
                eigenvec(j) = (*eigenvectors)[i][j];
            }
            
            // Normalize
            eigenvec.normalize();
            
            // Convert back to std::vector
            for (int j = 0; j < N; j++) {
                (*eigenvectors)[i][j] = eigenvec(j);
            }
        }
    }
}

// Lanczos algorithm with Conjugate Gradient refinement for eigenvectors
void lanczos_cg(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                double tol, std::vector<double>& eigenvalues, 
                std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // First, run the standard Lanczos algorithm to get initial approximations
    std::vector<ComplexVector> initial_eigenvectors;
    lanczos(H, N, max_iter, tol, eigenvalues, eigenvectors ? &initial_eigenvectors : nullptr);
    
    // If eigenvectors are not requested, we're done
    if (!eigenvectors) return;
    
    // Initialize output eigenvectors
    eigenvectors->clear();
    eigenvectors->resize(initial_eigenvectors.size(), ComplexVector(N, Complex(0.0, 0.0)));
    
    // Group eigenvalues into degenerate clusters
    const double degen_tol = 1e-10;
    std::vector<std::vector<int>> degen_clusters;
    
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        bool added_to_cluster = false;
        for (auto& cluster : degen_clusters) {
            if (std::abs(eigenvalues[i] - eigenvalues[cluster[0]]) < degen_tol) {
                cluster.push_back(i);
                added_to_cluster = true;
                break;
            }
        }
        if (!added_to_cluster) {
            degen_clusters.push_back({(int)i});
        }
    }
    
    // Process each cluster
    for (const auto& cluster : degen_clusters) {
        if (cluster.size() == 1) {
            // Non-degenerate case: standard CG refinement
            int idx = cluster[0];
            double lambda = eigenvalues[idx];
            
            // Convert std::vector to Eigen
            Eigen::VectorXcd v = Eigen::VectorXcd::Zero(N);
            for (int i = 0; i < N; i++) {
                v(i) = initial_eigenvectors[idx][i];
            }
            
            // Normalize
            v.normalize();
            
            // Apply refinement
            Eigen::VectorXcd r(N), p(N), Hp(N), Hv(N);
            
            // Convert v to std::vector for H
            ComplexVector v_std(N);
            ComplexVector Hv_std(N);
            for (int i = 0; i < N; i++) {
                v_std[i] = v(i);
            }
            
            // Apply H to v: Hv = H*v
            H(v_std.data(), Hv_std.data(), N);
            
            // Convert back to Eigen
            for (int i = 0; i < N; i++) {
                Hv(i) = Hv_std[i];
            }
            
            // Initial residual: r = Hv - λv
            r = Hv - lambda * v;
            
            // Initial search direction
            p = r;
            
            // CG iteration
            const int max_cg_iter = 50;
            const double cg_tol = tol * 0.1;
            double res_norm = r.norm();
            
            for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol; iter++) {
                // Apply (H - λI) to p
                ComplexVector p_std(N);
                ComplexVector Hp_std(N);
                for (int i = 0; i < N; i++) {
                    p_std[i] = p(i);
                }
                
                H(p_std.data(), Hp_std.data(), N);
                
                for (int i = 0; i < N; i++) {
                    Hp(i) = Hp_std[i] - lambda * p(i);
                }
                
                // α = (r·r) / (p·(H-λI)p)
                Complex r_dot_r = r.squaredNorm();
                Complex p_dot_Hp = p.dot(Hp);
                
                Complex alpha = r_dot_r / p_dot_Hp;
                
                // v = v + α*p
                v += alpha * p;
                
                // Store old r·r
                Complex r_dot_r_old = r_dot_r;
                
                // r = r - α*(H-λI)p
                r -= alpha * Hp;
                
                // Check convergence
                res_norm = r.norm();
                
                // β = (r_new·r_new) / (r_old·r_old)
                Complex r_dot_r_new = r.squaredNorm();
                Complex beta = r_dot_r_new / r_dot_r_old;
                
                // p = r + β*p
                p = r + beta * p;
            }
            
            // Normalize final eigenvector
            v.normalize();
            
            // Update eigenvalue using Rayleigh quotient
            ComplexVector v_final_std(N);
            ComplexVector Hv_final_std(N);
            
            for (int i = 0; i < N; i++) {
                v_final_std[i] = v(i);
            }
            
            H(v_final_std.data(), Hv_final_std.data(), N);
            
            Eigen::VectorXcd Hv_final(N);
            for (int i = 0; i < N; i++) {
                Hv_final(i) = Hv_final_std[i];
            }
            
            Complex lambda_new = v.adjoint() * Hv_final;
            eigenvalues[idx] = std::real(lambda_new);
            
            // Store the refined eigenvector
            for (int i = 0; i < N; i++) {
                (*eigenvectors)[idx][i] = v(i);
            }
        } else {
            // Degenerate case: block refinement
            int subspace_dim = cluster.size();
            std::vector<Eigen::VectorXcd> subspace_vectors(subspace_dim, Eigen::VectorXcd(N));
            
            // Start with initial approximations
            for (int c = 0; c < subspace_dim; c++) {
                for (int i = 0; i < N; i++) {
                    subspace_vectors[c](i) = initial_eigenvectors[cluster[c]][i];
                }
                subspace_vectors[c].normalize();
            }
            
            // Make sure the initial set is orthogonal
            for (int i = 0; i < subspace_dim; i++) {
                // Orthogonalize against previous vectors
                for (int j = 0; j < i; j++) {
                    Complex overlap = subspace_vectors[j].adjoint() * subspace_vectors[i];
                    subspace_vectors[i] -= overlap * subspace_vectors[j];
                }
                
                // Renormalize
                double norm = subspace_vectors[i].norm();
                if (norm > tol) {
                    subspace_vectors[i] /= norm;
                }
            }
            
            // Workspace for matrix-vector operations
            std::vector<Eigen::VectorXcd> HV(subspace_dim, Eigen::VectorXcd(N));
            std::vector<Eigen::VectorXcd> Y(subspace_dim, Eigen::VectorXcd(N));
            
            // Optimize the entire degenerate subspace together
            for (int iter = 0; iter < 20; iter++) {  // Fixed number of iterations
                // Apply H to each vector
                for (int i = 0; i < subspace_dim; i++) {
                    ComplexVector v_std(N);
                    ComplexVector Hv_std(N);
                    for (int j = 0; j < N; j++) {
                        v_std[j] = subspace_vectors[i](j);
                    }
                    
                    H(v_std.data(), Hv_std.data(), N);
                    
                    for (int j = 0; j < N; j++) {
                        HV[i](j) = Hv_std[j];
                    }
                }
                
                // Compute the projection matrix <v_i|H|v_j>
                Eigen::MatrixXcd projection = Eigen::MatrixXcd::Zero(subspace_dim, subspace_dim);
                for (int i = 0; i < subspace_dim; i++) {
                    for (int j = 0; j < subspace_dim; j++) {
                        projection(i, j) = subspace_vectors[i].adjoint() * HV[j];
                    }
                }
                
                // Diagonalize the projection matrix
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(projection);
                
                // Compute new vectors Y = V * evecs
                for (int i = 0; i < subspace_dim; i++) {
                    Y[i] = Eigen::VectorXcd::Zero(N);
                    for (int j = 0; j < subspace_dim; j++) {
                        Y[i] += solver.eigenvectors()(j, i) * subspace_vectors[j];
                    }
                }
                
                // Replace old vectors with the new ones
                subspace_vectors = Y;
                
                // Re-orthogonalize for stability
                for (int i = 0; i < subspace_dim; i++) {
                    // Orthogonalize against previous vectors
                    for (int j = 0; j < i; j++) {
                        Complex overlap = subspace_vectors[j].adjoint() * subspace_vectors[i];
                        subspace_vectors[i] -= overlap * subspace_vectors[j];
                    }
                    
                    // Normalize
                    subspace_vectors[i].normalize();
                }
            }
            
            // Update eigenvalues and store eigenvectors
            for (int c = 0; c < subspace_dim; c++) {
                int idx = cluster[c];
                ComplexVector v_std(N);
                ComplexVector Hv_std(N);
                
                for (int i = 0; i < N; i++) {
                    v_std[i] = subspace_vectors[c](i);
                }
                
                H(v_std.data(), Hv_std.data(), N);
                
                Eigen::VectorXcd Hv(N);
                for (int i = 0; i < N; i++) {
                    Hv(i) = Hv_std[i];
                }
                
                // Compute Rayleigh quotient
                Complex lambda_new = subspace_vectors[c].adjoint() * Hv;
                eigenvalues[idx] = std::real(lambda_new);
                
                // Store eigenvector
                for (int i = 0; i < N; i++) {
                    (*eigenvectors)[idx][i] = subspace_vectors[c](i);
                }
            }
        }
    }
}

// Automatically estimate spectral bounds and optimal parameters for Chebyshev filtering
struct ChebysehvFilterParams {
    double a;          // Lower bound of interval
    double b;          // Upper bound of interval
    int filter_degree; // Optimal filter degree
    int lanczos_iter;  // Recommended Lanczos iterations
};

ChebysehvFilterParams estimate_filter_parameters(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,               // Matrix dimension
    int num_eigenvalues, // Number of desired eigenvalues
    bool lowest = true,  // Whether to target lowest (true) or highest (false) eigenvalues
    int sample_iter = 30 // Number of Lanczos iterations for estimation
) {
    // 1. Run quick Lanczos to estimate spectral range
    std::vector<double> sample_eigenvalues;
    sample_iter = std::min(N, sample_iter);
    lanczos(H, N, sample_iter, 1e-10, sample_eigenvalues);
    
    // Sort eigenvalues
    std::sort(sample_eigenvalues.begin(), sample_eigenvalues.end());
    
    // 2. Estimate the full spectrum bounds
    double min_eig = sample_eigenvalues.front();
    double max_eig = sample_eigenvalues.back();
    
    // Add some margin to ensure we cover the full spectrum
    double buffer = (max_eig - min_eig) * 0.1;
    double global_min = min_eig - buffer;
    double global_max = max_eig + buffer;
    
    // 3. Define the target interval [a, b] based on which eigenvalues are desired
    double a, b;
    if (lowest) {
        // For lowest eigenvalues, set [a,b] to lower portion of spectrum
        a = global_min;
        // Set b to cover a bit more than the desired eigenvalue range
        int idx = std::min<int>(static_cast<int>(sample_eigenvalues.size() * 0.8), 
                              static_cast<int>(num_eigenvalues * 2));
        if (idx < sample_eigenvalues.size()) {
            b = sample_eigenvalues[idx];
        } else {
            b = global_max * 0.5;
        }
    } else {
        // For highest eigenvalues, set [a,b] to upper portion of spectrum
        b = global_max;
        int idx = std::max(0, static_cast<int>(sample_eigenvalues.size()) - 
                               static_cast<int>(num_eigenvalues * 2));
        if (idx < sample_eigenvalues.size()) {
            a = sample_eigenvalues[idx];
        } else {
            a = global_min * 0.5;
        }
    }
    
    // 4. Calculate filter degree based on spectrum width and desired accuracy
    // A heuristic: use larger degree for wider spectrum
    double spectrum_width = global_max - global_min;
    double target_width = b - a;
    int filter_degree = static_cast<int>(15 * std::sqrt(spectrum_width / target_width));
    // Clamp to reasonable values
    filter_degree = std::min(std::max(filter_degree, 5), 50);
    
    // 5. Recommend Lanczos iterations - typically 2-3× the number of desired eigenvalues
    int lanczos_iter = std::min(N, std::max(2 * num_eigenvalues, 30));
    std::cout << "Estimated spectral bounds: [" << a << ", " << b << "]" << std::endl;
    std::cout << "Estimated filter degree: " << filter_degree << std::endl;
    return {a, b, filter_degree, lanczos_iter};
}

// Apply Chebyshev polynomial filter to a vector
void chebyshev_filter(std::function<void(const Complex*, Complex*, int)> H,
                     const ComplexVector& v, ComplexVector& result,
                     int N, double a, double b, int degree) {    

    // Scale and shift parameters for mapping [a, b] to [-1, 1]
    double e = (b - a) / 2;    // Half-width of interval
    double c = (b + a) / 2;    // Center of interval
    
    ComplexVector v_prev(N), v_curr(N), v_next(N), temp(N);
    
    // T_0(x) = 1, so v_curr = v
    v_curr = v;
    
    // T_1(x) = x, so v_next = (H-c*I)*v / e
    H(v.data(), temp.data(), N);
    
    for (int i = 0; i < N; i++) {
        v_next[i] = (temp[i] - Complex(c, 0) * v[i]) / e;
    }
    
    // Apply Chebyshev recurrence: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
    for (int k = 1; k < degree; k++) {
        // Store current as previous
        v_prev = v_curr;
        v_curr = v_next;
        
        // v_next = 2*(H-c*I)*v_curr/e - v_prev
        H(v_curr.data(), temp.data(), N);
        
        for (int i = 0; i < N; i++) {
            v_next[i] = 2.0 * (temp[i] - Complex(c, 0) * v_curr[i]) / e - v_prev[i];
        }
    }
    
    // Copy the result
    result = v_next;
    
    // Normalize the result
    double norm = cblas_dznrm2(N, result.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, result.data(), 1);
}

// Block Chebyshev filtered Lanczos algorithm for better handling of degenerate eigenvalues
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, 
                               int N, int max_iter, double tol, 
                               std::vector<double>& eigenvalues,
                               std::vector<ComplexVector>* eigenvectors = nullptr, double a = 0.0, double b = 0.0, int filter_degree = 0) {
    
    // Block size for handling degenerate eigenvalues
    const int block_size = 4;  // Can adjust based on expected degeneracy
    
    // Initialize random starting vectors
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Generate orthonormal set of starting vectors
    block_vectors[0] = generateRandomVector(N, gen, dist);
    for (int i = 1; i < block_size; i++) {
        block_vectors[i] = generateOrthogonalVector(N, 
                                                  std::vector<ComplexVector>(block_vectors.begin(), 
                                                                          block_vectors.begin() + i), 
                                                  gen, dist);
    }
    
    // Get filter parameters if not provided
    if (a == 0.0 && b == 0.0 && filter_degree == 0) {
        ChebysehvFilterParams params = estimate_filter_parameters(H, N, max_iter, true);
        a = params.a;
        b = params.b;
        filter_degree = params.filter_degree;
    }
    
    // Apply initial Chebyshev filter to each starting vector
    for (int i = 0; i < block_size; i++) {
        chebyshev_filter(H, block_vectors[i], block_vectors[i], N, a, b, filter_degree);
    }
    
    // Re-orthonormalize after filtering
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < i; j++) {
            Complex overlap;
            cblas_zdotc_sub(N, block_vectors[j].data(), 1, block_vectors[i].data(), 1, &overlap);
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, block_vectors[j].data(), 1, block_vectors[i].data(), 1);
        }
        double norm = cblas_dznrm2(N, block_vectors[i].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, block_vectors[i].data(), 1);
    }
    
    // Initialize Lanczos vectors and coefficients for block Lanczos
    std::vector<ComplexVector> basis_vectors;
    for (int i = 0; i < block_size; i++) {
        basis_vectors.push_back(block_vectors[i]);
    }
    
    std::vector<ComplexVector> prev_block = block_vectors;
    std::vector<ComplexVector> curr_block = block_vectors;
    std::vector<ComplexVector> next_block(block_size, ComplexVector(N));
    std::vector<ComplexVector> work_block(block_size, ComplexVector(N));
    
    // Block tridiagonal matrix elements
    std::vector<std::vector<std::vector<Complex>>> alpha;  // Diagonal blocks
    std::vector<std::vector<std::vector<Complex>>> beta;   // Off-diagonal blocks
    
    // First empty beta block
    beta.push_back(std::vector<std::vector<Complex>>(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0))));
    
    // Number of Lanczos steps (each processes a block)
    int num_steps = max_iter / block_size;
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        // Current alpha block
        std::vector<std::vector<Complex>> curr_alpha(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        // Apply H to each vector in the current block
        for (int b = 0; b < block_size; b++) {
            H(curr_block[b].data(), work_block[b].data(), N);
        }
        
        // Subtract beta_j * prev_block
        if (j > 0) {
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    Complex neg_beta = -beta[j][i][k];
                    cblas_zaxpy(N, &neg_beta, prev_block[k].data(), 1, work_block[i].data(), 1);
                }
            }
        }
        
        // Compute alpha_j block and residuals
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                Complex dot;
                cblas_zdotc_sub(N, curr_block[k].data(), 1, work_block[i].data(), 1, &dot);
                curr_alpha[i][k] = dot;
                
                // Subtract from work vector: work -= dot * curr
                Complex neg_dot = -dot;
                cblas_zaxpy(N, &neg_dot, curr_block[k].data(), 1, work_block[i].data(), 1);
            }
        }
        
        alpha.push_back(curr_alpha);
        
        // Full reorthogonalization against all previous basis vectors
        for (int b = 0; b < block_size; b++) {
            for (int iter = 0; iter < 2; iter++) {  // Do twice for numerical stability
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, basis_vectors[k].data(), 1, work_block[b].data(), 1, &overlap);
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, work_block[b].data(), 1);
                }
            }
        }
        
        // QR factorization of the work block to get next orthonormal block
        // We'll use a simplified Gram-Schmidt for this
        std::vector<std::vector<Complex>> next_beta(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        for (int i = 0; i < block_size; i++) {
            // Compute the norm of the work vector
            double norm = cblas_dznrm2(N, work_block[i].data(), 1);
            
            // If nearly zero, generate a new orthogonal vector
            if (norm < tol) {
                next_block[i] = generateOrthogonalVector(N, basis_vectors, gen, dist);
                chebyshev_filter(H, next_block[i], next_block[i], N, a, b, filter_degree);
                
                // Re-orthogonalize against basis
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, basis_vectors[k].data(), 1, next_block[i].data(), 1, &overlap);
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, next_block[i].data(), 1);
                }
                
                norm = cblas_dznrm2(N, next_block[i].data(), 1);
            } else {
                // Copy work to next
                next_block[i] = work_block[i];
            }
            
            // Set the diagonal beta element
            next_beta[i][i] = Complex(norm, 0.0);
            
            // Normalize
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, next_block[i].data(), 1);
            
            // Orthogonalize remaining work vectors against this one
            for (int j = i + 1; j < block_size; j++) {
                Complex overlap;
                cblas_zdotc_sub(N, next_block[i].data(), 1, work_block[j].data(), 1, &overlap);
                next_beta[j][i] = overlap;  // Off-diagonal beta element
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, next_block[i].data(), 1, work_block[j].data(), 1);
            }
        }
        
        beta.push_back(next_beta);
        
        // Store the new basis vectors
        if (j < num_steps - 1) {
            for (int i = 0; i < block_size; i++) {
                basis_vectors.push_back(next_block[i]);
            }
        }
        
        // Update for next iteration
        prev_block = curr_block;
        curr_block = next_block;
    }
    
    // Convert block tridiagonal matrix to regular format for solving
    int total_dim = basis_vectors.size();
    std::vector<Complex> block_matrix(total_dim * total_dim, Complex(0.0, 0.0));
    
    // Fill diagonal blocks (alpha)
    for (size_t j = 0; j < alpha.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = j * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix[col * total_dim + row] = alpha[j][r][c];
                }
            }
        }
    }
    
    // Fill off-diagonal blocks (beta)
    for (size_t j = 1; j < beta.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = (j-1) * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix[col * total_dim + row] = beta[j][r][c];
                    block_matrix[row * total_dim + col] = std::conj(beta[j][r][c]);
                }
            }
        }
    }
    
    // Diagonalize the block tridiagonal matrix
    std::vector<double> evals(total_dim);
    std::vector<Complex> evecs(total_dim * total_dim);
    
    if (eigenvectors) {
        evecs = block_matrix;  // Copy for LAPACK which overwrites input
    }
    
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, eigenvectors ? 'V' : 'N', 'U', 
                           total_dim, reinterpret_cast<lapack_complex_double*>(eigenvectors ? evecs.data() : block_matrix.data()), 
                           total_dim, evals.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        return;
    }
    
    // Store eigenvalues
    eigenvalues = evals;
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors) {
        eigenvectors->clear();
        eigenvectors->resize(total_dim, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Group eigenvalues into degenerate clusters
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < total_dim; i++) {
            bool added_to_cluster = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(evals[i] - evals[cluster[0]]) < degen_tol) {
                    cluster.push_back(i);
                    added_to_cluster = true;
                    break;
                }
            }
            if (!added_to_cluster) {
                degen_clusters.push_back({i});
            }
        }
        
        // Process each cluster separately
        for (const auto& cluster : degen_clusters) {
            if (cluster.size() == 1) {
                // Non-degenerate case
                int idx = cluster[0];
                for (int i = 0; i < N; i++) {
                    for (size_t k = 0; k < basis_vectors.size(); k++) {
                        (*eigenvectors)[idx][i] += evecs[k*total_dim + idx] * basis_vectors[k][i];
                    }
                }
            } else {
                // Degenerate case
                int subspace_dim = cluster.size();
                std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (int i = 0; i < N; i++) {
                        for (size_t k = 0; k < basis_vectors.size(); k++) {
                            subspace_vectors[c][i] += evecs[k*total_dim + idx] * basis_vectors[k][i];
                        }
                    }
                }
                
                // Orthogonalize within degenerate subspace
                for (int c = 0; c < subspace_dim; c++) {
                    for (int prev = 0; prev < c; prev++) {
                        Complex overlap;
                        cblas_zdotc_sub(N, subspace_vectors[prev].data(), 1, 
                                      subspace_vectors[c].data(), 1, &overlap);
                        
                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, subspace_vectors[prev].data(), 1, 
                                   subspace_vectors[c].data(), 1);
                    }
                    
                    // Normalize
                    double norm = cblas_dznrm2(N, subspace_vectors[c].data(), 1);
                    if (norm > tol) {
                        Complex scale = Complex(1.0/norm, 0.0);
                        cblas_zscal(N, &scale, subspace_vectors[c].data(), 1);
                    }
                }
                
                // Verify that vectors are accurate eigenvectors
                for (int c = 0; c < subspace_dim; c++) {
                    ComplexVector Hv(N);
                    H(subspace_vectors[c].data(), Hv.data(), N);
                    
                    // Compute Rayleigh quotient
                    Complex lambda;
                    cblas_zdotc_sub(N, subspace_vectors[c].data(), 1, Hv.data(), 1, &lambda);
                    
                    // Store the refined eigenvector
                    int idx = cluster[c];
                    (*eigenvectors)[idx] = subspace_vectors[c];
                }
            }
        }
        
        // Final verification of orthogonality
        for (int i = 0; i < total_dim; i++) {
            // Normalize
            double norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
        }
    }
}


// Full diagonalization using LAPACK for Hermitian matrices
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, int N,
                          std::vector<double>& eigenvalues, 
                          std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Construct the full matrix representation
    std::vector<Complex> full_matrix(N * N, Complex(0.0, 0.0));
    ComplexVector basis_vector(N, Complex(0.0, 0.0));
    ComplexVector result(N);
    
    // Apply H to each standard basis vector to get columns of the matrix
    for (int j = 0; j < N; j++) {
        // Create standard basis vector e_j
        std::fill(basis_vector.begin(), basis_vector.end(), Complex(0.0, 0.0));
        basis_vector[j] = Complex(1.0, 0.0);
        
        // Apply H to e_j
        H(basis_vector.data(), result.data(), N);
        
        // Store the result in the j-th column of the matrix (column major for LAPACK)
        for (int i = 0; i < N; i++) {
            full_matrix[j*N + i] = result[i];
        }
    }
    
    // Allocate array for eigenvalues
    eigenvalues.resize(N);
    
    // Prepare working space for eigenvectors if requested
    std::vector<Complex> work_eigenvectors;
    if (eigenvectors) {
        work_eigenvectors = full_matrix; // Copy the matrix since LAPACK overwrites it
    }
    
    // Call LAPACK eigensolver
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 
                           eigenvectors ? 'V' : 'N', // 'V' to compute eigenvectors, 'N' for eigenvalues only
                           'U',                      // Upper triangular part of the matrix is used
                           N, 
                           reinterpret_cast<lapack_complex_double*>(eigenvectors ? work_eigenvectors.data() : full_matrix.data()), 
                           N, 
                           eigenvalues.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        return;
    }
    
    // Convert eigenvectors if requested
    if (eigenvectors) {
        eigenvectors->resize(N);
        for (int i = 0; i < N; i++) {
            (*eigenvectors)[i].resize(N);
            for (int j = 0; j < N; j++) {
                (*eigenvectors)[i][j] = work_eigenvectors[i*N + j];
            }
        }
    }
}

// Diagonalization using ezARPACK
void arpack_diagonalization(std::function<void(const Complex*, Complex*, int)> H, int N, 
                           int nev, bool lowest, 
                           std::vector<double>& eigenvalues,
                           std::vector<ComplexVector>* eigenvectors = nullptr) {


    using solver_t = ezarpack::arpack_solver<ezarpack::Symmetric, ezarpack::eigen_storage>;
    solver_t solver(N);


    auto matrix_op = [&H](solver_t::vector_const_view_t in,
        solver_t::vector_view_t out) {
        // Convert Eigen types to raw pointers for H
        H(reinterpret_cast<const Complex*>(in.data()), 
          reinterpret_cast<Complex*>(out.data()), 
          in.size());
    };

    nev = std::min(nev, N-1);
    // Specify parameters for the solver
    using params_t = solver_t::params_t;
    params_t params(nev,               // Number of low-lying eigenvalues
                    params_t::Smallest, // We want the smallest eigenvalues
                    true);              // Yes, we want the eigenvectors
                                        // (Ritz vectors) as well
    
    // Run diagonalization!
    solver(matrix_op, params);

    // Extract the results
    // Get eigenvalues and eigenvectors from the solver
    auto const& eigenvalues_vector = solver.eigenvalues();
    auto const& eigenvectors_matrix = solver.eigenvectors();

    // Copy eigenvalues to the output vector
    eigenvalues.resize(nev);
    for(int i = 0; i < nev; i++) {
        eigenvalues[i] = eigenvalues_vector(i);
    }

    // Copy eigenvectors if needed
    if(eigenvectors) {
        eigenvectors->resize(nev, ComplexVector(N));
        for(int i = 0; i < nev; i++) {
            for(int j = 0; j < N; j++) {
                // With a symmetric solver, eigenvectors are real
                (*eigenvectors)[i][j] = Complex(
                    eigenvectors_matrix(j, i),
                    0.0
                );
            }
        }
    }
}

// Shift-Invert Lanczos algorithm for finding eigenvalues close to a specified target
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, 
                         double sigma, int max_iter, double tol, 
                         std::vector<double>& eigenvalues,
                         std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Define block size for handling degenerate eigenvalues
    const int block_size = 4;
    
    // Initialize random starting vectors
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Generate orthonormal set of starting vectors
    block_vectors[0] = generateRandomVector(N, gen, dist);
    for (int i = 1; i < block_size; i++) {
        block_vectors[i] = generateOrthogonalVector(N, 
                                                  std::vector<ComplexVector>(block_vectors.begin(), 
                                                                          block_vectors.begin() + i), 
                                                  gen, dist);
    }
    
    // Create sparse matrix representation for shift-invert operator
    using SparseMatrix = Eigen::SparseMatrix<Complex>;
    using VectorXcd = Eigen::VectorXcd;
    
    // Build the sparse matrix representation of H
    SparseMatrix H_sparse(N, N);
    std::vector<Eigen::Triplet<Complex>> triplets;
    
    // We need to estimate the number of non-zeros per row to reserve memory
    const int est_nnz_per_row = 10;  // Adjust based on your matrix's sparsity
    triplets.reserve(N * est_nnz_per_row);
    
    // Apply H to each standard basis vector and capture non-zero entries
    ComplexVector basis(N, Complex(0.0, 0.0));
    ComplexVector result(N);
    
    for (int j = 0; j < N; j++) {
        basis[j] = Complex(1.0, 0.0);
        H(basis.data(), result.data(), N);
        
        for (int i = 0; i < N; i++) {
            if (std::abs(result[i]) > 1e-12) {
                triplets.push_back(Eigen::Triplet<Complex>(i, j, result[i]));
            }
        }
        
        basis[j] = Complex(0.0, 0.0);
    }
    
    H_sparse.setFromTriplets(triplets.begin(), triplets.end());
    
    // Create (H - σI) matrix
    SparseMatrix shifted_H = H_sparse;
    for (int i = 0; i < N; i++) {
        shifted_H.coeffRef(i, i) -= Complex(sigma, 0.0);
    }
    
    // Create shift-invert operator (H - σI)^(-1)
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(shifted_H);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "ERROR: LU decomposition failed for shift-invert operator!" << std::endl;
        return;
    }
    
    // Define the shift-invert function (H - σI)^(-1) * v
    auto shift_invert_op = [&solver, N](const Complex* v, Complex* result, int size) {
        // Convert input to Eigen vector
        VectorXcd x(size);
        for (int i = 0; i < size; i++) {
            x(i) = v[i];
        }
        
        // Apply (H - σI)^(-1)
        VectorXcd y = solver.solve(x);
        
        // Convert result back
        for (int i = 0; i < size; i++) {
            result[i] = y(i);
        }
    };
    
    // Run Chebyshev filtered Lanczos with the shift-invert operator
    std::vector<double> si_eigenvalues;
    std::vector<ComplexVector> si_eigenvectors;
    
    chebyshev_filtered_lanczos(shift_invert_op, N, max_iter, tol, 
                              si_eigenvalues, 
                              eigenvectors ? &si_eigenvectors : nullptr);
    
    // Transform eigenvalues back to original space: if μ is an eigenvalue of (H - σI)^(-1),
    // then λ = σ + 1/μ is an eigenvalue of H
    eigenvalues.resize(si_eigenvalues.size());
    for (size_t i = 0; i < si_eigenvalues.size(); i++) {
        if (std::abs(si_eigenvalues[i]) > tol) {
            eigenvalues[i] = sigma + 1.0/si_eigenvalues[i];
        } else {
            // Handle possible division by zero
            eigenvalues[i] = sigma + 1.0/(si_eigenvalues[i] + tol);
        }
    }
    
    // The eigenvectors are unchanged by the transformation
    if (eigenvectors) {
        *eigenvectors = si_eigenvectors;
        
        // Verify and refine eigenvectors
        for (size_t i = 0; i < eigenvectors->size(); i++) {
            // Apply original H to verify/refine
            ComplexVector Hv(N);
            H((*eigenvectors)[i].data(), Hv.data(), N);
            
            // Compute Rayleigh quotient for verification
            Complex rayleigh_quotient;
            cblas_zdotc_sub(N, (*eigenvectors)[i].data(), 1, Hv.data(), 1, &rayleigh_quotient);
            
            // Optional: Refine using a few steps of power iteration if needed
            double error = std::abs(rayleigh_quotient.real() - eigenvalues[i]);
            if (error > tol * 10) {
                // Apply one step of inverse iteration for refinement
                ComplexVector temp(N);
                shift_invert_op((*eigenvectors)[i].data(), temp.data(), N);
                
                // Normalize
                double norm = cblas_dznrm2(N, temp.data(), 1);
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, temp.data(), 1);
                
                // Replace with refined vector
                (*eigenvectors)[i] = temp;
            }
        }
    }
    
    // Sort eigenvalues and eigenvectors by distance from target sigma
    std::vector<size_t> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
             [&eigenvalues, sigma](size_t i1, size_t i2) {
                 return std::abs(eigenvalues[i1] - sigma) < std::abs(eigenvalues[i2] - sigma);
             });
    
    // Create sorted copies
    std::vector<double> sorted_eigenvalues(eigenvalues.size());
    std::vector<ComplexVector> sorted_eigenvectors;
    
    if (eigenvectors) {
        sorted_eigenvectors.resize(eigenvectors->size());
    }
    
    for (size_t i = 0; i < indices.size(); i++) {
        sorted_eigenvalues[i] = eigenvalues[indices[i]];
        if (eigenvectors) {
            sorted_eigenvectors[i] = (*eigenvectors)[indices[i]];
        }
    }
    
    // Replace with sorted versions
    eigenvalues = sorted_eigenvalues;
    if (eigenvectors) {
        *eigenvectors = sorted_eigenvectors;
    }
}

int main(){
    int num_site = 16;
    Operator op(num_site);
    op.loadFromFile("./ED_test/Trans.def");
    op.loadFromInterAllFile("./ED_test/InterAll.def");
    std::vector<double> eigenvalues;
    // std::vector<ComplexVector> eigenvectors;
    chebyshev_filtered_lanczos([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, (1<<num_site), 1000, 1e-10, eigenvalues);

    std::vector<double> eigenvalues_lanczos;
    block_lanczos([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, (1<<num_site), 1000, 1e-10, eigenvalues_lanczos);

    // Print the results
    // std::cout << "Eigenvalues:" << std::endl;
    // for (size_t i = 0; i < 20; i++) {
    //     std::cout << "Eigenvalue " << i << " Chebyshev Filtered Lanczos: " << eigenvalues[i] << " Lanczos: " << eigenvalues_lanczos[i] << std::endl;
    // }
    // Run full diagonalization for comparison
    // std::vector<double> full_eigenvalues;
    // full_diagonalization([&](const Complex* v, Complex* Hv, int N) {
    //     std::vector<Complex> vec(v, v + N);
    //     std::vector<Complex> result(N, Complex(0.0, 0.0));
    //     result = op.apply(vec);
    //     std::copy(result.begin(), result.end(), Hv);
    // }, 1<<num_site, full_eigenvalues);

    // // Sort both sets of eigenvalues for comparison
    // std::sort(eigenvalues.begin(), eigenvalues.end());
    // std::sort(full_eigenvalues.begin(), full_eigenvalues.end());

    // // Compare and print results
    // std::cout << "\nComparison between Lanczos and Full Diagonalization:" << std::endl;
    // std::cout << "Index | Lanczos        | Full          | Difference" << std::endl;
    // std::cout << "------------------------------------------------------" << std::endl;

    // int num_to_compare = std::min(eigenvalues.size(), full_eigenvalues.size());
    // num_to_compare = std::min(num_to_compare, 20);  // Limit to first 20 eigenvalues

    // for (int i = 0; i < num_to_compare; i++) {
    //     double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
    //     std::cout << std::setw(5) << i << " | " 
    //               << std::setw(14) << std::fixed << std::setprecision(10) << eigenvalues[i] << " | "
    //               << std::setw(14) << std::fixed << std::setprecision(10) << full_eigenvalues[i] << " | "
    //               << std::setw(10) << std::scientific << std::setprecision(3) << diff << std::endl;
    // }

    // // Calculate and print overall accuracy statistics
    // if (num_to_compare > 0) {
    //     double max_diff = 0.0;
    //     double sum_diff = 0.0;
    //     for (int i = 0; i < num_to_compare; i++) {
    //         double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
    //         max_diff = std::max(max_diff, diff);
    //         sum_diff += diff;
    //     }
    //     double avg_diff = sum_diff / num_to_compare;
        
    //     std::cout << "\nAccuracy statistics:" << std::endl;
    //     std::cout << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
    //     std::cout << "Average difference: " << std::scientific << std::setprecision(3) << avg_diff << std::endl;
        
    //     // Special focus on ground state and first excited state
    //     if (full_eigenvalues.size() > 0 && eigenvalues.size() > 0) {
    //         double ground_diff = std::abs(eigenvalues[0] - full_eigenvalues[0]);
    //         std::cout << "Ground state error: " << std::scientific << std::setprecision(3) << ground_diff << std::endl;
            
    //         if (full_eigenvalues.size() > 1 && eigenvalues.size() > 1) {
    //             double excited_diff = std::abs(eigenvalues[1] - full_eigenvalues[1]);
    //             std::cout << "First excited state error: " << std::scientific << std::setprecision(3) << excited_diff << std::endl;
    //         }
    //     }
    // }

    return 0;
}


// int main(){
//     // Matrix size (not too large to keep computation reasonable)
//     const int N = 500; 

//     // Generate a random Hermitian matrix
//     std::vector<std::vector<Complex>> randomMatrix(N, std::vector<Complex>(N));
//     std::mt19937 gen(42); // Fixed seed for reproducibility
//     std::uniform_real_distribution<double> dist(-1.0, 1.0);

//     // Fill with random values and make it Hermitian
//     for (int i = 0; i < N; i++) {
//         randomMatrix[i][i] = Complex(dist(gen), 0.0); // Real diagonal
//         for (int j = i+1; j < N; j++) {
//             randomMatrix[i][j] = Complex(dist(gen), dist(gen));
//             randomMatrix[j][i] = std::conj(randomMatrix[i][j]);
//         }
//     }

//     // Define matrix-vector multiplication function
//     auto matVecMult = [&](const Complex* v, Complex* result, int size) {
//         std::fill(result, result + size, Complex(0.0, 0.0));
//         for (int i = 0; i < size; i++) {
//             for (int j = 0; j < size; j++) {
//                 result[i] += randomMatrix[i][j] * v[j];
//             }
//         }
//     };

//     // Test all three methods
//     std::cout << "Testing with " << N << "x" << N << " random Hermitian matrix\n";

//     // Regular Lanczos
//     std::vector<double> lanczosEigenvalues;
//     std::vector<ComplexVector> lanczosEigenvectors;
//     chebyshev_filtered_lanczos(matVecMult, N, N/2, 1e-10, lanczosEigenvalues, &lanczosEigenvectors, -3, 3, 20);

//     // Lanczos with CG refinement
//     std::vector<double> lanczosCGEigenvalues;
//     std::vector<ComplexVector> lanczosCGEigenvectors;
//     lanczos(matVecMult, N, N/2, 1e-10, lanczosCGEigenvalues, &lanczosCGEigenvectors);

//     // Direct diagonalization
//     std::vector<Complex> flatMatrix(N * N);
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             flatMatrix[j*N + i] = randomMatrix[i][j];
//         }
//     }

//     std::vector<double> directEigenvalues(N);
//     int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', N, 
//                           reinterpret_cast<lapack_complex_double*>(flatMatrix.data()), 
//                           N, directEigenvalues.data());

//     if (info == 0) {
//         // Compare results
//         std::cout << "\nEigenvalue comparison:\n";
//         std::cout << "Index | Direct  | Lanczos | Diff    | Lanczos+CG | Diff\n";
//         std::cout << "--------------------------------------------------------\n";
//         int numToShow = std::min(10, N/2);
//         for (int i = 0; i < numToShow; i++) {
//             std::cout << std::setw(5) << i << " | "
//                     << std::fixed << std::setprecision(6)
//                     << std::setw(8) << directEigenvalues[i] << " | "
//                     << std::setw(7) << lanczosEigenvalues[i] << " | "
//                     << std::setw(7) << std::abs(directEigenvalues[i] - lanczosEigenvalues[i]) << " | "
//                     << std::setw(10) << lanczosCGEigenvalues[i] << " | "
//                     << std::setw(7) << std::abs(directEigenvalues[i] - lanczosCGEigenvalues[i]) << "\n";
//         }
//     }

// }
