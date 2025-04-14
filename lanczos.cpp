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
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>
#include <fstream>
#include <set>

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
#include <chrono>
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        #pragma omp critical
        {
            v_current[i] = Complex(real, imag);
        }
    }

    std::cout << "Lanczos: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Initialize Lanczos vectors and coefficients
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "Lanczos: Iterating..." << std::endl;   
    

    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization (twice for numerical stability)
        // for (int iter = 0; iter < 2; iter++) {
        //     #pragma omp parallel
        //     {
        //         ComplexVector private_w(N, Complex(0.0, 0.0));
                
        //         #pragma omp for nowait
        //         for (size_t k = 0; k <= j; k++) {
        //             Complex overlap;
        //             cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
        //             Complex neg_overlap = -overlap;
                    
        //             #pragma omp critical
        //             {
        //                 cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
        //             }
        //         }
        //     }
        // }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        
        // Check for invariant subspace
        // if (norm < tol) {
        //     // Generate a random vector orthogonal to basis
        //     v_next = generateRandomVector(N, gen, dist);
            
        //     // Orthogonalize against all basis vectors
        //     for (int iter = 0; iter < 2; iter++) {
        //         #pragma omp parallel
        //         {
        //             #pragma omp for nowait
        //             for (size_t k = 0; k < basis_vectors.size(); k++) {
        //                 Complex overlap;
        //                 cblas_zdotc_sub(N, basis_vectors[k].data(), 1, v_next.data(), 1, &overlap);
        //                 Complex neg_overlap = -overlap;
                        
        //                 #pragma omp critical
        //                 {
        //                     cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, v_next.data(), 1);
        //                 }
        //             }
        //         }
        //     }
            
        //     // Update the norm
        //     norm = cblas_dznrm2(N, v_next.data(), 1);
            
        //     // If still too small, we've reached an invariant subspace
        //     if (norm < tol) {
        //         break;
        //     }
        // } else {
        //     #pragma omp parallel for
        //     for (int i = 0; i < N; i++) {
        //         v_next[i] = w[i];
        //     }
        // }
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i];
        }
        beta.push_back(norm);
        
        // Normalize v_next
        scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, v_next.data(), 1);
        
        // Store basis vector
        if (eigenvectors){
            if (j < max_iter - 1) {
                basis_vectors.push_back(v_next);
            }
        }
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "Lanczos: Constructing tridiagonal matrix" << std::endl;

    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }

    std::cout << "Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    std::vector<double> evals(m);        // For eigenvalues
    std::vector<double> evecs;           // For eigenvectors if needed
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    char range = 'A';                      // Compute all eigenvalues
    int info;
    
    if (eigenvectors) {
        // Need space for eigenvectors
        evecs.resize(m*m);
        
        // Call LAPACK to compute eigenvalues and eigenvectors of tridiagonal matrix
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        return;
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        // Identify clusters of degenerate eigenvalues (only for the first n_eigenvalues)
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < n_eigenvalues; i++) {
            bool added_to_cluster = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(diag[i] - diag[cluster[0]]) < degen_tol) {
                    cluster.push_back(i);
                    added_to_cluster = true;
                    break;
                }
            }
            if (!added_to_cluster) {
                degen_clusters.push_back({i});
            }
        }
        
        // Transform to original basis and handle degeneracy - only for first n_eigenvalues
        eigenvectors->clear();
        eigenvectors->resize(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Process each cluster separately - this can be parallelized
        #pragma omp parallel for schedule(dynamic)
        for (size_t cl = 0; cl < degen_clusters.size(); cl++) {
            const auto& cluster = degen_clusters[cl];
            
            if (cluster.size() == 1) {
                // Non-degenerate case - standard treatment
                int idx = cluster[0];
                ComplexVector evec(N, Complex(0.0, 0.0));
                
                // Transform: evec = sum_k z(k,idx) * basis_vectors[k]
                for (int k = 0; k < m; k++) {
                    Complex coeff(evecs[k*m + idx], 0.0);
                    cblas_zaxpy(N, &coeff, basis_vectors[k].data(), 1, evec.data(), 1);
                }
                
                // Normalize
                double norm = cblas_dznrm2(N, evec.data(), 1);
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, evec.data(), 1);
                
                (*eigenvectors)[idx] = evec;
            } else {
                // Degenerate case - special handling
                int subspace_dim = cluster.size();
                std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N, Complex(0.0, 0.0)));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (int k = 0; k < m; k++) {
                        Complex coeff(evecs[k*m + idx], 0.0);
                        cblas_zaxpy(N, &coeff, basis_vectors[k].data(), 1, subspace_vectors[c].data(), 1);
                    }
                }
                
                // Re-orthogonalize within degenerate subspace
                for (int c = 0; c < subspace_dim; c++) {
                    // Normalize current vector
                    double norm = cblas_dznrm2(N, subspace_vectors[c].data(), 1);
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, subspace_vectors[c].data(), 1);
                    
                    // Orthogonalize against previous vectors
                    for (int prev = 0; prev < c; prev++) {
                        Complex overlap;
                        cblas_zdotc_sub(N, subspace_vectors[prev].data(), 1, subspace_vectors[c].data(), 1, &overlap);
                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, subspace_vectors[prev].data(), 1, subspace_vectors[c].data(), 1);
                    }
                    
                    // Renormalize if necessary
                    norm = cblas_dznrm2(N, subspace_vectors[c].data(), 1);
                    if (norm > tol) {
                        scale = Complex(1.0/norm, 0.0);
                        cblas_zscal(N, &scale, subspace_vectors[c].data(), 1);
                    }
                    
                    // Store vector
                    int idx = cluster[c];
                    (*eigenvectors)[idx] = subspace_vectors[c];
                }
            }
        }
        
        // Optionally refine eigenvectors using conjugate gradient - only for n_eigenvalues
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_eigenvalues; i++) {
            refine_eigenvector_with_cg(H, (*eigenvectors)[i], eigenvalues[i], N, tol);
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
    lanczos(H, N, num_eigenvalues, sample_iter, 1e-10, sample_eigenvalues);
    
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
                               int N, int max_iter, int exct, double tol, 
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
        ChebysehvFilterParams params = estimate_filter_parameters(H, N, exct, true);
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
    
    // Store only the first exct eigenvalues
    int n_eigenvalues = std::min(exct, total_dim);
    eigenvalues.resize(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        eigenvalues[i] = evals[i];
    }
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors) {
        eigenvectors->clear();
        eigenvectors->resize(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Group eigenvalues into degenerate clusters (only for the first n_eigenvalues)
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < n_eigenvalues; i++) {
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
        
        // Final verification of orthogonality (only for first n_eigenvalues)
        for (int i = 0; i < n_eigenvalues; i++) {
            // Normalize
            double norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
        }
    }
}

// Shift-and-invert Lanczos algorithm for better convergence to interior eigenvalues
// H: Function representing the Hamiltonian operator (H|v⟩)
// N: Dimension of the Hilbert space
// max_iter: Maximum number of Lanczos iterations
// shift: The shift value (σ) targeting eigenvalues near this value
// tol: Tolerance for convergence and detecting invariant subspaces
// eigenvalues: Output vector to store the eigenvalues
// eigenvectors: Output matrix to store the eigenvectors (optional)
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H,
                         int N, int max_iter, double shift, double tol,
                         std::vector<double>& eigenvalues,
                         std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize random starting vector
    ComplexVector v_current(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    v_current = generateRandomVector(N, gen, dist);
    
    // Initialize Lanczos vectors and coefficients
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N), temp(N);
    
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // Linear system solver parameters
    const int max_cg_iter = 1000;
    const double cg_tol = tol * 0.1;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // Apply shift-and-invert operator: w = (H - σI)^(-1) * v_current
        // This requires solving the linear system (H - σI)w = v_current
        
        // Initialize solution vector w to zeroes
        std::fill(w.begin(), w.end(), Complex(0.0, 0.0));
        
        // Use Conjugate Gradient to solve (H - σI)w = v_current
        ComplexVector r = v_current;  // Initial residual
        ComplexVector p = r;          // Initial search direction
        ComplexVector Hp(N);          // Temporary vector for H*p
        
        double res_norm = cblas_dznrm2(N, r.data(), 1);
        double init_norm = res_norm;
        
        for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol * init_norm; iter++) {
            // Apply (H - σI) to p
            H(p.data(), Hp.data(), N);
            for (int i = 0; i < N; i++) {
                Hp[i] -= Complex(shift, 0.0) * p[i];
            }
            
            // Calculate step size α = (r·r) / (p·(H-σI)p)
            Complex r_dot_r, p_dot_Hp;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &p_dot_Hp);
            
            Complex alpha_cg = r_dot_r / p_dot_Hp;
            
            // Update solution: w += α*p
            cblas_zaxpy(N, &alpha_cg, p.data(), 1, w.data(), 1);
            
            // Store old r·r
            Complex r_dot_r_old = r_dot_r;
            
            // Update residual: r -= α*(H-σI)p
            Complex neg_alpha_cg = -alpha_cg;
            cblas_zaxpy(N, &neg_alpha_cg, Hp.data(), 1, r.data(), 1);
            
            // Check convergence
            res_norm = cblas_dznrm2(N, r.data(), 1);
            
            // Compute β = (r_new·r_new) / (r_old·r_old)
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            Complex beta_cg = r_dot_r / r_dot_r_old;
            
            // Update search direction: p = r + β*p
            for (int k = 0; k < N; k++) {
                p[k] = r[k] + beta_cg * p[k];
            }
        }
        
        // Subtract projections: w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // Compute alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization (twice for numerical stability)
        for (int iter = 0; iter < 2; iter++) {
            for (int k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
            }
        }
        
        // beta_{j+1} = ||w||
        double norm = cblas_dznrm2(N, w.data(), 1);
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random orthogonal vector
            v_next = generateOrthogonalVector(N, basis_vectors, gen, dist);
            norm = cblas_dznrm2(N, v_next.data(), 1);
            if (norm < tol) {
                break;  // No more orthogonal vectors can be found
            }
        } else {
            cblas_zcopy(N, w.data(), 1, v_next.data(), 1);
        }
        
        beta.push_back(norm);
        
        // Scale v_next by 1/norm
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, v_next.data(), 1);
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct tridiagonal matrix
    int m = alpha.size();
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(beta.begin() + 1, beta.end()); // Skip β_0
    
    // Diagonalize tridiagonal matrix
    std::vector<double> evals(m);
    std::vector<double> z(m * m, 0.0);
    
    int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, eigenvectors ? 'V' : 'N', m, 
                           diag.data(), offdiag.data(), z.data(), m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstev failed with error code " << info << std::endl;
        return;
    }
    
    // Convert eigenvalues back to original problem: λ = σ + 1/θ
    eigenvalues.resize(m);
    for (int i = 0; i < m; i++) {
        // Make sure we don't divide by zero
        if (std::abs(diag[i]) > 1e-12) {
            eigenvalues[i] = shift + 1.0/diag[i];
        } else {
            eigenvalues[i] = shift;  // In case of zero eigenvalue
        }
    }
    
    // Sort eigenvalues by distance from shift (closest first)
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), 
             [&](int a, int b) { 
                 return std::abs(eigenvalues[a] - shift) < std::abs(eigenvalues[b] - shift); 
             });
    
    // Create sorted copies
    std::vector<double> sorted_eigenvalues(m);
    std::vector<double> sorted_z;
    if (eigenvectors) sorted_z.resize(m * m);
    
    for (int i = 0; i < m; i++) {
        sorted_eigenvalues[i] = eigenvalues[indices[i]];
        if (eigenvectors) {
            for (int j = 0; j < m; j++) {
                sorted_z[j*m + i] = z[j*m + indices[i]];
            }
        }
    }
    
    // Replace with sorted versions
    eigenvalues = sorted_eigenvalues;
    if (eigenvectors) z = sorted_z;
    
    // Transform eigenvectors if requested
    if (eigenvectors) {
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < m; k++) {
                    (*eigenvectors)[i][j] += z[k*m + i] * basis_vectors[k][j];
                }
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
        }
        
        // Refine eigenvectors if needed
        for (int i = 0; i < m; i++) {
            double lambda = eigenvalues[i];
            refine_eigenvector_with_cg(H, (*eigenvectors)[i], lambda, N, tol);
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

// Shift-Invert Lanczos algorithm implementation
// H: Function representing the Hamiltonian operator (H|v⟩)
// N: Dimension of the Hilbert space
// max_iter: Maximum number of Lanczos iterations
// shift: Value around which to find eigenvalues (σ)
// exct: Number of eigenvalues to return
// tol: Tolerance for convergence and detecting invariant subspaces
// eigenvalues: Output vector to store the eigenvalues
// eigenvectors: Output matrix to store the eigenvectors (optional)
void shift_invert_lanczos(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N, int max_iter, double shift, int exct,
    double tol, std::vector<double>& eigenvalues,
    std::vector<ComplexVector>* eigenvectors = nullptr
) {
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        #pragma omp critical
        {
            v_current[i] = Complex(real, imag);
        }
    }
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Initialize Lanczos vectors and coefficients
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // Parameters for Conjugate Gradient solver
    const int max_cg_iter = 1000;
    const double cg_tol = tol * 0.1;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // Apply (H - σI)^(-1) to v_j using Conjugate Gradient
        // We're solving (H - σI)w = v_current
        
        // Initialize solution vector w to zero
        std::fill(w.begin(), w.end(), Complex(0.0, 0.0));
        
        // Initialize residual and search direction for CG
        ComplexVector r = v_current;  // Initial residual = right-hand side
        ComplexVector p = r;          // Initial search direction
        ComplexVector Hp(N);          // Temporary vector
        
        double res_norm = cblas_dznrm2(N, r.data(), 1);
        double init_norm = res_norm;
        
        // CG iteration to solve (H - σI)w = v_current
        for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol * init_norm; iter++) {
            // Apply (H - σI) to p
            H(p.data(), Hp.data(), N);
            for (int i = 0; i < N; i++) {
                Hp[i] -= Complex(shift, 0.0) * p[i];
            }
            
            // Calculate step size α = (r·r) / (p·(H-σI)p)
            Complex r_dot_r, p_dot_Hp;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &p_dot_Hp);
            
            Complex alpha_cg = r_dot_r / p_dot_Hp;
            
            // Update solution: w += α*p
            cblas_zaxpy(N, &alpha_cg, p.data(), 1, w.data(), 1);
            
            // Store old r·r
            Complex r_dot_r_old = r_dot_r;
            
            // Update residual: r -= α*(H-σI)p
            Complex neg_alpha_cg = -alpha_cg;
            cblas_zaxpy(N, &neg_alpha_cg, Hp.data(), 1, r.data(), 1);
            
            // Check convergence
            res_norm = cblas_dznrm2(N, r.data(), 1);
            
            // Update β = (r_new·r_new) / (r_old·r_old)
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            Complex beta_cg = r_dot_r / r_dot_r_old;
            
            // Update search direction: p = r + β*p
            for (int k = 0; k < N; k++) {
                p[k] = r[k] + beta_cg * p[k];
            }
        }
        
        // Now w contains (H - σI)^(-1) * v_current
        
        // Subtract projections: w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // Compute α_j = ⟨v_j, w⟩
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - α_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization for numerical stability
        for (int iter = 0; iter < 2; iter++) {  // Do twice for better stability
            for (size_t k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
            }
        }
        
        // β_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector orthogonal to the basis
            v_next = generateOrthogonalVector(N, basis_vectors, gen, dist);
            
            // Double check orthogonality
            for (int iter = 0; iter < 2; iter++) {
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, basis_vectors[k].data(), 1, v_next.data(), 1, &overlap);
                    
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, v_next.data(), 1);
                }
            }
            
            // Re-normalize
            norm = cblas_dznrm2(N, v_next.data(), 1);
            if (norm < tol) {
                // No more orthogonal vectors can be found
                break;
            }
        } else {
            // v_next = w / norm
            for (int i = 0; i < N; i++) {
                v_next[i] = w[i] / norm;
            }
        }
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
        }
        
        // Update vectors for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct tridiagonal matrix
    int m = alpha.size();
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m-1);
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Save only the requested number of eigenvalues, or all if m < exct
    int n_eigenvalues = std::min(exct, m);
    std::vector<double> evals(m);
    std::vector<double> evecs;
    
    char jobz = eigenvectors ? 'V' : 'N';
    int info;
    
    if (eigenvectors) {
        evecs.resize(m*m);
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        return;
    }
    
    // Convert eigenvalues back to the original problem
    // If θ is an eigenvalue of (H - σI)^(-1), then λ = σ + 1/θ is an eigenvalue of H
    eigenvalues.resize(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        // Eigenvalues of the tridiagonal matrix are in ascending order
        // We want those closest to the shift, which are the largest in magnitude
        if (std::abs(diag[i]) > 1e-10) {  // Avoid division by zero
            eigenvalues[i] = shift + 1.0 / diag[i];
        } else {
            eigenvalues[i] = shift;  // If θ ≈ 0, then λ ≈ ∞, but realistically it's just a poor approximation
        }
    }
    
    // Sort eigenvalues by distance from shift
    std::vector<int> indices(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return std::abs(eigenvalues[a] - shift) < std::abs(eigenvalues[b] - shift);
    });
    
    // Reorder eigenvalues
    std::vector<double> sorted_evals(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        sorted_evals[i] = eigenvalues[indices[i]];
    }
    eigenvalues = sorted_evals;
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        eigenvectors->resize(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));
        
        for (int i = 0; i < n_eigenvalues; i++) {
            int idx = indices[i];
            
            // Combine Lanczos basis vectors to get eigenvector in original basis
            for (int j = 0; j < m; j++) {
                Complex coef(evecs[idx*m + j], 0.0);
                cblas_zaxpy(N, &coef, basis_vectors[j].data(), 1, (*eigenvectors)[i].data(), 1);
            }
            
            // Normalize the eigenvector
            double evec_norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/evec_norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
        }
        
        // Optionally refine eigenvectors
        for (int i = 0; i < n_eigenvalues; i++) {
            refine_eigenvector_with_cg(H, (*eigenvectors)[i], eigenvalues[i], N, tol);
        }
    }
}



// Calculate thermodynamic quantities directly from eigenvalues
struct ThermodynamicData {
    std::vector<double> temperatures;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
};

ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min = 0.01,
    double T_max = 10.0,
    int num_points = 100,
    double k_B = 1.0  // Set to 1.0 for natural units, or use 8.6173e-5 for eV/K
) {
    // Initialize results structure
    ThermodynamicData results;
    results.temperatures.resize(num_points);
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Find ground state energy for normalization
    double E_0 = eigenvalues.empty() ? 0.0 : *std::min_element(eigenvalues.begin(), eigenvalues.end());
    
    // Generate logarithmically spaced temperature points
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // For each temperature point
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        if (eigenvalues.empty()) {
            results.energy[i] = 0.0;
            results.specific_heat[i] = 0.0;
            results.free_energy[i] = 0.0;
            results.entropy[i] = 0.0;
            continue;
        }
        
        // Calculate log(Z) using log-sum-exp trick to avoid overflow
        double max_exp = -beta * (eigenvalues[0] - E_0);
        for (size_t j = 1; j < eigenvalues.size(); j++) {
            max_exp = std::max(max_exp, -beta * (eigenvalues[j] - E_0));
        }
        
        // Calculate partition function in log space
        std::vector<double> exp_terms(eigenvalues.size());
        for (size_t j = 0; j < eigenvalues.size(); j++) {
            exp_terms[j] = -beta * (eigenvalues[j] - E_0) - max_exp;
        }
        
        double sum_exp = 0.0;
        for (double term : exp_terms) {
            sum_exp += std::exp(term);
        }
        double log_Z = max_exp + std::log(sum_exp) - beta * E_0;
        
        // Calculate energy using logarithmic weights
        double avg_energy = 0.0;
        double Z = std::exp(log_Z);
        for (size_t j = 0; j < eigenvalues.size(); j++) {
            double weight = std::exp(-beta * eigenvalues[j]) / Z;
            avg_energy += eigenvalues[j] * weight;
        }
        
        // Calculate energy squared directly
        double avg_energy_squared = 0.0;
        for (size_t j = 0; j < eigenvalues.size(); j++) {
            double weight = std::exp(-beta * eigenvalues[j]) / Z;
            avg_energy_squared += eigenvalues[j] * eigenvalues[j] * weight;
        }
        
        // Calculate free energy
        double free_energy = -T * log_Z;
        
        // Calculate specific heat
        double specific_heat = k_B * beta * beta * (avg_energy_squared - avg_energy * avg_energy);
        
        // Calculate entropy
        double entropy = k_B * (beta * avg_energy - free_energy / T);
        
        // Store results
        results.energy[i] = avg_energy;
        results.specific_heat[i] = specific_heat;
        results.free_energy[i] = free_energy;
        results.entropy[i] = entropy;
    }
    
    return results;
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


// Calculate the expectation value <ψ_a|A|ψ_a> for the a-th eigenstate of H
Complex calculate_expectation_value(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                                // Dimension of Hilbert space
    int a = 0,                                           // Index of eigenstate (default: ground state)
    int max_iter = 100,                                  // Maximum iterations for eigenstate calculation
    double tol = 1e-10                                   // Tolerance
) {
    // First, calculate the a-th eigenstate of H
    std::vector<double> eigenvalues;
    std::vector<ComplexVector> eigenvectors;
    
    // Use Chebyshev filtered Lanczos for better accuracy
    chebyshev_filtered_lanczos(H, N, max_iter, max_iter, tol, eigenvalues, &eigenvectors);
    
    // Check if we have enough eigenstates
    if (a >= eigenvectors.size()) {
        std::cerr << "Error: Requested eigenstate index " << a 
                  << " but only " << eigenvectors.size() << " states computed." << std::endl;
        return Complex(0.0, 0.0);
    }
    
    // Get the a-th eigenstate
    const ComplexVector& psi = eigenvectors[a];
    ComplexVector A_psi(N);
    
    // Apply operator A to the eigenstate
    A(psi.data(), A_psi.data(), N);
    
    // Calculate <ψ_a|A|ψ_a>
    Complex expectation_value;
    cblas_zdotc_sub(N, psi.data(), 1, A_psi.data(), 1, &expectation_value);
    
    return expectation_value;
}

// Calculate the expectation value <ψ_a|A|ψ_b> between two different eigenstates
Complex calculate_matrix_element(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                                // Dimension of Hilbert space
    int a = 0,                                           // Index of first eigenstate
    int b = 1,                                           // Index of second eigenstate
    int max_iter = 100,                                  // Maximum iterations for eigenstate calculation
    double tol = 1e-10                                   // Tolerance
) {
    // Calculate eigenstates of H
    std::vector<double> eigenvalues;
    std::vector<ComplexVector> eigenvectors;
    
    // Compute enough states to cover both indices
    int max_index = std::max(a, b);
    chebyshev_filtered_lanczos(H, N, max_index + 30, max_index +30, tol, eigenvalues, &eigenvectors);
    
    // Check if we have enough eigenstates
    if (a >= eigenvectors.size() || b >= eigenvectors.size()) {
        std::cerr << "Error: Requested eigenstate indices " << a << " and " << b
                  << " but only " << eigenvectors.size() << " states computed." << std::endl;
        return Complex(0.0, 0.0);
    }
    
    // Get the eigenstates
    const ComplexVector& psi_a = eigenvectors[a];
    const ComplexVector& psi_b = eigenvectors[b];
    ComplexVector A_psi_b(N);
    
    // Apply operator A to |ψ_b⟩
    A(psi_b.data(), A_psi_b.data(), N);
    
    // Calculate <ψ_a|A|ψ_b>
    Complex matrix_element;
    cblas_zdotc_sub(N, psi_a.data(), 1, A_psi_b.data(), 1, &matrix_element);
    
    return matrix_element;
}



// Finite Temperature Lanczos Method (FTLM)
// Calculates <A> = Tr(A*e^(-βH))/Tr(e^(-βH)) for inverse temperature β
Complex FTLM(
    std::function<void(const Complex*, Complex*, int)> H, // Hamiltonian matrix-vector product
    std::function<void(const Complex*, Complex*, int)> A, // Observable matrix-vector product
    int N,             // Dimension of Hilbert space
    double beta,          // Inverse temperature (β = 1/kT)
    int r_max = 20,    // Number of random vectors for sampling the trace
    int m_max = 100,   // Maximum Lanczos iterations per random vector
    double tol = 1e-10 // Tolerance
) {
    // Initialize random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Accumulators for traces
    Complex trace_A_exp_H = 0.0;
    double trace_exp_H = 0.0;
    
    // For each random vector
    for (int r = 0; r < r_max; r++) {
        // Generate random starting vector
        ComplexVector v0(N);
        for (int i = 0; i < N; i++) {
            v0[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v0.data(), 1);
        
        // Store the starting vector as first basis vector
        std::vector<ComplexVector> basis_vectors;
        basis_vectors.push_back(v0);
        
        // Run Chebyshev filtered Lanczos to get eigenvalues and eigenvectors
        std::vector<double> evals;
        std::vector<ComplexVector> evecs;
        
        // Call the already implemented chebyshev_filtered_lanczos function
        chebyshev_filtered_lanczos(H, N, m_max, m_max, tol, evals, &evecs);
        
        int m = evals.size();
        
        // Compute reduced matrix elements of A in the Lanczos basis
        std::vector<std::vector<Complex>> A_reduced(m, std::vector<Complex>(m, 0.0));
        
        ComplexVector Av(N);
        for (int i = 0; i < m; i++) {
            // Apply A to |ψᵢ⟩
            A(evecs[i].data(), Av.data(), N);
            
            // Compute matrix elements ⟨ψⱼ|A|ψᵢ⟩
            for (int j = 0; j < m; j++) {
                // Calculate ⟨ψⱼ|A|ψᵢ⟩
                Complex matrix_element;
                cblas_zdotc_sub(N, evecs[j].data(), 1, Av.data(), 1, &matrix_element);
                A_reduced[j][i] = matrix_element;
            }
        }
        
        // Calculate contributions to the thermal traces
        for (int i = 0; i < m; i++) {
            double exp_factor = std::exp(-beta * evals[i]);
            trace_exp_H += exp_factor;
            trace_A_exp_H += A_reduced[i][i] * exp_factor;
        }
    }
    
    // Return thermal average <A> = Tr(A*e^(-βH))/Tr(e^(-βH))
    return trace_A_exp_H / trace_exp_H;
}

// Calculate the dynamical correlation function S_AB(ω) using FTLM
// S_AB(ω) = (1/Z) ∑_n,m e^(-βE_n) ⟨n|A|m⟩⟨m|B|n⟩ δ(ω - (E_m - E_n))
std::vector<std::pair<double, Complex>> FTLM_dynamical(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A,  // First operator
    std::function<void(const Complex*, Complex*, int)> B,  // Second operator (use A for auto-correlation)
    int N,                    // Dimension of Hilbert space
    double beta,              // Inverse temperature β = 1/kT
    double omega_min,         // Minimum frequency
    double omega_max,         // Maximum frequency
    int n_points,             // Number of frequency points
    double eta,               // Broadening parameter (half-width of Lorentzian)
    int r_max = 30,           // Number of random vectors for sampling
    int m_max = 100           // Maximum Lanczos iterations per random vector
) {
    // Generate frequency grid
    std::vector<double> omega_values(n_points);
    double delta_omega = (omega_max - omega_min) / (n_points - 1);
    for (int i = 0; i < n_points; i++) {
        omega_values[i] = omega_min + i * delta_omega;
    }
    
    // Initialize result vector with zeros
    std::vector<Complex> response(n_points, Complex(0.0, 0.0));
    
    // Initialize random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Accumulators for the partition function
    double Z = 0.0;
    
    // For each random vector
    for (int r = 0; r < r_max; r++) {
        // Generate random starting vector
        ComplexVector v0(N);
        for (int i = 0; i < N; i++) {
            v0[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v0.data(), 1);
        
        // Create a modified Hamiltonian operator that starts from our random vector
        auto H_from_v0 = [&H, &v0, N](const Complex* v, Complex* result, int size) {
            // If v is the unit vector e_0 = [1,0,0,...], we return H*v0
            // Otherwise we apply H normally
            if (std::abs(v[0] - Complex(1.0, 0.0)) < 1e-10) {
                bool is_e0 = true;
                for (int i = 1; i < size; i++) {
                    if (std::abs(v[i]) > 1e-10) {
                        is_e0 = false;
                        break;
                    }
                }
                
                if (is_e0) {
                    H(v0.data(), result, size);
                    return;
                }
            }
            
            // Regular application of H
            H(v, result, size);
        };
        
        // Use Chebyshev filtered Lanczos to get eigenvalues and eigenvectors
        std::vector<double> eigenvalues;
        std::vector<ComplexVector> eigenvectors;
        
        // Call chebyshev_filtered_lanczos with appropriate parameters
        const double tol = 1e-10;
        chebyshev_filtered_lanczos(H, N, m_max, m_max, tol, eigenvalues, &eigenvectors);
        
        int m = eigenvalues.size();
        
        // Calculate partition function contribution
        double Z_r = 0.0;
        for (int n = 0; n < m; n++) {
            Z_r += std::exp(-beta * eigenvalues[n]);
        }
        Z += Z_r;
        
        // Apply operators A and B to eigenvectors
        std::vector<ComplexVector> A_eigenvectors(m, ComplexVector(N));
        std::vector<ComplexVector> B_eigenvectors(m, ComplexVector(N));
        
        for (int n = 0; n < m; n++) {
            A(eigenvectors[n].data(), A_eigenvectors[n].data(), N);
            B(eigenvectors[n].data(), B_eigenvectors[n].data(), N);
        }
        
        // Calculate matrix elements and dynamical response
        for (int n = 0; n < m; n++) {
            double weight = std::exp(-beta * eigenvalues[n]);
            
            for (int p = 0; p < m; p++) {
                // Compute <n|A|p>
                Complex A_np;
                cblas_zdotc_sub(N, eigenvectors[n].data(), 1, A_eigenvectors[p].data(), 1, &A_np);
                
                // Compute <p|B|n>
                Complex B_pn;
                cblas_zdotc_sub(N, eigenvectors[p].data(), 1, B_eigenvectors[n].data(), 1, &B_pn);
                
                // Matrix element product
                Complex matrix_element = A_np * B_pn;
                
                // Energy difference
                double omega_np = eigenvalues[p] - eigenvalues[n];
                
                // Add contribution to all frequency points with Lorentzian broadening
                for (int i = 0; i < n_points; i++) {
                    double omega = omega_values[i];
                    // Lorentzian: 1/π * η/((ω-ω_0)² + η²)
                    Complex lorentzian = Complex(eta / (M_PI * ((omega - omega_np)*(omega - omega_np) + eta*eta)), 0.0);
                    response[i] += weight * matrix_element * lorentzian;
                }
            }
        }
    }
    
    // Normalize by partition function
    for (int i = 0; i < n_points; i++) {
        response[i] /= Z;
    }
    
    // Create result pair vector
    std::vector<std::pair<double, Complex>> result(n_points);
    for (int i = 0; i < n_points; i++) {
        result[i] = std::make_pair(omega_values[i], response[i]);
    }
    
    return result;
}

// Low-Temperature Lanczos Method (LTLM) for thermal expectation values
Complex LTLM(
    std::function<void(const Complex*, Complex*, int)> H, // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A, // Observable operator
    int N,              // Dimension of Hilbert space
    double beta,        // Inverse temperature (β = 1/kT)
    int R,              // Number of random samples
    int M,              // Lanczos iterations per sample
    double tol = 1e-10  // Tolerance for convergence
) {
    // Random number generator for random states
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Accumulators for numerator and denominator
    Complex num(0.0, 0.0);
    double denom = 0.0;
    
    // For each random sample
    for (int r = 0; r < R; r++) {
        // Generate random vector |r⟩
        ComplexVector r_vec(N);
        for (int i = 0; i < N; i++) {
            r_vec[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, r_vec.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, r_vec.data(), 1);
        
        // Create a matrix-vector product that starts from our random vector
        auto H_from_r = [&H, &r_vec, N](const Complex* v, Complex* result, int size) {
            // If v is the unit vector e_0 = [1,0,0,...], we return H*r_vec
            if (std::abs(v[0] - Complex(1.0, 0.0)) < 1e-10) {
                bool is_e0 = true;
                for (int i = 1; i < size; i++) {
                    if (std::abs(v[i]) > 1e-10) {
                        is_e0 = false;
                        break;
                    }
                }
                
                if (is_e0) {
                    H(r_vec.data(), result, size);
                    return;
                }
            }
            
            // Regular application of H
            H(v, result, size);
        };
        
        // Use Chebyshev filtered Lanczos to get eigenvalues and eigenvectors
        std::vector<double> eigenvalues;
        std::vector<ComplexVector> eigenvectors;
        
        chebyshev_filtered_lanczos(H, N, M, M, tol, eigenvalues, &eigenvectors);
        
        int m = eigenvalues.size();
        
        // Calculate ⟨r|e^{-βH}A|r⟩ and ⟨r|e^{-βH}|r⟩
        Complex exp_H_A_r(0.0, 0.0);
        double exp_H_r = 0.0;
        
        // For each eigenvalue/vector pair
        for (int j = 0; j < m; j++) {
            // Calculate |⟨r|ψ_j⟩|²
            Complex r_psi_j;
            cblas_zdotc_sub(N, r_vec.data(), 1, eigenvectors[j].data(), 1, &r_psi_j);
            double weight = std::exp(-beta * eigenvalues[j]) * std::norm(r_psi_j);
            
            // Add to denominator
            exp_H_r += weight;
            
            // Apply A to |ψ_j⟩
            ComplexVector A_psi(N);
            A(eigenvectors[j].data(), A_psi.data(), N);
            
            // Calculate ⟨r|A|ψ_j⟩
            Complex r_A_psi;
            cblas_zdotc_sub(N, r_vec.data(), 1, A_psi.data(), 1, &r_A_psi);
            
            // Add contribution to numerator
            exp_H_A_r += weight * r_A_psi;
        }
        
        // Add to accumulators
        num += exp_H_A_r;
        denom += exp_H_r;
    }
    
    // Return thermal average ⟨A⟩ = Tr(e^{-βH}A)/Tr(e^{-βH})
    return num / denom;
}

// LTLM for dynamical correlation function at low temperatures
// Calculates S_AB(ω) = (1/Z) ∑_j e^{-βE_j} ∑_i |⟨i|A|j⟩|^2 δ(ω - (E_i - E_j))
std::vector<std::pair<double, Complex>> LTLM_dynamical(
    std::function<void(const Complex*, Complex*, int)> H, // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A, // First operator
    std::function<void(const Complex*, Complex*, int)> B, // Second operator
    int N,              // Dimension of Hilbert space
    double beta,        // Inverse temperature (β = 1/kT)
    double omega_min,   // Minimum frequency
    double omega_max,   // Maximum frequency
    int n_points,       // Number of frequency points
    double eta,         // Broadening parameter
    int R,              // Number of random samples
    int M,              // Lanczos iterations per sample
    double tol = 1e-10  // Tolerance for convergence
) {
    // Generate frequency grid
    std::vector<double> omega_values(n_points);
    double delta_omega = (omega_max - omega_min) / (n_points - 1);
    for (int i = 0; i < n_points; i++) {
        omega_values[i] = omega_min + i * delta_omega;
    }
    
    // Initialize result vector
    std::vector<Complex> S_AB(n_points, Complex(0.0, 0.0));
    
    // Initialize random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Accumulator for partition function
    double Z = 0.0;
    
    // For each random sample
    for (int r = 0; r < R; r++) {
        // Generate random vector |r⟩
        ComplexVector r_vec(N);
        for (int i = 0; i < N; i++) {
            r_vec[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, r_vec.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, r_vec.data(), 1);
        
        // Create a function that applies H but starts from our random vector
        auto H_from_r = [&H, &r_vec, N](const Complex* v, Complex* result, int size) {
            // If v is the unit vector e_0 = [1,0,0,...], we return H*r_vec
            if (std::abs(v[0] - Complex(1.0, 0.0)) < 1e-10) {
                bool is_e0 = true;
                for (int i = 1; i < size; i++) {
                    if (std::abs(v[i]) > 1e-10) {
                        is_e0 = false;
                        break;
                    }
                }
                
                if (is_e0) {
                    H(r_vec.data(), result, size);
                    return;
                }
            }
            
            // Regular application of H
            H(v, result, size);
        };
        
        // Run Chebyshev filtered Lanczos to get eigenpairs
        std::vector<double> eigvals;
        std::vector<ComplexVector> eigvecs;
        
        chebyshev_filtered_lanczos(H_from_r, N, M, M, tol, eigvals, &eigvecs);
        
        // Calculate Z contribution for this sample
        double Z_r = 0.0;
        for (int j = 0; j < eigvals.size(); j++) {
            // Calculate |⟨r|ψ_j⟩|^2
            Complex r_psi_j;
            cblas_zdotc_sub(N, r_vec.data(), 1, eigvecs[j].data(), 1, &r_psi_j);
            Z_r += std::exp(-beta * eigvals[j]) * std::norm(r_psi_j);
        }
        Z += Z_r;
        
        // For each eigenstate |j⟩
        for (int j = 0; j < eigvals.size(); j++) {
            // Calculate |⟨r|ψ_j⟩|^2
            Complex r_psi_j;
            cblas_zdotc_sub(N, r_vec.data(), 1, eigvecs[j].data(), 1, &r_psi_j);
            
            // Thermodynamic weight factor e^{-βE_j}|⟨r|ψ_j⟩|^2
            double weight = std::exp(-beta * eigvals[j]) * std::norm(r_psi_j);
            
            // Apply A to |ψ_j⟩
            ComplexVector A_psi_j(N);
            A(eigvecs[j].data(), A_psi_j.data(), N);
            
            // Normalize A|ψ_j⟩
            double A_psi_norm = cblas_dznrm2(N, A_psi_j.data(), 1);
            if (A_psi_norm < tol) continue;  // Skip if A|ψ_j⟩ is approximately 0
            
            Complex A_scale = Complex(1.0/A_psi_norm, 0.0);
            cblas_zscal(N, &A_scale, A_psi_j.data(), 1);
            
            // Create a function that applies H but starts from A|ψ_j⟩
            auto H_from_Apsi = [&H, &A_psi_j, N](const Complex* v, Complex* result, int size) {
                // If v is the unit vector e_0 = [1,0,0,...], we return H*A_psi_j
                if (std::abs(v[0] - Complex(1.0, 0.0)) < 1e-10) {
                    bool is_e0 = true;
                    for (int i = 1; i < size; i++) {
                        if (std::abs(v[i]) > 1e-10) {
                            is_e0 = false;
                            break;
                        }
                    }
                    
                    if (is_e0) {
                        H(A_psi_j.data(), result, size);
                        return;
                    }
                }
                
                // Regular application of H
                H(v, result, size);
            };
            
            // Run second Chebyshev filtered Lanczos with A|ψ_j⟩ as starting vector
            std::vector<double> eigvals2;
            std::vector<ComplexVector> eigvecs2;
            
            chebyshev_filtered_lanczos(H_from_Apsi, N, M, M, tol, eigvals2, &eigvecs2);
            
            // Calculate B matrix elements if B is different from A
            bool B_is_A = (&B == &A);
            ComplexVector B_psi_j;
            double B_psi_norm = 0.0;
            
            if (!B_is_A) {
                B_psi_j.resize(N);
                B(eigvecs[j].data(), B_psi_j.data(), N);
                B_psi_norm = cblas_dznrm2(N, B_psi_j.data(), 1);
            }
            
            // For each final state |i⟩
            for (int i = 0; i < eigvals2.size(); i++) {
                // Energy difference
                double omega_ij = eigvals2[i] - eigvals[j];
                
                // Calculate ⟨ψ_i|A|ψ_j⟩
                Complex A_matrix_element;
                cblas_zdotc_sub(N, eigvecs2[i].data(), 1, A_psi_j.data(), 1, &A_matrix_element);
                A_matrix_element *= A_psi_norm; // Adjust for normalization
                
                // For cross-correlation <AB>, compute <i|B|j> matrix element
                Complex B_matrix_element;
                if (B_is_A) {
                    B_matrix_element = A_matrix_element;
                } else {
                    // Calculate ⟨ψ_i|B|ψ_j⟩
                    cblas_zdotc_sub(N, eigvecs2[i].data(), 1, B_psi_j.data(), 1, &B_matrix_element);
                }
                
                // Contribution to correlation function with Lorentzian broadening
                Complex contrib = weight * A_matrix_element * std::conj(B_matrix_element);
                
                // Add to all frequency points with broadening
                for (int p = 0; p < n_points; p++) {
                    double omega = omega_values[p];
                    // Lorentzian: η/π / [(ω-ω_ij)^2 + η^2]
                    double lorentz = eta / (M_PI * ((omega - omega_ij)*(omega - omega_ij) + eta*eta));
                    S_AB[p] += contrib * Complex(lorentz, 0.0);
                }
            }
        }
    }
    
    // Normalize by partition function and prepare result
    std::vector<std::pair<double, Complex>> result(n_points);
    for (int i = 0; i < n_points; i++) {
        result[i] = std::make_pair(omega_values[i], S_AB[i] / Z);
    }
    
    return result;
}

// LTLM for calculating thermal real-time correlation function
// Computes C_AB(t) = (1/Z) ∑_j e^{-βE_j} ⟨j|A(t)B|j⟩
std::vector<std::pair<double, Complex>> LTLM_real_time_correlation(
    std::function<void(const Complex*, Complex*, int)> H, // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A, // First operator
    std::function<void(const Complex*, Complex*, int)> B, // Second operator
    int N,              // Dimension of Hilbert space
    double b,        // Inverse temperature (β = 1/kT)
    double t_min,       // Minimum time
    double t_max,       // Maximum time
    int n_points,       // Number of time points
    int R,              // Number of random samples
    int M,              // Lanczos iterations per sample
    double tol = 1e-10  // Tolerance for convergence
) {
    // Generate time grid
    std::vector<double> time_values(n_points);
    double delta_t = (t_max - t_min) / (n_points - 1);
    for (int i = 0; i < n_points; i++) {
        time_values[i] = t_min + i * delta_t;
    }
    
    // Initialize result
    std::vector<Complex> C_AB(n_points, Complex(0.0, 0.0));
    
    // Random generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Partition function accumulator
    double Z = 0.0;
    
    // For each random sample
    for (int r = 0; r < R; r++) {
        // Generate random vector |r⟩
        ComplexVector r_vec(N);
        for (int i = 0; i < N; i++) {
            r_vec[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, r_vec.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, r_vec.data(), 1);
        
        // Create a function that applies H but starts from our random vector
        auto H_from_r = [&H, &r_vec, N](const Complex* v, Complex* result, int size) {
            // If v is the unit vector e_0 = [1,0,0,...], we return H*r_vec
            if (std::abs(v[0] - Complex(1.0, 0.0)) < 1e-10) {
                bool is_e0 = true;
                for (int i = 1; i < size; i++) {
                    if (std::abs(v[i]) > 1e-10) {
                        is_e0 = false;
                        break;
                    }
                }
                
                if (is_e0) {
                    H(r_vec.data(), result, size);
                    return;
                }
            }
            
            // Regular application of H
            H(v, result, size);
        };
        
        // Run Chebyshev filtered Lanczos to get eigenpairs
        std::vector<double> eigvals;
        std::vector<ComplexVector> eigvecs;
        
        chebyshev_filtered_lanczos(H_from_r, N, M, M, tol, eigvals, &eigvecs);
        
        // Update partition function
        double Z_r = 0.0;
        for (int j = 0; j < eigvals.size(); j++) {
            // Calculate |⟨r|ψ_j⟩|^2
            Complex r_psi_j;
            cblas_zdotc_sub(N, r_vec.data(), 1, eigvecs[j].data(), 1, &r_psi_j);
            Z_r += std::exp(-b * eigvals[j]) * std::norm(r_psi_j);
        }
        Z += Z_r;
        
        // For each eigenstate |j⟩
        for (int j = 0; j < eigvals.size(); j++) {
            // Calculate |⟨r|ψ_j⟩|^2
            Complex r_psi_j;
            cblas_zdotc_sub(N, r_vec.data(), 1, eigvecs[j].data(), 1, &r_psi_j);
            
            // Thermal weight
            double weight = std::exp(-b * eigvals[j]) * std::norm(r_psi_j);
            
            // Apply B to |ψ_j⟩: |φ⟩ = B|ψ_j⟩
            ComplexVector phi(N);
            B(eigvecs[j].data(), phi.data(), N);
            
            // For each time point
            for (int t_idx = 0; t_idx < n_points; t_idx++) {
                double t = time_values[t_idx];
                
                // Time-evolved state |φ(t)⟩ = exp(-iHt)|φ⟩ = exp(-iHt)B|ψ_j⟩
                ComplexVector phi_t(N, Complex(0.0, 0.0));
                
                // For each eigenvalue/vector pair
                for (int m = 0; m < eigvals.size(); m++) {
                    // Calculate ⟨ψ_m|φ⟩
                    Complex psi_m_phi;
                    cblas_zdotc_sub(N, eigvecs[m].data(), 1, phi.data(), 1, &psi_m_phi);
                    
                    // Apply time evolution: exp(-iE_m t)⟨ψ_m|φ⟩|ψ_m⟩
                    Complex phase = std::exp(Complex(0.0, -eigvals[m] * t));
                    Complex coef = phase * psi_m_phi;
                    
                    cblas_zaxpy(N, &coef, eigvecs[m].data(), 1, phi_t.data(), 1);
                }
                
                // Calculate ⟨φ(t)|A|ψ_j⟩
                ComplexVector A_psi(N);
                A(eigvecs[j].data(), A_psi.data(), N);
                
                Complex corr;
                cblas_zdotc_sub(N, phi_t.data(), 1, A_psi.data(), 1, &corr);
                
                // Add contribution
                C_AB[t_idx] += weight * corr;
            }
        }
    }
    
    // Normalize by partition function
    std::vector<std::pair<double, Complex>> result(n_points);
    for (int i = 0; i < n_points; i++) {
        result[i] = std::make_pair(time_values[i], C_AB[i] / Z);
    }
    
    return result;
}

// Advanced FTLM with adaptive convergence strategy
Complex FTLM_adaptive(
    std::function<void(const Complex*, Complex*, int)> H, // Hamiltonian matrix-vector product
    std::function<void(const Complex*, Complex*, int)> A, // Observable matrix-vector product
    int N,             // Dimension of Hilbert space
    double b,          // Inverse temperature (β = 1/kT)
    double conv_tol = 1e-4, // Convergence tolerance
    int r_start = 20,  // Starting number of random vectors
    int m_start = 100, // Starting max Lanczos iterations
    int max_steps = 3,// Maximum number of refinement steps
    double tol = 1e-10 // Tolerance for Lanczos algorithm
) {
    // Keep track of results for different parameters
    struct FTLMResult {
        int r_val;
        int m_val;
        Complex result;
        double rel_change;
    };
    std::vector<FTLMResult> results;
    
    // First calculation
    Complex initial_result = FTLM(H, A, N, b, r_start, m_start, tol);
    results.push_back({r_start, m_start, initial_result, 0.0});
    
    std::cout << "FTLM adaptive initial: r=" << r_start << ", m=" << m_start 
              << ", result=" << initial_result << std::endl;
    
    // Two separate parameters to vary independently
    int r_current = r_start;
    int m_current = m_start;
    
    // First try increasing r (more random vectors)
    for (int step = 0; step < max_steps/2 && results.size() < max_steps; step++) {
        r_current = static_cast<int>(r_current * 1.5);
        
        Complex r_result = FTLM(H, A, N, b, r_current, m_start, tol);
        double rel_diff = std::abs(r_result - results.back().result) / 
                          (std::abs(results.back().result) > 1e-10 ? std::abs(results.back().result) : 1.0);
        
        results.push_back({r_current, m_start, r_result, rel_diff});
        
        std::cout << "FTLM r-refinement: r=" << r_current << ", m=" << m_start 
                  << ", result=" << r_result << " (rel. change: " << rel_diff << ")" << std::endl;
        
        if (rel_diff < conv_tol/2) {
            break; // Converged on r parameter
        }
    }
    
    // Then try increasing m (Lanczos subspace size)
    for (int step = 0; step < max_steps/2 && results.size() < max_steps; step++) {
        m_current = static_cast<int>(m_current * 1.3);
        
        Complex m_result = FTLM(H, A, N, b, r_current, m_current, tol);
        double rel_diff = std::abs(m_result - results.back().result) / 
                          (std::abs(results.back().result) > 1e-10 ? std::abs(results.back().result) : 1.0);
        
        results.push_back({r_current, m_current, m_result, rel_diff});
        
        std::cout << "FTLM m-refinement: r=" << r_current << ", m=" << m_current 
                  << ", result=" << m_result << " (rel. change: " << rel_diff << ")" << std::endl;
        
        if (rel_diff < conv_tol/2) {
            break; // Converged on m parameter
        }
    }
    
    // Check if we've converged
    if (results.back().rel_change < conv_tol) {
        std::cout << "FTLM adaptive converged with r=" << results.back().r_val 
                  << ", m=" << results.back().m_val << std::endl;
    } else {
        std::cout << "FTLM adaptive did not fully converge. Using best result with r=" 
                  << results.back().r_val << ", m=" << results.back().m_val << std::endl;
    }
    
    return results.back().result;
}

// Thermodynamic quantities calculation using Finite Temperature Lanczos Method
struct ThermodynamicResults {
    std::vector<double> temperatures; // Temperature points
    std::vector<double> energy;       // Internal energy <E>
    std::vector<double> specific_heat; // Specific heat C_v
    std::vector<double> entropy;      // Entropy S
    std::vector<double> free_energy;  // Free energy F
};

// Calculate thermodynamic quantities using Low-Temperature Lanczos Method (LTLM)
ThermodynamicResults calculate_thermodynamics_LTLM(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N,                    // Dimension of Hilbert space
    double T_min = 0.01,     // Minimum temperature
    double T_max = 10.0,     // Maximum temperature
    int num_points = 100,    // Number of temperature points
    int R = 20,              // Number of random samples
    int M = 100,             // Lanczos iterations per sample
    double tol = 1e-10       // Tolerance for Lanczos algorithm
) {
    // Initialize results structure
    ThermodynamicResults results;
    results.temperatures.resize(num_points);
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Generate logarithmically spaced temperature points
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Define identity operator for calculating partition function
    auto identity_op = [](const Complex* v, Complex* result, int size) {
        std::copy(v, v + size, result);
    };
    
    // For each temperature point
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        std::cout << "Processing T = " << T << " (point " << (i+1) << "/" << num_points << ")" << std::endl;
        
        // Calculate <H> using LTLM
        Complex avg_energy = LTLM(H, H, N, beta, R, M, tol);
        results.energy[i] = avg_energy.real();
        
        // Calculate <H²> for specific heat
        auto H_squared = [&H, N](const Complex* v, Complex* result, int size) {
            // Apply H twice
            std::vector<Complex> temp(size);
            H(v, temp.data(), size);
            H(temp.data(), result, size);
        };
        
        Complex avg_energy_squared = LTLM(H, H_squared, N, beta, R, M, tol);
        
        // Calculate specific heat: C_v = β²(<H²> - <H>²)
        double var_energy = avg_energy_squared.real() - avg_energy.real() * avg_energy.real();
        results.specific_heat[i] = beta * beta * var_energy;
        
        // Calculate partition function Z = Tr[e^(-βH)]
        Complex Z_complex = LTLM(H, identity_op, N, beta, R, M, tol) * Complex(N, 0.0);
        double Z = Z_complex.real();
        
        // Calculate free energy F = -T * ln(Z)
        results.free_energy[i] = -T * std::log(Z);
        
        // Calculate entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
        
        std::cout << "  E = " << results.energy[i] 
                  << ", C_v = " << results.specific_heat[i]
                  << ", S = " << results.entropy[i] << std::endl;
    }
    
    return results;
}

ThermodynamicResults calculate_thermodynamics(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N,                    // Dimension of Hilbert space
    double T_min = 0.01,     // Minimum temperature
    double T_max = 10.0,     // Maximum temperature
    int num_points = 100,    // Number of temperature points
    int r_max = 20,          // Number of random vectors for FTLM
    int m_max = 100,         // Maximum Lanczos iterations per random vector
    double tol = 1e-10       // Tolerance for Lanczos algorithm
) {
    // Initialize results structure
    ThermodynamicResults results;
    results.temperatures.resize(num_points);
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Generate logarithmically spaced temperature points
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Define identity operator for calculating partition function
    auto identity_op = [](const Complex* v, Complex* result, int size) {
        std::copy(v, v + size, result);
    };
    
    // Calculate energy using FTLM for each temperature point
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        // Calculate <H> using FTLM
        Complex avg_energy = FTLM(H, H, N, beta, r_max, m_max, tol);
        results.energy[i] = avg_energy.real();
        
        // Calculate <H²> for specific heat
        auto H_squared = [&H, N](const Complex* v, Complex* result, int size) {
            // First apply H to v
            std::vector<Complex> temp(size);
            H(v, temp.data(), size);
            
            // Then apply H to the result
            H(temp.data(), result, size);
        };
        
        Complex avg_energy_squared = FTLM(H, H_squared, N, beta, r_max, m_max, tol);
        
        // Calculate specific heat: C_v = β²(<H²> - <H>²)
        double var_energy = avg_energy_squared.real() - avg_energy.real() * avg_energy.real();
        results.specific_heat[i] = beta * beta * var_energy;
        
        // Calculate partition function Z = Tr[e^(-βH)]
        // We use the identity operator for the observable
        Complex Z_complex = FTLM(H, identity_op, N, beta, r_max, m_max, tol) * Complex(N, 0.0);
        double Z = Z_complex.real();
        
        // Calculate free energy F = -T * ln(Z)
        results.free_energy[i] = -T * std::log(Z);
        
        // Calculate entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
    }
    
    return results;
}
// Function to output thermodynamic results to file
void output_thermodynamic_data(const ThermodynamicResults& results, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    outfile << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
    for (size_t i = 0; i < results.temperatures.size(); i++) {
        outfile << std::fixed << std::setprecision(6)
                << results.temperatures[i] << " "
                << results.energy[i] << " "
                << results.specific_heat[i] << " "
                << results.entropy[i] << " "
                << results.free_energy[i] << std::endl;
    }
    
    outfile.close();
    std::cout << "Thermodynamic data written to " << filename << std::endl;
}

// Improved version with adaptive FTLM calculations
ThermodynamicResults calculate_thermodynamics_adaptive(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N,                    // Dimension of Hilbert space
    double T_min = 0.01,     // Minimum temperature
    double T_max = 10.0,     // Maximum temperature
    int num_points = 100,    // Number of temperature points
    double conv_tol = 1e-4,  // FTLM convergence tolerance
    int r_start = 20,        // Starting number of random vectors
    int m_start = 100        // Starting Lanczos iterations
) {
    // Initialize results structure
    ThermodynamicResults results;
    results.temperatures.resize(num_points);
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Generate logarithmically spaced temperature points
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Define identity operator for calculating partition function
    auto identity_op = [](const Complex* v, Complex* result, int size) {
        std::copy(v, v + size, result);
    };
    
    // Calculate energy using adaptive FTLM for each temperature point
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        std::cout << "Processing T = " << T << " (point " << (i+1) << "/" << num_points << ")" << std::endl;
        
        // Calculate <H> using adaptive FTLM
        Complex avg_energy = FTLM_adaptive(H, H, N, beta, conv_tol, r_start, m_start);
        results.energy[i] = avg_energy.real();
        
        // Calculate <H²> for specific heat
        auto H_squared = [&H, N](const Complex* v, Complex* result, int size) {
            // First apply H to v
            std::vector<Complex> temp(size);
            H(v, temp.data(), size);
            
            // Then apply H to the result
            H(temp.data(), result, size);
        };
        
        Complex avg_energy_squared = FTLM_adaptive(H, H_squared, N, beta, conv_tol, r_start, m_start);
        
        // Calculate specific heat: C_v = β²(<H²> - <H>²)
        double var_energy = avg_energy_squared.real() - avg_energy.real() * avg_energy.real();
        results.specific_heat[i] = beta * beta * var_energy;
        
        // Calculate partition function Z = Tr[e^(-βH)]
        Complex Z_complex = FTLM_adaptive(H, identity_op, N, beta, conv_tol, r_start, m_start) * Complex(N, 0.0);
        double Z = Z_complex.real();
        
        // Calculate free energy F = -T * ln(Z)
        results.free_energy[i] = -T * std::log(Z);
        
        // Calculate entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
        
        std::cout << "  E = " << results.energy[i] 
                  << ", C_v = " << results.specific_heat[i]
                  << ", S = " << results.entropy[i] << std::endl;
    }
    
    return results;
}

// ShiftedKrylovSolver class to calculate Green's function using
// the shifted Krylov subspace method from the paper
class ShiftedKrylovSolver {
public:
    // Calculate Green's function G(z) = <a|(zI-H)^(-1)|b> for multiple z-values
    static std::vector<Complex> calculateGreenFunction(
        std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
        const ComplexVector& b,                               // Source vector
        const ComplexVector& a,                               // Projection vector
        const std::vector<Complex>& shifts,                   // List of energy points z_k
        int max_iter,                                         // Maximum iterations
        double tol,                                          // Convergence tolerance
        bool hermitian = true                                // Whether H is Hermitian
    ) {
        int N = b.size();
        
        if (hermitian) {
            return solveShiftedCG(H, b, a, shifts, N, max_iter, tol);
        } else {
            return solveShiftedCOCG(H, b, a, shifts, N, max_iter, tol);
        }
    }

private:
    // Shifted CG method for Hermitian matrices
    static std::vector<Complex> solveShiftedCG(
        std::function<void(const Complex*, Complex*, int)> H,
        const ComplexVector& b,
        const ComplexVector& a,
        const std::vector<Complex>& shifts,
        int N, int max_iter, double tol
    ) {
        int n_shifts = shifts.size();
        
        // Select the first shift as the seed
        Complex sigma_seed = shifts[0];
        
        // Define the seed shifted matrix A = sigma_seed*I - H
        auto A = [&H, &sigma_seed, N](const Complex* v, Complex* result, int size) {
            // First compute H*v
            H(v, result, size);
            
            // Then compute (sigma_seed*I - H)*v
            for (int i = 0; i < size; i++) {
                result[i] = sigma_seed * v[i] - result[i];
            }
        };
        
        // Initialize residual vectors for the seed equation
        ComplexVector r_prev(N, Complex(0.0, 0.0));
        ComplexVector r_curr(b);  // r_0 = b
        ComplexVector r_next(N);
        
        // Initialize coefficients
        Complex rho_curr, rho_prev = Complex(0.0, 0.0);
        Complex alpha, alpha_prev = Complex(0.0, 0.0);
        Complex beta, beta_prev = Complex(0.0, 0.0);
        
        // Calculate initial rho_0 = <r_0|r_0>
        cblas_zdotc_sub(N, r_curr.data(), 1, r_curr.data(), 1, &rho_curr);
        
        // Initialize collinearity factors for each shift
        std::vector<Complex> pi_prev(n_shifts, Complex(1.0, 0.0));
        std::vector<Complex> pi_curr(n_shifts, Complex(1.0, 0.0));
        std::vector<Complex> pi_next(n_shifts);
        
        // Initialize projected solution and search vectors
        std::vector<Complex> y_sigma(n_shifts, Complex(0.0, 0.0));
        
        // Calculate initial projection u_0^sigma = <a|b>
        Complex a_dot_b;
        cblas_zdotc_sub(N, a.data(), 1, b.data(), 1, &a_dot_b);
        std::vector<Complex> u_sigma(n_shifts, a_dot_b);
        
        // Main iteration loop
        for (int n = 0; n < max_iter; n++) {
            // Apply matrix A to current residual
            ComplexVector Ar(N);
            A(r_curr.data(), Ar.data(), N);
            
            // Calculate <r_n|A*r_n>
            Complex r_dot_Ar;
            cblas_zdotc_sub(N, r_curr.data(), 1, Ar.data(), 1, &r_dot_Ar);
            
            // Calculate alpha_n using three-term recurrence
            if (n == 0) {
                alpha = rho_curr / r_dot_Ar;
            } else {
                alpha = rho_curr / (r_dot_Ar - (beta_prev / alpha_prev) * rho_curr);
            }
            
            // Update residual using three-term recurrence (Eq. 9)
            if (n == 0) {
                for (int i = 0; i < N; i++) {
                    r_next[i] = r_curr[i] - alpha * Ar[i];
                }
            } else {
                Complex factor = alpha * beta_prev / alpha_prev;
                for (int i = 0; i < N; i++) {
                    r_next[i] = (1.0 + factor) * r_curr[i] - factor * r_prev[i] - alpha * Ar[i];
                }
            }
            
            // Calculate the projection <a|r_{n+1}>
            Complex a_dot_r_next;
            cblas_zdotc_sub(N, a.data(), 1, r_next.data(), 1, &a_dot_r_next);
            
            // Calculate rho_{n+1}
            rho_prev = rho_curr;
            cblas_zdotc_sub(N, r_next.data(), 1, r_next.data(), 1, &rho_curr);
            
            // Calculate beta_n
            beta = rho_curr / rho_prev;
            
            // Update for all shifts
            for (int k = 0; k < n_shifts; k++) {
                Complex sigma_diff = shifts[k] - sigma_seed;
                
                // Calculate pi_{n+1}^sigma using recurrence (Eq. 17)
                if (n == 0) {
                    pi_next[k] = (1.0 + alpha * sigma_diff) * pi_curr[k];
                } else {
                    Complex factor = alpha * beta_prev / alpha_prev;
                    pi_next[k] = (1.0 + factor + alpha * sigma_diff) * pi_curr[k] - 
                                  factor * pi_prev[k];
                }
                
                // Calculate alpha_n^sigma and beta_n^sigma (Eq. 18 and 19)
                Complex alpha_sigma = alpha * (pi_curr[k] / pi_next[k]);
                Complex beta_sigma = beta * std::pow(pi_curr[k] / pi_next[k], 2);
                
                // Update projected solution (Eq. 21)
                y_sigma[k] += alpha_sigma * u_sigma[k];
                
                // Update projected search vector (Eq. 22)
                u_sigma[k] = a_dot_r_next / pi_next[k] + beta_sigma * u_sigma[k];
            }
            
            // Check convergence
            double res_norm = cblas_dznrm2(N, r_next.data(), 1);
            if (res_norm < tol) {
                std::cout << "Converged after " << n+1 << " iterations." << std::endl;
                break;
            }
            
            // Update for next iteration
            r_prev = r_curr;
            r_curr = r_next;
            
            alpha_prev = alpha;
            beta_prev = beta;
            
            pi_prev = pi_curr;
            pi_curr = pi_next;
            
            // Optional: Implement seed switching for better performance
        }
        
        return y_sigma;
    }
    
    // Shifted COCG method for complex symmetric matrices
    static std::vector<Complex> solveShiftedCOCG(
        std::function<void(const Complex*, Complex*, int)> H,
        const ComplexVector& b,
        const ComplexVector& a,
        const std::vector<Complex>& shifts,
        int N, int max_iter, double tol
    ) {
        int n_shifts = shifts.size();
        
        // Select the first shift as the seed
        Complex sigma_seed = shifts[0];
        
        // Define the seed shifted matrix A = sigma_seed*I - H
        auto A = [&H, &sigma_seed, N](const Complex* v, Complex* result, int size) {
            H(v, result, size);
            for (int i = 0; i < size; i++) {
                result[i] = sigma_seed * v[i] - result[i];
            }
        };
        
        // Initialize vectors for COCG algorithm
        ComplexVector r(b);  // r_0 = b
        ComplexVector p(r);  // p_0 = r_0
        
        // Initialize coefficients
        std::vector<Complex> pi_curr(n_shifts, Complex(1.0, 0.0));
        std::vector<Complex> pi_prev(n_shifts, Complex(1.0, 0.0));
        
        // Initialize projected solution and search vectors
        std::vector<Complex> y_sigma(n_shifts, Complex(0.0, 0.0));
        Complex a_dot_p;
        cblas_zdotc_sub(N, a.data(), 1, p.data(), 1, &a_dot_p);
        std::vector<Complex> u_sigma(n_shifts, a_dot_p);
        
        // Main iteration
        for (int n = 0; n < max_iter; n++) {
            // Calculate (r_n, r_n)
            Complex r_dot_r;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            
            // Apply A to p_n
            ComplexVector Ap(N);
            A(p.data(), Ap.data(), N);
            
            // Calculate (p_n, Ap_n) - note: transpose, not conjugate transpose for COCG
            Complex p_dot_Ap = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                p_dot_Ap += p[i] * Ap[i];  // Complex multiplication without conjugation
            }
            
            // Calculate alpha_n
            Complex alpha = r_dot_r / p_dot_Ap;
            
            // Calculate a^T * Ap_n (transpose)
            Complex a_dot_Ap = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                a_dot_Ap += a[i] * Ap[i];  // Complex multiplication without conjugation
            }
            
            // Update residual: r_{n+1} = r_n - alpha_n * Ap_n
            ComplexVector r_next(N);
            for (int i = 0; i < N; i++) {
                r_next[i] = r[i] - alpha * Ap[i];
            }
            
            // Calculate a^T * r_{n+1}
            Complex a_dot_r_next = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                a_dot_r_next += a[i] * r_next[i];  // Complex multiplication without conjugation
            }
            
            // Update for all shifts
            for (int k = 0; k < n_shifts; k++) {
                Complex sigma_diff = shifts[k] - sigma_seed;
                
                // Update collinearity factor pi
                Complex pi_next = pi_curr[k] * (1.0 + sigma_diff * alpha) / 
                                (1.0 + sigma_diff * alpha * pi_curr[k] / pi_prev[k]);
                
                // Calculate alpha_sigma
                Complex alpha_sigma = alpha * (pi_curr[k] / pi_next);
                
                // Update projected solution
                y_sigma[k] += alpha_sigma * u_sigma[k];
                
                // Update parameters for next iteration
                Complex beta = (r_next[0] * r_next[0]) / (r[0] * r[0]);  // Simplified
                Complex beta_sigma = beta * std::pow(pi_next / pi_curr[k], 2);
                
                // Update projected search direction
                u_sigma[k] = a_dot_r_next / pi_next + beta_sigma * u_sigma[k];
                
                // Update pi for next iteration
                pi_prev[k] = pi_curr[k];
                pi_curr[k] = pi_next;
            }
            
            // Check for convergence
            double res_norm = cblas_dznrm2(N, r_next.data(), 1);
            if (res_norm < tol) {
                std::cout << "COCG converged after " << n+1 << " iterations." << std::endl;
                break;
            }
            
            // Calculate beta_n = (r_{n+1}, r_{n+1})/(r_n, r_n)
            Complex r_next_dot_r_next;
            cblas_zdotc_sub(N, r_next.data(), 1, r_next.data(), 1, &r_next_dot_r_next);
            Complex beta = r_next_dot_r_next / r_dot_r;
            
            // Update search direction: p_{n+1} = r_{n+1} + beta_n * p_n
            ComplexVector p_next(N);
            for (int i = 0; i < N; i++) {
                p_next[i] = r_next[i] + beta * p[i];
            }
            
            // Update for next iteration
            r = r_next;
            p = p_next;
        }
        
        return y_sigma;
    }
};

// Function to calculate dynamical Green's function for a range of frequencies
std::vector<std::pair<double, Complex>> calculateDynamicalGreenFunction(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                               // Dimension of Hilbert space
    const ComplexVector& a,                              // Left vector <a|
    const ComplexVector& b,                              // Right vector |b>
    double omega_min,                                    // Minimum frequency
    double omega_max,                                    // Maximum frequency
    int n_points,                                        // Number of frequency points
    double eta,                                          // Small imaginary broadening
    int max_iter = 1000,                                 // Maximum iterations
    double tol = 1e-10                                   // Convergence tolerance
) {
    // Generate frequency grid
    std::vector<double> omega_values(n_points);
    double delta_omega = (omega_max - omega_min) / (n_points - 1);
    for (int i = 0; i < n_points; i++) {
        omega_values[i] = omega_min + i * delta_omega;
    }
    
    // Generate complex shifts z = omega + i*eta
    std::vector<Complex> shifts(n_points);
    for (int i = 0; i < n_points; i++) {
        shifts[i] = Complex(omega_values[i], eta);
    }
    
    // Calculate Green's function using shifted Krylov solver
    std::vector<Complex> green_function = ShiftedKrylovSolver::calculateGreenFunction(
        H, b, a, shifts, max_iter, tol, true  // Assuming Hermitian H
    );
    
    // Combine frequencies and Green's function values
    std::vector<std::pair<double, Complex>> result(n_points);
    for (int i = 0; i < n_points; i++) {
        result[i] = std::make_pair(omega_values[i], green_function[i]);
    }
    
    return result;
}

#include <chrono>
int main(int argc, char* argv[]) {
    // Load the operator from ED_test directory
    int num_site = 16;  // Assuming 8 sites based on previous code
    Operator op(num_site);
    std::string dir = argv[1];
    op.loadFromFile(dir + "/Trans.def");
    op.loadFromInterAllFile(dir + "/InterAll.def");
    
    // Create Hamiltonian function
    auto H = [&op](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    };
    
    // Hilbert space dimension
    int N = 1 << num_site;  // 2^num_site
    
    std::cout << "Hilbert space dimension: " << N << std::endl;
    
    // Calculate full spectrum using full diagonalization
    std::cout << "Starting full diagonalization..." << std::endl;
    std::vector<double> eigenvalues;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // arpack_diagonalization(H, N, 2e4, true, eigenvalues);
    // full_diagonalization(H, N, eigenvalues);
    lanczos(H, N, N, N, 1e-10, eigenvalues);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Full diagonalization completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Save eigenvalues to file
    std::ofstream eigenvalue_file("ED_test_full_spectrum.dat");
    if (eigenvalue_file.is_open()) {
        for (const auto& eigenvalue : eigenvalues) {
            eigenvalue_file << eigenvalue << std::endl;
        }
        eigenvalue_file.close();
        std::cout << "Full spectrum saved to ED_test_full_spectrum.dat" << std::endl;
    }
    
    // Load eigenvalues from file instead of full diagonalization
    // std::cout << "Reading eigenvalues from ED_test_full_spectrum.dat..." << std::endl;
    // std::ifstream eigenvalue_input("ED_test_full_spectrum.dat");
    // eigenvalues.clear();

    // if (eigenvalue_input.is_open()) {
    //     double value;
    //     while (eigenvalue_input >> value) {
    //         eigenvalues.push_back(value);
    //     }
    //     eigenvalue_input.close();
    //     std::cout << "Read " << eigenvalues.size() << " eigenvalues from file" << std::endl;
    // } else {
    //     std::cerr << "Failed to open ED_test_full_spectrum.dat" << std::endl;
    //     return 1;
    // }

    // Calculate thermodynamics from spectrum
    std::cout << "Calculating thermodynamic properties..." << std::endl;
    double T_min = 0.001;
    double T_max = 10.0;
    int num_points = 2000;
    
    ThermodynamicData thermo = calculate_thermodynamics_from_spectrum(
        eigenvalues, T_min, T_max, num_points
    );
    
    // Save thermodynamic data
    std::ofstream thermo_file("ED_test_thermodynamics_full.dat");
    if (thermo_file.is_open()) {
        thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
        for (size_t i = 0; i < thermo.temperatures.size(); i++) {
            thermo_file << std::fixed << std::setprecision(6)
                      << thermo.temperatures[i] << " "
                      << thermo.energy[i] << " "
                      << thermo.specific_heat[i] << " "
                      << thermo.entropy[i] << " "
                      << thermo.free_energy[i] << std::endl;
        }
        thermo_file.close();
        std::cout << "Thermodynamic data saved to ED_test_thermodynamics_full.dat" << std::endl;
    }
    
    // Print some statistics about the spectrum
    std::sort(eigenvalues.begin(), eigenvalues.end());
    std::cout << "Spectrum statistics:" << std::endl;
    std::cout << "  Ground state energy: " << eigenvalues.front() << std::endl;
    std::cout << "  Maximum energy: " << eigenvalues.back() << std::endl;
    std::cout << "  Energy span: " << eigenvalues.back() - eigenvalues.front() << std::endl;
    
    return 0;
}

// int main() {
//     // Load the operator from ED_test directory
//     int num_site = 8;  // Assuming 8 sites based on previous code
//     Operator op(num_site);
//     op.loadFromFile("./ED_test/Trans.def");
//     op.loadFromInterAllFile("./ED_test/InterAll.def");
    
//     // Create Hamiltonian function for thermodynamic calculations
//     auto H = [&op](const Complex* v, Complex* Hv, int N) {
//         std::vector<Complex> vec(v, v + N);
//         std::vector<Complex> result = op.apply(vec);
//         std::copy(result.begin(), result.end(), Hv);
//     };
    
//     // Hilbert space dimension
//     int N = 1 << num_site;  // 2^num_site
    
//     std::cout << "Hilbert space dimension: " << N << std::endl;
    
//     // High temperature range using FTLM (T >= 0.1)
//     double T_max_high = 10.0;
//     double T_min_high = 0.1;
//     int num_points_high = 50;
    
//     std::cout << "Calculating high temperature thermodynamics with FTLM..." << std::endl;
//     auto start_high = std::chrono::high_resolution_clock::now();
    
//     ThermodynamicResults high_temp_results = calculate_thermodynamics(
//         H, N, T_min_high, T_max_high, num_points_high, 30, 100, 1e-10
//     );
    
//     auto end_high = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_high = end_high - start_high;
//     std::cout << "FTLM completed in " << elapsed_high.count() << " seconds" << std::endl;
    
//     // Save high temperature results
//     output_thermodynamic_data(high_temp_results, "ED_test_high_temp_thermo_FTLM.dat");
    
//     // Low temperature range using LTLM (T < 0.1)
//     double T_max_low = 0.1;
//     double T_min_low = 0.01;
//     int num_points_low = 30;
    
//     std::cout << "Calculating low temperature thermodynamics with LTLM..." << std::endl;
//     auto start_low = std::chrono::high_resolution_clock::now();
    
//     ThermodynamicResults low_temp_results = calculate_thermodynamics_LTLM(
//         H, N, T_min_low, T_max_low, num_points_low, 30, 100, 1e-10
//     );
    
//     auto end_low = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_low = end_low - start_low;
//     std::cout << "LTLM completed in " << elapsed_low.count() << " seconds" << std::endl;
    
//     // Save low temperature results
//     output_thermodynamic_data(low_temp_results, "ED_test_low_temp_thermo_LTLM.dat");
    
//     // Combine results for plotting
//     ThermodynamicResults combined_results;
//     combined_results.temperatures.insert(combined_results.temperatures.end(), 
//                                         low_temp_results.temperatures.begin(), 
//                                         low_temp_results.temperatures.end());
//     combined_results.temperatures.insert(combined_results.temperatures.end(), 
//                                         high_temp_results.temperatures.begin(), 
//                                         high_temp_results.temperatures.end());
                                        
//     combined_results.energy.insert(combined_results.energy.end(), 
//                                 low_temp_results.energy.begin(), 
//                                 low_temp_results.energy.end());
//     combined_results.energy.insert(combined_results.energy.end(), 
//                                 high_temp_results.energy.begin(), 
//                                 high_temp_results.energy.end());
                                
//     combined_results.specific_heat.insert(combined_results.specific_heat.end(), 
//                                         low_temp_results.specific_heat.begin(), 
//                                         low_temp_results.specific_heat.end());
//     combined_results.specific_heat.insert(combined_results.specific_heat.end(), 
//                                         high_temp_results.specific_heat.begin(), 
//                                         high_temp_results.specific_heat.end());
                                        
//     combined_results.entropy.insert(combined_results.entropy.end(), 
//                                 low_temp_results.entropy.begin(), 
//                                 low_temp_results.entropy.end());
//     combined_results.entropy.insert(combined_results.entropy.end(), 
//                                 high_temp_results.entropy.begin(), 
//                                 high_temp_results.entropy.end());
                                
//     combined_results.free_energy.insert(combined_results.free_energy.end(), 
//                                     low_temp_results.free_energy.begin(), 
//                                     low_temp_results.free_energy.end());
//     combined_results.free_energy.insert(combined_results.free_energy.end(), 
//                                     high_temp_results.free_energy.begin(), 
//                                     high_temp_results.free_energy.end());
    
//     // Save combined results
//     output_thermodynamic_data(combined_results, "ED_test_combined_thermo.dat");
    
//     std::cout << "Thermodynamic calculations completed successfully!" << std::endl;
//     std::cout << "Results saved to:" << std::endl;
//     std::cout << "  ED_test_high_temp_thermo_FTLM.dat" << std::endl;
//     std::cout << "  ED_test_low_temp_thermo_LTLM.dat" << std::endl;
//     std::cout << "  ED_test_combined_thermo.dat" << std::endl;
    
//     return 0;
// }




// int main(){
//     int num_site = 16;
//     Operator op(num_site);
//     op.loadFromFile("./ED_test/Trans.def");
//     op.loadFromInterAllFile("./ED_test/InterAll.def");
//     std::vector<double> eigenvalues;
//     // std::vector<ComplexVector> eigenvectors;
//     lanczos([&](const Complex* v, Complex* Hv, int N) {
//         std::vector<Complex> vec(v, v + N);
//         std::vector<Complex> result(N, Complex(0.0, 0.0));
//         result = op.apply(vec);
//         std::copy(result.begin(), result.end(), Hv);
//     }, (1<<num_site), 1000, 1000, 1e-10, eigenvalues);

//     // Write all eigenvalues to a file


//     std::vector<double> eigenvalues_lanczos;
//     // shift_invert_lanczos([&](const Complex* v, Complex* Hv, int N) {
//     //     std::vector<Complex> vec(v, v + N);
//     //     std::vector<Complex> result(N, Complex(0.0, 0.0));
//     //     result = op.apply(vec);
//     //     std::copy(result.begin(), result.end(), Hv);
//     // }, (1<<num_site), 1000, 0, 20, 1e-10, eigenvalues_lanczos);

//     eigenvalues_lanczos = full_spectrum_lanczos([&](const Complex* v, Complex* Hv, int N) {
//         std::vector<Complex> vec(v, v + N);
//         std::vector<Complex> result(N, Complex(0.0, 0.0));
//         result = op.apply(vec);
//         std::copy(result.begin(), result.end(), Hv);
//     }, (1<<num_site), 100);

//     std::ofstream eigenvalues_file("lanczos_eigenvalues.txt");
//     if (eigenvalues_file.is_open()) {
//         for (const auto& eigenvalue : eigenvalues) {
//             eigenvalues_file << eigenvalue << std::endl;
//         }
//         eigenvalues_file.close();
//     } else {
//         std::cerr << "Unable to open file for writing eigenvalues." << std::endl;
//     }
//     std::ofstream eigenvalues_lanczos_file("lanczos_eigenvalues_lanczos.txt");
//     if (eigenvalues_lanczos_file.is_open()) {
//         for (const auto& eigenvalue : eigenvalues_lanczos) {
//             eigenvalues_lanczos_file << eigenvalue << std::endl;
//         }
//         eigenvalues_lanczos_file.close();
//     } else {
//         std::cerr << "Unable to open file for writing eigenvalues." << std::endl;
//     }

//     // Print the results
//     std::cout << "Eigenvalues:" << std::endl;
//     for (size_t i = 0; i < 20; i++) {
//         std::cout << "Eigenvalue " << i << " Chebyshev Filtered Lanczos: " << eigenvalues[i] << " Lanczos: " << eigenvalues_lanczos[i] << std::endl;
//     }
//     // Run full diagonalization for comparison
//     std::vector<double> full_eigenvalues;
//     full_diagonalization([&](const Complex* v, Complex* Hv, int N) {
//         std::vector<Complex> vec(v, v + N);
//         std::vector<Complex> result(N, Complex(0.0, 0.0));
//         result = op.apply(vec);
//         std::copy(result.begin(), result.end(), Hv);
//     }, 1<<num_site, full_eigenvalues);

//     // Sort both sets of eigenvalues for comparison
//     std::sort(eigenvalues.begin(), eigenvalues.end());
//     std::sort(full_eigenvalues.begin(), full_eigenvalues.end());

//     // Compare and print results
//     std::cout << "\nComparison between Lanczos and Full Diagonalization:" << std::endl;
//     std::cout << "Index | Lanczos        | Full          | Difference" << std::endl;
//     std::cout << "------------------------------------------------------" << std::endl;

//     int num_to_compare = std::min(eigenvalues.size(), full_eigenvalues.size());
//     num_to_compare = std::min(num_to_compare, 20);  // Limit to first 20 eigenvalues

//     for (int i = 0; i < num_to_compare; i++) {
//         double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
//         std::cout << std::setw(5) << i << " | " 
//                   << std::setw(14) << std::fixed << std::setprecision(10) << eigenvalues[i] << " | "
//                   << std::setw(14) << std::fixed << std::setprecision(10) << full_eigenvalues[i] << " | "
//                   << std::setw(10) << std::scientific << std::setprecision(3) << diff << std::endl;
//     }

//     // Calculate and print overall accuracy statistics
//     if (num_to_compare > 0) {
//         double max_diff = 0.0;
//         double sum_diff = 0.0;
//         for (int i = 0; i < num_to_compare; i++) {
//             double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
//             max_diff = std::max(max_diff, diff);
//             sum_diff += diff;
//         }
//         double avg_diff = sum_diff / num_to_compare;
        
//         std::cout << "\nAccuracy statistics:" << std::endl;
//         std::cout << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
//         std::cout << "Average difference: " << std::scientific << std::setprecision(3) << avg_diff << std::endl;
        
//         // Special focus on ground state and first excited state
//         if (full_eigenvalues.size() > 0 && eigenvalues.size() > 0) {
//             double ground_diff = std::abs(eigenvalues[0] - full_eigenvalues[0]);
//             std::cout << "Ground state error: " << std::scientific << std::setprecision(3) << ground_diff << std::endl;
            
//             if (full_eigenvalues.size() > 1 && eigenvalues.size() > 1) {
//                 double excited_diff = std::abs(eigenvalues[1] - full_eigenvalues[1]);
//                 std::cout << "First excited state error: " << std::scientific << std::setprecision(3) << excited_diff << std::endl;
//             }
//         }
//     }

//     return 0;
// }


// int main(){
//     // Matrix size (not too large to keep computation reasonable)
//     const int N = 1000; 

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
//     chebyshev_filtered_lanczos(matVecMult, N, N, N, 1e-10, lanczosEigenvalues);

//     // Lanczos with CG refinement
//     std::vector<double> lanczosCGEigenvalues;
//     std::vector<ComplexVector> lanczosCGEigenvectors;
//     // chebyshev_filtered_lanczos(matVecMult, N, N/2, 1e-10, lanczosCGEigenvalues, &lanczosCGEigenvectors);

//     lanczosCGEigenvalues = full_spectrum_lanczos(matVecMult, N, 100, N/2);

//     // Write eigenvalues to file for comparison
//     std::ofstream outfile("eigenvalues_comparison.txt");
//     if (outfile.is_open()) {
//         outfile << "# Index   Lanczos         LanczosCG\n";
//         int max_vals = std::min(lanczosEigenvalues.size(), lanczosCGEigenvalues.size());
        
//         for (int i = 0; i < max_vals; i++) {
//             outfile << std::setw(5) << i << "  " 
//                     << std::setprecision(12) << std::fixed << lanczosEigenvalues[i] << "  "
//                     << std::setprecision(12) << std::fixed << lanczosCGEigenvalues[i] << "\n";
//         }
//         outfile.close();
//         std::cout << "Eigenvalues written to eigenvalues_comparison.txt\n";
//     }

//     // Compute statistics on eigenvalue differences
//     double max_diff = 0.0;
//     double sum_diff = 0.0;
//     int min_size = std::min(lanczosEigenvalues.size(), lanczosCGEigenvalues.size());

//     for (int i = 0; i < min_size; i++) {
//         double diff = std::abs(lanczosEigenvalues[i] - lanczosCGEigenvalues[i]);
//         max_diff = std::max(max_diff, diff);
//         sum_diff += diff;
//     }

//     double avg_diff = sum_diff / min_size;
//     std::cout << "Eigenvalue difference statistics:\n";
//     std::cout << "  Maximum difference: " << std::scientific << max_diff << "\n";
//     std::cout << "  Average difference: " << std::scientific << avg_diff << "\n";

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