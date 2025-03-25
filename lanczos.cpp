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
    ComplexVector w(N);
    
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    double norm;
    max_iter = std::min(N, max_iter);
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex alpha(-beta[j], 0.0);  // -beta[j] as complex number
            cblas_zaxpy(N, &alpha, v_prev.data(), 1, w.data(), 1);
        }

        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization (twice for degenerate eigenvalues)
        for (int iter = 0; iter < 2; iter++) {
            for (int k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
                
                // Subtract projection: w -= overlap * basis_vectors[k]
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector instead of breaking
            v_next = generateOrthogonalVector(N, basis_vectors, gen, dist);
            
            // Update the norm for use in scaling
            norm = cblas_dznrm2(N, v_next.data(), 1);
            
            // If after orthogonalization the norm is still too small, we should break
            if (norm < tol) {
                break;
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
    
    // Allocate arrays for eigenvalues and eigenvectors
    std::vector<double> evals(m);
    std::vector<double> z(m * m, 0.0);
    
    // Diagonalize tridiagonal matrix using LAPACK
    int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, eigenvectors ? 'V' : 'N', m, 
                           diag.data(), offdiag.data(), z.data(), m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstev failed with error code " << info << std::endl;
        return;
    }
    
    // Copy eigenvalues to output
    eigenvalues = diag;
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        // Create sorted indices
        std::vector<int> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices based on eigenvalues
        std::sort(indices.begin(), indices.end(),
                 [&](int a, int b) { return diag[a] < diag[b]; });
        
        // Create sorted copies
        std::vector<double> sorted_ev = diag;
        std::vector<double> sorted_z(m * m);
        
        for (int i = 0; i < m; i++) {
            sorted_ev[i] = diag[indices[i]];
            for (int j = 0; j < m; j++) {
                sorted_z[j*m + i] = z[j*m + indices[i]];
            }
        }
        
        // Replace with sorted versions
        diag = sorted_ev;
        eigenvalues = diag;
        z = sorted_z;
        
        // Transform Lanczos eigenvectors to original basis
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Identify clusters of degenerate eigenvalues (within tolerance)
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < m; i++) {
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
        
        // Process each cluster separately
        for (const auto& cluster : degen_clusters) {
            if (cluster.size() == 1) {
                // Non-degenerate case - standard treatment
                int idx = cluster[0];
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < m; k++) {
                        (*eigenvectors)[idx][i] += z[k*m + idx] * basis_vectors[k][i];
                    }
                }
            } else {
                // Degenerate case - special handling
                int subspace_dim = cluster.size();
                std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    for (int i = 0; i < N; i++) {
                        for (int k = 0; k < m; k++) {
                            subspace_vectors[c][i] += z[k*m + idx] * basis_vectors[k][i];
                        }
                    }
                }
                
                // Re-orthogonalize within the degenerate subspace using QR
                // We'll use a simple Gram-Schmidt for this
                for (int c = 0; c < subspace_dim; c++) {
                    // Normalize current vector
                    double norm = cblas_dznrm2(N, subspace_vectors[c].data(), 1);
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, subspace_vectors[c].data(), 1);
                    
                    // Orthogonalize against previous vectors
                    for (int prev = 0; prev < c; prev++) {
                        Complex overlap;
                        cblas_zdotc_sub(N, subspace_vectors[prev].data(), 1, 
                                      subspace_vectors[c].data(), 1, &overlap);
                        
                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, subspace_vectors[prev].data(), 1, 
                                   subspace_vectors[c].data(), 1);
                    }
                    
                    // Renormalize
                    norm = cblas_dznrm2(N, subspace_vectors[c].data(), 1);
                    if (norm > tol) {  // Check if still valid after orthogonalization
                        scale = Complex(1.0/norm, 0.0);
                        cblas_zscal(N, &scale, subspace_vectors[c].data(), 1);
                    }
                }
                
                // Store the orthogonalized eigenvectors
                for (int c = 0; c < subspace_dim; c++) {
                    int idx = cluster[c];
                    (*eigenvectors)[idx] = subspace_vectors[c];
                }
            }
        }
        
        // Final verification of orthogonality between all eigenvectors
        for (int i = 0; i < m; i++) {
            // Normalize
            double norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
            
            // Verify Hv = λv (optional, for debugging)
            // ComplexVector Hv(N);
            // H((*eigenvectors)[i].data(), Hv.data(), N);
            // Complex lambda_actual;
            // cblas_zdotc_sub(N, (*eigenvectors)[i].data(), 1, Hv.data(), 1, &lambda_actual);
            // std::cout << "Eigenvalue " << i << ": expected=" << eigenvalues[i] 
            //          << ", actual=" << std::real(lambda_actual) << std::endl;
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
            ComplexVector v = initial_eigenvectors[idx];
            
            // Standard CG refinement
            refine_eigenvector_with_cg(H, v, lambda, N, tol);
            
            // Update eigenvalue using Rayleigh quotient
            ComplexVector Hv(N);
            H(v.data(), Hv.data(), N);
            Complex lambda_new;
            cblas_zdotc_sub(N, v.data(), 1, Hv.data(), 1, &lambda_new);
            eigenvalues[idx] = std::real(lambda_new);
            
            // Store the refined eigenvector
            (*eigenvectors)[idx] = v;
        } else {
            // Degenerate case: block refinement
            int subspace_dim = cluster.size();
            std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N));
            
            // Start with initial approximations
            for (int c = 0; c < subspace_dim; c++) {
                subspace_vectors[c] = initial_eigenvectors[cluster[c]];
            }
            
            // Perform block conjugate gradient refinement
            refine_degenerate_eigenvectors(H, subspace_vectors, eigenvalues[cluster[0]], N, tol);
            
            // Update eigenvalue (all use the same value in the cluster)
            ComplexVector Hv(N);
            for (int c = 0; c < subspace_dim; c++) {
                int idx = cluster[c];
                H(subspace_vectors[c].data(), Hv.data(), N);
                Complex lambda_new;
                cblas_zdotc_sub(N, subspace_vectors[c].data(), 1, Hv.data(), 1, &lambda_new);
                eigenvalues[idx] = std::real(lambda_new);
                (*eigenvectors)[idx] = subspace_vectors[c];
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

int main(){
    int num_site = 8;
    Operator op(num_site);
    op.loadFromFile("./ED_test/Trans.def");
    op.loadFromInterAllFile("./ED_test/InterAll.def");
    Matrix matrix = op.returnMatrix();
    std::vector<double> eigenvalues;
    std::vector<ComplexVector> eigenvectors;
    chebyshev_filtered_lanczos([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, 1<<num_site, 100, 1e-6, eigenvalues);
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << std::endl;
    } 


    // Run full diagonalization for comparison
    std::vector<double> full_eigenvalues;
    full_diagonalization([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, 1<<num_site, full_eigenvalues);

    // Sort both sets of eigenvalues for comparison
    std::sort(eigenvalues.begin(), eigenvalues.end());
    std::sort(full_eigenvalues.begin(), full_eigenvalues.end());

    // Compare and print results
    std::cout << "\nComparison between Lanczos and Full Diagonalization:" << std::endl;
    std::cout << "Index | Lanczos        | Full          | Difference" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    int num_to_compare = std::min(eigenvalues.size(), full_eigenvalues.size());
    num_to_compare = std::min(num_to_compare, 20);  // Limit to first 20 eigenvalues

    for (int i = 0; i < num_to_compare; i++) {
        double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(14) << std::fixed << std::setprecision(10) << eigenvalues[i] << " | "
                  << std::setw(14) << std::fixed << std::setprecision(10) << full_eigenvalues[i] << " | "
                  << std::setw(10) << std::scientific << std::setprecision(3) << diff << std::endl;
    }

    // Calculate and print overall accuracy statistics
    if (num_to_compare > 0) {
        double max_diff = 0.0;
        double sum_diff = 0.0;
        for (int i = 0; i < num_to_compare; i++) {
            double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
        }
        double avg_diff = sum_diff / num_to_compare;
        
        std::cout << "\nAccuracy statistics:" << std::endl;
        std::cout << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
        std::cout << "Average difference: " << std::scientific << std::setprecision(3) << avg_diff << std::endl;
        
        // Special focus on ground state and first excited state
        if (full_eigenvalues.size() > 0 && eigenvalues.size() > 0) {
            double ground_diff = std::abs(eigenvalues[0] - full_eigenvalues[0]);
            std::cout << "Ground state error: " << std::scientific << std::setprecision(3) << ground_diff << std::endl;
            
            if (full_eigenvalues.size() > 1 && eigenvalues.size() > 1) {
                double excited_diff = std::abs(eigenvalues[1] - full_eigenvalues[1]);
                std::cout << "First excited state error: " << std::scientific << std::setprecision(3) << excited_diff << std::endl;
            }
        }
    }

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
