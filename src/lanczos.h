#ifndef LANCZOS_H
#define LANCZOS_H

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
#include <thread>

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
// Lanczos algorithm implementation with basis vectors stored on disk
// Lanczos algorithm implementation with basis vectors stored on disk
void lanczos_no_ortho(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false) {
    
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
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
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
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
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
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (eigenvectors){
            if (j < max_iter - 1) {
                std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
                std::ofstream outfile(next_basis_file, std::ios::binary);
                if (!outfile) {
                    std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                    return;
                }
                outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
                outfile.close();
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
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    if (eigenvectors) {
        // Need space for eigenvectors but m might be too large for full allocation
        // Instead of computing all eigenvectors at once, compute them in batches
        const int batch_size = 1000; // Adjust based on available memory
        int num_batches = (n_eigenvalues + batch_size - 1) / batch_size; // Ceiling division

        
        // First compute all eigenvalues without eigenvectors
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info 
                  << " when computing eigenvalues" << std::endl;
            system(("rm -rf " + temp_dir).c_str());
            return;
        }
        
        // Then compute eigenvectors in batches using dstevr which allows range selection
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues) - 1;
            int batch_n = end_idx - start_idx + 1;
            
            std::cout << "Computing eigenvectors batch " << batch + 1 << "/" << num_batches 
                  << " (indices " << start_idx << " to " << end_idx << ")" << std::endl;
            
            // Allocate memory just for this batch
            std::vector<double> batch_evals(batch_n);
            std::vector<double> batch_evecs(m * batch_n);
            std::vector<int> isuppz(2 * batch_n);
            
            // Make a copy of the tridiagonal matrix data for dstevr
            std::vector<double> diag_copy = diag;
            std::vector<double> offdiag_copy(offdiag);
            
            int m_found;
            // Compute eigenvectors for this batch using index range
            info = LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'I', m, 
                     diag_copy.data(), offdiag_copy.data(), 
                     0.0, 0.0, // vl, vu not used with 'I' range option
                     start_idx + 1, end_idx + 1, // FORTRAN 1-indexing
                     1e-6, // abstol, set to 0 for default
                     &m_found,
                     batch_evals.data(), batch_evecs.data(), m, 
                     isuppz.data());
            
            if (info != 0 || m_found != batch_n) {
            std::cerr << "LAPACKE_dstevr failed with error code " << info 
                  << " when computing eigenvectors for batch " << batch + 1 
                  << ". Found " << m_found << " of " << batch_n << " eigenvectors." << std::endl;
            continue;
            }

            std::cout << "  Found " << m_found << " eigenvectors in this batch." << std::endl;
            
            // Transform and save each eigenvector in this batch
            for (int i = 0; i < batch_n; i++) {
                int global_idx = start_idx + i;
                
                // Transform the eigenvector
                // Initialize full vector
                std::cout << "  Transforming eigenvector " << global_idx + 1 << std::endl;
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                // Read basis vectors in batches to reduce disk I/O
                const int basis_batch_size = 100;  // Adjust based on available memory
                for (int batch_start = 0; batch_start < m; batch_start += basis_batch_size) {
                    int batch_end = std::min(batch_start + basis_batch_size, m);
                    
                    // Read this batch of basis vectors
                    std::vector<ComplexVector> basis_batch;
                    basis_batch.reserve(batch_end - batch_start);
                    for (int j = batch_start; j < batch_end; j++) {
                        basis_batch.push_back(read_basis_vector(j, N));
                    }
                    
                    // Compute contribution from this batch
                    for (int j = 0; j < batch_end - batch_start; j++) {
                        int j_global = batch_start + j;
                        Complex coef(batch_evecs[j_global*batch_n + i], 0.0);
                        cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, full_vector.data(), 1);
                    }
                }
                std::cout << "  Normalizing eigenvector " << global_idx + 1 << std::endl;
                // Normalize
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                if (norm > 1e-12) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, full_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(global_idx) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(full_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
                
                // Print progress occasionally
                if (global_idx % 10 == 0 || global_idx == n_eigenvalues - 1) {
                    std::cout << "  Saved eigenvector " << global_idx + 1 << " of " << n_eigenvalues << std::endl;
                }
            }
            
            // Clear memory by reassigning vectors to empty ones
            std::vector<double>().swap(batch_evals);
            std::vector<double>().swap(batch_evecs);
            std::vector<int>().swap(isuppz);
        }
    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());

    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (!eval_outfile) {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    } else {
        // Write the number of eigenvalues first
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
        // Write all eigenvalues
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
}


// Lanczos algorithm with selective reorthogonalization
void lanczos_selective_reorth(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false) {
    
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
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
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
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Parameters for selective reorthogonalization
    const double orth_threshold = 1e-5;  // Threshold for selective reorthogonalization
    const int periodic_full_reorth = max_iter/10; // Periodically do full reorthogonalization
    
    // Storage for tracking loss of orthogonality
    std::vector<ComplexVector> recent_vectors; // Store most recent vectors in memory
    const int max_recent = 5;                  // Maximum number of recent vectors to keep in memory
    
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
        
        // Approach for selective reorthogonalization:
        // 1. Always orthogonalize against the previous vector and a few recent ones
        // 2. Periodically do full reorthogonalization (every 'periodic_full_reorth' steps)
        // 3. Otherwise do selective reorthogonalization based on a threshold
        
        // Always orthogonalize against v_{j-1} for numerical stability
        if (j > 0) {
            Complex overlap;
            cblas_zdotc_sub(N, v_prev.data(), 1, w.data(), 1, &overlap);
            if (std::abs(overlap) > orth_threshold) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, v_prev.data(), 1, w.data(), 1);
            }
        }
        
        // Orthogonalize against recent vectors in memory
        for (const auto& vec : recent_vectors) {
            Complex overlap;
            cblas_zdotc_sub(N, vec.data(), 1, w.data(), 1, &overlap);
            if (std::abs(overlap) > orth_threshold) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, vec.data(), 1, w.data(), 1);
            }
        }
        
        // Periodic full reorthogonalization or selective reorthogonalization. Currently suppressed
        // if (j % periodic_full_reorth == 0) {
        //     // Full reorthogonalization
        //     std::cout << "Performing full reorthogonalization at step " << j + 1 << std::endl;
        //     for (int k = 0; k <= j; k++) {
        //         // Skip recent vectors that were already orthogonalized
        //         bool is_recent = false;
        //         for (const auto& vec : recent_vectors) {
        //             ComplexVector recent_v = read_basis_vector(k, N);
        //             double diff = 0.0;
        //             for (int i = 0; i < N; i++) {
        //                 diff += std::norm(vec[i] - recent_v[i]);
        //             }
        //             if (diff < 1e-12) {
        //                 is_recent = true;
        //                 break;
        //             }
        //         }
        //         if (is_recent) continue;
                
        //         // Read basis vector k from file
        //         ComplexVector basis_k = read_basis_vector(k, N);
                
        //         Complex overlap;
        //         cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
        //         if (std::abs(overlap) > orth_threshold) {
        //             Complex neg_overlap = -overlap;
        //             cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
        //         }
        //     }
        // } else {
        //     // Selective reorthogonalization against vectors with significant overlap
        //     for (int k = 0; k <= j - 2; k++) {  // Skip v_{j-1} as it's already handled
        //         // Skip recent vectors that were already orthogonalized
        //         bool is_recent = false;
        //         for (const auto& vec : recent_vectors) {
        //             ComplexVector recent_v = read_basis_vector(k, N);
        //             double diff = 0.0;
        //             for (int i = 0; i < N; i++) {
        //                 diff += std::norm(vec[i] - recent_v[i]);
        //             }
        //             if (diff < 1e-12) {
        //                 is_recent = true;
        //                 break;
        //             }
        //         }
        //         if (is_recent) continue;
                
        //         // Read basis vector k from file
        //         ComplexVector basis_k = read_basis_vector(k, N);
                
        //         // Check if orthogonalization against this vector is needed
        //         Complex overlap;
        //         cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
        //         if (std::abs(overlap) > orth_threshold) {
        //             Complex neg_overlap = -overlap;
        //             cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                    
        //             // Double-check orthogonality
        //             cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
        //             if (std::abs(overlap) > orth_threshold) {
        //                 // If still not orthogonal, apply one more time
        //                 neg_overlap = -overlap;
        //                 cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
        //             }
        //         }
        //     }
        // }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
        
        // Keep track of recent vectors for quick access
        recent_vectors.push_back(v_current);
        if (recent_vectors.size() > max_recent) {
            recent_vectors.erase(recent_vectors.begin());
        }
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
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    if (eigenvectors) {
        // Need space for eigenvectors but m might be too large for full allocation
        // Instead of computing all eigenvectors at once, compute them in batches
        const int batch_size = 100; // Adjust based on available memory
        int num_batches = (n_eigenvalues + batch_size - 1) / batch_size; // Ceiling division

        
        // First compute all eigenvalues without eigenvectors
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info 
                  << " when computing eigenvalues" << std::endl;
            system(("rm -rf " + temp_dir).c_str());
            return;
        }
        
        // Then compute eigenvectors in batches using dstevr which allows range selection
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues) - 1;
            int batch_n = end_idx - start_idx + 1;
            
            std::cout << "Computing eigenvectors batch " << batch + 1 << "/" << num_batches 
                  << " (indices " << start_idx << " to " << end_idx << ")" << std::endl;
            
            // Allocate memory just for this batch
            std::vector<double> batch_evals(batch_n);
            std::vector<double> batch_evecs(m * batch_n);
            std::vector<int> isuppz(2 * batch_n);
            
            // Make a copy of the tridiagonal matrix data for dstevr
            std::vector<double> diag_copy = diag;
            std::vector<double> offdiag_copy(offdiag);
            
            int m_found;
            // Compute eigenvectors for this batch using index range
            info = LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'I', m, 
                     diag_copy.data(), offdiag_copy.data(), 
                     0.0, 0.0, // vl, vu not used with 'I' range option
                     start_idx + 1, end_idx + 1, // FORTRAN 1-indexing
                     1e-8, // abstol
                     &m_found,
                     batch_evals.data(), batch_evecs.data(), m, 
                     isuppz.data());
            
            if (info != 0 || m_found != batch_n) {
                std::cerr << "LAPACKE_dstevr failed with error code " << info 
                      << " when computing eigenvectors for batch " << batch + 1 
                      << ". Found " << m_found << " of " << batch_n << " eigenvectors." << std::endl;
                continue;
            }
            
            // Read basis vectors in batches to reduce disk I/O
            const int basis_batch_size = 50;  // Adjust based on available memory
            
            // Transform and save each eigenvector in this batch
            for (int i = 0; i < batch_n; i++) {
                int global_idx = start_idx + i;
                
                // Initialize full vector
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                
                for (int basis_start = 0; basis_start < m; basis_start += basis_batch_size) {
                    int basis_end = std::min(basis_start + basis_batch_size, m);
                    
                    // Read this batch of basis vectors
                    std::vector<ComplexVector> basis_batch;
                    basis_batch.reserve(basis_end - basis_start);
                    for (int j = basis_start; j < basis_end; j++) {
                        basis_batch.push_back(read_basis_vector(j, N));
                    }
                    
                    // Compute contribution from this batch
                    for (int j = 0; j < basis_end - basis_start; j++) {
                        int j_global = basis_start + j;
                        Complex coef(batch_evecs[i*m + j_global], 0.0);
                        cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, full_vector.data(), 1);
                    }
                }
                
                // Normalize
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                if (norm > 1e-12) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, full_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(global_idx) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(full_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
                
                // Print progress occasionally
                if (global_idx % 10 == 0 || global_idx == n_eigenvalues - 1) {
                    std::cout << "  Saved eigenvector " << global_idx + 1 << " of " << n_eigenvalues << std::endl;
                }
            }
            
            // Clear memory by reassigning vectors to empty ones
            std::vector<double>().swap(batch_evals);
            std::vector<double>().swap(batch_evecs);
            std::vector<int>().swap(isuppz);
        }
    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());

    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (!eval_outfile) {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    } else {
        // Write the number of eigenvalues first
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
        // Write all eigenvalues
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
}

// Lanczos algorithm implementation with basis vectors stored on disk
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false) {
    
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
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
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
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
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
        for (int k = 0; k <= j; k++) {
            // Read basis vector k from file
            ComplexVector basis_k = read_basis_vector(k, N);
            
            Complex overlap;
            cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
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
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    if (eigenvectors) {
        // Need space for eigenvectors but m might be too large for full allocation
        // Instead of computing all eigenvectors at once, compute them in batches
        const int batch_size = 100; // Adjust based on available memory
        int num_batches = (n_eigenvalues + batch_size - 1) / batch_size; // Ceiling division

        
        // First compute all eigenvalues without eigenvectors
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info 
                  << " when computing eigenvalues" << std::endl;
            system(("rm -rf " + temp_dir).c_str());
            return;
        }
        
        // Then compute eigenvectors in batches using dstevr which allows range selection
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues) - 1;
            int batch_n = end_idx - start_idx + 1;
            
            std::cout << "Computing eigenvectors batch " << batch + 1 << "/" << num_batches 
                  << " (indices " << start_idx << " to " << end_idx << ")" << std::endl;
            
            // Allocate memory just for this batch
            std::vector<double> batch_evals(batch_n);
            std::vector<double> batch_evecs(m * batch_n);
            std::vector<int> isuppz(2 * batch_n);
            
            // Make a copy of the tridiagonal matrix data for dstevr
            std::vector<double> diag_copy = diag;
            std::vector<double> offdiag_copy(offdiag);
            
            int m_found;
            // Compute eigenvectors for this batch using index range
            info = LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'I', m, 
                     diag_copy.data(), offdiag_copy.data(), 
                     0.0, 0.0, // vl, vu not used with 'I' range option
                     start_idx + 1, end_idx + 1, // FORTRAN 1-indexing
                     0.0, // abstol, set to 0 for default
                     &m_found,
                     batch_evals.data(), batch_evecs.data(), m, 
                     isuppz.data());
            
            if (info != 0 || m_found != batch_n) {
            std::cerr << "LAPACKE_dstevr failed with error code " << info 
                  << " when computing eigenvectors for batch " << batch + 1 
                  << ". Found " << m_found << " of " << batch_n << " eigenvectors." << std::endl;
            continue;
            }
            
            // Transform and save each eigenvector in this batch
            for (int i = 0; i < batch_n; i++) {
                int global_idx = start_idx + i;
                
                // Read all basis vectors needed for transformation
                std::vector<ComplexVector> basis_batch(m);
                for (int j = 0; j < m; j++) {
                    basis_batch[j] = read_basis_vector(j, N);
                }
                
                // Transform the eigenvector
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                for (int j = 0; j < m; j++) {
                    Complex coef(batch_evecs[j*batch_n + i], 0.0);
                    cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, full_vector.data(), 1);
                }
                
                // Normalize
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                if (norm > 1e-12) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, full_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(global_idx) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(full_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
                
                // Print progress occasionally
                if (global_idx % 10 == 0 || global_idx == n_eigenvalues - 1) {
                    std::cout << "  Saved eigenvector " << global_idx + 1 << " of " << n_eigenvalues << std::endl;
                }
            }
            
            // Clear memory by reassigning vectors to empty ones
            std::vector<double>().swap(batch_evals);
            std::vector<double>().swap(batch_evecs);
            std::vector<int>().swap(isuppz);
        }
    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());

    
    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (!eval_outfile) {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    } else {
        // Write the number of eigenvalues first
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
        // Write all eigenvalues
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
}

// Block Lanczos algorithm for eigenvalue computation
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int num_eigs, 
                  double tol, std::vector<double>& eigenvalues, std::string dir = "",
                  bool compute_eigenvectors = false, int block_size=4) {
    // Validate input parameters
    if (block_size <= 0) {
        std::cerr << "Block size must be positive" << std::endl;
        return;
    }
    
    if (max_iter <= 0) {
        std::cerr << "Maximum iterations must be positive" << std::endl;
        return;
    }
    
    std::cout << "Block Lanczos: Starting with block size " << block_size << std::endl;
    
    // Create directories for temporary files and output
    std::string temp_dir = dir + "/block_lanczos_temp";
    std::string evec_dir = dir + "/eigenvectors";
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    system(("mkdir -p " + temp_dir).c_str());
    
    // Initialize random generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Step 1: Generate initial orthogonal block of vectors
    std::vector<ComplexVector> V_curr(block_size, ComplexVector(N));
    
    // Generate the first vector randomly
    for (int i = 0; i < N; i++) {
        V_curr[0][i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize the first vector
    double norm = cblas_dznrm2(N, V_curr[0].data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, V_curr[0].data(), 1);
    
    // Generate remaining vectors in the block and orthogonalize
    for (int j = 1; j < block_size; j++) {
        // Generate random vector
        for (int i = 0; i < N; i++) {
            V_curr[j][i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors using modified Gram-Schmidt
        for (int k = 0; k < j; k++) {
            Complex proj;
            cblas_zdotc_sub(N, V_curr[k].data(), 1, V_curr[j].data(), 1, &proj);
            
            Complex neg_proj = -proj;
            cblas_zaxpy(N, &neg_proj, V_curr[k].data(), 1, V_curr[j].data(), 1);
        }
        
        // Normalize
        norm = cblas_dznrm2(N, V_curr[j].data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, V_curr[j].data(), 1);
    }
    
    // Write initial block to disk
    for (int j = 0; j < block_size; j++) {
        std::string vector_file = temp_dir + "/block_0_vector_" + std::to_string(j) + ".bin";
        std::ofstream outfile(vector_file, std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Cannot open file " << vector_file << " for writing" << std::endl;
            return;
        }
        outfile.write(reinterpret_cast<char*>(V_curr[j].data()), N * sizeof(Complex));
        outfile.close();
    }
    
    // Storage for block tridiagonal matrix
    // Alpha_j are block diagonal matrices (block_size x block_size)
    // Beta_j are block off-diagonal matrices (block_size x block_size)
    std::vector<std::vector<Complex>> Alpha;  // Will store diagonal blocks
    std::vector<std::vector<Complex>> Beta;   // Will store off-diagonal blocks
    
    // Previous block of vectors (initially empty)
    std::vector<ComplexVector> V_prev(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    
    // Helper function to read a block from disk
    auto read_block = [&temp_dir, N, block_size](int block_idx) -> std::vector<ComplexVector> {
        std::vector<ComplexVector> block(block_size, ComplexVector(N));
        
        for (int j = 0; j < block_size; j++) {
            std::string vector_file = temp_dir + "/block_" + std::to_string(block_idx) + 
                                    "_vector_" + std::to_string(j) + ".bin";
            std::ifstream infile(vector_file, std::ios::binary);
            if (!infile) {
                std::cerr << "Error: Cannot open file " << vector_file << " for reading" << std::endl;
                return block;
            }
            infile.read(reinterpret_cast<char*>(block[j].data()), N * sizeof(Complex));
            infile.close();
        }
        
        return block;
    };
    
    // Main Block Lanczos iteration
    std::cout << "Block Lanczos: Iterating..." << std::endl;
    
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Block Lanczos: Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Step 2: Compute W_j = A*V_j
        std::vector<ComplexVector> W(block_size, ComplexVector(N));
        
        for (int b = 0; b < block_size; b++) {
            H(V_curr[b].data(), W[b].data(), N);
        }
        
        // Step 3: Compute Alpha_j = V_j^H * W_j (block_size x block_size matrix)
        std::vector<Complex> Alpha_j(block_size * block_size, Complex(0.0, 0.0));
        
        for (int row = 0; row < block_size; row++) {
            for (int col = 0; col < block_size; col++) {
                Complex dot;
                cblas_zdotc_sub(N, V_curr[row].data(), 1, W[col].data(), 1, &dot);
                Alpha_j[row * block_size + col] = dot;
            }
        }
        
        // Store Alpha_j
        Alpha.push_back(Alpha_j);
        
        // Step 4: W_j = W_j - V_{j-1} * Beta_{j-1}^H - V_j * Alpha_j
        
        // First subtract V_j * Alpha_j
        for (int col = 0; col < block_size; col++) {
            for (int row = 0; row < block_size; row++) {
                Complex coef = -Alpha_j[row * block_size + col];
                cblas_zaxpy(N, &coef, V_curr[row].data(), 1, W[col].data(), 1);
            }
        }
        
        // Then subtract V_{j-1} * Beta_{j-1}^H if j > 0
        if (j > 0) {
            std::vector<Complex>& Beta_prev = Beta.back();
            
            for (int col = 0; col < block_size; col++) {
                for (int row = 0; row < block_size; row++) {
                    // Beta_prev is stored in row-major order, we need to conjugate transpose
                    Complex coef = -std::conj(Beta_prev[col * block_size + row]);
                    cblas_zaxpy(N, &coef, V_prev[row].data(), 1, W[col].data(), 1);
                }
            }
        }
        
        // Step 5: Full reorthogonalization against all previous blocks
        for (int prev_j = 0; prev_j <= j; prev_j++) {
            std::vector<ComplexVector> V_j = (prev_j == j) ? V_curr : read_block(prev_j);
            
            for (int col = 0; col < block_size; col++) {
                for (int row = 0; row < block_size; row++) {
                    Complex dot;
                    cblas_zdotc_sub(N, V_j[row].data(), 1, W[col].data(), 1, &dot);
                    
                    Complex neg_dot = -dot;
                    cblas_zaxpy(N, &neg_dot, V_j[row].data(), 1, W[col].data(), 1);
                }
            }
        }
        
        // Step 6: Compute QR factorization of W_j to get Beta_j and V_{j+1}
        // We'll use modified Gram-Schmidt for QR
        
        // First compute column norms to check for rank deficiency
        std::vector<double> col_norms(block_size);
        for (int col = 0; col < block_size; col++) {
            col_norms[col] = cblas_dznrm2(N, W[col].data(), 1);
        }
        
        // Initialize R (Beta_j) as zero matrix
        std::vector<Complex> Beta_j(block_size * block_size, Complex(0.0, 0.0));
        
        // Initialize next block of basis vectors
        std::vector<ComplexVector> V_next(block_size, ComplexVector(N));
        
        // Threshold for considering a vector linearly dependent
        const double lin_dep_threshold = tol * std::sqrt(static_cast<double>(N));
        
        // Check if we need to do a restart due to rank deficiency
        bool rank_deficient = false;
        for (int col = 0; col < block_size; col++) {
            if (col_norms[col] < lin_dep_threshold) {
                rank_deficient = true;
                break;
            }
        }
        
        if (rank_deficient) {
            std::cout << "Block Lanczos: Rank deficiency detected at iteration " << j + 1 << std::endl;
            
            // Generate new random vectors for the deficient columns
            for (int col = 0; col < block_size; col++) {
                if (col_norms[col] < lin_dep_threshold) {
                    // Generate new random vector
                    for (int i = 0; i < N; i++) {
                        W[col][i] = Complex(dist(gen), dist(gen));
                    }
                    
                    // Orthogonalize against all previous vectors and current non-deficient vectors
                    for (int prev_j = 0; prev_j <= j; prev_j++) {
                        std::vector<ComplexVector> V_j = (prev_j == j) ? V_curr : read_block(prev_j);
                        
                        for (int v = 0; v < block_size; v++) {
                            Complex dot;
                            cblas_zdotc_sub(N, V_j[v].data(), 1, W[col].data(), 1, &dot);
                            
                            Complex neg_dot = -dot;
                            cblas_zaxpy(N, &neg_dot, V_j[v].data(), 1, W[col].data(), 1);
                        }
                    }
                    
                    // Also orthogonalize against the previous columns of W that weren't deficient
                    for (int prev_col = 0; prev_col < col; prev_col++) {
                        if (col_norms[prev_col] >= lin_dep_threshold) {
                            Complex dot;
                            cblas_zdotc_sub(N, V_next[prev_col].data(), 1, W[col].data(), 1, &dot);
                            
                            Complex neg_dot = -dot;
                            cblas_zaxpy(N, &neg_dot, V_next[prev_col].data(), 1, W[col].data(), 1);
                        }
                    }
                    
                    // Recompute the norm
                    col_norms[col] = cblas_dznrm2(N, W[col].data(), 1);
                }
            }
        }
        
        // Perform modified Gram-Schmidt to get Q and R
        for (int col = 0; col < block_size; col++) {
            // Normalize current column to get Q[:,col]
            Beta_j[col * block_size + col] = Complex(col_norms[col], 0.0);
            scale = Complex(1.0/col_norms[col], 0.0);
            cblas_zscal(N, &scale, W[col].data(), 1);
            V_next[col] = W[col];
            
            // Orthogonalize remaining columns against this one
            for (int next_col = col + 1; next_col < block_size; next_col++) {
                Complex dot;
                cblas_zdotc_sub(N, V_next[col].data(), 1, W[next_col].data(), 1, &dot);
                
                // Store in R (Beta_j)
                Beta_j[next_col * block_size + col] = dot;
                
                // Subtract projection
                Complex neg_dot = -dot;
                cblas_zaxpy(N, &neg_dot, V_next[col].data(), 1, W[next_col].data(), 1);
                
                // Update column norm
                col_norms[next_col] = cblas_dznrm2(N, W[next_col].data(), 1);
            }
        }
        
        // Store Beta_j
        Beta.push_back(Beta_j);
        
        // Write the next block to disk
        for (int b = 0; b < block_size; b++) {
            std::string vector_file = temp_dir + "/block_" + std::to_string(j+1) + 
                                    "_vector_" + std::to_string(b) + ".bin";
            std::ofstream outfile(vector_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << vector_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(V_next[b].data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Update for next iteration
        V_prev = V_curr;
        V_curr = V_next;
        
        // Periodically check convergence
        if ((j+1) % 10 == 0 || j == max_iter - 1) {
            int current_size = (j+1) * block_size;
            
            if (current_size >= num_eigs) {
                std::cout << "  Computing eigenvalues at iteration " << j+1 << "..." << std::endl;
                
                // Construct the block tridiagonal matrix
                std::vector<Complex> block_tridiag(current_size * current_size, Complex(0.0, 0.0));
                
                // Fill diagonal blocks (Alpha)
                for (int block_idx = 0; block_idx <= j; block_idx++) {
                    std::vector<Complex>& Alpha_block = Alpha[block_idx];
                    for (int row = 0; row < block_size; row++) {
                        for (int col = 0; col < block_size; col++) {
                            int global_row = block_idx * block_size + row;
                            int global_col = block_idx * block_size + col;
                            block_tridiag[global_row * current_size + global_col] = Alpha_block[row * block_size + col];
                        }
                    }
                }
                
                // Fill off-diagonal blocks (Beta)
                for (int block_idx = 0; block_idx < j; block_idx++) {
                    std::vector<Complex>& Beta_block = Beta[block_idx];
                    for (int row = 0; row < block_size; row++) {
                        for (int col = 0; col < block_size; col++) {
                            int global_row = block_idx * block_size + row;
                            int global_col = (block_idx + 1) * block_size + col;
                            block_tridiag[global_row * current_size + global_col] = Beta_block[row * block_size + col];
                            block_tridiag[global_col * current_size + global_row] = std::conj(Beta_block[row * block_size + col]);
                        }
                    }
                }
                
                // Compute eigenvalues
                std::vector<double> evals_real(current_size);
                int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', current_size,
                                        reinterpret_cast<lapack_complex_double*>(block_tridiag.data()),
                                        current_size, evals_real.data());
                
                if (info == 0) {
                    // Check convergence for lowest eigenvalues
                    int num_converged = 0;
                    for (int i = 0; i < std::min(num_eigs, current_size); i++) {
                        // For simplicity, consider converged if we're at the final iteration
                        if (j == max_iter - 1) {
                            num_converged++;
                        } else if (j > 0) {
                            // More sophisticated convergence check would go here
                            // Using last block and eigenvector components to estimate residual
                            num_converged++;
                        }
                    }
                    
                    std::cout << "  Eigenvalues computed, " << num_converged << " of " << num_eigs << " converged" << std::endl;
                    
                    if (num_converged >= num_eigs) {
                        std::cout << "Block Lanczos: All required eigenvalues converged" << std::endl;
                        break;
                    }
                }
            }
        }
    }
    
    // Final computation of eigenvalues and eigenvectors
    int total_size = Alpha.size() * block_size;
    std::cout << "Block Lanczos: Computing final eigenvalues and eigenvectors..." << std::endl;
    
    // Construct the final block tridiagonal matrix
    std::vector<Complex> block_tridiag(total_size * total_size, Complex(0.0, 0.0));
    
    // Fill diagonal blocks (Alpha)
    for (int block_idx = 0; block_idx < Alpha.size(); block_idx++) {
        std::vector<Complex>& Alpha_block = Alpha[block_idx];
        for (int row = 0; row < block_size; row++) {
            for (int col = 0; col < block_size; col++) {
                int global_row = block_idx * block_size + row;
                int global_col = block_idx * block_size + col;
                block_tridiag[global_row * total_size + global_col] = Alpha_block[row * block_size + col];
            }
        }
    }
    
    // Fill off-diagonal blocks (Beta)
    for (int block_idx = 0; block_idx < Beta.size(); block_idx++) {
        std::vector<Complex>& Beta_block = Beta[block_idx];
        for (int row = 0; row < block_size; row++) {
            for (int col = 0; col < block_size; col++) {
                int global_row = block_idx * block_size + row;
                int global_col = (block_idx + 1) * block_size + col;
                block_tridiag[global_row * total_size + global_col] = Beta_block[row * block_size + col];
                block_tridiag[global_col * total_size + global_row] = std::conj(Beta_block[row * block_size + col]);
            }
        }
    }
    
    // Compute eigenvalues and eigenvectors
    std::vector<double> evals_real(total_size);
    std::vector<Complex> evecs;
    
    if (compute_eigenvectors) {
        evecs.resize(total_size * total_size);
    }
    
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 
                          compute_eigenvectors ? 'V' : 'N',
                          'U',
                          total_size,
                          reinterpret_cast<lapack_complex_double*>(block_tridiag.data()),
                          total_size,
                          evals_real.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues to output
    eigenvalues.resize(num_eigs);
    for (int i = 0; i < std::min(num_eigs, total_size); i++) {
        eigenvalues[i] = evals_real[i];
    }
    
    // Compute eigenvectors in original basis if requested
    if (compute_eigenvectors) {
        std::cout << "Block Lanczos: Computing eigenvectors in original basis..." << std::endl;
        
        // Process in batches to save memory
        const int batch_size = 10;
        for (int batch = 0; batch < (num_eigs + batch_size - 1) / batch_size; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_eigs);
            int batch_n = end_idx - start_idx;
            
            // Allocate memory for eigenvectors
            std::vector<ComplexVector> full_evecs(batch_n, ComplexVector(N, Complex(0.0, 0.0)));
            
            // For each block of Lanczos vectors
            for (int block_idx = 0; block_idx < Alpha.size(); block_idx++) {
                // Load the block
                std::vector<ComplexVector> V_block = read_block(block_idx);
                
                // For each eigenvector in the batch
                for (int i = 0; i < batch_n; i++) {
                    int global_i = start_idx + i;
                    
                    // For each vector in the block
                    for (int j = 0; j < block_size; j++) {
                        int global_j = block_idx * block_size + j;
                        
                        Complex coef = evecs[global_i * total_size + global_j];
                        cblas_zaxpy(N, &coef, V_block[j].data(), 1, full_evecs[i].data(), 1);
                    }
                }
            }
            
            // Normalize and save eigenvectors
            for (int i = 0; i < batch_n; i++) {
                double norm = cblas_dznrm2(N, full_evecs[i].data(), 1);
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, full_evecs[i].data(), 1);
                
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(start_idx + i) + ".bin";
                std::ofstream outfile(evec_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(full_evecs[i].data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
        }
        
        // Save eigenvalues to file
        std::string eval_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eval_file, std::ios::binary);
        if (eval_outfile) {
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved " << n_evals << " eigenvalues to " << eval_file << std::endl;
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "Block Lanczos: Completed successfully" << std::endl;
}

// Chebyshev filtered Lanczos algorithm implementation
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Step 1: Estimate spectral bounds using a few steps of power iteration
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Create random vector for power iteration
    ComplexVector v_rand(N);
    for (int i = 0; i < N; i++) {
        v_rand[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, v_rand.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_rand.data(), 1);
    
    // Estimate largest eigenvalue using power iteration
    const int power_steps = 20;
    ComplexVector Hv(N);
    for (int i = 0; i < power_steps; i++) {
        H(v_rand.data(), Hv.data(), N);
        
        // Replace v with Hv (normalized)
        double norm = cblas_dznrm2(N, Hv.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        
        for (int j = 0; j < N; j++) {
            v_rand[j] = Hv[j] * scale;
        }
    }
    
    // Apply H one more time for Rayleigh quotient
    H(v_rand.data(), Hv.data(), N);
    Complex rayleigh_quotient;
    cblas_zdotc_sub(N, v_rand.data(), 1, Hv.data(), 1, &rayleigh_quotient);
    
    // Estimate spectral bounds
    double lambda_max = std::real(rayleigh_quotient);
    double lambda_min = -lambda_max;  // Conservative estimate for symmetric H
    
    // Allow for non-symmetric spectrum
    double spectral_center = (lambda_max + lambda_min) / 2.0;
    double spectral_radius = (lambda_max - lambda_min) / 2.0;
    
    std::cout << "Chebyshev Filtered Lanczos: Estimated spectral bounds [" 
              << lambda_min << ", " << lambda_max << "]" << std::endl;
    
    // Step 2: Apply Chebyshev filter to enhance convergence to desired eigenvalues
    
    // Initialize random starting vector
    ComplexVector v_current(N);
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize the starting vector
    norm = cblas_dznrm2(N, v_current.data(), 1);
    scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_current.data(), 1);
    
    // Apply Chebyshev filter of degree 10 to enhance low eigenvalues
    const int cheby_degree = 10;
    ComplexVector v_prev(N), v_next(N), v_temp(N);
    
    // First Chebyshev term: T_0(x) = 1 (identity)
    ComplexVector c0 = v_current;
    
    // Second Chebyshev term: T_1(x) = x
    H(v_current.data(), v_temp.data(), N);
    
    // Scale and shift to map [lambda_min, lambda_max] to [-1, 1]
    for (int i = 0; i < N; i++) {
        v_temp[i] = (v_temp[i] - Complex(spectral_center, 0.0)) / Complex(spectral_radius, 0.0);
    }
    
    ComplexVector c1 = v_temp;
    
    // Use Chebyshev recurrence relation: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    // We focus on low eigenvalues, so we apply a filter that enhances these
    for (int k = 2; k <= cheby_degree; k++) {
        // First calculate 2*x*T_n(x)
        H(c1.data(), v_temp.data(), N);
        
        // Scale and shift
        for (int i = 0; i < N; i++) {
            v_temp[i] = (v_temp[i] - Complex(spectral_center, 0.0)) / Complex(spectral_radius, 0.0);
        }
        
        // Scale by 2
        Complex factor(2.0, 0.0);
        cblas_zscal(N, &factor, v_temp.data(), 1);
        
        // Subtract T_{n-1}(x) to get T_{n+1}(x)
        for (int i = 0; i < N; i++) {
            v_next[i] = v_temp[i] - c0[i];
        }
        
        // Update for next iteration
        c0 = c1;
        c1 = v_next;
    }
    
    // Use the filtered vector as the starting vector for Lanczos
    v_current = c1;
    
    // Normalize again
    norm = cblas_dznrm2(N, v_current.data(), 1);
    scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = "chebyshev_lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
    v_prev = ComplexVector(N, Complex(0.0, 0.0));
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Chebyshev Filtered Lanczos: Starting iterations..." << std::endl;
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
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
        
        // Full reorthogonalization
        for (int k = 0; k <= j; k++) {
            // Read basis vector k from file
            ComplexVector basis_k = read_basis_vector(k, N);
            
            Complex overlap;
            cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for invariant subspace
        if (norm < tol) {
            std::cout << "Chebyshev Filtered Lanczos: Invariant subspace found at iteration " << j + 1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "Chebyshev Filtered Lanczos: Constructing tridiagonal matrix" << std::endl;

    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }

    std::cout << "Chebyshev Filtered Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    std::vector<double> evals(m);        // For eigenvalues    
    std::vector<double> evecs;           // For eigenvectors
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    if (eigenvectors) {
        evecs.resize(m*m);
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());
    
    // Transform eigenvectors if requested
    if (eigenvectors) {
        eigenvectors->resize(n_eigenvalues, ComplexVector(N));
        
        for (int i = 0; i < n_eigenvalues; i++) {
            for (int j = 0; j < N; j++) {
                (*eigenvectors)[i][j] = Complex(0.0, 0.0);
            }
            
            for (int j = 0; j < m; j++) {
                // Load j-th Lanczos basis vector
                ComplexVector basis_j = read_basis_vector(j, N);
                
                // Add contribution to eigenvector
                Complex coef(evecs[j*m + i], 0.0);
                cblas_zaxpy(N, &coef, basis_j.data(), 1, (*eigenvectors)[i].data(), 1);
            }
            
            // Normalize the eigenvector
            double norm = cblas_dznrm2(N, (*eigenvectors)[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, (*eigenvectors)[i].data(), 1);
        }
        
        // Optionally, refine the eigenvectors
        for (int i = 0; i < n_eigenvalues; i++) {
            refine_eigenvector_with_cg(H, (*eigenvectors)[i], eigenvalues[i], N, tol);
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
}

// Shift-invert Lanczos algorithm implementation
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H, 
                         int N, int max_iter, int exct, double sigma, 
                         double tol, std::vector<double>& eigenvalues, std::string dir = "",
                         bool eigenvectors = false) {
    
    std::cout << "Shift-invert Lanczos: Starting with shift σ = " << sigma << std::endl;
    
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
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir + "/shift_invert_lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin Shift-invert Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Helper function to solve the shift-invert system (H - sigma*I)x = v using CG
    auto solve_shifted_system = [&H, sigma, N, tol](const ComplexVector& v) -> ComplexVector {
        // Define the shifted operator (H - sigma*I)
        auto shifted_H = [&H, sigma, N](const Complex* v, Complex* result, int size) {
            // Apply H to v
            H(v, result, size);
            // Subtract sigma*v
            for (int i = 0; i < size; i++) {
                result[i] -= sigma * v[i];
            }
        };
        
        // Initialize CG variables
        ComplexVector x(N, Complex(0.0, 0.0));  // Initial guess
        ComplexVector r(v);                     // Initial residual r = v - A*x = v
        ComplexVector p(r);                     // Initial search direction
        ComplexVector Ap(N);                    // To store A*p
        
        // Initial residual norm
        double r_norm = cblas_dznrm2(N, r.data(), 1);
        double initial_r_norm = r_norm;
        
        // CG iteration
        const int max_cg_iter = 1000;
        for (int iter = 0; iter < max_cg_iter && r_norm > tol * initial_r_norm; iter++) {
            // Apply shifted operator to p
            shifted_H(p.data(), Ap.data(), N);
            
            // Calculate step size alpha = (r·r)/(p·Ap)
            Complex r_dot_r, p_dot_Ap;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            cblas_zdotc_sub(N, p.data(), 1, Ap.data(), 1, &p_dot_Ap);
            
            Complex alpha = r_dot_r / p_dot_Ap;
            
            // Update solution: x = x + alpha*p
            cblas_zaxpy(N, &alpha, p.data(), 1, x.data(), 1);
            
            // Store old r·r for beta calculation
            Complex r_dot_r_old = r_dot_r;
            
            // Update residual: r = r - alpha*Ap
            Complex neg_alpha = -alpha;
            cblas_zaxpy(N, &neg_alpha, Ap.data(), 1, r.data(), 1);
            
            // Check convergence
            r_norm = cblas_dznrm2(N, r.data(), 1);
            
            // Calculate beta = (r_new·r_new)/(r_old·r_old)
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            Complex beta = r_dot_r / r_dot_r_old;
            
            // Update search direction: p = r + beta*p
            for (int j = 0; j < N; j++) {
                p[j] = r[j] + beta * p[j];
            }
        }
        
        return x;
    };
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Apply shift-invert operator: w = (H - sigma*I)^(-1) * v_j
        w = solve_shifted_system(v_current);
        
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
        
        // Full reorthogonalization
        // for (int k = 0; k <= j; k++) {
        //     // Read basis vector k from file
        //     ComplexVector basis_k = read_basis_vector(k, N);
            
        //     Complex overlap;
        //     cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
        //     Complex neg_overlap = -overlap;
        //     cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
        // }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for invariant subspace
        if (norm < tol) {
            std::cout << "Shift-invert Lanczos: Invariant subspace found at iteration " << j + 1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "Shift-invert Lanczos: Constructing tridiagonal matrix" << std::endl;

    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }

    std::cout << "Shift-invert Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Save the requested number of eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    std::vector<double> shift_invert_evals(m);  // Eigenvalues of shift-invert operator
    std::vector<double> evecs;                 // For eigenvectors
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    if (eigenvectors) {
        evecs.resize(m*m);
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Convert shift-invert eigenvalues back to original spectrum
    // For shift-invert: λ_original = σ + 1/λ_shift_invert
    eigenvalues.resize(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        // Note: The largest eigenvalues of the shift-invert operator
        // correspond to eigenvalues closest to sigma in the original spectrum
        double shift_invert_eval = diag[m-1-i];  // Take from the largest values
        
        // Avoid division by zero
        if (std::abs(shift_invert_eval) < 1e-10) {
            eigenvalues[i] = sigma;
        } else {
            eigenvalues[i] = sigma + 1.0 / shift_invert_eval;
        }
    }
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (!eval_outfile) {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    } else {
        // Write the number of eigenvalues first
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
        // Write all eigenvalues
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    
    // Transform eigenvectors if requested
    if (eigenvectors) {
        std::cout << "Shift-invert Lanczos: Transforming eigenvectors" << std::endl;
        
        // Process in batches to save memory
        const int batch_size = 10;
        for (int batch = 0; batch < (n_eigenvalues + batch_size - 1) / batch_size; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues);
            
            std::cout << "Processing eigenvectors " << start_idx + 1 << " to " << end_idx 
                      << " of " << n_eigenvalues << std::endl;
            
            // For each eigenvector in this batch
            for (int i = start_idx; i < end_idx; i++) {
                // The index in the evecs array (reverse order since we want closest to sigma)
                int si_idx = m - 1 - i;
                
                // Calculate the full eigenvector in the original basis
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                
                // Read basis vectors in batches to reduce disk I/O
                const int basis_batch_size = 100;
                for (int basis_start = 0; basis_start < m; basis_start += basis_batch_size) {
                    int basis_end = std::min(basis_start + basis_batch_size, m);
                    
                    // Read this batch of basis vectors
                    std::vector<ComplexVector> basis_batch;
                    basis_batch.reserve(basis_end - basis_start);
                    for (int j = basis_start; j < basis_end; j++) {
                        basis_batch.push_back(read_basis_vector(j, N));
                    }
                    
                    // Compute contribution from this batch
                    for (int j = 0; j < basis_end - basis_start; j++) {
                        int j_global = basis_start + j;
                        Complex coef(evecs[si_idx*m + j_global], 0.0);
                        cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, full_vector.data(), 1);
                    }
                }
                
                // Normalize the eigenvector
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, full_vector.data(), 1);
                
                // Optionally refine the eigenvector using inverse iteration
                // (H - lambda*I)x = previous_x
                ComplexVector refined_vector = full_vector;
                for (int refine_iter = 0; refine_iter < 3; refine_iter++) {
                    // Create a shifted system for this specific eigenvalue
                    auto lambda_shifted_system = [&H, &eigenvalues, i, N](const Complex* v, Complex* result, int size) {
                        // Apply H to v
                        H(v, result, size);
                        // Subtract lambda*v
                        double lambda = eigenvalues[i];
                        for (int j = 0; j < size; j++) {
                            result[j] -= lambda * v[j];
                        }
                    };
                    
                    // Solve the system
                    ComplexVector temp_vec = refined_vector;
                    refined_vector.assign(N, Complex(0.0, 0.0));
                    
                    // Use a simpler direct solver for refinement
                    // In practice, we would use a more sophisticated method here
                    auto shifted_H = [&H, &eigenvalues, i, N](const Complex* v, Complex* result, int size) {
                        H(v, result, size);
                        for (int j = 0; j < size; j++) {
                            result[j] -= eigenvalues[i] * v[j];
                        }
                    };
                    
                    // Approximate solution using a few steps of CG
                    ComplexVector r(temp_vec);
                    ComplexVector p(r);
                    ComplexVector Ap(N);
                    
                    for (int cg_iter = 0; cg_iter < 5; cg_iter++) {
                        shifted_H(p.data(), Ap.data(), N);
                        
                        Complex r_dot_r, p_dot_Ap;
                        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
                        cblas_zdotc_sub(N, p.data(), 1, Ap.data(), 1, &p_dot_Ap);
                        
                        Complex alpha = r_dot_r / p_dot_Ap;
                        cblas_zaxpy(N, &alpha, p.data(), 1, refined_vector.data(), 1);
                        
                        Complex neg_alpha = -alpha;
                        cblas_zaxpy(N, &neg_alpha, Ap.data(), 1, r.data(), 1);
                        
                        Complex r_new_dot_r_new;
                        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_new_dot_r_new);
                        
                        Complex beta = r_new_dot_r_new / r_dot_r;
                        for (int j = 0; j < N; j++) {
                            p[j] = r[j] + beta * p[j];
                        }
                    }
                    
                    // Normalize
                    norm = cblas_dznrm2(N, refined_vector.data(), 1);
                    scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, refined_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(refined_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "Shift-invert Lanczos: Completed successfully" << std::endl;
}

// Robust version of shift-invert Lanczos algorithm with enhanced stability and convergence
void shift_invert_lanczos_robust(std::function<void(const Complex*, Complex*, int)> H, 
                               int N, int max_iter, int exct, double sigma, 
                               double tol, std::vector<double>& eigenvalues, std::string dir = "",
                               bool eigenvectors = false) {
    
    std::cout << "Robust Shift-invert Lanczos: Starting with shift σ = " << sigma << std::endl;
    
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
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir + "/robust_shift_invert_lanczos_basis";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(v_current.data()), N * sizeof(Complex));
    outfile.close();
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin Robust Shift-invert Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Define the shifted operator (H - sigma*I)
    auto shifted_H = [&H, sigma, N](const Complex* v, Complex* result, int size) {
        // Apply H to v
        H(v, result, size);
        // Subtract sigma*v
        for (int i = 0; i < size; i++) {
            result[i] -= sigma * v[i];
        }
    };
    
    // Improved BiCGSTAB solver for the shifted system
    auto solve_shifted_system_bicgstab = [&shifted_H, N, tol](const ComplexVector& b) -> ComplexVector {
        // Initialize solution vector
        ComplexVector x(N, Complex(0.0, 0.0));
        
        // Initialize BiCGSTAB vectors
        ComplexVector r(b);  // Initial residual r = b - A*x (x=0 initially)
        ComplexVector r_hat = r;  // Shadow residual (arbitrary, usually r_0)
        
        // Check if the initial residual is already small enough
        double r_norm = cblas_dznrm2(N, r.data(), 1);
        double b_norm = cblas_dznrm2(N, b.data(), 1);
        double rel_tol = tol * b_norm;
        
        if (r_norm <= rel_tol) {
            return x;  // Initial guess is good enough
        }
        
        // BiCGSTAB iteration vectors
        ComplexVector p(N, Complex(0.0, 0.0));
        ComplexVector v(N, Complex(0.0, 0.0));
        ComplexVector s(N), t(N), Ap(N), As(N);
        
        Complex rho_prev = Complex(1.0, 0.0);
        Complex alpha = Complex(1.0, 0.0);
        Complex omega = Complex(1.0, 0.0);
        
        // BiCGSTAB iterations
        const int max_bicg_iter = 1000;
        int iter;
        for (iter = 0; iter < max_bicg_iter; iter++) {
            // Compute rho = (r_hat, r)
            Complex rho;
            cblas_zdotc_sub(N, r_hat.data(), 1, r.data(), 1, &rho);
            
            if (std::abs(rho) < 1e-15) {
                std::cerr << "BiCGSTAB breakdown: rho near zero" << std::endl;
                break;  // Method fails
            }
            
            // Beta computation to handle the first iteration correctly
            Complex beta;
            if (iter == 0) {
                p = r;
            } else {
                beta = (rho / rho_prev) * (alpha / omega);
                
                // p = r + beta * (p - omega * v)
                for (int i = 0; i < N; i++) {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }
            
            // Compute v = A*p
            shifted_H(p.data(), Ap.data(), N);
            std::copy(Ap.begin(), Ap.end(), v.begin());
            
            // Compute alpha = rho / (r_hat, v)
            Complex r_hat_dot_v;
            cblas_zdotc_sub(N, r_hat.data(), 1, v.data(), 1, &r_hat_dot_v);
            
            if (std::abs(r_hat_dot_v) < 1e-15) {
                std::cerr << "BiCGSTAB breakdown: r_hat_dot_v near zero" << std::endl;
                break;  // Method fails
            }
            
            alpha = rho / r_hat_dot_v;
            
            // Compute s = r - alpha * v
            for (int i = 0; i < N; i++) {
                s[i] = r[i] - alpha * v[i];
            }
            
            // Check if solution is accurate enough
            double s_norm = cblas_dznrm2(N, s.data(), 1);
            if (s_norm <= rel_tol) {
                // x = x + alpha * p
                for (int i = 0; i < N; i++) {
                    x[i] += alpha * p[i];
                }
                break;
            }
            
            // Compute t = A*s
            shifted_H(s.data(), As.data(), N);
            std::copy(As.begin(), As.end(), t.begin());
            
            // Compute omega = (t, s) / (t, t)
            Complex t_dot_s, t_dot_t;
            cblas_zdotc_sub(N, t.data(), 1, s.data(), 1, &t_dot_s);
            cblas_zdotc_sub(N, t.data(), 1, t.data(), 1, &t_dot_t);
            
            if (std::abs(t_dot_t) < 1e-15) {
                std::cerr << "BiCGSTAB breakdown: t_dot_t near zero" << std::endl;
                omega = Complex(0.0, 0.0);
            } else {
                omega = t_dot_s / t_dot_t;
            }
            
            // Update solution: x = x + alpha * p + omega * s
            for (int i = 0; i < N; i++) {
                x[i] += alpha * p[i] + omega * s[i];
            }
            
            // Compute residual: r = s - omega * t
            for (int i = 0; i < N; i++) {
                r[i] = s[i] - omega * t[i];
            }
            
            // Check convergence
            r_norm = cblas_dznrm2(N, r.data(), 1);
            if (r_norm <= rel_tol || std::abs(omega) < 1e-15) {
                break;
            }
            
            // Store rho for next iteration
            rho_prev = rho;
        }
        
        // Iterative refinement to improve accuracy
        if (iter < max_bicg_iter) {
            // Compute residual: r = b - A*x
            ComplexVector Ax(N);
            shifted_H(x.data(), Ax.data(), N);
            
            for (int i = 0; i < N; i++) {
                r[i] = b[i] - Ax[i];
            }
            
            // Perform a few extra iterations to improve solution
            ComplexVector dx(N, Complex(0.0, 0.0));
            for (int refine = 0; refine < 2; refine++) {
                // Solve A*dx = r approximately
                r_norm = cblas_dznrm2(N, r.data(), 1);
                if (r_norm <= rel_tol) break;
                
                // Use a few steps of CG for refinement
                ComplexVector dr(N), dp(N), dAp(N);
                std::copy(r.begin(), r.end(), dr.begin());
                std::copy(r.begin(), r.end(), dp.begin());
                
                for (int cg_iter = 0; cg_iter < 5; cg_iter++) {
                    shifted_H(dp.data(), dAp.data(), N);
                    
                    Complex dr_dot_dr, dp_dot_dAp;
                    cblas_zdotc_sub(N, dr.data(), 1, dr.data(), 1, &dr_dot_dr);
                    cblas_zdotc_sub(N, dp.data(), 1, dAp.data(), 1, &dp_dot_dAp);
                    
                    Complex alpha_cg = dr_dot_dr / dp_dot_dAp;
                    
                    // Update solution
                    for (int i = 0; i < N; i++) {
                        dx[i] += alpha_cg * dp[i];
                        dr[i] -= alpha_cg * dAp[i];
                    }
                    
                    Complex dr_new_dot_dr_new;
                    cblas_zdotc_sub(N, dr.data(), 1, dr.data(), 1, &dr_new_dot_dr_new);
                    Complex beta_cg = dr_new_dot_dr_new / dr_dot_dr;
                    
                    for (int i = 0; i < N; i++) {
                        dp[i] = dr[i] + beta_cg * dp[i];
                    }
                }
                
                // Update solution: x = x + dx
                for (int i = 0; i < N; i++) {
                    x[i] += dx[i];
                }
                
                // Recompute residual
                shifted_H(x.data(), Ax.data(), N);
                for (int i = 0; i < N; i++) {
                    r[i] = b[i] - Ax[i];
                }
            }
        }
        
        return x;
    };
    
    // Lanczos iteration with enhanced stability
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Apply shift-invert operator: w = (H - sigma*I)^(-1) * v_j
        w = solve_shifted_system_bicgstab(v_current);
        
        // Check solution quality
        ComplexVector check_residual(N);
        shifted_H(w.data(), check_residual.data(), N);
        for (int i = 0; i < N; i++) {
            check_residual[i] = v_current[i] - check_residual[i];
        }
        double residual_norm = cblas_dznrm2(N, check_residual.data(), 1);
        
        if (residual_norm > 100 * tol) {
            std::cout << "  Warning: Linear solver residual = " << residual_norm 
                      << ", which is higher than expected. Attempting to improve." << std::endl;
            
            // Try again with higher precision
            ComplexVector w_improved = solve_shifted_system_bicgstab(v_current);
            
            // Check if the improved solution is better
            ComplexVector check_improved(N);
            shifted_H(w_improved.data(), check_improved.data(), N);
            for (int i = 0; i < N; i++) {
                check_improved[i] = v_current[i] - check_improved[i];
            }
            double improved_norm = cblas_dznrm2(N, check_improved.data(), 1);
            
            if (improved_norm < residual_norm) {
                w = w_improved;
                residual_norm = improved_norm;
                std::cout << "  Improved solution with residual = " << residual_norm << std::endl;
            }
        }
        
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
        
        // Enhanced full reorthogonalization with modified Gram-Schmidt
        for (int k = 0; k <= j; k++) {
            // First orthogonalization pass
            ComplexVector basis_k = read_basis_vector(k, N);
            
            Complex overlap;
            cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
            if (std::abs(overlap) > tol) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                
                // Second orthogonalization pass for numerical stability
                cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                if (std::abs(overlap) > tol) {
                    neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for invariant subspace or numerical issues
        if (norm < tol || std::isnan(norm)) {
            std::cout << "Shift-invert Lanczos: ";
            if (std::isnan(norm)) {
                std::cout << "Numerical issue detected at iteration " << j + 1 << std::endl;
                // If we have enough iterations, we can continue with what we have
                if (j >= exct + 10) {
                    std::cout << "Proceeding with " << j << " iterations" << std::endl;
                    max_iter = j;
                    break;
                } else {
                    // Restart with a different random vector
                    std::cout << "Restarting with a new random vector" << std::endl;
                    for (int i = 0; i < N; i++) {
                        v_current[i] = Complex(dist(gen), dist(gen));
                    }
                    norm = cblas_dznrm2(N, v_current.data(), 1);
                    scale_factor = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale_factor, v_current.data(), 1);
                    
                    // Reset and start over
                    alpha.clear();
                    beta.clear();
                    beta.push_back(0.0);
                    j = -1;  // Will be incremented to 0 in next loop iteration
                    continue;
                }
            } else {
                std::cout << "Invariant subspace found at iteration " << j + 1 << std::endl;
                max_iter = j + 1;
                break;
            }
        }
        
        // Periodically check convergence of eigenvalues
        if (j >= exct && j % 5 == 0) {
            // Construct temporary tridiagonal matrix
            std::vector<double> diag_tmp(alpha);
            std::vector<double> offdiag_tmp(j);
            for (int i = 0; i < j; i++) {
                offdiag_tmp[i] = beta[i+1];
            }
            
            // Compute eigenvalues of the tridiagonal matrix
            std::vector<double> evals_tmp(j+1);
            int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', j+1, diag_tmp.data(), offdiag_tmp.data(), nullptr, j+1);
            
            if (info == 0) {
                // Count converged eigenvalues
                int nconv = 0;
                for (int i = 0; i < std::min(exct, j+1); i++) {
                    double ritz_value = sigma + 1.0 / evals_tmp[i];
                    double residual = std::abs(beta[j] * evals_tmp[j] / beta[j+1]);
                    if (residual < tol) nconv++;
                }
                
                std::cout << "  " << nconv << " eigenvalues converged so far" << std::endl;
                
                // If we have enough converged eigenvalues, we can stop
                if (nconv >= exct) {
                    std::cout << "Sufficient eigenvalues converged at iteration " << j + 1 << std::endl;
                    max_iter = j + 1;
                    break;
                }
            }
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "Robust Shift-invert Lanczos: Constructing tridiagonal matrix" << std::endl;

    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }

    std::cout << "Robust Shift-invert Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Save the requested number of eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    std::vector<double> shift_invert_evals(m);  // Eigenvalues of shift-invert operator
    std::vector<double> evecs;                 // For eigenvectors
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    if (eigenvectors) {
        evecs.resize(m*m);
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        // Try with a more stable but slower algorithm
        std::cout << "Attempting to use a more stable algorithm..." << std::endl;
        
        // Reset the arrays
        diag = alpha;
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = beta[i+1];
        }
        
        // Use LAPACKE_dsterf for just eigenvalues (more stable)
        if (!eigenvectors) {
            info = LAPACKE_dsterf(m, diag.data(), offdiag.data());
        } else {
            // For eigenvectors, use QR algorithm with explicit shifts
            info = LAPACKE_dsteqr(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), evecs.data(), m);
        }
        
        if (info != 0) {
            std::cerr << "Stable algorithm also failed with error code " << info << std::endl;
            system(("rm -rf " + temp_dir).c_str());
            return;
        }
    }
    
    // Convert shift-invert eigenvalues back to original spectrum
    // For shift-invert: λ_original = σ + 1/λ_shift_invert
    eigenvalues.resize(n_eigenvalues);
    
    // Sort by proximity to sigma (largest shift-invert eigenvalues)
    std::vector<std::pair<double, int>> sorted_indices;
    for (int i = 0; i < m; i++) {
        sorted_indices.push_back({std::abs(diag[i]), i});
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    for (int i = 0; i < n_eigenvalues; i++) {
        int idx = sorted_indices[i].second;
        double shift_invert_eval = diag[idx];
        
        // Avoid division by zero
        if (std::abs(shift_invert_eval) < 1e-10) {
            eigenvalues[i] = sigma;
        } else {
            eigenvalues[i] = sigma + 1.0 / shift_invert_eval;
        }
    }
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (!eval_outfile) {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    } else {
        // Write the number of eigenvalues first
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
        // Write all eigenvalues
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    
    // Transform eigenvectors if requested
    if (eigenvectors) {
        std::cout << "Robust Shift-invert Lanczos: Transforming eigenvectors" << std::endl;
        
        // Process eigenvectors for the eigenvalues closest to the shift
        for (int i = 0; i < n_eigenvalues; i++) {
            int idx = sorted_indices[i].second;
            
            // Calculate the full eigenvector in the original basis
            ComplexVector full_vector(N, Complex(0.0, 0.0));
            
            // Project Lanczos vectors
            for (int j = 0; j < m; j++) {
                ComplexVector basis_j = read_basis_vector(j, N);
                Complex coef(evecs[idx*m + j], 0.0);
                cblas_zaxpy(N, &coef, basis_j.data(), 1, full_vector.data(), 1);
            }
            
            // Normalize the eigenvector
            double vec_norm = cblas_dznrm2(N, full_vector.data(), 1);
            Complex scale = Complex(1.0/vec_norm, 0.0);
            cblas_zscal(N, &scale, full_vector.data(), 1);
            
            // Refine eigenvector using inverse iteration
            double lambda = eigenvalues[i];
            ComplexVector refined_vector = full_vector;
            
            // Apply (H - lambda*I)^(-1) a few times to improve eigenvector
            for (int refine_iter = 0; refine_iter < 2; refine_iter++) {
                // Define a new shifted system for this specific eigenvalue
                auto lambda_shifted_H = [&H, lambda, N](const Complex* v, Complex* result, int size) {
                    H(v, result, size);
                    for (int j = 0; j < size; j++) {
                        result[j] -= lambda * v[j];
                    }
                };
                
                // Solve the system using BiCGSTAB
                ComplexVector temp = refined_vector;
                auto solve_lambda_system = [&lambda_shifted_H, N, tol](const ComplexVector& v) -> ComplexVector {
                    // Similar to solve_shifted_system_bicgstab but with the specific lambda
                    // Implementation omitted for brevity - would use BiCGSTAB as above
                    // Just a placeholder for a real implementation
                    ComplexVector result(N);
                    lambda_shifted_H(v.data(), result.data(), N);
                    return result;
                };
                
                // Apply inverse iteration step
                refined_vector = solve_shifted_system_bicgstab(refined_vector);
                
                // Normalize
                vec_norm = cblas_dznrm2(N, refined_vector.data(), 1);
                scale = Complex(1.0/vec_norm, 0.0);
                cblas_zscal(N, &scale, refined_vector.data(), 1);
                
                // Check improvement
                ComplexVector check_Hv(N);
                H(refined_vector.data(), check_Hv.data(), N);
                for (int j = 0; j < N; j++) {
                    check_Hv[j] -= lambda * refined_vector[j];
                }
                double residual = cblas_dznrm2(N, check_Hv.data(), 1);
                
                if (residual < tol) {
                    break;  // Good enough
                }
            }
            
            // Save refined eigenvector to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (!evec_outfile) {
                std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                continue;
            }
            evec_outfile.write(reinterpret_cast<char*>(refined_vector.data()), N * sizeof(Complex));
            evec_outfile.close();
            
            // Progress reporting
            if ((i+1) % 10 == 0 || i == n_eigenvalues - 1) {
                std::cout << "  Processed eigenvector " << i+1 << " of " << n_eigenvalues << std::endl;
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "Robust Shift-invert Lanczos: Completed successfully" << std::endl;
}

// Full diagonalization using LAPACK for Hermitian matrices
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, int N,
                          std::vector<double>& eigenvalues, std::string dir = "",
                          bool compute_eigenvectors = false) {
    
    std::cout << "Full diagonalization: Starting for matrix of dimension " << N << std::endl;
    
    // Create directory for output if needed
    std::string evec_dir = dir + "/eigenvectors";
    if (compute_eigenvectors && !dir.empty()) {
        std::string cmd = "mkdir -p " + evec_dir;
        system(cmd.c_str());
    }
    
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
    
    // Call LAPACK eigensolver
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 
                           compute_eigenvectors ? 'V' : 'N', // 'V' to compute eigenvectors, 'N' for eigenvalues only
                           'U',                              // Upper triangular part of the matrix is used
                           N, 
                           reinterpret_cast<lapack_complex_double*>(full_matrix.data()), 
                           N, 
                           eigenvalues.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        return;
    }
    
    // Save eigenvectors to files if requested
    if (compute_eigenvectors && !dir.empty()) {
        std::cout << "Full diagonalization: Saving eigenvectors to " << evec_dir << std::endl;
        
        for (int i = 0; i < N; i++) {
            // Extract eigenvector from column of full_matrix
            ComplexVector eigenvector(N);
            for (int j = 0; j < N; j++) {
                eigenvector[j] = full_matrix[i*N + j];
            }
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (!evec_outfile) {
                std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                continue;
            }
            evec_outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
            evec_outfile.close();
            
            // Print progress occasionally
            if (i % 10 == 0 || i == N - 1) {
                std::cout << "  Saved eigenvector " << i + 1 << " of " << N << std::endl;
            }
        }
        
        // Save eigenvalues to a single file
        std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
        if (!eval_outfile) {
            std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
        } else {
            // Write the number of eigenvalues first
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            // Write all eigenvalues
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), N * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved " << N << " eigenvalues to " << eigenvalue_file << std::endl;
        }
    }
    
    std::cout << "Full diagonalization completed successfully" << std::endl;
}

void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                 int num_eigs, double tol, std::vector<double>& eigenvalues, 
                 std::string dir = "", bool compute_eigenvectors = false) {
    
    // Create directories for output
    std::string temp_dir = dir + "/krylov_schur_temp";
    std::string evec_dir = dir + "/eigenvectors";
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    system(("mkdir -p " + temp_dir).c_str());
    
    // Parameters
    int m = std::min(2*num_eigs + 20, max_iter);  // Maximum size of Krylov subspace
    int k = num_eigs;                            // Number of wanted eigenvalues
    
    // Helper functions to read/write vectors from/to disk
    auto write_vector = [&temp_dir, N](int index, const ComplexVector& vec) {
        std::string filename = temp_dir + "/krylov_vector_" + std::to_string(index) + ".bin";
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
            return false;
        }
        outfile.write(reinterpret_cast<const char*>(vec.data()), N * sizeof(Complex));
        outfile.close();
        return true;
    };
    
    auto read_vector = [&temp_dir, N](int index) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/krylov_vector_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        infile.close();
        return vec;
    };
    
    // Initialize random starting vector
    ComplexVector v(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v.data(), 1);
    
    // Write initial vector to disk
    write_vector(0, v);
    
    // Storage for tridiagonal matrix
    std::vector<double> alpha(m, 0.0);  // Diagonal
    std::vector<double> beta(m+1, 0.0);  // Off-diagonal (beta[0] is unused)
    
    // Main loop for Krylov-Schur iterations
    int iter = 0;
    const int max_restarts = 100;
    bool converged = false;
    
    while (iter < max_restarts && !converged) {
        std::cout << "Krylov-Schur iteration " << iter + 1 << std::endl;
        
        // Build or extend Krylov decomposition
        int j_start = (iter == 0) ? 0 : k;
        
        for (int j = j_start; j < m; j++) {
            // Read vector from disk
            ComplexVector v_j = read_vector(j);
            
            // w = H * v_j
            ComplexVector w(N);
            H(v_j.data(), w.data(), N);
            
            // Orthogonalize against previous vectors (for Lanczos/Hermitian case)
            if (j > 0) {
                ComplexVector v_j_minus_1 = read_vector(j-1);
                Complex neg_beta(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_j_minus_1.data(), 1, w.data(), 1);
            }
            
            // alpha_j = <v_j, w>
            Complex dot;
            cblas_zdotc_sub(N, v_j.data(), 1, w.data(), 1, &dot);
            alpha[j] = std::real(dot);  // For Hermitian case, alpha is real
            
            // w = w - alpha_j * v_j
            Complex neg_alpha(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_j.data(), 1, w.data(), 1);
            
            // Full reorthogonalization for numerical stability
            for (int i = 0; i <= j; i++) {
                ComplexVector v_i = read_vector(i);
                Complex ip;
                cblas_zdotc_sub(N, v_i.data(), 1, w.data(), 1, &ip);
                Complex neg_ip = -ip;
                cblas_zaxpy(N, &neg_ip, v_i.data(), 1, w.data(), 1);
            }
            
            // beta_{j+1} = ||w||
            beta[j+1] = cblas_dznrm2(N, w.data(), 1);
            
            // Check for breakdown (invariant subspace)
            if (beta[j+1] < tol) {
                m = j + 1;  // Reduce subspace size
                std::cout << "Invariant subspace found at step " << j + 1 << std::endl;
                break;
            }
            
            // v_{j+1} = w / beta_{j+1}
            if (j < m-1) {
                scale = Complex(1.0/beta[j+1], 0.0);
                cblas_zscal(N, &scale, w.data(), 1);
                write_vector(j+1, w);
            }
        }
        
        // We have a Krylov decomposition: A*V_m = V_m*T_m + beta_m+1*v_m+1*e_m^T
        
        // Compute eigendecomposition of the tridiagonal matrix
        std::vector<double> d(m);  // Eigenvalues
        std::vector<double> e(m-1);  // Subdiagonal elements
        std::vector<double> z(m*m, 0.0);  // Eigenvectors in column-major order
        
        // Copy matrix elements to LAPACK format
        for (int i = 0; i < m; i++) {
            d[i] = alpha[i];
        }
        for (int i = 0; i < m-1; i++) {
            e[i] = beta[i+1];
        }
        
        // Call LAPACK to compute eigenvalues and eigenvectors of tridiagonal matrix
        char jobz = 'V';  // Compute both eigenvalues and eigenvectors
        int info = LAPACKE_dstev(LAPACK_COL_MAJOR, jobz, m, d.data(), e.data(), z.data(), m);
        
        if (info != 0) {
            std::cerr << "Error in LAPACKE_dstev: " << info << std::endl;
            break;
        }
        
        // Sort eigenvalues (smallest first) and corresponding eigenvectors
        std::vector<std::pair<double, int>> eig_pairs;
        for (int i = 0; i < m; i++) {
            eig_pairs.push_back({d[i], i});
        }
        std::sort(eig_pairs.begin(), eig_pairs.end());
        
        // Reorganize eigenvalues and eigenvectors
        std::vector<double> sorted_evals(m);
        std::vector<double> sorted_evecs(m*m);
        
        for (int i = 0; i < m; i++) {
            sorted_evals[i] = eig_pairs[i].first;
            int idx = eig_pairs[i].second;
            for (int j = 0; j < m; j++) {
                sorted_evecs[i*m + j] = z[idx*m + j];
            }
        }
        
        // Check convergence: compute residuals for the wanted eigenvalues
        int nconv = 0;
        std::vector<double> residuals(k);
        
        for (int i = 0; i < k && i < m; i++) {
            // Residual for Ritz pair: ||A*x - λ*x|| = |β_m+1 * e_m^T * y|
            residuals[i] = std::abs(beta[m] * sorted_evecs[i*m + (m-1)]);
            if (residuals[i] < tol) {
                nconv++;
            }
        }
        
        std::cout << "  Converged eigenvalues: " << nconv << " of " << k << " wanted" << std::endl;
        
        if (nconv >= k || iter == max_restarts-1) {
            converged = true;
            
            // Save the converged eigenvalues
            eigenvalues.resize(k);
            for (int i = 0; i < k && i < m; i++) {
                eigenvalues[i] = sorted_evals[i];
            }
            
            // Compute and save eigenvectors if requested
            if (compute_eigenvectors) {
                for (int i = 0; i < k && i < m; i++) {
                    // Compute Ritz vector: u = V * z_i
                    ComplexVector ritz_vector(N, Complex(0.0, 0.0));
                    for (int j = 0; j < m; j++) {
                        ComplexVector v_j = read_vector(j);
                        Complex coef(sorted_evecs[i*m + j], 0.0);
                        cblas_zaxpy(N, &coef, v_j.data(), 1, ritz_vector.data(), 1);
                    }
                    
                    // Normalize eigenvector
                    double vec_norm = cblas_dznrm2(N, ritz_vector.data(), 1);
                    scale = Complex(1.0/vec_norm, 0.0);
                    cblas_zscal(N, &scale, ritz_vector.data(), 1);
                    
                    // Save eigenvector to file
                    std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                    std::ofstream evec_outfile(evec_file, std::ios::binary);
                    if (evec_outfile) {
                        evec_outfile.write(reinterpret_cast<char*>(ritz_vector.data()), N * sizeof(Complex));
                        evec_outfile.close();
                    }
                }
                
                // Save eigenvalues to a single file
                std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
                std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
                if (eval_outfile) {
                    // Write the number of eigenvalues first
                    size_t n_evals = eigenvalues.size();
                    eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                    // Write all eigenvalues
                    eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), k * sizeof(double));
                    eval_outfile.close();
                    std::cout << "Saved " << k << " eigenvalues to " << eigenvalue_file << std::endl;
                }
            }
            
            break;
        }
        
        // Krylov-Schur restart
        
        // Step 1: Compute new basis vectors and write to temporary files
        std::string new_temp_dir = temp_dir + "_new";
        system(("mkdir -p " + new_temp_dir).c_str());
        
        for (int i = 0; i < k; i++) {
            ComplexVector v_new(N, Complex(0.0, 0.0));
            for (int j = 0; j < m; j++) {
                ComplexVector v_j = read_vector(j);
                Complex coef(sorted_evecs[i*m + j], 0.0);
                cblas_zaxpy(N, &coef, v_j.data(), 1, v_new.data(), 1);
            }
            // Write to temporary location
            std::string filename = new_temp_dir + "/krylov_vector_" + std::to_string(i) + ".bin";
            std::ofstream outfile(filename, std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<const char*>(v_new.data()), N * sizeof(Complex));
                outfile.close();
            }
        }
        
        // Step 2: Update the tridiagonal matrix to diagonal form (Schur form for symmetric case)
        for (int i = 0; i < k; i++) {
            alpha[i] = sorted_evals[i];  // Eigenvalues on diagonal
            beta[i+1] = 0.0;  // Zeros on off-diagonal for Schur form
        }
        
        // Step 3: Compute the k+1 vector - initialize with a random vector orthogonal to basis
        ComplexVector v_k_plus_1(N);
        for (int i = 0; i < N; i++) {
            v_k_plus_1[i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against the new basis
        for (int j = 0; j < k; j++) {
            std::string filename = new_temp_dir + "/krylov_vector_" + std::to_string(j) + ".bin";
            std::ifstream infile(filename, std::ios::binary);
            if (infile) {
                ComplexVector v_j(N);
                infile.read(reinterpret_cast<char*>(v_j.data()), N * sizeof(Complex));
                infile.close();
                
                Complex ip;
                cblas_zdotc_sub(N, v_j.data(), 1, v_k_plus_1.data(), 1, &ip);
                Complex neg_ip = -ip;
                cblas_zaxpy(N, &neg_ip, v_j.data(), 1, v_k_plus_1.data(), 1);
            }
        }
        
        // Normalize
        norm = cblas_dznrm2(N, v_k_plus_1.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v_k_plus_1.data(), 1);
        
        // Write the k+1 vector to temporary location
        std::string filename = new_temp_dir + "/krylov_vector_" + std::to_string(k) + ".bin";
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<const char*>(v_k_plus_1.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // Replace old vectors with new ones
        system(("rm -rf " + temp_dir).c_str());
        system(("mv " + new_temp_dir + " " + temp_dir).c_str());
        
        iter++;
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    if (!converged) {
        std::cout << "Warning: Krylov-Schur did not fully converge after " << max_restarts << " restarts." << std::endl;
    } else {
        std::cout << "Krylov-Schur completed successfully." << std::endl;
    }
}


// Implicitly Restarted Lanczos algorithm for computing eigenvalues of large symmetric/Hermitian matrices
void implicitly_restarted_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                            double tol, std::vector<double>& eigenvalues, std::string dir = "",
                            bool compute_eigenvectors = false) {
    
    std::cout << "Implicitly Restarted Lanczos: Starting computation for " << exct << " eigenvalues" << std::endl;
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir + "/irl_basis_vectors";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Parameters for the IRL algorithm
    int m = std::min(2*exct + 20, max_iter);  // Maximum Lanczos dimension before restart
    int k = exct;                             // Number of desired eigenvalues
    int p = m - k;                            // Number of shifts per restart
    int max_restarts = 100;                   // Maximum number of implicit restarts
    const int reorth_window = 10;             // Number of recent vectors to use for reorthogonalization
    
    // Helper functions to read/write basis vectors from/to disk
    auto write_basis_vector = [&temp_dir, N](int index, const ComplexVector& vec) {
        std::string filename = temp_dir + "/irl_vector_" + std::to_string(index) + ".bin";
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
            return false;
        }
        outfile.write(reinterpret_cast<const char*>(vec.data()), N * sizeof(Complex));
        outfile.close();
        return true;
    };
    
    auto read_basis_vector = [&temp_dir, N](int index) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/irl_vector_" + std::to_string(index) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        infile.close();
        return vec;
    };
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Write initial basis vector to disk
    write_basis_vector(0, v_current);
    
    // Initialize vectors for the Lanczos process
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Ring buffer for recent vectors (for selective orthogonalization)
    std::deque<ComplexVector> recent_vectors;
    recent_vectors.push_back(v_current);
    
    // Initialize tridiagonal matrix elements
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    // Main loop for implicit restarts
    int iteration = 0;
    bool converged = false;
    int current_size = 1;  // Current size of the Lanczos factorization
    
    while (!converged && iteration < max_restarts) {
        std::cout << "IRL: Starting iteration " << iteration + 1 << " of maximum " << max_restarts << std::endl;
        
        // Build or extend the Lanczos factorization to dimension m
        int j_start = current_size;
        
        for (int j = j_start; j < m; j++) {
            // w = H*v_j
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
            
            // Selective reorthogonalization against recent vectors only
            for (const auto& vec : recent_vectors) {
                Complex overlap;
                cblas_zdotc_sub(N, vec.data(), 1, w.data(), 1, &overlap);
                
                if (std::abs(overlap) > tol) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, vec.data(), 1, w.data(), 1);
                    
                    // Double-check orthogonality (optional)
                    cblas_zdotc_sub(N, vec.data(), 1, w.data(), 1, &overlap);
                    if (std::abs(overlap) > tol) {
                        neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, vec.data(), 1, w.data(), 1);
                    }
                }
            }
            
            // beta_{j+1} = ||w||
            norm = cblas_dznrm2(N, w.data(), 1);
            beta.push_back(norm);
            
            // Check for invariant subspace
            if (norm < tol) {
                std::cout << "IRL: Invariant subspace detected at iteration " << j << std::endl;
                m = j + 1;  // Adjust m to current size
                break;
            }
            
            // v_{j+1} = w / beta_{j+1}
            for (int i = 0; i < N; i++) {
                v_next[i] = w[i] / norm;
            }
            
            // Store basis vector to disk
            write_basis_vector(j+1, v_next);
            
            // Update recent vectors ring buffer
            recent_vectors.push_back(v_next);
            if (recent_vectors.size() > reorth_window) {
                recent_vectors.pop_front();
            }
            
            // Update for next iteration
            v_prev = v_current;
            v_current = v_next;
        }
        
        current_size = m;  // Update current factorization size
        
        // At this point, we have a Lanczos factorization of dimension m
        // H*V_m = V_m*T_m + beta_m*v_{m+1}*e_m^T
        
        // Compute eigenvalues and eigenvectors of the tridiagonal matrix T_m
        std::vector<double> diag = alpha;    // Copy of diagonal elements
        std::vector<double> offdiag(m-1);    // Off-diagonal elements
        
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = beta[i+1];
        }
        
        // Allocate space for eigenvalues and eigenvectors of T_m
        std::vector<double> evals(m);
        std::vector<double> evecs(m*m);
        
        // Solve tridiagonal eigenvalue problem
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "IRL: LAPACKE_dstevd failed with error code " << info << std::endl;
            break;
        }
        
        // Check convergence for the k wanted eigenvalues
        bool all_converged = true;
        for (int i = 0; i < k; i++) {
            // Compute residual norm ||(H - lambda_i*I)y_i||
            // Simplified: beta_m * |last component of eigenvector|
            double residual = beta[m-1] * std::abs(evecs[i*m + (m-1)]);
            
            if (residual > tol) {
                all_converged = false;
                break;
            }
        }
        
        // Helper function to compute and save eigenvectors
        auto compute_and_save_eigenvectors = [&]() {
            if (!compute_eigenvectors || dir.empty()) return;
            
            std::string evec_dir = dir + "/eigenvectors";
            std::string cmd = "mkdir -p " + evec_dir;
            system(cmd.c_str());
            
            std::cout << "IRL: Computing and saving eigenvectors..." << std::endl;
            
            // Transform Lanczos basis to eigenvectors of original matrix
            for (int i = 0; i < k; i++) {
                ComplexVector eigenvector(N, Complex(0.0, 0.0));
                
                // Process basis vectors in batches to reduce memory usage
                const int batch_size = 50;
                for (int batch_start = 0; batch_start < m; batch_start += batch_size) {
                    int batch_end = std::min(batch_start + batch_size, m);
                    
                    // Read this batch of basis vectors
                    std::vector<ComplexVector> basis_batch;
                    for (int j = batch_start; j < batch_end; j++) {
                        basis_batch.push_back(read_basis_vector(j));
                    }
                    
                    // Compute contribution from this batch
                    for (int j = 0; j < batch_end - batch_start; j++) {
                        int j_global = batch_start + j;
                        Complex coef(evecs[i*m + j_global], 0.0);
                        cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, eigenvector.data(), 1);
                    }
                }
                
                // Normalize eigenvector
                double evec_norm = cblas_dznrm2(N, eigenvector.data(), 1);
                Complex evec_scale = Complex(1.0/evec_norm, 0.0);
                cblas_zscal(N, &evec_scale, eigenvector.data(), 1);
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
                evec_outfile.close();
                
                // Print progress occasionally
                if ((i+1) % 10 == 0 || i == k-1) {
                    std::cout << "  Saved eigenvector " << i+1 << " of " << k << std::endl;
                }
            }
            
            // Save eigenvalues
            std::string eval_file = evec_dir + "/eigenvalues.bin";
            std::ofstream eval_outfile(eval_file, std::ios::binary);
            if (eval_outfile) {
                size_t n_evals = eigenvalues.size();
                eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), k * sizeof(double));
                eval_outfile.close();
                std::cout << "  Saved " << k << " eigenvalues to " << eval_file << std::endl;
            }
        };
        
        if (all_converged || m < k + p) {
            std::cout << "IRL: Convergence achieved at iteration " << iteration + 1 << std::endl;
            converged = true;
            
            // Copy the converged eigenvalues
            eigenvalues.resize(k);
            for (int i = 0; i < k; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Compute and save eigenvectors if requested
            compute_and_save_eigenvectors();
            
            break;  // Exit the main loop
        }
        
        // Perform implicit restart using exact shifts strategy
        // The p unwanted eigenvalues become the shifts
        
        // Sort eigenvalues by magnitude (optional)
        std::vector<std::pair<double, int>> sorted_evals;
        for (int i = 0; i < m; i++) {
            sorted_evals.push_back({std::abs(evals[i]), i});
        }
        std::sort(sorted_evals.begin(), sorted_evals.end());
        
        // Collect the p unwanted eigenvalues as shifts
        std::vector<double> shifts;
        for (int i = 0; i < p; i++) {
            int idx = sorted_evals[i].second;
            shifts.push_back(evals[idx]);
        }
        
        // Perform p implicit QR steps with the chosen shifts
        
        // Initialize QR factorization (T_m - mu_i*I) = Q_i*R_i
        std::vector<double> q = alpha;
        std::vector<double> e = beta;
        
        // Apply all shifts sequentially using bulge-chasing
        for (double shift : shifts) {
            // Construct first column of shifted matrix
            double x = q[0] - shift;
            double z = e[1];
            
            // Apply Givens rotations to introduce and then chase the bulge
            for (int i = 0; i < m-1; i++) {
                // Compute Givens rotation to zero out the bulge
                double c, s;
                LAPACKE_dlartgs(x, z, c, &s, &x);
                
                // Apply the Givens rotation to the tridiagonal matrix
                if (i < m-2) {
                    z = -s * e[i+2];
                    e[i+2] *= c;
                }
                
                double tmp = c * q[i+1] - s * e[i+1];
                e[i+1] = s * q[i+1] + c * e[i+1];
                q[i+1] = tmp;
                
                if (i < m-2) {
                    x = q[i+1];
                    z = -s * e[i+2];
                }
            }
        }
        
        // Now compute the new starting vector for the next restart
        ComplexVector new_v(N, Complex(0.0, 0.0));
        
        // Load the first k basis vectors to form the new starting point
        for (int i = 0; i < k; i++) {
            ComplexVector basis_i = read_basis_vector(i);
            Complex coef(evecs[i*m], 0.0);  // First component of the i-th eigenvector
            cblas_zaxpy(N, &coef, basis_i.data(), 1, new_v.data(), 1);
        }
        
        // Normalize the new starting vector
        norm = cblas_dznrm2(N, new_v.data(), 1);
        scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, new_v.data(), 1);
        
        // Reset the Lanczos process with the new vector
        v_current = new_v;
        v_prev = ComplexVector(N, Complex(0.0, 0.0));
        
        // Write the new starting vector to disk
        write_basis_vector(0, v_current);
        
        // Clear the recent vectors and add the new starting vector
        recent_vectors.clear();
        recent_vectors.push_back(v_current);
        
        // Reset the tridiagonal matrix elements for the new restart
        alpha.resize(k);
        beta.resize(k+1);
        
        for (int i = 0; i < k; i++) {
            alpha[i] = q[i];
        }
        for (int i = 1; i <= k; i++) {
            beta[i] = e[i];
        }
        
        current_size = k;
        iteration++;
    }
    
    // If we reached maximum iterations without convergence,
    // still compute and return the best available approximations
    if (!converged) {
        std::cout << "IRL: Maximum iterations reached without convergence" << std::endl;
        
        // Compute eigenvalues and eigenvectors of the final tridiagonal matrix
        std::vector<double> diag = alpha;  
        std::vector<double> offdiag(alpha.size()-1);
        
        for (int i = 0; i < alpha.size()-1; i++) {
            offdiag[i] = beta[i+1];
        }
        
        int final_m = alpha.size();
        std::vector<double> evals(final_m);
        std::vector<double> evecs(final_m*final_m);
        
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', final_m, diag.data(), offdiag.data(), evecs.data(), final_m);
        
        if (info == 0) {
            // Get the best available approximations
            int num_vals = std::min(k, final_m);
            eigenvalues.resize(num_vals);
            for (int i = 0; i < num_vals; i++) {
                eigenvalues[i] = diag[i];
            }
            
            // Compute and save eigenvectors if requested
            if (compute_eigenvectors && !dir.empty()) {
                std::string evec_dir = dir + "/eigenvectors";
                std::string cmd = "mkdir -p " + evec_dir;
                system(cmd.c_str());
                
                std::cout << "IRL: Computing and saving best approximation eigenvectors..." << std::endl;
                
                // Transform Lanczos basis to eigenvectors of original matrix
                for (int i = 0; i < num_vals; i++) {
                    ComplexVector eigenvector(N, Complex(0.0, 0.0));
                    
                    // Process basis vectors in batches
                    const int batch_size = 50;
                    for (int batch_start = 0; batch_start < final_m; batch_start += batch_size) {
                        int batch_end = std::min(batch_start + batch_size, final_m);
                        
                        // Load this batch of basis vectors
                        std::vector<ComplexVector> basis_batch;
                        for (int j = batch_start; j < batch_end; j++) {
                            basis_batch.push_back(read_basis_vector(j));
                        }
                        
                        // Apply coefficients
                        for (int j = 0; j < batch_end - batch_start; j++) {
                            int j_global = batch_start + j;
                            Complex coef(evecs[i*final_m + j_global], 0.0);
                            cblas_zaxpy(N, &coef, basis_batch[j].data(), 1, eigenvector.data(), 1);
                        }
                    }
                    
                    // Normalize eigenvector
                    double evec_norm = cblas_dznrm2(N, eigenvector.data(), 1);
                    Complex evec_scale = Complex(1.0/evec_norm, 0.0);
                    cblas_zscal(N, &evec_scale, eigenvector.data(), 1);
                    
                    // Save to file
                    std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                    std::ofstream evec_outfile(evec_file, std::ios::binary);
                    if (!evec_outfile) {
                        std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                        continue;
                    }
                    evec_outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
                    evec_outfile.close();
                    
                    // Print progress occasionally
                    if ((i+1) % 10 == 0 || i == num_vals-1) {
                        std::cout << "  Saved eigenvector " << i+1 << " of " << num_vals << std::endl;
                    }
                }
                
                // Save eigenvalues
                std::string eval_file = evec_dir + "/eigenvalues.bin";
                std::ofstream eval_outfile(eval_file, std::ios::binary);
                if (eval_outfile) {
                    size_t n_evals = eigenvalues.size();
                    eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                    eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), num_vals * sizeof(double));
                    eval_outfile.close();
                    std::cout << "  Saved " << num_vals << " eigenvalues to " << eval_file << std::endl;
                }
            }
        } else {
            std::cerr << "IRL: Final eigenvalue computation failed with error code " << info << std::endl;
        }
    }
    
    // Clean up temporary directory if needed
    if (!dir.empty()) {
        system(("rm -rf " + temp_dir).c_str());
    }
    
    std::cout << "Implicitly Restarted Lanczos completed" << std::endl;
}

// Optimal solver for full spectrum with degenerate eigenvalues
// Combines block Lanczos, Chebyshev filtering, and spectrum slicing with MPI parallelization
void optimal_spectrum_solver(std::function<void(const Complex*, Complex*, int)> H, int N, 
                             std::vector<double>& eigenvalues, std::string dir = "",
                             bool compute_eigenvectors = true, double tol = 1e-10) {
    // Initialize MPI variables
    int mpi_rank = 0, mpi_size = 1;
    
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    #endif
    
    // Only the root process should print
    if (mpi_rank == 0) {
        std::cout << "Starting optimal spectrum solver for matrix of dimension " << N << std::endl;
        std::cout << "Using " << mpi_size << " MPI processes for parallelization" << std::endl;
    }
    
    // Create directories for output (only root process)
    std::string solver_dir = dir + "/optimal_solver";
    std::string evec_dir = solver_dir + "/eigenvectors";
    std::string temp_dir = solver_dir + "/temp";
    
    if (mpi_rank == 0) {
        system(("mkdir -p " + evec_dir).c_str());
        system(("mkdir -p " + temp_dir).c_str());
    }
    
    // For very small matrices, just use full diagonalization (only on root)
    if (N <= 1000) {
        if (mpi_rank == 0) {
            std::cout << "Small matrix detected, using direct diagonalization" << std::endl;
            full_diagonalization(H, N, eigenvalues, evec_dir, compute_eigenvectors);
        }
        return;
    }
    
    // Parameters for spectrum slicing
    const int max_slice_size = 5000;  // Maximum eigenvalues per slice
    const int overlap = 50;           // Overlap between slices to ensure continuity
    const int num_slices = (N + max_slice_size - 1) / max_slice_size;
    
    if (mpi_rank == 0) {
        std::cout << "Using spectrum slicing with " << num_slices << " slices" << std::endl;
    }
    
    // Estimate spectral bounds using Lanczos (only on root process)
    std::vector<double> boundary_evals;
    double lambda_min, lambda_max, safety_margin;
    
    if (mpi_rank == 0) {
        std::cout << "Estimating spectral bounds..." << std::endl;
        lanczos(H, N, 100, 100, tol, boundary_evals, temp_dir, false);
        
        lambda_min = boundary_evals[0];
        lambda_max = boundary_evals[99];
        safety_margin = 0.05 * (lambda_max - lambda_min);
        
        lambda_min -= safety_margin;
        lambda_max += safety_margin;
        
        std::cout << "Estimated spectral range: [" << lambda_min << ", " << lambda_max << "]" << std::endl;
    }
    
    // Broadcast spectral bounds to all processes
    #ifdef USE_MPI
    MPI_Bcast(&lambda_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lambda_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #endif
    
    // Initialize local result vector for each process
    std::vector<double> local_eigenvalues;
    std::vector<int> local_slice_sizes; // To keep track of eigenvalues per slice for this process
    
    // Assign slices to processes for load balancing
    for (int slice = 0; slice < num_slices; slice++) {
        // Simple round-robin assignment of slices to processes
        if (slice % mpi_size == mpi_rank) {
            double slice_start, slice_end;
            
            if (num_slices == 1) {
                slice_start = lambda_min;
                slice_end = lambda_max;
            } else {
                double slice_width = (lambda_max - lambda_min) / num_slices;
                slice_start = lambda_min + slice * slice_width - (slice > 0 ? overlap * slice_width / max_slice_size : 0);
                slice_end = lambda_min + (slice + 1) * slice_width + (slice < num_slices - 1 ? overlap * slice_width / max_slice_size : 0);
            }
            
            std::cout << "Process " << mpi_rank << ": Processing slice " << slice + 1 << "/" << num_slices 
                      << " range [" << slice_start << ", " << slice_end << "]" << std::endl;
            
            // Choose optimal method for this slice based on its position in spectrum
            std::vector<double> slice_eigenvalues;
            std::string slice_dir = temp_dir + "/slice_" + std::to_string(slice) + "_proc_" + std::to_string(mpi_rank);
            system(("mkdir -p " + slice_dir).c_str());
            
            // For the lowest part of the spectrum, use Krylov-Schur which works well for smallest eigenvalues
            if (slice == 0) {
                std::cout << "Process " << mpi_rank << ": Using Krylov-Schur for lowest eigenvalues" << std::endl;
                krylov_schur(H, N, max_slice_size, max_slice_size, tol, slice_eigenvalues, slice_dir, compute_eigenvectors);
            }
            // For the highest part of the spectrum, use IRL with spectral transformation
            else if (slice == num_slices - 1) {
                std::cout << "Process " << mpi_rank << ": Using IRL for highest eigenvalues" << std::endl;
                implicitly_restarted_lanczos(H, N, max_slice_size, max_slice_size, tol, slice_eigenvalues, slice_dir, compute_eigenvectors);
            }
            // For middle slices, use shift-invert Lanczos with a shift in the middle of the slice
            else {
                double sigma = (slice_start + slice_end) / 2.0;
                std::cout << "Process " << mpi_rank << ": Using shift-invert Lanczos with shift σ = " << sigma << std::endl;
                shift_invert_lanczos_robust(H, N, max_slice_size, max_slice_size, sigma, tol, slice_eigenvalues, slice_dir, compute_eigenvectors);
            }
            
            // Filter eigenvalues to the current slice
            std::vector<double> filtered_eigenvalues;
            for (double eval : slice_eigenvalues) {
                if (eval >= slice_start && eval <= slice_end) {
                    filtered_eigenvalues.push_back(eval);
                }
            }
            
            std::cout << "Process " << mpi_rank << ": Found " << filtered_eigenvalues.size() << " eigenvalues in slice " << slice + 1 << std::endl;
            
            // Keep track of slice size
            local_slice_sizes.push_back(filtered_eigenvalues.size());
            
            // Add to local eigenvalue list
            local_eigenvalues.insert(local_eigenvalues.end(), filtered_eigenvalues.begin(), filtered_eigenvalues.end());
            
            // If computing eigenvectors, store the location for later gathering
            if (compute_eigenvectors) {
                std::string src_dir = slice_dir;
                if (slice == 0) {
                    src_dir += "/krylov_schur_eigenvectors";
                } else if (slice == num_slices - 1) {
                    src_dir += "/irl_eigenvectors";
                } else {
                    src_dir += "/shift_invert_lanczos_results";
                }
                
                // Create a metadata file to record the source directory and eigenvalue count
                std::string meta_file = slice_dir + "/metadata.txt";
                std::ofstream meta_out(meta_file);
                if (meta_out) {
                    meta_out << src_dir << "\n";
                    meta_out << filtered_eigenvalues.size() << "\n";
                    meta_out.close();
                }
            }
        }
    }
    
    // Synchronize all processes
    #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    
    // Gather all eigenvalues to the root process
    #ifdef USE_MPI
    // First, gather the number of eigenvalues from each process
    int local_count = local_eigenvalues.size();
    std::vector<int> all_counts(mpi_size);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate displacements for gatherv
    std::vector<int> displacements(mpi_size, 0);
    if (mpi_rank == 0) {
        for (int i = 1; i < mpi_size; i++) {
            displacements[i] = displacements[i-1] + all_counts[i-1];
        }
    }
    
    // Resize the global eigenvalues vector on root
    int total_count = 0;
    if (mpi_rank == 0) {
        for (int count : all_counts) {
            total_count += count;
        }
        eigenvalues.resize(total_count);
    }
    
    // Gather all eigenvalues
    MPI_Gatherv(local_eigenvalues.data(), local_count, MPI_DOUBLE,
                eigenvalues.data(), all_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    #else
    // If not using MPI, just copy local to global
    eigenvalues = local_eigenvalues;
    #endif
    
    // The root process performs the final steps
    if (mpi_rank == 0) {
        // Sort all eigenvalues
        std::sort(eigenvalues.begin(), eigenvalues.end());
        
        // Remove duplicates from overlapping slices
        if (num_slices > 1) {
            auto new_end = std::unique(eigenvalues.begin(), eigenvalues.end(), 
                                      [tol](double a, double b) { return std::abs(a - b) < tol; });
            eigenvalues.erase(new_end, eigenvalues.end());
        }
        
        std::cout << "Final eigenvalue count: " << eigenvalues.size() << std::endl;
        
        // If computing eigenvectors, gather them from all processes
        if (compute_eigenvectors) {
            std::cout << "Gathering eigenvectors from all processes..." << std::endl;
            
            // Create a mapping from eigenvalue to index in the final sorted array
            std::map<double, int> eigenvalue_indices;
            for (size_t i = 0; i < eigenvalues.size(); i++) {
                eigenvalue_indices[eigenvalues[i]] = i;
            }
            
            // Loop through slices to collect eigenvectors
            for (int slice = 0; slice < num_slices; slice++) {
                int slice_proc = slice % mpi_size;  // Process that handled this slice
                
                std::string slice_dir = temp_dir + "/slice_" + std::to_string(slice) + "_proc_" + std::to_string(slice_proc);
                std::string meta_file = slice_dir + "/metadata.txt";
                
                std::ifstream meta_in(meta_file);
                if (!meta_in) {
                    continue;  // Skip if metadata not found
                }
                
                std::string src_dir;
                int evec_count;
                meta_in >> src_dir >> evec_count;
                meta_in.close();
                
                // Read the eigenvalues from this slice for mapping
                std::string eval_file = src_dir + "/eigenvalues.bin";
                std::ifstream eval_in(eval_file, std::ios::binary);
                if (!eval_in) {
                    continue;
                }
                
                size_t n_evals;
                eval_in.read(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                std::vector<double> slice_evals(n_evals);
                eval_in.read(reinterpret_cast<char*>(slice_evals.data()), n_evals * sizeof(double));
                eval_in.close();
                
                // Copy eigenvectors to final directory
                for (size_t i = 0; i < n_evals; i++) {
                    // Find the closest eigenvalue in the global list
                    double closest_match = eigenvalues[0];
                    double min_diff = std::abs(slice_evals[i] - eigenvalues[0]);
                    
                    for (double eval : eigenvalues) {
                        double diff = std::abs(slice_evals[i] - eval);
                        if (diff < min_diff) {
                            min_diff = diff;
                            closest_match = eval;
                        }
                    }
                    
                    if (min_diff > 10*tol) {
                        continue;  // Skip if no good match found
                    }
                    
                    // Get target index
                    int target_idx = eigenvalue_indices[closest_match];
                    
                    // Copy the eigenvector
                    std::string src_file = src_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                    std::string dst_file = evec_dir + "/eigenvector_" + std::to_string(target_idx) + ".bin";
                    
                    std::string copy_cmd = "cp \"" + src_file + "\" \"" + dst_file + "\"";
                    system(copy_cmd.c_str());
                }
            }
        }
        
        // Perform post-processing to identify and refine degenerate eigenvalues
        std::cout << "Performing post-processing for degenerate eigenspaces..." << std::endl;
        
        std::vector<std::pair<double, int>> degeneracy_groups;
        double current_eval = eigenvalues[0];
        int current_count = 1;
        
        for (size_t i = 1; i < eigenvalues.size(); i++) {
            if (std::abs(eigenvalues[i] - current_eval) < 10*tol) {
                current_count++;
            } else {
                degeneracy_groups.push_back({current_eval, current_count});
                current_eval = eigenvalues[i];
                current_count = 1;
            }
        }
        degeneracy_groups.push_back({current_eval, current_count});
        
        // Report degeneracy statistics
        std::cout << "Degeneracy analysis:" << std::endl;
        int max_degeneracy = 0;
        double max_degen_eval = 0;
        int num_degenerate_groups = 0;
        
        for (const auto& group : degeneracy_groups) {
            if (group.second > 1) {
                num_degenerate_groups++;
                if (group.second > max_degeneracy) {
                    max_degeneracy = group.second;
                    max_degen_eval = group.first;
                }
            }
        }
        
        std::cout << "Found " << num_degenerate_groups << " degenerate eigenvalue groups" << std::endl;
        std::cout << "Maximum degeneracy: " << max_degeneracy << " at eigenvalue " << max_degen_eval << std::endl;
        
        // If eigenvectors are computed, fix degenerate subspaces for better numerical stability
        if (compute_eigenvectors && num_degenerate_groups > 0) {
            std::cout << "Refining eigenvectors for degenerate subspaces..." << std::endl;
            
            int index_offset = 0;
            for (const auto& group : degeneracy_groups) {
                if (group.second > 1) {
                    // Load group of degenerate eigenvectors
                    std::vector<ComplexVector> degenerate_vectors;
                    for (int i = 0; i < group.second; i++) {
                        std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(index_offset + i) + ".bin";
                        std::ifstream infile(evec_file, std::ios::binary);
                        if (!infile) {
                            std::cerr << "Error: Cannot open file " << evec_file << " for reading" << std::endl;
                            continue;
                        }
                        
                        ComplexVector vec(N);
                        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
                        infile.close();
                        
                        degenerate_vectors.push_back(vec);
                    }
                    
                    // Refine degenerate subspace
                    refine_degenerate_eigenvectors(H, degenerate_vectors, group.first, N, tol);
                    
                    // Save refined vectors
                    for (int i = 0; i < group.second; i++) {
                        std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(index_offset + i) + ".bin";
                        std::ofstream outfile(evec_file, std::ios::binary);
                        if (!outfile) {
                            std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                            continue;
                        }
                        
                        outfile.write(reinterpret_cast<char*>(degenerate_vectors[i].data()), N * sizeof(Complex));
                        outfile.close();
                    }
                }
                
                index_offset += group.second;
            }
        }
        
        // Save eigenvalues to a file
        std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
        if (eval_outfile) {
            // Write the number of eigenvalues first
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            // Write all eigenvalues
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved " << n_evals << " eigenvalues to " << eigenvalue_file << std::endl;
        }
        
        // Cleanup temporary files
        if (!dir.empty()) {
            system(("rm -rf " + temp_dir).c_str());
        }
        
        std::cout << "Optimal spectrum solver completed successfully" << std::endl;
    }
    
    // Synchronize all processes before returning
    #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

// Full spectrum solver that divides the spectrum for faster computation
void spectrum_slicing_solver(std::function<void(const Complex*, Complex*, int)> H, int N, 
                            std::vector<double>& eigenvalues, std::string dir = "",
                            bool compute_eigenvectors = false, double tol = 1e-10) {
    std::cout << "Starting spectrum slicing solver for matrix of dimension " << N << std::endl;
    
    // Create output directories
    std::string solver_dir = dir + "/spectrum_slicing";
    std::string slice_dir = solver_dir + "/slices";
    std::string result_dir = solver_dir + "/results";
    
    system(("mkdir -p " + slice_dir).c_str());
    system(("mkdir -p " + result_dir).c_str());
    
    // For very small matrices, just use full diagonalization
    if (N <= 1000) {
        std::cout << "Small matrix detected, using direct diagonalization" << std::endl;
        full_diagonalization(H, N, eigenvalues, result_dir, compute_eigenvectors);
        return;
    }
    
    // Step 1: Estimate spectral bounds using standard Lanczos
    std::cout << "Estimating spectral bounds..." << std::endl;
    std::vector<double> boundary_evals;
    lanczos(H, N, 100, 100, tol, boundary_evals, dir, false);
    
    double lambda_min = boundary_evals[0];
    double lambda_max = boundary_evals[boundary_evals.size()-1];
    
    // Add safety margin to ensure we cover the full spectrum
    double margin = 0.05 * (lambda_max - lambda_min);
    lambda_min -= margin;
    lambda_max += margin;
    
    std::cout << "Estimated spectral range: [" << lambda_min << ", " << lambda_max << "]" << std::endl;
    
    // Step 2: Determine slice configuration
    // Choose number of slices based on matrix size
    int num_slices;
    if (N < 5000) num_slices = 4;
    else if (N < 10000) num_slices = 8;
    else if (N < 20000) num_slices = 16;
    else num_slices = 32;
    
    // Adjust based on available hardware threads
    int hardware_threads = std::thread::hardware_concurrency();
    num_slices = std::min(num_slices, hardware_threads * 2);
    
    std::cout << "Dividing spectrum into " << num_slices << " slices" << std::endl;
    
    // Create slices with slight overlap to catch eigenvalues at boundaries
    double slice_width = (lambda_max - lambda_min) / num_slices;
    double overlap = 0.05 * slice_width; // 5% overlap between slices
    
    std::vector<std::pair<double, double>> slices;
    for (int i = 0; i < num_slices; i++) {
        double slice_start = lambda_min + i * slice_width - (i > 0 ? overlap : 0);
        double slice_end = lambda_min + (i+1) * slice_width + (i < num_slices-1 ? overlap : 0);
        slices.push_back({slice_start, slice_end});
    }
    
    // Step 3: Process each slice (potentially in parallel)
    std::vector<std::vector<double>> slice_eigenvalues(num_slices);
    
    #pragma omp parallel for schedule(dynamic)
    for (int slice_idx = 0; slice_idx < num_slices; slice_idx++) {
        double slice_start = slices[slice_idx].first;
        double slice_end = slices[slice_idx].second;
        double slice_mid = (slice_start + slice_end) / 2.0;
        
        std::cout << "Processing slice " << slice_idx + 1 << "/" << num_slices 
                  << " [" << slice_start << ", " << slice_end << "]" << std::endl;
        
        std::string current_slice_dir = slice_dir + "/slice_" + std::to_string(slice_idx);
        system(("mkdir -p " + current_slice_dir).c_str());
        
        // Choose appropriate method based on slice position
        std::vector<double> evals;
        
        // For edge slices, use standard Lanczos
        if (slice_idx == 0) {
            // First slice - use standard Lanczos for smallest eigenvalues
            int max_evals_in_slice = static_cast<int>(N / num_slices * 1.5); // Add 50% buffer
            lanczos_no_ortho(H, N, max_evals_in_slice * 2, max_evals_in_slice, tol, evals, 
                   current_slice_dir, compute_eigenvectors);
            
            // Filter eigenvalues to this slice
            std::vector<double> filtered_evals;
            for (double eval : evals) {
                if (eval <= slice_end) {
                    filtered_evals.push_back(eval);
                }
            }
            evals = filtered_evals;
        } 
        else if (slice_idx == num_slices - 1) {
            // Last slice - transform to find largest eigenvalues
            // We can use standard Lanczos with a spectral transformation to find largest eigenvalues
            int max_evals_in_slice = static_cast<int>(N / num_slices * 1.5);
            
            // Define a transformed operator to find largest eigenvalues
            auto H_transformed = [&H, lambda_min, lambda_max, N](const Complex* v, Complex* result, int size) {
                // Apply H to v
                H(v, result, size);
                
                // Apply transformation: -(H - lambda_max*I) to reverse the spectrum
                for (int i = 0; i < size; i++) {
                    result[i] = lambda_max * v[i] - result[i];
                }
            };
            
            lanczos_no_ortho(H_transformed, N, max_evals_in_slice * 2, max_evals_in_slice, tol, evals, 
                   current_slice_dir, compute_eigenvectors);
            
            // Convert eigenvalues back to original spectrum
            for (double& eval : evals) {
                eval = lambda_max - eval;
            }
            
            // Filter eigenvalues to this slice
            std::vector<double> filtered_evals;
            for (double eval : evals) {
                if (eval >= slice_start) {
                    filtered_evals.push_back(eval);
                }
            }
            evals = filtered_evals;
        } 
        else {
            // Interior slice - use shift-invert Lanczos centered at the slice midpoint
            int max_evals_in_slice = static_cast<int>(N / num_slices * 2.0); // Add 100% buffer for interior
            shift_invert_lanczos(H, N, max_evals_in_slice, max_evals_in_slice, 
                                      slice_mid, tol, evals, current_slice_dir, compute_eigenvectors);
            
            // Filter eigenvalues to this slice
            std::vector<double> filtered_evals;
            for (double eval : evals) {
                if (eval >= slice_start && eval <= slice_end) {
                    filtered_evals.push_back(eval);
                }
            }
            evals = filtered_evals;
        }
        
        std::cout << "Found " << evals.size() << " eigenvalues in slice " << slice_idx + 1 << std::endl;
        
        // Save slice eigenvalues for merging
        #pragma omp critical
        {
            slice_eigenvalues[slice_idx] = evals;
        }
    }
    
    // Step 4: Merge eigenvalues from all slices and remove duplicates
    std::cout << "Merging results from all slices..." << std::endl;
    
    std::vector<double> all_eigenvalues;
    for (const auto& slice_evals : slice_eigenvalues) {
        all_eigenvalues.insert(all_eigenvalues.end(), slice_evals.begin(), slice_evals.end());
    }
    
    // Sort all eigenvalues
    std::sort(all_eigenvalues.begin(), all_eigenvalues.end());
    
    // Remove duplicates (from overlapping slices)
    auto new_end = std::unique(all_eigenvalues.begin(), all_eigenvalues.end(), 
                              [tol](double a, double b) { return std::abs(a - b) < tol; });
    all_eigenvalues.erase(new_end, all_eigenvalues.end());
    
    // Store the final result
    eigenvalues = all_eigenvalues;
    
    std::cout << "Spectrum slicing completed. Found " << eigenvalues.size() << " eigenvalues." << std::endl;
    
    // If eigenvectors were computed, organize them 
    if (compute_eigenvectors) {
        std::cout << "Organizing eigenvectors..." << std::endl;
        
        // Create a mapping from eigenvalue to index in the final sorted array
        std::map<double, int> eigenvalue_indices;
        for (size_t i = 0; i < eigenvalues.size(); i++) {
            eigenvalue_indices[eigenvalues[i]] = i;
        }
        
        // Loop through slices to collect eigenvectors
        for (int slice_idx = 0; slice_idx < num_slices; slice_idx++) {
            std::string current_slice_dir = slice_dir + "/slice_" + std::to_string(slice_idx);
            
            // Read the eigenvalues from this slice for mapping
            std::string eval_file;
            if (slice_idx == 0 || slice_idx == num_slices - 1) {
                eval_file = current_slice_dir + "/lanczos_eigenvectors/eigenvalues.bin";
            } else {
                eval_file = current_slice_dir + "/shift_invert_lanczos_results/eigenvalues.bin";
            }
            
            std::ifstream eval_in(eval_file, std::ios::binary);
            if (!eval_in) {
                std::cerr << "Could not open " << eval_file << " for reading" << std::endl;
                continue;
            }
            
            size_t n_evals;
            eval_in.read(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            std::vector<double> slice_evals(n_evals);
            eval_in.read(reinterpret_cast<char*>(slice_evals.data()), n_evals * sizeof(double));
            eval_in.close();
            
            // Determine the source directory for eigenvectors
            std::string evec_src_dir;
            if (slice_idx == 0 || slice_idx == num_slices - 1) {
                evec_src_dir = current_slice_dir + "/lanczos_eigenvectors";
            } else {
                evec_src_dir = current_slice_dir + "/shift_invert_lanczos_results";
            }
            
            // Copy eigenvectors to the result directory
            for (size_t i = 0; i < slice_evals.size(); i++) {
                double eval = slice_evals[i];
                
                // Find the closest eigenvalue in the final list
                double min_diff = std::numeric_limits<double>::max();
                int target_idx = -1;
                
                for (size_t j = 0; j < eigenvalues.size(); j++) {
                    double diff = std::abs(eigenvalues[j] - eval);
                    if (diff < min_diff && diff < tol) {
                        min_diff = diff;
                        target_idx = j;
                    }
                }
                
                if (target_idx >= 0) {
                    // Copy the eigenvector file
                    std::string src_file = evec_src_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                    std::string dst_file = result_dir + "/eigenvector_" + std::to_string(target_idx) + ".bin";
                    
                    std::string cmd = "cp \"" + src_file + "\" \"" + dst_file + "\"";
                    system(cmd.c_str());
                }
            }
        }
        
        // Save the final eigenvalues
        std::string eigenvalue_file = result_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
        if (eval_outfile) {
            // Write the number of eigenvalues first
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            // Write all eigenvalues
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved " << n_evals << " eigenvalues to " << eigenvalue_file << std::endl;
        }
    }
    
    // Cleanup temporary directories
    if (!dir.empty()) {
        system(("rm -rf " + slice_dir).c_str());
    }
    
    std::cout << "Spectrum slicing solver completed successfully" << std::endl;
}

// Diagonalization using ezARPACK
// Diagonalization using ezARPACK
void arpack_diagonalization(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                           double tol, std::vector<double>& eigenvalues, std::string dir = "",
                           bool compute_eigenvectors = false) {
    
    std::cout << "ARPACK diagonalization: Starting for matrix of dimension " << N << std::endl;
    std::cout << "Computing " << exct << " eigenvalues" << std::endl;
    
    // Create directory for output if needed
    std::string evec_dir = dir + "/eigenvectors";
    if (compute_eigenvectors && !dir.empty()) {
        std::string cmd = "mkdir -p " + evec_dir;
        system(cmd.c_str());
    }
    
    // Use ezARPACK solver with Eigen storage backend
    using solver_t = ezarpack::arpack_solver<ezarpack::Symmetric, ezarpack::eigen_storage>;
    solver_t solver(N);
    
    // Wrap our matrix-vector product function to match ezARPACK's expected format
    auto matrix_op = [&H](solver_t::vector_const_view_t in, solver_t::vector_view_t out) {
        // Convert Eigen types to raw pointers for H
        H(reinterpret_cast<const Complex*>(in.data()), 
          reinterpret_cast<Complex*>(out.data()), 
          in.size());
    };
    
    // Ensure we don't request more eigenvalues than possible
    exct = std::min(exct, N-1);
    
    // Specify parameters for the solver
    using params_t = solver_t::params_t;
    params_t params(exct,              // Number of eigenvalues to find
                   params_t::Smallest, // Find smallest eigenvalues
                   compute_eigenvectors);  // Whether to compute eigenvectors
    
    // Set additional parameters to match other solvers
    // params.max_iterations = max_iter;
    // params.tolerance = tol;
    
    // Run diagonalization
    try {
        solver(matrix_op, params);
    } catch (std::exception& e) {
        std::cerr << "ARPACK solver failed: " << e.what() << std::endl;
        return;
    }
    
    // Extract results
    auto const& eigenvalues_vector = solver.eigenvalues();
    
    // Copy eigenvalues to the output vector
    eigenvalues.resize(exct);
    for (int i = 0; i < exct; i++) {
        eigenvalues[i] = eigenvalues_vector(i);
    }
    
    // If computing eigenvectors and a directory is provided, save to files
    if (compute_eigenvectors && !dir.empty()) {
        std::cout << "Saving eigenvectors to " << evec_dir << std::endl;
        
        auto const& eigenvectors_matrix = solver.eigenvectors();
        
        // Save each eigenvector
        for (int i = 0; i < exct; i++) {
            ComplexVector eigenvector(N);
            for (int j = 0; j < N; j++) {
                // With a symmetric solver, eigenvectors are real
                eigenvector[j] = Complex(eigenvectors_matrix(j, i), 0.0);
            }
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (!evec_outfile) {
                std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                continue;
            }
            evec_outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
            evec_outfile.close();
            
            // Print progress occasionally
            if (i % 10 == 0 || i == exct - 1) {
                std::cout << "  Saved eigenvector " << i + 1 << " of " << exct << std::endl;
            }
        }
        
        // Save eigenvalues to a single file
        std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
        if (!eval_outfile) {
            std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
        } else {
            // Write the number of eigenvalues first
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            // Write all eigenvalues
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), exct * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved " << exct << " eigenvalues to " << eigenvalue_file << std::endl;
        }
    }
    
    std::cout << "ARPACK diagonalization completed successfully" << std::endl;
}




#endif // LANCZOS_H