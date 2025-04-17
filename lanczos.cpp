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
    std::string evec_dir = dir + "/lanczos_eigenvectors";
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
    std::string evec_dir = dir + "/lanczos_eigenvectors";
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

// Lanczos algorithm implementation with periodic reorthogonalization
void lanczos_periodic_reorth(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false, int reorth_period = 20) {
    
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
    std::cout << "Reorthogonalization period = " << reorth_period << std::endl;
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
        
        // Periodic reorthogonalization
        // Only reorthogonalize every reorth_period iterations or if j is close to max_iter
        if (j % reorth_period == reorth_period - 1 || j >= max_iter - 5) {
            std::cout << "  Performing reorthogonalization at step " << j + 1 << std::endl;
            
            // Full reorthogonalization against all previous basis vectors
            for (int k = 0; k <= j; k++) {
                // Read basis vector k from file
                ComplexVector basis_k = read_basis_vector(k, N);
                
                // Calculate projection and subtract
                Complex overlap;
                cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
            }
            
            // Optional: repeat for enhanced numerical stability
            for (int k = 0; k <= j; k++) {
                ComplexVector basis_k = read_basis_vector(k, N);
                
                Complex overlap;
                cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
            }
        }
        
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
    std::string evec_dir = dir + "/lanczos_eigenvectors";
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


// Block Lanczos algorithm without reorthogonalization
void block_lanczos_no_ortho(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct,
                          double tol, std::vector<double>& eigenvalues, std::string dir = "",
                          bool eigenvectors = false, int block_size = 4) {
    
    // Initialize random block of orthogonal vectors
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    
    // Generate first random vector
    block_vectors[0] = generateRandomVector(N, gen, dist);
    
    // Generate remaining orthogonal vectors in the block
    for (int i = 1; i < block_size; i++) {
        block_vectors[i] = generateOrthogonalVector(N, 
            std::vector<ComplexVector>(block_vectors.begin(), block_vectors.begin() + i), 
            gen, dist);
    }
    
    // Create directory for temporary basis vector blocks
    std::string temp_dir = dir + "/block_lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());
    
    // Write the first block of basis vectors to files
    for (int i = 0; i < block_size; i++) {
        std::string basis_file = temp_dir + "/basis_0_" + std::to_string(i) + ".bin";
        std::ofstream outfile(basis_file, std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
            return;
        }
        outfile.write(reinterpret_cast<char*>(block_vectors[i].data()), N * sizeof(Complex));
        outfile.close();
    }
    
    // Previous, current and next blocks
    std::vector<ComplexVector> prev_block(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    std::vector<ComplexVector> curr_block = block_vectors;
    std::vector<ComplexVector> next_block(block_size, ComplexVector(N));
    std::vector<ComplexVector> work_block(block_size, ComplexVector(N));
    
    // Adjust max_iter to account for block size
    int num_steps = std::min(max_iter / block_size, N / block_size);
    
    // Use flat vectors for alpha and beta blocks
    std::vector<Complex> alpha_flat(num_steps * block_size * block_size, Complex(0.0, 0.0));
    std::vector<Complex> beta_flat((num_steps+1) * block_size * block_size, Complex(0.0, 0.0));
    
    std::cout << "Block Lanczos: Iterating with block size " << block_size << ", steps " << num_steps << std::endl;
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int block_idx, int vec_idx, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(block_idx) + "_" + std::to_string(vec_idx) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Helper functions to access flattened arrays
    auto alpha = [&alpha_flat, block_size](int j, int r, int c) -> Complex& {
        return alpha_flat[j * block_size * block_size + r * block_size + c];
    };
    
    auto beta = [&beta_flat, block_size](int j, int r, int c) -> Complex& {
        return beta_flat[j * block_size * block_size + r * block_size + c];
    };
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        std::cout << "Block iteration " << j + 1 << " of " << num_steps << std::endl;
        
        // Apply H to each vector in the current block
        for (int b = 0; b < block_size; b++) {
            H(curr_block[b].data(), work_block[b].data(), N);
        }
        
        // Subtract beta_j * prev_block
        if (j > 0) {
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    Complex neg_beta = -beta(j, i, k);
                    cblas_zaxpy(N, &neg_beta, prev_block[k].data(), 1, work_block[i].data(), 1);
                }
            }
        }
        
        // Compute alpha_j block
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                Complex dot;
                cblas_zdotc_sub(N, curr_block[k].data(), 1, work_block[i].data(), 1, &dot);
                alpha(j, i, k) = dot;
                
                // Subtract from work vector
                Complex neg_dot = -dot;
                cblas_zaxpy(N, &neg_dot, curr_block[k].data(), 1, work_block[i].data(), 1);
            }
        }
        
        // QR factorization of the work block to get next orthonormal block
        for (int i = 0; i < block_size; i++) {
            // Compute the norm of the work vector
            double norm = cblas_dznrm2(N, work_block[i].data(), 1);
            
            // If nearly zero, generate a new orthogonal vector
            if (norm < tol) {
                // Generate a set of all vectors we've seen so far
                std::vector<ComplexVector> all_vectors;
                for (int b = 0; b < block_size; b++) {
                    all_vectors.push_back(curr_block[b]);
                    if (j > 0) all_vectors.push_back(prev_block[b]);
                }
                for (int b = 0; b < i; b++) {
                    all_vectors.push_back(next_block[b]);
                }
                
                next_block[i] = generateOrthogonalVector(N, all_vectors, gen, dist);
                norm = 1.0; // Already normalized by generateOrthogonalVector
            } else {
                // Copy work to next
                next_block[i] = work_block[i];
                
                // Normalize
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, next_block[i].data(), 1);
            }
            
            // Set the diagonal beta element
            beta(j+1, i, i) = Complex(norm, 0.0);
            
            // Orthogonalize remaining work vectors against this one
            for (int k = i + 1; k < block_size; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, next_block[i].data(), 1, work_block[k].data(), 1, &overlap);
                beta(j+1, k, i) = overlap;  // Off-diagonal beta element
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, next_block[i].data(), 1, work_block[k].data(), 1);
            }
        }
        
        // Store the new basis vectors to disk
        if (eigenvectors) {
            if (j < num_steps - 1) {
                for (int b = 0; b < block_size; b++) {
                    std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + "_" + std::to_string(b) + ".bin";
                    std::ofstream outfile(next_basis_file, std::ios::binary);
                    if (!outfile) {
                        std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                        return;
                    }
                    outfile.write(reinterpret_cast<char*>(next_block[b].data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
        }
        
        // Update for next iteration
        prev_block = curr_block;
        curr_block = next_block;
    }
    
    // Convert block tridiagonal matrix to regular format for solving
    int total_dim = block_size * num_steps;
    std::vector<Complex> block_matrix(total_dim * total_dim, Complex(0.0, 0.0));
    
    std::cout << "Block Lanczos: Constructing block tridiagonal matrix of size " << total_dim << "x" << total_dim << std::endl;
    
    // Fill diagonal blocks (alpha)
    for (int j = 0; j < num_steps; j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = j * block_size + r;
                int col = j * block_size + c;
                block_matrix[col * total_dim + row] = alpha(j, r, c);
            }
        }
    }
    
    // Fill off-diagonal blocks (beta)
    for (int j = 1; j < num_steps + 1; j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = (j-1) * block_size + r;
                int col = j * block_size + c;
                if (col < total_dim) {
                    block_matrix[col * total_dim + row] = beta(j, r, c);
                    block_matrix[row * total_dim + col] = std::conj(beta(j, r, c));
                }
            }
        }
    }
    
    std::cout << "Block Lanczos: Solving block tridiagonal matrix" << std::endl;
    
    // Diagonalize the block tridiagonal matrix
    std::vector<double> evals(total_dim);
    std::vector<Complex> evecs;
    
    char jobz = eigenvectors ? 'V' : 'N';
    int info;
    
    if (eigenvectors) {
        evecs = block_matrix;  // Copy for LAPACK which overwrites input
    }
    
    info = LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, 'U', 
                       total_dim, reinterpret_cast<lapack_complex_double*>(eigenvectors ? evecs.data() : block_matrix.data()), 
                       total_dim, evals.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    int n_eigenvalues = std::min(exct, total_dim);
    eigenvalues.resize(n_eigenvalues);
    std::copy(evals.begin(), evals.begin() + n_eigenvalues, eigenvalues.begin());
    
    // Save eigenvalues to a single file
    std::string evec_dir = dir + "/block_lanczos_eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
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
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors) {
        std::cout << "Block Lanczos: Transforming eigenvectors back to original basis" << std::endl;
        
        // Process in batches to save memory
        const int batch_size = 10;
        for (int start_idx = 0; start_idx < n_eigenvalues; start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues);
            
            std::cout << "Processing eigenvectors " << start_idx + 1 << " to " << end_idx 
                      << " of " << n_eigenvalues << std::endl;
            
            // Transform each eigenvector in the batch
            for (int i = start_idx; i < end_idx; i++) {
                std::cout << "  Transforming eigenvector " << i + 1 << std::endl;
                
                // Initialize full eigenvector
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                
                // For each block step
                for (int j = 0; j < num_steps; j++) {
                    // For each vector in the block
                    for (int b = 0; b < block_size; b++) {
                        // Load basis vector
                        ComplexVector basis = read_basis_vector(j, b, N);
                        
                        // Get coefficient
                        Complex coef = evecs[(j * block_size + b) * total_dim + i];
                        
                        // Add contribution
                        cblas_zaxpy(N, &coef, basis.data(), 1, full_vector.data(), 1);
                    }
                }
                
                // Normalize the eigenvector
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                if (norm > 1e-12) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, full_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(full_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
}

// Block Lanczos algorithm with periodic reorthogonalization
void block_lanczos_periodic_reorth(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct,
                                 double tol, std::vector<double>& eigenvalues, std::string dir = "",
                                 bool eigenvectors = false, int block_size = 4, int reorth_period = 5) {
    
    // Initialize random block of orthogonal vectors
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    
    // Generate first random vector
    block_vectors[0] = generateRandomVector(N, gen, dist);
    
    // Generate remaining orthogonal vectors in the block
    for (int i = 1; i < block_size; i++) {
        block_vectors[i] = generateOrthogonalVector(N, 
            std::vector<ComplexVector>(block_vectors.begin(), block_vectors.begin() + i), 
            gen, dist);
    }
    
    // Create directory for temporary basis vector blocks
    std::string temp_dir = dir + "/block_lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());
    
    // Store basis vectors for reorthogonalization
    std::vector<ComplexVector> all_basis_vectors;
    for (int i = 0; i < block_size; i++) {
        all_basis_vectors.push_back(block_vectors[i]);
        
        // Write the first block of basis vectors to files
        std::string basis_file = temp_dir + "/basis_0_" + std::to_string(i) + ".bin";
        std::ofstream outfile(basis_file, std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
            return;
        }
        outfile.write(reinterpret_cast<char*>(block_vectors[i].data()), N * sizeof(Complex));
        outfile.close();
    }
    
    // Previous, current and next blocks
    std::vector<ComplexVector> prev_block(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    std::vector<ComplexVector> curr_block = block_vectors;
    std::vector<ComplexVector> next_block(block_size, ComplexVector(N));
    std::vector<ComplexVector> work_block(block_size, ComplexVector(N));
    
    // Block tridiagonal matrix elements
    std::vector<std::vector<std::vector<Complex>>> alpha; // Diagonal blocks
    std::vector<std::vector<std::vector<Complex>>> beta;  // Off-diagonal blocks
    
    // First empty beta block
    beta.push_back(std::vector<std::vector<Complex>>(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0))));
    
    // Adjust max_iter to account for block size
    int num_steps = std::min(max_iter / block_size, N / block_size);
    
    std::cout << "Block Lanczos with periodic reorthogonalization: Iterating with block size " << block_size 
              << ", steps " << num_steps << ", reorth period " << reorth_period << std::endl;
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int block_idx, int vec_idx, int N) -> ComplexVector {
        ComplexVector vec(N);
        std::string filename = temp_dir + "/basis_" + std::to_string(block_idx) + "_" + std::to_string(vec_idx) + ".bin";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
            return vec;
        }
        infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
        return vec;
    };
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        std::cout << "Block iteration " << j + 1 << " of " << num_steps << std::endl;
        
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
        
        // Compute alpha_j block
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                Complex dot;
                cblas_zdotc_sub(N, curr_block[k].data(), 1, work_block[i].data(), 1, &dot);
                curr_alpha[i][k] = dot;
                
                // Subtract from work vector
                Complex neg_dot = -dot;
                cblas_zaxpy(N, &neg_dot, curr_block[k].data(), 1, work_block[i].data(), 1);
            }
        }
        
        alpha.push_back(curr_alpha);
        
        // Periodic reorthogonalization
        if (j % reorth_period == reorth_period - 1 || j >= num_steps - 3) {
            std::cout << "  Performing reorthogonalization at step " << j + 1 << std::endl;
            
            // Full reorthogonalization against all previous basis vectors
            for (int b = 0; b < block_size; b++) {
                for (int iter = 0; iter < 2; iter++) { // Twice for numerical stability
                    for (int step = 0; step <= j; step++) {
                        for (int vec = 0; vec < block_size; vec++) {
                            // Read basis vector from file
                            ComplexVector basis = read_basis_vector(step, vec, N);
                            
                            // Calculate projection and subtract
                            Complex overlap;
                            cblas_zdotc_sub(N, basis.data(), 1, work_block[b].data(), 1, &overlap);
                            
                            Complex neg_overlap = -overlap;
                            cblas_zaxpy(N, &neg_overlap, basis.data(), 1, work_block[b].data(), 1);
                        }
                    }
                }
            }
        }
        
        // QR factorization of the work block to get next orthonormal block
        std::vector<std::vector<Complex>> next_beta(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        // Simplified Gram-Schmidt for QR factorization
        for (int i = 0; i < block_size; i++) {
            // Compute the norm of the work vector
            double norm = cblas_dznrm2(N, work_block[i].data(), 1);
            
            // If nearly zero, generate a new orthogonal vector
            if (norm < tol) {
                // Generate a new vector orthogonal to all previously stored vectors
                std::vector<ComplexVector> all_vectors;
                for (int step = 0; step <= j; step++) {
                    for (int vec = 0; vec < block_size; vec++) {
                        all_vectors.push_back(read_basis_vector(step, vec, N));
                    }
                }
                for (int b = 0; b < i; b++) {
                    all_vectors.push_back(next_block[b]);
                }
                
                next_block[i] = generateOrthogonalVector(N, all_vectors, gen, dist);
                norm = 1.0; // Already normalized by generateOrthogonalVector
            } else {
                // Copy work to next
                next_block[i] = work_block[i];
                
                // Normalize
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, next_block[i].data(), 1);
            }
            
            // Set the diagonal beta element
            next_beta[i][i] = Complex(norm, 0.0);
            
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
        
        // Store the new basis vectors to disk
        if (eigenvectors || j < num_steps - 1) {
            for (int b = 0; b < block_size; b++) {
                std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + "_" + std::to_string(b) + ".bin";
                std::ofstream outfile(next_basis_file, std::ios::binary);
                if (!outfile) {
                    std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                    return;
                }
                outfile.write(reinterpret_cast<char*>(next_block[b].data()), N * sizeof(Complex));
                outfile.close();
            }
        }
        
        // Update for next iteration
        prev_block = curr_block;
        curr_block = next_block;
    }
    
    // Convert block tridiagonal matrix to regular format for solving
    int total_dim = block_size * num_steps;
    std::vector<Complex> block_matrix(total_dim * total_dim, Complex(0.0, 0.0));
    
    std::cout << "Block Lanczos: Constructing block tridiagonal matrix of size " << total_dim << "x" << total_dim << std::endl;
    
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
    
    std::cout << "Block Lanczos: Solving block tridiagonal matrix" << std::endl;
    
    // Diagonalize the block tridiagonal matrix
    std::vector<double> evals(total_dim);
    std::vector<Complex> evecs;
    
    char jobz = eigenvectors ? 'V' : 'N';
    int info;
    
    if (eigenvectors) {
        evecs = block_matrix;  // Copy for LAPACK which overwrites input
    }
    
    info = LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, 'U', 
                       total_dim, reinterpret_cast<lapack_complex_double*>(eigenvectors ? evecs.data() : block_matrix.data()), 
                       total_dim, evals.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues
    int n_eigenvalues = std::min(exct, total_dim);
    eigenvalues.resize(n_eigenvalues);
    std::copy(evals.begin(), evals.begin() + n_eigenvalues, eigenvalues.begin());
    
    // Save eigenvalues to a single file
    std::string evec_dir = dir + "/block_lanczos_eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
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
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors) {
        std::cout << "Block Lanczos: Transforming eigenvectors back to original basis" << std::endl;
        
        // Process in batches to save memory
        const int batch_size = 10;
        for (int start_idx = 0; start_idx < n_eigenvalues; start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues);
            int batch_n = end_idx - start_idx;
            
            std::cout << "Processing eigenvectors " << start_idx + 1 << " to " << end_idx 
                      << " of " << n_eigenvalues << std::endl;
            
            // Transform each eigenvector in the batch
            for (int i = start_idx; i < end_idx; i++) {
                std::cout << "  Transforming eigenvector " << i + 1 << std::endl;
                
                // Initialize full eigenvector
                ComplexVector full_vector(N, Complex(0.0, 0.0));
                
                // For each block step
                for (int j = 0; j < num_steps; j++) {
                    // For each vector in the block
                    for (int b = 0; b < block_size; b++) {
                        // Load basis vector
                        ComplexVector basis = read_basis_vector(j, b, N);
                        
                        // Get coefficient
                        Complex coef = evecs[(j * block_size + b) * total_dim + i];
                        
                        // Add contribution
                        cblas_zaxpy(N, &coef, basis.data(), 1, full_vector.data(), 1);
                    }
                }
                
                // Normalize the eigenvector
                double norm = cblas_dznrm2(N, full_vector.data(), 1);
                if (norm > 1e-12) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, full_vector.data(), 1);
                }
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (!evec_outfile) {
                    std::cerr << "Error: Cannot open file " << evec_file << " for writing" << std::endl;
                    continue;
                }
                evec_outfile.write(reinterpret_cast<char*>(full_vector.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Observables

// Calculate thermodynamic quantities directly from eigenvalues
struct ThermodynamicData {
    std::vector<double> temperatures;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
};

// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min = 0.01,        // Minimum temperature
    double T_max = 10.0,        // Maximum temperature
    int num_points = 100        // Number of temperature points
) {
    ThermodynamicData results;
    
    // Generate logarithmically spaced temperature points
    results.temperatures.resize(num_points);
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Resize other arrays
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Find ground state energy (useful for numerical stability)
    double E0 = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    
    // For each temperature
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        // Use log-sum-exp trick for numerical stability in calculating Z
        // Find the maximum value for normalization
        double max_exp = -beta * E0;  // Start with ground state
        
        // Calculate partition function Z and energy using log-sum-exp trick
        double sum_exp = 0.0;
        double sum_E_exp = 0.0;
        double sum_E2_exp = 0.0;
        
        for (double E : eigenvalues) {
            double delta_E = E - E0;
            double exp_term = std::exp(-beta * delta_E);
            
            sum_exp += exp_term;
            sum_E_exp += E * exp_term;
            sum_E2_exp += E * E * exp_term;
        }
        
        // Calculate log(Z) = log(sum_exp) + (-beta*E0)
        double log_Z = std::log(sum_exp) - beta * E0;
        
        // Free energy F = -T * log(Z)
        results.free_energy[i] = -T * log_Z;
        
        // Energy E = (1/Z) * sum_i E_i * exp(-beta*E_i)
        results.energy[i] = sum_E_exp / sum_exp;
        
        // Specific heat C_v = beta^2 * (⟨E^2⟩ - ⟨E⟩^2)
        double avg_E2 = sum_E2_exp / sum_exp;
        double avg_E_squared = results.energy[i] * results.energy[i];
        results.specific_heat[i] = beta * beta * (avg_E2 - avg_E_squared);
        
        // Entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
    }
    
    // Handle special case for T → 0 (avoid numerical issues)
    if (T_min < 1e-6) {
        // In the limit T → 0, only the ground state contributes
        // Energy → E0
        results.energy[0] = E0;
        
        // Specific heat → 0
        results.specific_heat[0] = 0.0;
        
        // Entropy → 0 (third law of thermodynamics) or ln(g) if g-fold degenerate
        int degeneracy = 0;
        for (double E : eigenvalues) {
            if (std::abs(E - E0) < 1e-10) degeneracy++;
        }
        results.entropy[0] = (degeneracy > 1) ? std::log(degeneracy) : 0.0;
        
        // Free energy → E0 - TS
        results.free_energy[0] = E0 - results.temperatures[0] * results.entropy[0];
    }
    
    return results;
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


// Calculate thermal expectation value of operator A using eigenvalues and eigenvectors
// <A> = (1/Z) * ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
Complex calculate_thermal_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                               // Hilbert space dimension
    double beta,                                         // Inverse temperature β = 1/kT
    const std::string& eig_dir                           // Directory with eigenvector files
) {

    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.bin";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return Complex(0.0, 0.0);
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();

    // Using the log-sum-exp trick for numerical stability
    // Find the maximum value for normalization
    double max_val = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_val = std::max(max_val, -beta * eigenvalues[i]);
    }
    
    // Calculate the numerator <A> = ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
    Complex numerator(0.0, 0.0);
    double sum_exp = 0.0;
    
    // Temporary vector to store A|ψ_i⟩
    ComplexVector A_psi(N);
    ComplexVector psi_i(N);
    
    // Calculate both the numerator and Z in one loop
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        // Calculate the Boltzmann factor with numerical stability
        double boltzmann = std::exp(-beta * eigenvalues[i] - max_val);
        sum_exp += boltzmann;
        
        // Load eigenvector from file
        std::string evec_file = eig_dir + "/eigenvector_" + std::to_string(i) + ".bin";
        std::ifstream infile(evec_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        infile.read(reinterpret_cast<char*>(psi_i.data()), N * sizeof(Complex));
        infile.close();
        
        // Calculate <ψ_i|A|ψ_i>
        A(psi_i.data(), A_psi.data(), N);
        
        Complex expectation;
        cblas_zdotc_sub(N, psi_i.data(), 1, A_psi.data(), 1, &expectation);
        
        // Add contribution to numerator
        numerator += boltzmann * expectation;
    }
    
    // Return <A> = numerator/Z
    return numerator / sum_exp;
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
    int m_max = 1000           // Maximum Lanczos iterations per random vector
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
    int R=30,              // Number of random samples
    int M=1000,              // Lanczos iterations per sample
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
    std::string dir = argv[1];

    // Read num_site from the second line of Trans.dat
    std::ifstream trans_file(dir + "/Trans.dat");
    if (!trans_file.is_open()) {
        std::cerr << "Error: Cannot open file " << dir + "/Trans.dat" << std::endl;
        return 1;
    }

    // Skip the first line
    std::string dummy_line;
    std::getline(trans_file, dummy_line);

    // Read the second line to get num_site
    std::string dum;
    int num_site;
    trans_file >> dum >> num_site;
    trans_file.close();

    std::cout << "Number of sites: " << num_site << std::endl;

    Operator op(num_site);
    int eigenvector = std::atoi(argv[2]);

    op.loadFromFile(dir + "/Trans.dat");
    op.loadFromInterAllFile(dir + "/InterAll.dat");
    
    // Matrix Ham = op.returnMatrix();
    // std::cout << "Loaded operator from " << dir << std::endl;
    // std::cout << "Matrix size: " << Ham.size() << " x " << Ham[0].size() << std::endl;
    // std::cout << "Matrix elements: " << std::endl;
    // for (const auto& row : Ham) {
    //     for (const auto& elem : row) {
    //         std::cout << elem << " ";
    //     }
    //     std::cout << std::endl;
    // }

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
<<<<<<< HEAD
    block_lanczos_no_ortho(H, N, N, N, 1e-10, eigenvalues, dir);
=======
    lanczos_no_ortho(H, N, N, N, 1e-10, eigenvalues, dir, eigenvector);


>>>>>>> aed7f1dfc20be09fcbaa544e8539c56170ce9c59

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Full diagonalization completed in " << elapsed.count() << " seconds" << std::endl;
    
    std::string output_dir = dir + "/output/";

    // Create output directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());
    std::cout << "Output directory created: " << output_dir << std::endl;

    // Save eigenvalues to file
    std::ofstream eigenvalue_file(output_dir+"spectrum.dat");
    if (eigenvalue_file.is_open()) {
        for (const auto& eigenvalue : eigenvalues) {
            eigenvalue_file << eigenvalue << std::endl;
        }
        eigenvalue_file.close();
        std::cout << "Full spectrum saved to ED_test_full_spectrum.dat" << std::endl;
    }
    
    std::cout << "Reading eigenvalues from ED_test_full_spectrum.dat..." << std::endl;
    std::ifstream eigenvalue_input("ED_test_full_spectrum.dat");
    eigenvalues.clear();

    if (eigenvalue_input.is_open()) {
        double value;
        while (eigenvalue_input >> value) {
            eigenvalues.push_back(value);
        }
        eigenvalue_input.close();
        std::cout << "Read " << eigenvalues.size() << " eigenvalues from file" << std::endl;
    } else {
        std::cerr << "Failed to open ED_test_full_spectrum.dat" << std::endl;
        return 1;
    }

    // Calculate thermodynamics from spectrum
    std::cout << "Calculating thermodynamic properties and expectation values of observables..." << std::endl;
    double T_min = 0.01;
    double T_max = 20.0;
    int num_points = 2000;

    if (eigenvector){
    // Find and process operators from files in the directory
    std::cout << "Looking for operator files in directory: " << dir << std::endl;

    // Vector to store operators and their names
    std::vector<std::pair<std::string, Operator>> operators;

    // Helper function to process a file
    auto process_operator_file = [&](const std::string& filepath) {
        // Extract the base filename from the path
        std::string filename = filepath;
        size_t last_slash_pos = filepath.find_last_of('/');
        if (last_slash_pos != std::string::npos) {
            filename = filepath.substr(last_slash_pos + 1);
        }
        
        // Determine if one-body or two-body operator
        if (filename.find("one_body") == 0) {
            std::cout << "Processing one-body operator from file: " << filename << std::endl;
            
            // Create operator
            Operator new_op(num_site);
            
            try {
                // Load operator
                new_op.loadFromFile(filepath);
                
                // Store the operator with its name
                operators.push_back({filename, new_op});
            } catch (const std::exception& e) {
                std::cerr << "Error loading " << filename << ": " << e.what() << std::endl;
            }
        } 
        else if (filename.find("two_body") == 0) {
            std::cout << "Processing two-body operator from file: " << filename << std::endl;
            
            // Create operator
            Operator new_op(num_site);
            
            try {
                // Load operator
                new_op.loadFromInterAllFile(filepath);
                
                // Store the operator with its name
                operators.push_back({filename, new_op});
            } catch (const std::exception& e) {
                std::cerr << "Error loading " << filename << ": " << e.what() << std::endl;
            }
        }
    };

    // List files in directory using system command
    std::string cmd = "find \"" + dir + "\" -type f \\( -name \"one_body*\" -o -name \"two_body*\" \\) 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string filepath(buffer);
            
            // Remove trailing newline if present
            if (!filepath.empty() && filepath[filepath.size() - 1] == '\n') {
                filepath.erase(filepath.size() - 1);
            }
            
            process_operator_file(filepath);
        }
        pclose(pipe);
    } else {
        std::cerr << "Error executing directory listing command" << std::endl;
    }

    std::cout << "Found " << operators.size() << " operator files in directory" << std::endl;

    // Compute thermal expectation values for each operator
    if (operators.empty()) {
        std::cout << "No operator files found. Skipping expectation value calculations." << std::endl;
    } else {
        std::cout << "Computing thermal expectation values for " << operators.size() << " operators" << std::endl;

        std::vector<double> temperatures(num_points);
        double log_T_min = std::log(T_min);
        double log_T_max = std::log(T_max);
        double log_T_step = (log_T_max - log_T_min) / (num_points - 1);

        for (int i = 0; i < num_points; i++) {
            temperatures[i] = std::exp(log_T_min + i * log_T_step);
        }

        // For each operator, calculate expectation values
        for (const auto& op_pair : operators) {
            std::string op_name = op_pair.first;
            const Operator& curr_op = op_pair.second;
            
            std::cout << "Calculating thermal expectation values for: " << op_name << std::endl;
            
            // Create operator function for this operator
            auto op_func = [&curr_op](const Complex* v, Complex* result, int size) {
                std::vector<Complex> vec(v, v + size);
                std::vector<Complex> op_result = curr_op.apply(vec);
                std::copy(op_result.begin(), op_result.end(), result);
            };
            
            // Calculate expectation values at each temperature
            std::vector<double> expectation_real;
            std::vector<double> expectation_imag;
            
            for (double temp : temperatures) {
                double beta = 1.0 / temp;
                
                // Use FTLM method to compute thermal average
                Complex exp_val = calculate_thermal_expectation(
                    op_func, N, beta, dir+"/lanczos_eigenvectors"
                );
                
                expectation_real.push_back(exp_val.real());
                expectation_imag.push_back(exp_val.imag());
                
                // Occasional progress reporting
                if (temperatures.size() > 10 && 
                    (temp == temperatures.front() || temp == temperatures.back() || 
                     expectation_real.size() % (num_points/10) == 0)) {
                    std::cout << "  T = " << std::fixed << std::setprecision(4) << temp 
                              << ", <O> = " << exp_val.real() << " + " << exp_val.imag() << "i" << std::endl;
                }
            }
            
            // Save results to file
            std::string outfile_name = op_name + "_thermal_expectation.dat";
            std::ofstream outfile(outfile_name);
            if (outfile.is_open()) {
                outfile << "# Temperature ExpectationValue_Real ExpectationValue_Imag" << std::endl;
                for (size_t i = 0; i < temperatures.size(); i++) {
                    outfile << std::fixed << std::setprecision(6)
                            << temperatures[i] << " "
                            << expectation_real[i] << " "
                            << expectation_imag[i] << std::endl;
                }
                outfile.close();
                std::cout << "Thermal expectation values for " << op_name << " saved to " << outfile_name << std::endl;
            } else {
                std::cerr << "Error: Could not open file " << outfile_name << " for writing" << std::endl;
            }
        }
    }

    }   
    
    ThermodynamicData thermo = calculate_thermodynamics_from_spectrum(
        eigenvalues, T_min, T_max, num_points
    );
    
    // Save thermodynamic data
    std::ofstream thermo_file(output_dir+"thermodynamics.dat");
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