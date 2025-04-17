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
    std::string evec_dir = dir + "/shift_invert_lanczos_results";
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

// Thermal Pure Quantum (TPQ) state methods for thermodynamic calculations

// Generate a canonical TPQ state by applying (H - E)^k to a random vector
ComplexVector generate_tpq_state(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    double E_offset,                                       // Energy offset
    int k = 1,                                             // Power of (H - E)
    double tol = 1e-10                                     // Tolerance
) {
    // Create a random initial state
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector tpq_state(N);
    for (int i = 0; i < N; i++) {
        tpq_state[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, tpq_state.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, tpq_state.data(), 1);
    
    // Apply (H - E)^k
    ComplexVector temp_state(N);
    
    for (int i = 0; i < k; i++) {
        // Apply H to current state
        H(tpq_state.data(), temp_state.data(), N);
        
        // Subtract E * current state
        for (int j = 0; j < N; j++) {
            temp_state[j] -= E_offset * tpq_state[j];
        }
        
        // Normalize
        norm = cblas_dznrm2(N, temp_state.data(), 1);
        if (norm < tol) {
            // If norm is too small, we've probably hit an eigenstate
            std::cout << "Warning: TPQ generation may have converged to an eigenstate at step " << i + 1 << std::endl;
            break;
        }
        
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, temp_state.data(), 1);
        
        // Update for next iteration
        tpq_state = temp_state;
    }
    
    return tpq_state;
}

// Calculate the effective inverse temperature (beta) of a TPQ state
double calculate_tpq_beta(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N,                                                 // Hilbert space dimension
    double E_ref = 0.0                                     // Energy reference (usually ground state)
) {
    // Calculate <H>
    ComplexVector H_tpq(N);
    H(tpq_state.data(), H_tpq.data(), N);
    
    Complex energy_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H_tpq.data(), 1, &energy_exp);
    double energy = std::real(energy_exp);
    
    // Calculate <H²>
    ComplexVector H2_tpq(N);
    H(H_tpq.data(), H2_tpq.data(), N);
    
    Complex energy2_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H2_tpq.data(), 1, &energy2_exp);
    double energy2 = std::real(energy2_exp);
    
    // Variance of H
    double var_H = energy2 - energy * energy;
    
    // Effective inverse temperature: β = 2*(⟨H⟩ - E_ref)/⟨(H-⟨H⟩)²⟩
    if (var_H < 1e-10) {
        return std::numeric_limits<double>::infinity(); // If variance is zero, we have an eigenstate
    }
    
    return 2.0 * (energy - E_ref) / var_H;
}

// Calculate thermodynamic quantities from a TPQ state
struct TPQThermodynamics {
    double beta;       // Inverse temperature
    double energy;     // Energy
    double specific_heat;
    double entropy;
    double free_energy;
};

TPQThermodynamics calculate_tpq_thermodynamics(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N,                                                 // Hilbert space dimension
    double E_ref = 0.0                                     // Energy reference (usually ground state)
) {
    // Calculate <H>
    ComplexVector H_tpq(N);
    H(tpq_state.data(), H_tpq.data(), N);
    
    Complex energy_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H_tpq.data(), 1, &energy_exp);
    double energy = std::real(energy_exp);
    
    // Calculate <H²>
    ComplexVector H2_tpq(N);
    H(H_tpq.data(), H2_tpq.data(), N);
    
    Complex energy2_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H2_tpq.data(), 1, &energy2_exp);
    double energy2 = std::real(energy2_exp);
    
    // Variance of H
    double var_H = energy2 - energy * energy;
    
    // Effective inverse temperature
    double beta = 2.0 * (energy - E_ref) / var_H;
    double temperature = (beta > 1e-10) ? 1.0 / beta : std::numeric_limits<double>::infinity();
    
    // Specific heat: C = beta² * var_H
    double specific_heat = beta * beta * var_H;
    
    // Entropy and free energy require additional approximations
    // For canonical TPQ states, entropy can be approximated as S ≈ ln(D) - β²*var_H/2
    // where D is the Hilbert space dimension
    double entropy = std::log(N) - beta * beta * var_H / 2.0;
    
    // Free energy: F = E - TS
    double free_energy = energy - temperature * entropy;
    
    return {beta, energy, specific_heat, entropy, free_energy};
}

// Calculate expectation value of an observable using a TPQ state
Complex calculate_tpq_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N                                                 // Hilbert space dimension
) {
    ComplexVector A_tpq(N);
    A(tpq_state.data(), A_tpq.data(), N);
    
    Complex expectation;
    cblas_zdotc_sub(N, tpq_state.data(), 1, A_tpq.data(), 1, &expectation);
    
    return expectation;
}

// Main TPQ implementation for temperature scanning
struct TPQResults {
    std::vector<double> betas;           // Inverse temperatures
    std::vector<double> temperatures;    // Temperatures
    std::vector<double> energies;        // Energies
    std::vector<double> specific_heats;  // Specific heats
    std::vector<double> entropies;       // Entropies
    std::vector<double> free_energies;   // Free energies
    std::vector<std::vector<Complex>> observables; // Observable expectation values
};

TPQResults perform_tpq_calculation(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int num_samples = 20,                                 // Number of TPQ samples
    int max_k = 50,                                       // Maximum power for (H-E)^k
    double E_min = -1.0,                                  // Minimum energy offset 
    double E_max = 1.0,                                   // Maximum energy offset
    double E_ref = 0.0,                                   // Energy reference
    double beta_min = 0.01,                               // Minimum inverse temperature
    double beta_max = 100.0,                              // Maximum inverse temperature
    int num_beta_bins = 100,                             // Number of temperature bins
    std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {}  // Optional observables
) {
    TPQResults results;
    
    // Initialize result containers
    results.betas.resize(num_beta_bins, 0.0);
    results.temperatures.resize(num_beta_bins, 0.0);
    results.energies.resize(num_beta_bins, 0.0);
    results.specific_heats.resize(num_beta_bins, 0.0);
    results.entropies.resize(num_beta_bins, 0.0);
    results.free_energies.resize(num_beta_bins, 0.0);
    
    // Initialize counters for each bin
    std::vector<int> bin_counts(num_beta_bins, 0);
    
    // Initialize bins for beta values
    double log_beta_min = std::log(beta_min);
    double log_beta_max = std::log(beta_max);
    double log_beta_step = (log_beta_max - log_beta_min) / (num_beta_bins - 1);
    
    for (int i = 0; i < num_beta_bins; i++) {
        results.betas[i] = std::exp(log_beta_min + i * log_beta_step);
        results.temperatures[i] = 1.0 / results.betas[i];
    }
    
    // Initialize observable containers if provided
    if (!observables.empty()) {
        results.observables.resize(observables.size(), std::vector<Complex>(num_beta_bins, Complex(0.0, 0.0)));
    }
    
    std::cout << "TPQ: Starting calculations with " << num_samples << " samples" << std::endl;
    
    // Generate TPQ states with different energy offsets and powers
    for (int sample = 0; sample < num_samples; sample++) {
        // Randomly choose energy offset between E_min and E_max
        double energy_offset = E_min + (E_max - E_min) * static_cast<double>(sample) / num_samples;
        
        // Generate multiple states with different powers
        for (int k = 1; k <= max_k; k += 2) { // Increment by 2 for efficiency
            // Generate TPQ state
            ComplexVector tpq_state = generate_tpq_state(H, N, energy_offset, k);
            
            // Calculate thermodynamics
            auto thermo = calculate_tpq_thermodynamics(H, tpq_state, N, E_ref);
            
            // Determine which beta bin this state belongs to
            double log_beta = std::log(thermo.beta);
            int bin = static_cast<int>((log_beta - log_beta_min) / log_beta_step);
            
            // Skip if outside our temperature range
            if (bin < 0 || bin >= num_beta_bins) {
                continue;
            }
            
            // Accumulate thermodynamic data
            bin_counts[bin]++;
            results.energies[bin] += thermo.energy;
            results.specific_heats[bin] += thermo.specific_heat;
            results.entropies[bin] += thermo.entropy;
            results.free_energies[bin] += thermo.free_energy;
            
            // Calculate observables if provided
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                Complex exp_val = calculate_tpq_expectation(observables[obs_idx], tpq_state, N);
                results.observables[obs_idx][bin] += exp_val;
            }
            
            // Progress reporting
            if ((sample * max_k + k) % 10 == 0) {
                std::cout << "TPQ: Sample " << sample + 1 << "/" << num_samples 
                          << ", k = " << k << ", β = " << thermo.beta
                          << ", E = " << thermo.energy << std::endl;
            }
        }
    }
    
    // Average the results over the number of samples in each bin
    for (int i = 0; i < num_beta_bins; i++) {
        if (bin_counts[i] > 0) {
            results.energies[i] /= bin_counts[i];
            results.specific_heats[i] /= bin_counts[i];
            results.entropies[i] /= bin_counts[i];
            results.free_energies[i] /= bin_counts[i];
            
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                results.observables[obs_idx][i] /= bin_counts[i];
            }
            
            std::cout << "TPQ: Bin " << i << " (β = " << results.betas[i] 
                      << ") has " << bin_counts[i] << " samples" << std::endl;
        } else {
            std::cout << "TPQ: Warning - no samples in bin " << i 
                      << " (β = " << results.betas[i] << ")" << std::endl;
        }
    }
    
    return results;
}

// Generate microcanonical TPQ states for better control over temperature
ComplexVector generate_microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    double target_energy,                                  // Target energy
    double energy_window = 0.1,                           // Energy window width
    int max_iter = 100,                                   // Maximum iterations
    double tol = 1e-6                                     // Tolerance
) {
    // Create a random initial state
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector psi(N);
    for (int i = 0; i < N; i++) {
        psi[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, psi.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, psi.data(), 1);
    
    // Apply a filter to target the desired energy window
    // This is approximated by exp(-γ(H-E)²)
    ComplexVector H_psi(N);
    ComplexVector H2_psi(N);
    ComplexVector psi_new(N);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate current energy expectation
        H(psi.data(), H_psi.data(), N);
        Complex energy_exp;
        cblas_zdotc_sub(N, psi.data(), 1, H_psi.data(), 1, &energy_exp);
        double current_energy = std::real(energy_exp);
        
        // Calculate energy variance
        H(H_psi.data(), H2_psi.data(), N);
        Complex energy2_exp;
        cblas_zdotc_sub(N, psi.data(), 1, H2_psi.data(), 1, &energy2_exp);
        double energy_var = std::real(energy2_exp) - current_energy * current_energy;
        
        std::cout << "Microcanonical TPQ: iter " << iter << ", E = " << current_energy 
                  << ", var = " << energy_var << std::endl;
        
        // Check if we're close enough to target energy with small variance
        if (std::abs(current_energy - target_energy) < tol && energy_var < energy_window) {
            std::cout << "Microcanonical TPQ: Converged at iteration " << iter << std::endl;
            break;
        }
        
        // Adjust filtering parameter based on current energy
        double gamma = 1.0 / (2.0 * energy_window);
        if (std::abs(current_energy - target_energy) > energy_window) {
            gamma = 0.1 / energy_var; // Faster approach when far from target
        }
        
        // Apply filter exp(-γ(H-E)²) using Chebyshev approximation
        ComplexVector temp1(N), temp2(N);
        for (int i = 0; i < N; i++) {
            psi_new[i] = psi[i];
            temp1[i] = psi[i];
        }
        
        // Subtract target energy: (H-E)|ψ⟩
        for (int i = 0; i < N; i++) {
            H_psi[i] -= target_energy * psi[i];
        }
        
        // Apply approximation of exp(-γ(H-E)²) using series expansion
        Complex coef = Complex(1.0, 0.0);
        cblas_zaxpy(N, &coef, psi.data(), 1, psi_new.data(), 1);
        
        coef = Complex(-gamma, 0.0);
        for (int i = 0; i < N; i++) {
            temp2[i] = H_psi[i] * H_psi[i]; // (H-E)²|ψ⟩
        }
        cblas_zaxpy(N, &coef, temp2.data(), 1, psi_new.data(), 1);
        
        // Normalize the new state
        norm = cblas_dznrm2(N, psi_new.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, psi_new.data(), 1);
        
        // Update state for next iteration
        psi = psi_new;
    }
    
    return psi;
}

// Save TPQ results to file
void save_tpq_results(const TPQResults& results, const std::string& filename, 
                     int num_observables = 0) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    outfile << "# Beta Temperature Energy SpecificHeat Entropy FreeEnergy";
    for (int i = 0; i < num_observables; i++) {
        outfile << " Observable" << i << "_Real Observable" << i << "_Imag";
    }
    outfile << std::endl;
    
    for (size_t i = 0; i < results.betas.size(); i++) {
        outfile << results.betas[i] << " "
               << results.temperatures[i] << " "
               << results.energies[i] << " "
               << results.specific_heats[i] << " "
               << results.entropies[i] << " "
               << results.free_energies[i];
        
        for (int j = 0; j < num_observables; j++) {
            outfile << " " << results.observables[j][i].real()
                   << " " << results.observables[j][i].imag();
        }
        
        outfile << std::endl;
    }
    
    outfile.close();
    std::cout << "TPQ results saved to " << filename << std::endl;
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

// Finite Temperature Lanczos Method (FTLM) for thermal expectation values
ThermodynamicData finite_temperature_lanczos_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                // Hilbert space dimension
    int num_samples = 20,                                 // Number of random starting vectors
    int m = 100,                                          // Lanczos iterations per sample
    double T_min = 0.01,                                  // Minimum temperature
    double T_max = 10.0,                                  // Maximum temperature
    int num_points = 100,                                 // Number of temperature points
    std::string dir = "",                                 // Directory for temporary files
    std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {} // Optional observables
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
    
    // Initialize results vectors
    results.energy.resize(num_points, 0.0);
    results.specific_heat.resize(num_points, 0.0);
    results.entropy.resize(num_points, 0.0);
    results.free_energy.resize(num_points, 0.0);
    
    // Vectors for accumulating values across samples
    std::vector<double> Z_T(num_points, 0.0);  // Partition function
    std::vector<double> E_T(num_points, 0.0);  // Energy
    std::vector<double> E2_T(num_points, 0.0); // Energy squared
    
    // Vectors for observable expectation values if provided
    std::vector<std::vector<Complex>> obs_values;
    if (!observables.empty()) {
        obs_values.resize(observables.size(), std::vector<Complex>(num_points, Complex(0.0, 0.0)));
    }
    
    // Create random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Create directory for temporary files
    std::string temp_dir = dir + "/ftlm_temp";
    system(("mkdir -p " + temp_dir).c_str());
    
    std::cout << "FTLM: Starting with " << num_samples << " samples, " 
              << m << " Lanczos iterations per sample" << std::endl;
    
    // Loop over random samples
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "FTLM: Processing sample " << sample + 1 << " of " << num_samples << std::endl;
        
        // Generate random starting vector
        ComplexVector v_start(N);
        for (int i = 0; i < N; i++) {
            v_start[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v_start.data(), 1);
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, v_start.data(), 1);
        
        // Perform Lanczos iteration to generate basis and tridiagonal matrix
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_curr = v_start;
        ComplexVector v_next(N);
        ComplexVector w(N);
        
        // Store Lanczos vectors for later reconstruction
        std::vector<ComplexVector> lanczos_vectors;
        lanczos_vectors.push_back(v_curr);
        
        // Initialize alpha and beta for tridiagonal matrix
        std::vector<double> alpha;  // Diagonal elements
        std::vector<double> beta;   // Off-diagonal elements
        beta.push_back(0.0);        // β_0 is not used
        
        // Lanczos iteration to build tridiagonal matrix
        for (int j = 0; j < m; j++) {
            // w = H*v_j
            H(v_curr.data(), w.data(), N);
            
            // w = w - beta_j * v_{j-1}
            if (j > 0) {
                Complex neg_beta = Complex(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            // alpha_j = <v_j, w>
            Complex dot_product;
            cblas_zdotc_sub(N, v_curr.data(), 1, w.data(), 1, &dot_product);
            alpha.push_back(std::real(dot_product));
            
            // w = w - alpha_j * v_j
            Complex neg_alpha = Complex(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_curr.data(), 1, w.data(), 1);
            
            // Reorthogonalization for numerical stability
            for (int k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, lanczos_vectors[k].data(), 1, w.data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, lanczos_vectors[k].data(), 1, w.data(), 1);
            }
            
            // beta_{j+1} = ||w||
            norm = cblas_dznrm2(N, w.data(), 1);
            beta.push_back(norm);
            
            // Check for invariant subspace
            if (norm < 1e-10) {
                std::cout << "FTLM: Invariant subspace found at iteration " << j + 1 << std::endl;
                break;
            }
            
            // v_{j+1} = w / beta_{j+1}
            for (int i = 0; i < N; i++) {
                v_next[i] = w[i] / norm;
            }
            
            // Store for next iteration
            v_prev = v_curr;
            v_curr = v_next;
            
            // Store Lanczos vector for later reconstruction
            if (j < m - 1) {
                lanczos_vectors.push_back(v_curr);
            }
        }
        
        // Diagonalize tridiagonal matrix
        int actual_m = alpha.size();
        std::vector<double> eigenvalues(actual_m);
        std::vector<double> eigenvectors_T(actual_m * actual_m);
        
        // Call LAPACK
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', actual_m, 
                                alpha.data(), &beta[1], eigenvectors_T.data(), actual_m);
        
        if (info != 0) {
            std::cerr << "FTLM: LAPACKE_dstevd failed with code " << info << std::endl;
            continue;
        }
        
        // Compute matrix elements of observables if provided
        std::vector<std::vector<Complex>> obs_matrix_elements;
        if (!observables.empty()) {
            obs_matrix_elements.resize(observables.size(), 
                                     std::vector<Complex>(actual_m, Complex(0.0, 0.0)));
            
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                auto& A = observables[obs_idx];
                
                // For each eigenstate in the Krylov subspace
                for (int i = 0; i < actual_m; i++) {
                    // Reconstruct eigenstate in original basis
                    ComplexVector psi_i(N, Complex(0.0, 0.0));
                    for (int j = 0; j < actual_m; j++) {
                        Complex coef(eigenvectors_T[j * actual_m + i], 0.0);
                        cblas_zaxpy(N, &coef, lanczos_vectors[j].data(), 1, psi_i.data(), 1);
                    }
                    
                    // Apply observable
                    ComplexVector A_psi(N);
                    A(psi_i.data(), A_psi.data(), N);
                    
                    // Compute <psi_i|A|psi_i>
                    Complex expectation;
                    cblas_zdotc_sub(N, psi_i.data(), 1, A_psi.data(), 1, &expectation);
                    
                    obs_matrix_elements[obs_idx][i] = expectation;
                }
            }
        }
        
        // For each temperature, compute contribution to partition function and thermal averages
        for (int t = 0; t < num_points; t++) {
            double beta = 1.0 / results.temperatures[t];
            double Z_sample = 0.0;
            double E_sample = 0.0;
            double E2_sample = 0.0;
            std::vector<Complex> obs_sample(observables.size(), Complex(0.0, 0.0));
            
            // Project the initial random vector onto the eigenbasis of the tridiagonal matrix
            std::vector<double> overlaps(actual_m);
            for (int i = 0; i < actual_m; i++) {
                // The overlap is just the first component of the eigenvector
                // because the initial Lanczos vector is the first standard basis vector
                // in the Krylov subspace
                overlaps[i] = eigenvectors_T[i * actual_m + 0];
            }
            
            // Accumulate contributions to the trace
            for (int i = 0; i < actual_m; i++) {
                double boltzmann = std::exp(-beta * eigenvalues[i]);
                double weight = overlaps[i] * overlaps[i] * boltzmann;
                
                Z_sample += weight;
                E_sample += eigenvalues[i] * weight;
                E2_sample += eigenvalues[i] * eigenvalues[i] * weight;
                
                // Accumulate observable expectation values
                for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                    obs_sample[obs_idx] += obs_matrix_elements[obs_idx][i] * weight;
                }
            }
            
            // Factor N accounts for the trace normalization
            Z_T[t] += Z_sample * N;
            E_T[t] += E_sample * N;
            E2_T[t] += E2_sample * N;
            
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                obs_values[obs_idx][t] += obs_sample[obs_idx] * N;
            }
        }
        
        // Progress reporting
        if ((sample + 1) % 5 == 0 || sample == num_samples - 1) {
            std::cout << "FTLM: Completed " << sample + 1 << " samples" << std::endl;
            // Show current estimate of ground state energy
            std::cout << "  Current estimate of ground state energy: " << eigenvalues[0] << std::endl;
        }
    }
    
    // Finalize results by averaging over samples
    for (int t = 0; t < num_points; t++) {
        double T = results.temperatures[t];
        double Z = Z_T[t] / num_samples;
        double E = E_T[t] / num_samples;
        double E2 = E2_T[t] / num_samples;
        
        // Thermal averages
        results.energy[t] = E / Z;
        results.specific_heat[t] = (E2 / Z - (E / Z) * (E / Z)) / (T * T);
        results.free_energy[t] = -T * std::log(Z);
        results.entropy[t] = (results.energy[t] - results.free_energy[t]) / T;
        
        // Normalize observable expectation values
        for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
            obs_values[obs_idx][t] /= (num_samples * Z);
        }
    }
    
    // Save observable results if provided
    if (!observables.empty()) {
        std::string obs_dir = dir + "/ftlm_observables";
        system(("mkdir -p " + obs_dir).c_str());
        
        for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
            std::string filename = obs_dir + "/observable_" + std::to_string(obs_idx) + ".dat";
            std::ofstream outfile(filename);
            
            if (outfile.is_open()) {
                outfile << "# Temperature Real Imaginary" << std::endl;
                for (int t = 0; t < num_points; t++) {
                    outfile << results.temperatures[t] << " " 
                          << obs_values[obs_idx][t].real() << " "
                          << obs_values[obs_idx][t].imag() << std::endl;
                }
                outfile.close();
                std::cout << "FTLM: Observable " << obs_idx << " saved to " << filename << std::endl;
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    return results;
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