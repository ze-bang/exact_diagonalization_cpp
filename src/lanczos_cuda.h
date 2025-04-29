// lanczos_cuda.cu - CUDA implementation of Lanczos algorithms
#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "construct_ham.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << status << std::endl; \
        exit(1); \
    } \
} while(0)

// cuSOLVER error checking macro
#define CUSOLVER_CHECK(call) \
do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << status << std::endl; \
        exit(1); \
    } \
} while(0)

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
using thrust_complex = thrust::complex<double>;

// Wrapper class for Hamiltonian operator in CUDA
class CudaOperator {
private:
    std::function<void(const Complex*, Complex*, int)> cpu_op;
    int dim;

public:
    CudaOperator(std::function<void(const Complex*, Complex*, int)> op, int N) 
        : cpu_op(op), dim(N) {}

    void apply(const thrust_complex* d_in, thrust_complex* d_out) {
        // Allocate temporary CPU memory
        ComplexVector h_in(dim);
        ComplexVector h_out(dim);
        
        // Copy from device to host
        CUDA_CHECK(cudaMemcpy(h_in.data(), d_in, dim * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        
        // Apply CPU operator (reinterpret between std::complex and thrust::complex)
        cpu_op(reinterpret_cast<const Complex*>(h_in.data()), 
               reinterpret_cast<Complex*>(h_out.data()), dim);
        
        // Copy result back to device
        CUDA_CHECK(cudaMemcpy(d_out, h_out.data(), dim * sizeof(thrust_complex), cudaMemcpyHostToDevice));
    }
};

// Generate a random vector on GPU
void generateRandomVectorCUDA(thrust_complex* d_v, int N, std::mt19937& gen, 
                             std::uniform_real_distribution<double>& dist, cublasHandle_t handle) {
    // Create host vector
    ComplexVector h_v(N);
    
    // Generate random values on CPU
    for (int i = 0; i < N; i++) {
        h_v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
    
    // Normalize vector on device
    double norm;
    CUBLAS_CHECK(cublasDznrm2(handle, N, reinterpret_cast<cuDoubleComplex*>(d_v), 1, &norm));
    
    thrust_complex scale(1.0/norm, 0.0);
    CUBLAS_CHECK(cublasZscal(handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                             reinterpret_cast<cuDoubleComplex*>(d_v), 1));
}

// Basic Lanczos algorithm implementation in CUDA
void lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                 double tol, std::vector<double>& eigenvalues, std::string dir = "",
                 bool eigenvectors = false) {
    
    std::cout << "CUDA Lanczos: Starting for matrix of dimension " << N << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir + "/lanczos_cuda_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate device memory
    thrust_complex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Initialize v_prev to zeros
    CUDA_CHECK(cudaMemset(d_v_prev, 0, N * sizeof(thrust_complex)));
    
    // Generate and normalize random starting vector
    generateRandomVectorCUDA(d_v_current, N, gen, dist, cublas_handle);
    
    // Create operator wrapper
    CudaOperator cuda_op(H, N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "CUDA Lanczos: Iterating..." << std::endl;
    
    // Save first basis vector to file
    ComplexVector h_v_current(N);
    CUDA_CHECK(cudaMemcpy(h_v_current.data(), d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
    
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<char*>(h_v_current.data()), N * sizeof(Complex));
        outfile.close();
    } else {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
    }
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // w = H*v_j
        cuda_op.apply(d_v_current, d_w);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            thrust_complex neg_beta(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_beta), 
                         reinterpret_cast<cuDoubleComplex*>(d_v_prev), 1,
                         reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        }
        
        // alpha_j = <v_j, w>
        thrust_complex dot_product;
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                    reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                    reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                    reinterpret_cast<cuDoubleComplex*>(&dot_product)));
        
        alpha.push_back(dot_product.real());
        
        // w = w - alpha_j * v_j
        thrust_complex neg_alpha(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_alpha), 
                     reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // beta_{j+1} = ||w||
        double beta_next;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
        beta.push_back(beta_next);
        
        // Check for invariant subspace
        if (beta_next < tol) {
            std::cout << "CUDA Lanczos: Invariant subspace found at iteration " << j+1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        thrust_complex scale(1.0/beta_next, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // Save basis vector to file
        ComplexVector h_v_next(N);
        CUDA_CHECK(cudaMemcpy(h_v_next.data(), d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        
        std::string basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
        std::ofstream outfile(basis_file, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<char*>(h_v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // v_{j-1} = v_j
        CUDA_CHECK(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        
        // v_j = v_{j+1}
        CUDA_CHECK(cudaMemcpy(d_v_current, d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Lanczos: Solving tridiagonal matrix of size " << m << std::endl;
    
    // Allocate arrays for solver
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    
    // Allocate device memory for solver
    double *d_diag, *d_offdiag, *d_evals;
    CUDA_CHECK(cudaMalloc(&d_diag, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_offdiag, (m-1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Workspace for cuSOLVER
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDstebz_bufferSize(cusolver_handle, m, &lwork));
    
    double *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Solve eigenvalue problem
    CUSOLVER_CHECK(cusolverDnDstebz(cusolver_handle, CUSOLVER_EIG_VALUE_SMALLEST_FIRST, 
                   m, d_diag, d_offdiag, lwork, d_work, d_evals, d_info));
    
    // Copy results back to host
    std::vector<double> evals(m);
    CUDA_CHECK(cudaMemcpy(evals.data(), d_evals, m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(evals.begin(), evals.begin() + n_eigenvalues, eigenvalues.begin());
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/lanczos_cuda_eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
    // Save eigenvalues to a single file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = n_eigenvalues;
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_eigenvalues * sizeof(double));
        eval_outfile.close();
        std::cout << "CUDA Lanczos: Saved " << n_eigenvalues << " eigenvalues to " << eigenvalue_file << std::endl;
    } else {
        std::cerr << "Error: Cannot open file " << eigenvalue_file << " for writing" << std::endl;
    }
    
    // Compute eigenvectors if requested
    if (eigenvectors) {
        std::cout << "CUDA Lanczos: Computing eigenvectors..." << std::endl;
        
        // Use cuSOLVER to compute eigenvectors of the tridiagonal matrix
        double *d_Z;
        CUDA_CHECK(cudaMalloc(&d_Z, m * m * sizeof(double)));
        
        // Recompute eigenvalues and eigenvectors together
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                       CUBLAS_FILL_MODE_LOWER, m, d_diag, m, d_evals, d_Z, m, d_work, lwork, d_info));
        
        // Copy eigenvectors of tridiagonal matrix
        std::vector<double> Z(m * m);
        CUDA_CHECK(cudaMemcpy(Z.data(), d_Z, m * m * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Transform back to original basis
        for (int i = 0; i < n_eigenvalues; i++) {
            // Allocate memory for the eigenvector in the original basis
            thrust_complex *d_evec;
            CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            
            // Read each basis vector and accumulate
            for (int j = 0; j < m; j++) {
                // Read j-th basis vector
                std::string basis_file = temp_dir + "/basis_" + std::to_string(j) + ".bin";
                std::ifstream infile(basis_file, std::ios::binary);
                if (!infile) {
                    std::cerr << "Error: Cannot open file " << basis_file << " for reading" << std::endl;
                    continue;
                }
                
                ComplexVector h_basis(N);
                infile.read(reinterpret_cast<char*>(h_basis.data()), N * sizeof(Complex));
                infile.close();
                
                // Load to device
                thrust_complex *d_basis;
                CUDA_CHECK(cudaMalloc(&d_basis, N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
                
                // Add contribution: evec += Z[j,i] * basis_j
                thrust_complex z_ji(Z[j + i*m], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&z_ji), 
                             reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                             reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
                
                CUDA_CHECK(cudaFree(d_basis));
            }
            
            // Copy eigenvector to host and save to file
            ComplexVector h_evec(N);
            CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
            
            CUDA_CHECK(cudaFree(d_evec));
        }
        
        CUDA_CHECK(cudaFree(d_Z));
    }
    
    // Clean up CUDA resources
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_offdiag));
    CUDA_CHECK(cudaFree(d_evals));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "CUDA Lanczos: Completed successfully" << std::endl;
}

// Block Lanczos algorithm implementation in CUDA
void block_lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                       int num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir = "",
                       bool compute_eigenvectors = false, int block_size = 4) {
    
    std::cout << "CUDA Block Lanczos: Starting with block size " << block_size << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directories for temporary files
    std::string temp_dir = dir + "/block_lanczos_cuda_temp";
    std::string evec_dir = dir + "/block_lanczos_cuda_eigenvectors";
    system(("mkdir -p " + temp_dir).c_str());
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Validate input parameters
    if (block_size <= 0) {
        std::cerr << "Block size must be positive" << std::endl;
        return;
    }
    
    if (max_iter <= 0) {
        std::cerr << "Maximum iterations must be positive" << std::endl;
        return;
    }
    
    // Initialize random generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate device memory for block vectors
    thrust_complex **d_V_curr = new thrust_complex*[block_size];
    thrust_complex **d_V_prev = new thrust_complex*[block_size];
    thrust_complex **d_W = new thrust_complex*[block_size];
    
    for (int i = 0; i < block_size; i++) {
        CUDA_CHECK(cudaMalloc(&d_V_curr[i], N * sizeof(thrust_complex)));
        CUDA_CHECK(cudaMalloc(&d_V_prev[i], N * sizeof(thrust_complex)));
        CUDA_CHECK(cudaMalloc(&d_W[i], N * sizeof(thrust_complex)));
        
        // Initialize previous vectors to zero
        CUDA_CHECK(cudaMemset(d_V_prev[i], 0, N * sizeof(thrust_complex)));
    }
    
    // Generate initial orthogonal block
    for (int i = 0; i < block_size; i++) {
        // Generate random vector
        generateRandomVectorCUDA(d_V_curr[i], N, gen, dist, cublas_handle);
        
        // Orthogonalize against previous vectors in the block
        for (int j = 0; j < i; j++) {
            // Calculate projection: proj = <V_j, V_curr>
            thrust_complex proj;
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_V_curr[j]), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_V_curr[i]), 1, 
                        reinterpret_cast<cuDoubleComplex*>(&proj)));
            
            // V_curr -= proj * V_j
            thrust_complex neg_proj = -proj;
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&neg_proj), 
                        reinterpret_cast<cuDoubleComplex*>(d_V_curr[j]), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_V_curr[i]), 1));
        }
        
        // Normalize
        double norm;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                    reinterpret_cast<cuDoubleComplex*>(d_V_curr[i]), 1, &norm));
        
        thrust_complex scale(1.0/norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                    reinterpret_cast<cuDoubleComplex*>(&scale), 
                    reinterpret_cast<cuDoubleComplex*>(d_V_curr[i]), 1));
        
        // Save to file
        ComplexVector h_vec(N);
        CUDA_CHECK(cudaMemcpy(h_vec.data(), d_V_curr[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        
        std::string vec_file = temp_dir + "/block_0_vec_" + std::to_string(i) + ".bin";
        std::ofstream outfile(vec_file, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<char*>(h_vec.data()), N * sizeof(Complex));
            outfile.close();
        }
    }
    
    // Create operator wrapper
    CudaOperator cuda_op(H, N);
    
    // Storage for block tridiagonal matrix
    std::vector<std::vector<Complex>> Alpha;
    std::vector<std::vector<Complex>> Beta;
    
    // Main Block Lanczos iteration
    std::cout << "CUDA Block Lanczos: Iterating..." << std::endl;
    
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Block iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Apply H to each vector in the current block
        for (int i = 0; i < block_size; i++) {
            cuda_op.apply(d_V_curr[i], d_W[i]);
        }
        
        // Compute Alpha_j = V_j^H * W
        std::vector<Complex> alpha_block(block_size * block_size);
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_V_curr[i]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                alpha_block[i * block_size + k] = Complex(dot.real(), dot.imag());
            }
        }
        Alpha.push_back(alpha_block);
        
        // W = W - V_j * Alpha_j - V_{j-1} * Beta_{j-1}^H
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                // W[i] -= Alpha[j][i,k] * V_curr[k]
                thrust_complex coef(-alpha_block[i * block_size + k].real(), 
                                 -alpha_block[i * block_size + k].imag());
                
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&coef), 
                            reinterpret_cast<cuDoubleComplex*>(d_V_curr[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1));
                
                // Also subtract V_prev contribution if not the first iteration
                if (j > 0) {
                    coef = thrust_complex(-Beta[j-1][k * block_size + i].real(), 
                                       -Beta[j-1][k * block_size + i].imag());
                    
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&coef), 
                                reinterpret_cast<cuDoubleComplex*>(d_V_prev[k]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1));
                }
            }
        }
        
        // QR factorization of W to compute Beta_j and next V block
        std::vector<Complex> beta_block(block_size * block_size);
        
        // Simplified QR for block orthogonalization
        for (int i = 0; i < block_size; i++) {
            // Orthogonalize W[i] against all V_next[j] for j < i
            for (int k = 0; k < i; k++) {
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                beta_block[k * block_size + i] = Complex(dot.real(), dot.imag());
                
                thrust_complex neg_dot = -dot;
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                            reinterpret_cast<cuDoubleComplex*>(d_W[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1));
            }
            
            // Compute norm of W[i]
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1, &norm));
            
            beta_block[i * block_size + i] = Complex(norm, 0.0);
            
            // Check for invariant subspace
            if (norm < tol) {
                std::cout << "CUDA Block Lanczos: Near-invariant subspace found at iteration " 
                          << j+1 << ", vector " << i << std::endl;
                
                // Generate a new random vector
                generateRandomVectorCUDA(d_W[i], N, gen, dist, cublas_handle);
                
                // Orthogonalize against all existing vectors
                for (int k = 0; k < i; k++) {
                    // Orthogonalize against W[k]
                    thrust_complex dot;
                    CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(d_W[k]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(&dot)));
                    
                    thrust_complex neg_dot = -dot;
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                                reinterpret_cast<cuDoubleComplex*>(d_W[k]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1));
                }
                
                // Normalize new vector
                CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1, &norm));
                
                beta_block[i * block_size + i] = Complex(norm, 0.0);
            }
            
            // Normalize W[i]
            thrust_complex scale(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&scale), 
                        reinterpret_cast<cuDoubleComplex*>(d_W[i]), 1));
            
            // Save to file
            ComplexVector h_vec(N);
            CUDA_CHECK(cudaMemcpy(h_vec.data(), d_W[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            std::string vec_file = temp_dir + "/block_" + std::to_string(j+1) + "_vec_" + std::to_string(i) + ".bin";
            std::ofstream outfile(vec_file, std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<char*>(h_vec.data()), N * sizeof(Complex));
                outfile.close();
            }
        }
        
        Beta.push_back(beta_block);
        
        // V_prev = V_curr, V_curr = W
        for (int i = 0; i < block_size; i++) {
            CUDA_CHECK(cudaMemcpy(d_V_prev[i], d_V_curr[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_V_curr[i], d_W[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        }
    }
    
    // Final computation of eigenvalues and eigenvectors
    int total_size = Alpha.size() * block_size;
    std::cout << "CUDA Block Lanczos: Computing final eigenvalues and eigenvectors..." << std::endl;
    
    // Construct the final block tridiagonal matrix
    std::vector<Complex> block_tridiag(total_size * total_size, Complex(0.0, 0.0));
    
    // Fill diagonal blocks (Alpha)
    for (int block_idx = 0; block_idx < Alpha.size(); block_idx++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int row = block_idx * block_size + i;
                int col = block_idx * block_size + j;
                block_tridiag[row * total_size + col] = Alpha[block_idx][i * block_size + j];
            }
        }
    }
    
    // Fill off-diagonal blocks (Beta)
    for (int block_idx = 0; block_idx < Beta.size() - 1; block_idx++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int row = block_idx * block_size + i;
                int col = (block_idx + 1) * block_size + j;
                block_tridiag[row * total_size + col] = Beta[block_idx][i * block_size + j];
                block_tridiag[col * total_size + row] = std::conj(Beta[block_idx][i * block_size + j]);
            }
        }
    }
    
    // Copy to device for cuSOLVER
    cuDoubleComplex *d_block_tridiag;
    double *d_evals;
    CUDA_CHECK(cudaMalloc(&d_block_tridiag, total_size * total_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_evals, total_size * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_block_tridiag, block_tridiag.data(), 
              total_size * total_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Workspace for cuSOLVER
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolver_handle, 
                  CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, 
                  total_size, d_block_tridiag, total_size, d_evals, &lwork));
    
    cuDoubleComplex *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Solve eigenvalue problem
    if (compute_eigenvectors) {
        CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle, 
                      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 
                      total_size, d_block_tridiag, total_size, d_evals, 
                      d_work, lwork, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle, 
                      CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, 
                      total_size, d_block_tridiag, total_size, d_evals, 
                      d_work, lwork, d_info));
    }
    
    // Copy results back to host
    std::vector<double> evals(total_size);
    CUDA_CHECK(cudaMemcpy(evals.data(), d_evals, total_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy eigenvalues to output
    eigenvalues.resize(num_eigs);
    for (int i = 0; i < std::min(num_eigs, total_size); i++) {
        eigenvalues[i] = evals[i];
    }
    
    // Save eigenvalues to file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
        eval_outfile.close();
        std::cout << "CUDA Block Lanczos: Saved " << eigenvalues.size() << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    
    // Compute eigenvectors in original basis if requested
    if (compute_eigenvectors) {
        std::vector<Complex> evecs(total_size * total_size);
        CUDA_CHECK(cudaMemcpy(evecs.data(), d_block_tridiag, 
                  total_size * total_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // Allocate device memory for original basis eigenvectors
        thrust_complex *d_evec;
        CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
        
        // Process only the requested number of eigenvectors
        for (int i = 0; i < std::min(num_eigs, total_size); i++) {
            // Clear eigenvector
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            
            // Reconstruct eigenvector in original basis
            for (int block_idx = 0; block_idx < Alpha.size(); block_idx++) {
                for (int j = 0; j < block_size; j++) {
                    // Index in the block tridiagonal eigenvector
                    int idx = block_idx * block_size + j;
                    thrust_complex coef(evecs[idx + i * total_size].real(), evecs[idx + i * total_size].imag());
                    
                    // Read basis vector
                    std::string vec_file = temp_dir + "/block_" + std::to_string(block_idx) + 
                                          "_vec_" + std::to_string(j) + ".bin";
                    std::ifstream infile(vec_file, std::ios::binary);
                    if (!infile) {
                        std::cerr << "Error: Cannot open file " << vec_file << " for reading" << std::endl;
                        continue;
                    }
                    
                    ComplexVector h_basis(N);
                    infile.read(reinterpret_cast<char*>(h_basis.data()), N * sizeof(Complex));
                    infile.close();
                    
                    // Load to device
                    thrust_complex *d_basis;
                    CUDA_CHECK(cudaMalloc(&d_basis, N * sizeof(thrust_complex)));
                    CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
                    
                    // Add contribution: evec += coef * basis
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&coef), 
                                reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
                    
                    CUDA_CHECK(cudaFree(d_basis));
                }
            }
            
            // Normalize eigenvector
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_evec), 1, &norm));
            
            thrust_complex scale(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&scale), 
                        reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
            
            // Copy to host and save
            ComplexVector h_evec(N);
            CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
        }
        
        CUDA_CHECK(cudaFree(d_evec));
    }
    
    // Clean up CUDA resources
    for (int i = 0; i < block_size; i++) {
        CUDA_CHECK(cudaFree(d_V_curr[i]));
        CUDA_CHECK(cudaFree(d_V_prev[i]));
        CUDA_CHECK(cudaFree(d_W[i]));
    }
    
    delete[] d_V_curr;
    delete[] d_V_prev;
    delete[] d_W;
    
    CUDA_CHECK(cudaFree(d_block_tridiag));
    CUDA_CHECK(cudaFree(d_evals));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "CUDA Block Lanczos: Completed successfully" << std::endl;
}
// Shift-Invert Lanczos algorithm in CUDA
void shift_invert_lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                             int num_eigs, double shift, double tol, std::vector<double>& eigenvalues, 
                             std::string dir = "", bool compute_eigenvectors = false) {
    std::cout << "CUDA Shift-Invert Lanczos: Starting with shift " << shift << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directories for temporary files
    std::string temp_dir = dir + "/shift_invert_lanczos_cuda_temp";
    std::string evec_dir = dir + "/shift_invert_lanczos_cuda_eigenvectors";
    system(("mkdir -p " + temp_dir).c_str());
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Formulate the shifted matrix (H - shift*I)
    std::function<void(const Complex*, Complex*, int)> shifted_h = 
        [&H, shift, N](const Complex* in, Complex* out, int dim) {
            // Apply H
            H(in, out, dim);
            
            // Subtract shift*I
            for (int i = 0; i < dim; i++) {
                out[i] -= shift * in[i];
            }
        };
    
    // Create matrix for shift-invert operator
    std::vector<Complex> matrix(N * N, Complex(0.0, 0.0));
    
    // Form the explicit matrix representation of H - shift*I
    std::vector<Complex> unit_vector(N, Complex(0.0, 0.0));
    std::vector<Complex> result_vector(N);
    
    for (int j = 0; j < N; j++) {
        // Create unit vector e_j
        std::fill(unit_vector.begin(), unit_vector.end(), Complex(0.0, 0.0));
        unit_vector[j] = Complex(1.0, 0.0);
        
        // Apply shifted operator
        shifted_h(unit_vector.data(), result_vector.data(), N);
        
        // Store result in j-th column
        for (int i = 0; i < N; i++) {
            matrix[i * N + j] = result_vector[i];
        }
    }
    
    // Copy matrix to device
    cuDoubleComplex *d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, N * N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Prepare for LU factorization
    int *d_pivots, *d_info;
    CUDA_CHECK(cudaMalloc(&d_pivots, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Calculate workspace size for LU factorization
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(cusolver_handle, N, N, d_matrix, N, &lwork));
    
    cuDoubleComplex *d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    
    // Perform LU factorization
    CUSOLVER_CHECK(cusolverDnZgetrf(cusolver_handle, N, N, d_matrix, N, d_work, d_pivots, d_info));
    
    // Wrap operator for shift-invert Lanczos
    CudaOperator cuda_op(
        [&](const Complex* in, Complex* out, int dim) {
            // Copy input to device
            cuDoubleComplex *d_in, *d_out;
            CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMemcpy(d_in, in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            
            // Solve (H - shift*I) x = in
            CUSOLVER_CHECK(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, N, 1, 
                           d_matrix, N, d_pivots, d_in, N, d_info));
            
            // Copy solution back to host
            CUDA_CHECK(cudaMemcpy(out, d_in, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            
            // Clean up
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
        }, N);
    
    // Allocate device memory
    thrust_complex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Initialize v_prev to zeros
    CUDA_CHECK(cudaMemset(d_v_prev, 0, N * sizeof(thrust_complex)));
    
    // Generate initial random vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    generateRandomVectorCUDA(d_v_current, N, gen, dist, cublas_handle);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    max_iter = std::min(N, max_iter);
    
    // Save first basis vector to file
    ComplexVector h_v_current(N);
    CUDA_CHECK(cudaMemcpy(h_v_current.data(), d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
    
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<char*>(h_v_current.data()), N * sizeof(Complex));
        outfile.close();
    }
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // w = (H - shift*I)^(-1) * v_j
        cuda_op.apply(d_v_current, d_w);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            thrust_complex neg_beta(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_beta), 
                         reinterpret_cast<cuDoubleComplex*>(d_v_prev), 1,
                         reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        }
        
        // alpha_j = <v_j, w>
        thrust_complex dot_product;
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                    reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                    reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                    reinterpret_cast<cuDoubleComplex*>(&dot_product)));
        
        alpha.push_back(dot_product.real());
        
        // w = w - alpha_j * v_j
        thrust_complex neg_alpha(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_alpha), 
                     reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // beta_{j+1} = ||w||
        double beta_next;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
        beta.push_back(beta_next);
        
        // Check for invariant subspace
        if (beta_next < tol) {
            std::cout << "CUDA Shift-Invert Lanczos: Invariant subspace found at iteration " << j+1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        thrust_complex scale(1.0/beta_next, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // Save basis vector to file
        ComplexVector h_v_next(N);
        CUDA_CHECK(cudaMemcpy(h_v_next.data(), d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        
        std::string basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
        std::ofstream outfile(basis_file, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<char*>(h_v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
        
        // v_{j-1} = v_j
        CUDA_CHECK(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        
        // v_j = v_{j+1}
        CUDA_CHECK(cudaMemcpy(d_v_current, d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Shift-Invert Lanczos: Solving tridiagonal matrix of size " << m << std::endl;
    
    // Allocate arrays for solver
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Allocate device memory for solver
    double *d_diag, *d_offdiag, *d_evals;
    CUDA_CHECK(cudaMalloc(&d_diag, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_offdiag, (m-1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Calculate workspace size for eigenvalue solver
    CUSOLVER_CHECK(cusolverDnDstebz_bufferSize(cusolver_handle, m, &lwork));
    
    double *d_work_trid;
    CUDA_CHECK(cudaMalloc(&d_work_trid, lwork * sizeof(double)));
    
    // Solve eigenvalue problem for tridiagonal matrix
    CUSOLVER_CHECK(cusolverDnDstebz(cusolver_handle, CUSOLVER_EIG_VALUE_SMALLEST_FIRST, 
                   m, d_diag, d_offdiag, lwork, d_work_trid, d_evals, d_info));
    
    // Copy results back to host
    std::vector<double> ritz_values(m);
    CUDA_CHECK(cudaMemcpy(ritz_values.data(), d_evals, m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert the eigenvalues back to original problem
    // For shift-invert: lambda = shift + 1/mu, where mu is the eigenvalue of (H - shift*I)^(-1)
    eigenvalues.resize(std::min(num_eigs, m));
    for (int i = 0; i < std::min(num_eigs, m); i++) {
        // Avoid division by zero
        if (std::abs(ritz_values[m-i-1]) > 1e-10) {
            eigenvalues[i] = shift + 1.0/ritz_values[m-i-1];
        } else {
            eigenvalues[i] = shift;
        }
    }
    
    // Save eigenvalues to file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
        eval_outfile.close();
    }
    
    // Compute eigenvectors if requested
    if (compute_eigenvectors) {
        // Allocate memory for eigenvectors
        double *d_Z;
        CUDA_CHECK(cudaMalloc(&d_Z, m * m * sizeof(double)));
        
        // Compute eigenvectors of tridiagonal matrix
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                       CUBLAS_FILL_MODE_LOWER, m, d_diag, m, d_evals, d_Z, m, d_work_trid, lwork, d_info));
        
        // Copy eigenvectors of tridiagonal matrix
        std::vector<double> Z(m * m);
        CUDA_CHECK(cudaMemcpy(Z.data(), d_Z, m * m * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Transform back to original basis for the requested eigenvalues
        for (int i = 0; i < std::min(num_eigs, m); i++) {
            // Use eigenvector for m-i-1 (since we want largest, not smallest)
            int eigen_idx = m - i - 1;
            
            // Allocate memory for eigenvector in original basis
            thrust_complex *d_evec;
            CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            
            // Read each basis vector and accumulate
            for (int j = 0; j < m; j++) {
                // Read j-th basis vector
                std::string basis_file = temp_dir + "/basis_" + std::to_string(j) + ".bin";
                std::ifstream infile(basis_file, std::ios::binary);
                if (!infile) {
                    std::cerr << "Error: Cannot open file " << basis_file << " for reading" << std::endl;
                    continue;
                }
                
                ComplexVector h_basis(N);
                infile.read(reinterpret_cast<char*>(h_basis.data()), N * sizeof(Complex));
                infile.close();
                
                // Load to device
                thrust_complex *d_basis;
                CUDA_CHECK(cudaMalloc(&d_basis, N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
                
                // Add contribution: evec += Z[j,eigen_idx] * basis_j
                thrust_complex z_ji(Z[j + eigen_idx*m], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&z_ji), 
                             reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                             reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
                
                CUDA_CHECK(cudaFree(d_basis));
            }
            
            // Copy eigenvector to host and save to file
            ComplexVector h_evec(N);
            CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
            
            CUDA_CHECK(cudaFree(d_evec));
        }
        
        CUDA_CHECK(cudaFree(d_Z));
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_pivots));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_offdiag));
    CUDA_CHECK(cudaFree(d_evals));
    CUDA_CHECK(cudaFree(d_work_trid));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "CUDA Shift-Invert Lanczos: Completed successfully" << std::endl;
}

// Chebyshev Lanczos algorithm in CUDA
void chebyshev_lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                          int num_eigs, double a, double b, double c, double tol, 
                          std::vector<double>& eigenvalues, std::string dir = "", 
                          bool compute_eigenvectors = false, int cheby_degree = 10) {
    std::cout << "CUDA Chebyshev Lanczos: Starting with spectral range [" << a << ", " << b 
              << "], target " << c << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directories for temporary files
    std::string temp_dir = dir + "/chebyshev_lanczos_cuda_temp";
    std::string evec_dir = dir + "/chebyshev_lanczos_cuda_eigenvectors";
    system(("mkdir -p " + temp_dir).c_str());
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Validate spectral bounds
    if (a >= b) {
        std::cerr << "Error: Invalid spectral range. Must have a < b." << std::endl;
        return;
    }
    
    // Center and half-width of the interval [a,b]
    double center = (a + b) / 2.0;
    double halfwidth = (b - a) / 2.0;
    
    // Create filter operator that applies Chebyshev polynomial p_m(H) to a vector
    CudaOperator filter_op(
        [&H, N, center, halfwidth, cheby_degree, c](const Complex* in, Complex* out, int dim) {
            // Temporary vectors for Chebyshev recurrence
            std::vector<Complex> v_prev(dim);
            std::vector<Complex> v_curr(dim);
            std::vector<Complex> v_next(dim);
            std::vector<Complex> temp(dim);
            
            // Initialize with v_0 = in, v_1 = (H - center*I) / halfwidth * in
            std::copy(in, in + dim, v_prev.data());
            
            // Apply (H - center*I) / halfwidth
            H(in, temp.data(), dim);
            for (int i = 0; i < dim; i++) {
                v_curr[i] = (temp[i] - center * in[i]) / halfwidth;
            }
            
            // Compute Chebyshev polynomial via recurrence
            // T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
            for (int k = 2; k <= cheby_degree; k++) {
                // First, compute v_next = (H - center*I) / halfwidth * v_curr
                H(v_curr.data(), temp.data(), dim);
                for (int i = 0; i < dim; i++) {
                    v_next[i] = (temp[i] - center * v_curr[i]) / halfwidth;
                }
                
                // Then, complete the recurrence: v_next = 2*v_next - v_prev
                for (int i = 0; i < dim; i++) {
                    v_next[i] = 2.0 * v_next[i] - v_prev[i];
                }
                
                // Shift for next iteration
                v_prev.swap(v_curr);
                v_curr.swap(v_next);
            }
            
            // Final result is in v_curr, copy to out
            std::copy(v_curr.begin(), v_curr.end(), out);
        }, N);
    
    // Allocate device memory
    thrust_complex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Initialize v_prev to zeros
    CUDA_CHECK(cudaMemset(d_v_prev, 0, N * sizeof(thrust_complex)));
    
    // Generate initial random vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    generateRandomVectorCUDA(d_v_current, N, gen, dist, cublas_handle);
    
    // Apply Chebyshev filter to initial vector
    filter_op.apply(d_v_current, d_w);
    CUDA_CHECK(cudaMemcpy(d_v_current, d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
    
    // Normalize filtered vector
    double norm;
    CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, &norm));
    thrust_complex scale(1.0/norm, 0.0);
    CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                 reinterpret_cast<cuDoubleComplex*>(d_v_current), 1));
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    max_iter = std::min(N, max_iter);
    
    // Save first basis vector to file
    ComplexVector h_v_current(N);
    CUDA_CHECK(cudaMemcpy(h_v_current.data(), d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
    
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<char*>(h_v_current.data()), N * sizeof(Complex));
        outfile.close();
    }
    
    // Lanczos iteration with original Hamiltonian (not the filtered one)
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // w = H*v_j
        cuda_op.apply(d_v_current, d_w);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            thrust_complex neg_beta(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_beta), 
                         reinterpret_cast<cuDoubleComplex*>(d_v_prev), 1,
                         reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        }
        
        // alpha_j = <v_j, w>
        thrust_complex dot_product;
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                    reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                    reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                    reinterpret_cast<cuDoubleComplex*>(&dot_product)));
        
        alpha.push_back(dot_product.real());
        
        // w = w - alpha_j * v_j
        thrust_complex neg_alpha(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_alpha), 
                     reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // beta_{j+1} = ||w||
        double beta_next;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
        beta.push_back(beta_next);
        
        // Check for invariant subspace
        if (beta_next < tol) {
            std::cout << "CUDA Chebyshev Lanczos: Invariant subspace found at iteration " << j+1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        thrust_complex scale(1.0/beta_next, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                     reinterpret_cast<cuDoubleComplex*>(d_w), 1));
        
        // Apply Chebyshev filter again to improve convergence towards target
        filter_op.apply(d_w, d_v_next);
        
        // Reorthogonalize
        for (int k = 0; k <= j; k++) {
            // Read k-th basis vector
            std::string basis_file = temp_dir + "/basis_" + std::to_string(k) + ".bin";
            std::ifstream infile(basis_file, std::ios::binary);
            if (!infile) {
                std::cerr << "Error: Cannot open file " << basis_file << " for reading" << std::endl;
                continue;
            }
            
            ComplexVector h_basis(N);
            infile.read(reinterpret_cast<char*>(h_basis.data()), N * sizeof(Complex));
            infile.close();
            
            // Load to device
            thrust_complex *d_basis;
            CUDA_CHECK(cudaMalloc(&d_basis, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
            
            // Calculate projection
            thrust_complex dot;
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_v_next), 1, 
                        reinterpret_cast<cuDoubleComplex*>(&dot)));
            
            // Subtract projection
            thrust_complex neg_dot = -dot;
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                         reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                         reinterpret_cast<cuDoubleComplex*>(d_v_next), 1));
            
            CUDA_CHECK(cudaFree(d_basis));
        }
        
        // Normalize
        CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_v_next), 1, &norm));
        scale = thrust_complex(1.0/norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                     reinterpret_cast<cuDoubleComplex*>(d_v_next), 1));
        
        // Save basis vector to file
        ComplexVector h_v_next(N);
        CUDA_CHECK(cudaMemcpy(h_v_next.data(), d_v_next, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        
        std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
        std::ofstream next_outfile(next_basis_file, std::ios::binary);
        if (next_outfile) {
            next_outfile.write(reinterpret_cast<char*>(h_v_next.data()), N * sizeof(Complex));
            next_outfile.close();
        }
        
        // v_{j-1} = v_j
        CUDA_CHECK(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        
        // v_j = v_{j+1}
        CUDA_CHECK(cudaMemcpy(d_v_current, d_v_next, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Chebyshev Lanczos: Solving tridiagonal matrix of size " << m << std::endl;
    
    // Allocate arrays for solver
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Allocate device memory for solver
    double *d_diag, *d_offdiag, *d_evals;
    CUDA_CHECK(cudaMalloc(&d_diag, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_offdiag, (m-1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Calculate workspace size for eigenvalue solver
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDstebz_bufferSize(cusolver_handle, m, &lwork));
    
    double *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Solve eigenvalue problem for tridiagonal matrix
    CUSOLVER_CHECK(cusolverDnDstebz(cusolver_handle, CUSOLVER_EIG_VALUE_SMALLEST_FIRST, 
                   m, d_diag, d_offdiag, lwork, d_work, d_evals, d_info));
    
    // Copy results back to host
    std::vector<double> evals(m);
    CUDA_CHECK(cudaMemcpy(evals.data(), d_evals, m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Sort eigenvalues by distance to target c
    std::vector<std::pair<double, int>> evals_with_index(m);
    for (int i = 0; i < m; i++) {
        evals_with_index[i] = std::make_pair(std::abs(evals[i] - c), i);
    }
    std::sort(evals_with_index.begin(), evals_with_index.end());
    
    // Copy the closest eigenvalues to the output
    eigenvalues.resize(std::min(num_eigs, m));
    for (int i = 0; i < std::min(num_eigs, m); i++) {
        eigenvalues[i] = evals[evals_with_index[i].second];
    }
    
    // Save eigenvalues to file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
        eval_outfile.close();
    }
    
    // Compute eigenvectors if requested
    if (compute_eigenvectors) {
        // Allocate memory for eigenvectors
        double *d_Z;
        CUDA_CHECK(cudaMalloc(&d_Z, m * m * sizeof(double)));
        
        // Compute eigenvectors of tridiagonal matrix
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                       CUBLAS_FILL_MODE_LOWER, m, d_diag, m, d_evals, d_Z, m, d_work, lwork, d_info));
        
        // Copy eigenvectors of tridiagonal matrix
        std::vector<double> Z(m * m);
        CUDA_CHECK(cudaMemcpy(Z.data(), d_Z, m * m * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Transform back to original basis for the requested eigenvalues
        for (int i = 0; i < std::min(num_eigs, m); i++) {
            int eigen_idx = evals_with_index[i].second;
            
            // Allocate memory for eigenvector in original basis
            thrust_complex *d_evec;
            CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            
            // Read each basis vector and accumulate
            for (int j = 0; j < m; j++) {
                // Read j-th basis vector
                std::string basis_file = temp_dir + "/basis_" + std::to_string(j) + ".bin";
                std::ifstream infile(basis_file, std::ios::binary);
                if (!infile) {
                    std::cerr << "Error: Cannot open file " << basis_file << " for reading" << std::endl;
                    continue;
                }
                
                ComplexVector h_basis(N);
                infile.read(reinterpret_cast<char*>(h_basis.data()), N * sizeof(Complex));
                infile.close();
                
                // Load to device
                thrust_complex *d_basis;
                CUDA_CHECK(cudaMalloc(&d_basis, N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), N * sizeof(thrust_complex), cudaMemcpyHostToDevice));
                
                // Add contribution: evec += Z[j,eigen_idx] * basis_j
                thrust_complex z_ji(Z[j + eigen_idx*m], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&z_ji), 
                             reinterpret_cast<cuDoubleComplex*>(d_basis), 1, 
                             reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
                
                CUDA_CHECK(cudaFree(d_basis));
            }
            
            // Copy eigenvector to host and save to file
            ComplexVector h_evec(N);
            CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
            
            CUDA_CHECK(cudaFree(d_evec));
        }
        
        CUDA_CHECK(cudaFree(d_Z));
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_offdiag));
    CUDA_CHECK(cudaFree(d_evals));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    std::cout << "CUDA Chebyshev Lanczos: Completed successfully" << std::endl;
}

// Full diagonalization with CUDA
void full_diagonalization_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, 
                             int num_eigs, std::vector<double>& eigenvalues, std::string dir = "",
                             bool compute_eigenvectors = false) {
    std::cout << "CUDA Full Diagonalization: Starting for matrix of dimension " << N << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directory for eigenvectors
    std::string evec_dir = dir + "/full_diagonalization_cuda_eigenvectors";
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Construct explicit matrix representation of H
    std::vector<Complex> matrix(N * N, Complex(0.0, 0.0));
    std::vector<Complex> unit_vector(N, Complex(0.0, 0.0));
    std::vector<Complex> result_vector(N);
    
    std::cout << "CUDA Full Diagonalization: Constructing explicit matrix..." << std::endl;
    
    for (int j = 0; j < N; j++) {
        // Create unit vector e_j
        std::fill(unit_vector.begin(), unit_vector.end(), Complex(0.0, 0.0));
        unit_vector[j] = Complex(1.0, 0.0);
        
        // Apply operator
        H(unit_vector.data(), result_vector.data(), N);
        
        // Store result in j-th column
        for (int i = 0; i < N; i++) {
            matrix[i * N + j] = result_vector[i];
        }
    }
    
    // Copy matrix to device
    cuDoubleComplex *d_matrix;
    double *d_eigenvalues;
    CUDA_CHECK(cudaMalloc(&d_matrix, N * N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_eigenvalues, N * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Workspace for cuSOLVER
    int lwork = 0;
    cusolverEigMode_t jobz = compute_eigenvectors ? 
                             CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolver_handle, jobz, 
                  CUBLAS_FILL_MODE_LOWER, N, d_matrix, N, d_eigenvalues, &lwork));
    
    cuDoubleComplex *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Solve eigenvalue problem
    std::cout << "CUDA Full Diagonalization: Solving eigenvalue problem..." << std::endl;
    
    CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle, jobz, CUBLAS_FILL_MODE_LOWER, 
                  N, d_matrix, N, d_eigenvalues, d_work, lwork, d_info));
    
    // Check for errors
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "Error: cuSOLVER eigenvalue solver failed with error code " << info << std::endl;
        // Clean up and return
        CUDA_CHECK(cudaFree(d_matrix));
        CUDA_CHECK(cudaFree(d_eigenvalues));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
        return;
    }
    
    // Copy eigenvalues back to host
    std::vector<double> all_eigenvalues(N);
    CUDA_CHECK(cudaMemcpy(all_eigenvalues.data(), d_eigenvalues, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy requested number of eigenvalues to output
    eigenvalues.resize(std::min(num_eigs, N));
    std::copy(all_eigenvalues.begin(), all_eigenvalues.begin() + std::min(num_eigs, N), eigenvalues.begin());
    
    // Save eigenvalues to file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
        eval_outfile.close();
        std::cout << "CUDA Full Diagonalization: Saved " << eigenvalues.size() 
                  << " eigenvalues to " << eigenvalue_file << std::endl;
    }
    
    // Save eigenvectors if requested
    if (compute_eigenvectors) {
        std::cout << "CUDA Full Diagonalization: Saving eigenvectors..." << std::endl;
        
        // Copy eigenvectors from device to host
        std::vector<Complex> eigenvectors(N * N);
        CUDA_CHECK(cudaMemcpy(eigenvectors.data(), d_matrix, N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // Save each requested eigenvector
        for (int i = 0; i < std::min(num_eigs, N); i++) {
            std::vector<Complex> eigenvector(N);
            for (int j = 0; j < N; j++) {
                eigenvector[j] = eigenvectors[j + i * N];
            }
            
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_eigenvalues));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    std::cout << "CUDA Full Diagonalization: Completed successfully" << std::endl;
}


// Krylov-Schur algorithm in CUDA
void krylov_schur_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                     int num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir = "",
                     bool compute_eigenvectors = false, int max_restarts = 10, int krylov_dim = 30) {
    std::cout << "CUDA Krylov-Schur: Starting for matrix of dimension " << N << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directories for temporary files
    std::string temp_dir = dir + "/krylov_schur_cuda_temp";
    std::string evec_dir = dir + "/krylov_schur_cuda_eigenvectors";
    system(("mkdir -p " + temp_dir).c_str());
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Ensure krylov_dim is at least 2*num_eigs for effective restarting
    krylov_dim = std::max(krylov_dim, 2 * num_eigs);
    krylov_dim = std::min(krylov_dim, N);
    
    // Initialize random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate device memory for vectors
    thrust_complex *d_v_current, *d_v_new, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_new, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Create operator wrapper
    CudaOperator cuda_op(H, N);
    
    // Setup storage for Krylov basis vectors
    std::vector<thrust_complex*> d_V(krylov_dim);
    for (int i = 0; i < krylov_dim; i++) {
        CUDA_CHECK(cudaMalloc(&d_V[i], N * sizeof(thrust_complex)));
    }
    
    // Generate initial random vector
    generateRandomVectorCUDA(d_V[0], N, gen, dist, cublas_handle);
    
    // Initialize Hessenberg matrix
    std::vector<Complex> H_matrix(krylov_dim * krylov_dim, Complex(0.0, 0.0));
    
    // Main Krylov-Schur iteration
    bool converged = false;
    int restart_count = 0;
    
    while (!converged && restart_count < max_restarts) {
        std::cout << "CUDA Krylov-Schur: Starting restart " << restart_count + 1 
                  << " of " << max_restarts << std::endl;
        
        // Arnoldi/Lanczos iteration to build Krylov subspace
        int k = 1; // Start from 1 since we already have the first vector
        
        while (k < krylov_dim) {
            // Apply operator: w = H*v
            cuda_op.apply(d_V[k-1], d_w);
            
            // Modified Gram-Schmidt orthogonalization
            for (int j = 0; j < k; j++) {
                // h_jk = <v_j, w>
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                H_matrix[j * krylov_dim + (k-1)] = Complex(dot.real(), dot.imag());
                
                // w = w - h_jk * v_j
                thrust_complex neg_dot = -dot;
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            }
            
            // Compute h_kk
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1, &norm));
            
            H_matrix[k * krylov_dim + (k-1)] = Complex(norm, 0.0);
            
            // Check for invariant subspace
            if (norm < tol) {
                std::cout << "CUDA Krylov-Schur: Invariant subspace found at iteration " << k << std::endl;
                // Generate a new random vector orthogonal to current basis
                generateRandomVectorCUDA(d_v_new, N, gen, dist, cublas_handle);
                
                // Orthogonalize against existing basis
                for (int j = 0; j < k; j++) {
                    thrust_complex dot;
                    CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_v_new), 1, 
                                reinterpret_cast<cuDoubleComplex*>(&dot)));
                    
                    thrust_complex neg_dot = -dot;
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                                reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_v_new), 1));
                }
                
                // Normalize
                CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_v_new), 1, &norm));
                
                thrust_complex scale(1.0/norm, 0.0);
                CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&scale), 
                            reinterpret_cast<cuDoubleComplex*>(d_v_new), 1));
                
                CUDA_CHECK(cudaMemcpy(d_w, d_v_new, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                
                // Reset beta
                norm = 1.0;
            }
            
            // v_{k+1} = w / norm
            if (norm > tol) {
                thrust_complex scale(1.0/norm, 0.0);
                CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&scale), 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1));
                
                CUDA_CHECK(cudaMemcpy(d_V[k], d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                k++;
            }
        }
        
        // Number of actual iterations performed
        int m = std::min(k, krylov_dim);
        
        // Compute Schur decomposition of the Hessenberg matrix
        // Prepare for cuSOLVER
        cuDoubleComplex *d_H, *d_schur;
        double *d_evals;
        CUDA_CHECK(cudaMalloc(&d_H, m * m * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_schur, m * m * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_H, H_matrix.data(), 
                  m * m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // Workspace for cuSOLVER
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZhegvd_bufferSize(cusolver_handle, 
                      CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, 
                      CUBLAS_FILL_MODE_LOWER, m, d_H, m, 
                      d_H, m, d_evals, &lwork));
        
        cuDoubleComplex *d_work;
        int *d_info;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        
        // Compute eigendecomposition
        CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle, 
                      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 
                      m, d_H, m, d_evals, d_work, lwork, d_info));
        
        // Copy results back to host
        std::vector<double> ritz_values(m);
        std::vector<Complex> ritz_vectors(m * m);
        CUDA_CHECK(cudaMemcpy(ritz_values.data(), d_evals, 
                  m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ritz_vectors.data(), d_H, 
                  m * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // Sort Ritz values by magnitude (or other criteria depending on which eigenvalues you want)
        std::vector<std::pair<double, int>> sorted_evals;
        for (int i = 0; i < m; i++) {
            sorted_evals.push_back(std::make_pair(ritz_values[i], i));
        }
        std::sort(sorted_evals.begin(), sorted_evals.end());
        
        // Check for convergence
        double residual_norm = 0.0;
        for (int i = 0; i < num_eigs; i++) {
            int idx = sorted_evals[i].second;
            // Compute residual for this eigenpair: r = H*v - lambda*v
            thrust_complex *d_evec, *d_Hv;
            CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMalloc(&d_Hv, N * sizeof(thrust_complex)));
            
            // Form eigenvector in original basis
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            for (int j = 0; j < m; j++) {
                thrust_complex coef(ritz_vectors[j + idx*m].real(), ritz_vectors[j + idx*m].imag());
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&coef), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
            }
            
            // Apply H to eigenvector
            cuda_op.apply(d_evec, d_Hv);
            
            // Subtract lambda*v: d_Hv -= lambda * d_evec
            thrust_complex neg_lambda(-ritz_values[idx], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&neg_lambda), 
                        reinterpret_cast<cuDoubleComplex*>(d_evec), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_Hv), 1));
            
            // Compute norm of residual
            double res_norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_Hv), 1, &res_norm));
            
            residual_norm = std::max(residual_norm, res_norm);
            
            CUDA_CHECK(cudaFree(d_evec));
            CUDA_CHECK(cudaFree(d_Hv));
        }
        
        // Check if converged
        if (residual_norm < tol) {
            converged = true;
            std::cout << "CUDA Krylov-Schur: Converged with residual norm " << residual_norm << std::endl;
        } else {
            std::cout << "CUDA Krylov-Schur: Residual norm after restart " 
                      << restart_count + 1 << ": " << residual_norm << std::endl;
        }
        
        // Prepare for next restart if not converged
        if (!converged) {
            // Transform basis to Schur basis
            std::vector<thrust_complex*> d_V_new(krylov_dim);
            for (int i = 0; i < krylov_dim; i++) {
                CUDA_CHECK(cudaMalloc(&d_V_new[i], N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemset(d_V_new[i], 0, N * sizeof(thrust_complex)));
            }
            
            // V_new = V * Q
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    thrust_complex q_ji(ritz_vectors[j + i*m].real(), ritz_vectors[j + i*m].imag());
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&q_ji), 
                                reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_V_new[i]), 1));
                }
            }
            
            // Keep only the first num_eigs+1 basis vectors
            int keep = num_eigs + 1;
            keep = std::min(keep, m);
            
            // Free old basis and copy new basis
            for (int i = 0; i < krylov_dim; i++) {
                CUDA_CHECK(cudaFree(d_V[i]));
                if (i < keep) {
                    CUDA_CHECK(cudaMalloc(&d_V[i], N * sizeof(thrust_complex)));
                    CUDA_CHECK(cudaMemcpy(d_V[i], d_V_new[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                }
            }
            
            // Free new temporary basis
            for (int i = 0; i < krylov_dim; i++) {
                CUDA_CHECK(cudaFree(d_V_new[i]));
            }
            
            // Clear Hessenberg matrix
            std::fill(H_matrix.begin(), H_matrix.end(), Complex(0.0, 0.0));
            
            // Update the upper-left block of H with kept eigenvalues
            for (int i = 0; i < keep; i++) {
                H_matrix[i * krylov_dim + i] = Complex(ritz_values[sorted_evals[i].second], 0.0);
            }
            
            // Generate a new random vector for the next Krylov vector
            generateRandomVectorCUDA(d_v_new, N, gen, dist, cublas_handle);
            
            // Orthogonalize against existing basis
            for (int j = 0; j < keep; j++) {
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_v_new), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                thrust_complex neg_dot = -dot;
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_v_new), 1));
            }
            
            // Normalize
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_v_new), 1, &norm));
            
            thrust_complex scale(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&scale), 
                        reinterpret_cast<cuDoubleComplex*>(d_v_new), 1));
            
            // Set as next basis vector
            if (keep < krylov_dim) {
                CUDA_CHECK(cudaMalloc(&d_V[keep], N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemcpy(d_V[keep], d_v_new, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            }
        }
        
        // Free temporary storage
        CUDA_CHECK(cudaFree(d_H));
        CUDA_CHECK(cudaFree(d_schur));
        CUDA_CHECK(cudaFree(d_evals));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
        
        restart_count++;
    }
    
    // Extract final eigenvalues and eigenvectors
    if (converged || restart_count >= max_restarts) {
        // Compute final eigendecomposition
        int m = krylov_dim;
        
        cuDoubleComplex *d_H;
        double *d_evals;
        CUDA_CHECK(cudaMalloc(&d_H, m * m * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_H, H_matrix.data(), 
                  m * m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // Workspace for cuSOLVER
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolver_handle, 
                      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 
                      m, d_H, m, d_evals, &lwork));
        
        cuDoubleComplex *d_work;
        int *d_info;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        
        // Compute eigendecomposition
        CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle, 
                      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 
                      m, d_H, m, d_evals, d_work, lwork, d_info));
        
        // Copy results back to host
        std::vector<double> ritz_values(m);
        std::vector<Complex> ritz_vectors(m * m);
        CUDA_CHECK(cudaMemcpy(ritz_values.data(), d_evals, 
                  m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ritz_vectors.data(), d_H, 
                  m * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // Sort Ritz values
        std::vector<std::pair<double, int>> sorted_evals;
        for (int i = 0; i < m; i++) {
            sorted_evals.push_back(std::make_pair(ritz_values[i], i));
        }
        std::sort(sorted_evals.begin(), sorted_evals.end());
        
        // Copy the num_eigs smallest eigenvalues
        eigenvalues.resize(std::min(num_eigs, m));
        for (int i = 0; i < std::min(num_eigs, m); i++) {
            eigenvalues[i] = sorted_evals[i].first;
        }
        
        // Save eigenvalues to file
        std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
        if (eval_outfile) {
            size_t count = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
            eval_outfile.close();
        }
        
        // Compute eigenvectors if requested
        if (compute_eigenvectors) {
            for (int i = 0; i < std::min(num_eigs, m); i++) {
                int idx = sorted_evals[i].second;
                
                // Allocate memory for eigenvector in original basis
                thrust_complex *d_evec;
                CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
                
                // Form eigenvector in original basis
                for (int j = 0; j < m; j++) {
                    thrust_complex coef(ritz_vectors[j + idx*m].real(), ritz_vectors[j + idx*m].imag());
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&coef), 
                                reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
                }
                
                // Copy eigenvector to host and save to file
                ComplexVector h_evec(N);
                CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), 
                          cudaMemcpyDeviceToHost));
                
                // Save to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream evec_outfile(evec_file, std::ios::binary);
                if (evec_outfile) {
                    evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                    evec_outfile.close();
                }
                
                CUDA_CHECK(cudaFree(d_evec));
            }
        }
        
        // Free resources
        CUDA_CHECK(cudaFree(d_H));
        CUDA_CHECK(cudaFree(d_evals));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
    }
    
    // Clean up resources
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_new));
    CUDA_CHECK(cudaFree(d_w));
    
    for (int i = 0; i < krylov_dim; i++) {
        if (d_V[i] != nullptr) {
            CUDA_CHECK(cudaFree(d_V[i]));
        }
    }
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    std::cout << "CUDA Krylov-Schur: Completed after " << restart_count 
              << " restarts, converged = " << converged << std::endl;
}

// Implicitly Restarted Lanczos algorithm in CUDA
void implicitly_restarted_lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, 
                                     int max_iter, int num_eigs, double tol, 
                                     std::vector<double>& eigenvalues, std::string dir = "",
                                     bool compute_eigenvectors = false, int lanczos_size = 20, 
                                     int max_restarts = 30) {
    std::cout << "CUDA Implicitly Restarted Lanczos: Starting for matrix of dimension " << N << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Create directories for temporary files
    std::string temp_dir = dir + "/irl_cuda_temp";
    std::string evec_dir = dir + "/irl_cuda_eigenvectors";
    system(("mkdir -p " + temp_dir).c_str());
    
    if (compute_eigenvectors) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Ensure lanczos_size is at least num_eigs+2 for effective restarting
    lanczos_size = std::max(lanczos_size, num_eigs + 2);
    lanczos_size = std::min(lanczos_size, N);
    
    // Storage for Lanczos vectors
    std::vector<thrust_complex*> d_V(lanczos_size + 1);  // +1 for the extra vector in recurrence
    for (int i = 0; i < lanczos_size + 1; i++) {
        CUDA_CHECK(cudaMalloc(&d_V[i], N * sizeof(thrust_complex)));
    }
    
    // Storage for tridiagonal matrix
    std::vector<double> alpha(lanczos_size);
    std::vector<double> beta(lanczos_size);
    
    // Create operator wrapper
    CudaOperator cuda_op(H, N);
    
    // Generate initial random vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    generateRandomVectorCUDA(d_V[0], N, gen, dist, cublas_handle);
    
    // Temporary workspace
    thrust_complex *d_w;
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Main IRL iteration
    bool converged = false;
    int restart_count = 0;
    
    while (!converged && restart_count < max_restarts) {
        std::cout << "CUDA IRL: Starting restart " << restart_count + 1 
                  << " of " << max_restarts << std::endl;
        
        // Run Lanczos iteration to build tridiagonal matrix
        int j;
        for (j = 0; j < lanczos_size; j++) {
            // Apply operator: w = H*v_j
            cuda_op.apply(d_V[j], d_w);
            
            // w = w - beta_{j-1} * v_{j-1} (if j > 0)
            if (j > 0) {
                thrust_complex neg_beta(-beta[j-1], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_beta), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j-1]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            }
            
            // alpha_j = <v_j, w>
            thrust_complex dot_product;
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                        reinterpret_cast<cuDoubleComplex*>(&dot_product)));
            
            alpha[j] = dot_product.real();
            
            // w = w - alpha_j * v_j
            thrust_complex neg_alpha(-alpha[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&neg_alpha), 
                        reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            
            // Full reorthogonalization against all previous vectors
            for (int k = 0; k <= j; k++) {
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_V[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                thrust_complex neg_dot = -dot;
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[k]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            }
            
            // beta_{j+1} = ||w||
            double beta_next;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
            
            // Check for invariant subspace
            if (beta_next < tol) {
                std::cout << "CUDA IRL: Invariant subspace found at iteration " << j+1 << std::endl;
                
                // Generate new random vector
                generateRandomVectorCUDA(d_w, N, gen, dist, cublas_handle);
                
                // Orthogonalize against existing vectors
                for (int k = 0; k <= j; k++) {
                    thrust_complex dot;
                    CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(d_V[k]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                                reinterpret_cast<cuDoubleComplex*>(&dot)));
                    
                    thrust_complex neg_dot = -dot;
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                                reinterpret_cast<cuDoubleComplex*>(d_V[k]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_w), 1));
                }
                
                // Normalize
                CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
                
                beta_next = std::max(beta_next, tol); // Ensure non-zero
            }
            
            beta[j] = beta_next;
            
            // v_{j+1} = w / beta_{j+1}
            thrust_complex scale(1.0/beta_next, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&scale), 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            
            CUDA_CHECK(cudaMemcpy(d_V[j+1], d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        }
        
        // Solve eigenvalue problem for tridiagonal matrix
        double *d_diag, *d_offdiag, *d_evals;
        CUDA_CHECK(cudaMalloc(&d_diag, lanczos_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_offdiag, (lanczos_size-1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_evals, lanczos_size * sizeof(double)));
        
        // Copy alpha and beta to device
        CUDA_CHECK(cudaMemcpy(d_diag, alpha.data(), lanczos_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offdiag, beta.data(), (lanczos_size-1) * sizeof(double), cudaMemcpyHostToDevice));
        
        // Workspace for cuSOLVER
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDstebz_bufferSize(cusolver_handle, lanczos_size, &lwork));
        
        double *d_work;
        int *d_info;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        
        // Solve tridiagonal eigenvalue problem
        CUSOLVER_CHECK(cusolverDnDstebz(cusolver_handle, CUSOLVER_EIG_VALUE_SMALLEST_FIRST, 
                       lanczos_size, d_diag, d_offdiag, lwork, d_work, d_evals, d_info));
        
        // Copy eigenvalues to host
        std::vector<double> ritz_values(lanczos_size);
        CUDA_CHECK(cudaMemcpy(ritz_values.data(), d_evals, lanczos_size * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Compute eigenvectors of tridiagonal matrix
        double *d_Z;
        CUDA_CHECK(cudaMalloc(&d_Z, lanczos_size * lanczos_size * sizeof(double)));
        
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                       CUBLAS_FILL_MODE_LOWER, lanczos_size, d_diag, lanczos_size, 
                       d_evals, d_work, lwork, d_info));
        
        CUDA_CHECK(cudaMemcpy(d_Z, d_diag, lanczos_size * lanczos_size * sizeof(double), cudaMemcpyDeviceToDevice));
        
        // Check convergence - compute residual for the wanted eigenvalues
        std::vector<double> Z(lanczos_size * lanczos_size);
        CUDA_CHECK(cudaMemcpy(Z.data(), d_Z, lanczos_size * lanczos_size * sizeof(double), cudaMemcpyDeviceToHost));
        
        double max_residual = 0.0;
        for (int i = 0; i < num_eigs; i++) {
            // Get last component of eigenvector
            double last_component = Z[(lanczos_size-1) + i*lanczos_size];
            double residual = std::abs(beta[lanczos_size-1] * last_component);
            max_residual = std::max(max_residual, residual);
        }
        
        std::cout << "CUDA IRL: Maximum residual after restart " 
                  << restart_count + 1 << ": " << max_residual << std::endl;
        
        if (max_residual < tol) {
            converged = true;
            std::cout << "CUDA IRL: Converged!" << std::endl;
        } else if (restart_count < max_restarts - 1) {
            // Perform implicit restart
            int p = lanczos_size - num_eigs;  // Number of shifts to apply
            
            // Select shifts - use unwanted eigenvalues
            std::vector<double> shifts(p);
            for (int i = 0; i < p; i++) {
                shifts[i] = ritz_values[num_eigs + i];
            }
            
            // Apply shifts using bulge-chasing method
            // (This is an approximation of the full QR algorithm)
            double beta_save = beta[lanczos_size-1];
            
            // Store a copy of the current basis vectors
            std::vector<thrust_complex*> d_V_copy(lanczos_size + 1);
            for (int i = 0; i < lanczos_size + 1; i++) {
                CUDA_CHECK(cudaMalloc(&d_V_copy[i], N * sizeof(thrust_complex)));
                CUDA_CHECK(cudaMemcpy(d_V_copy[i], d_V[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            }
            
            // Apply each shift
            for (int shift_idx = 0; shift_idx < p; shift_idx++) {
                double shift = shifts[shift_idx];
                
                // Initialize algorithm
                std::vector<double> alpha_temp = alpha;
                std::vector<double> beta_temp = beta;
                
                // Apply implicit Q algorithm step
                for (int i = 0; i < lanczos_size - 1; i++) {
                    // Calculate bulge
                    double alpha_i = alpha_temp[i];
                    double beta_i = beta_temp[i];
                    
                    // Construct the 2x2 bulge
                    double a = alpha_i - shift;
                    double b = beta_i;
                    
                    // Compute Givens rotation
                    double c, s;
                    if (std::abs(b) < 1e-14) {
                        c = 1.0;
                        s = 0.0;
                    } else {
                        double t = std::sqrt(a*a + b*b);
                        c = a / t;
                        s = -b / t;
                    }
                    
                    // Apply rotation to bulge
                    double alpha_ip1 = alpha_temp[i+1];
                    double new_alpha_i = c*c*alpha_i + 2*c*s*beta_i + s*s*alpha_ip1;
                    double new_alpha_ip1 = s*s*alpha_i - 2*c*s*beta_i + c*c*alpha_ip1;
                    double new_beta_i = c*s*(alpha_ip1 - alpha_i) + (c*c - s*s)*beta_i;
                    
                    if (i+2 < lanczos_size) {
                        double beta_ip1 = beta_temp[i+1];
                        double new_beta_ip1 = c*beta_ip1;
                        beta_temp[i+1] = new_beta_ip1;
                    }
                    
                    alpha_temp[i] = new_alpha_i;
                    alpha_temp[i+1] = new_alpha_ip1;
                    beta_temp[i] = new_beta_i;
                    
                    // Apply the same rotation to the basis vectors
                    thrust_complex *d_temp;
                    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(thrust_complex)));
                    
                    // temp = c*V[i] - s*V[i+1]
                    CUDA_CHECK(cudaMemcpy(d_temp, d_V_copy[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                    thrust_complex c_val(c, 0.0);
                    thrust_complex s_val(-s, 0.0);
                    CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&c_val), 
                                reinterpret_cast<cuDoubleComplex*>(d_temp), 1));
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&s_val), 
                                reinterpret_cast<cuDoubleComplex*>(d_V_copy[i+1]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_temp), 1));
                    
                    // V[i+1] = s*V[i] + c*V[i+1]
                    s_val = thrust_complex(s, 0.0);
                    CUDA_CHECK(cudaMemcpy(d_w, d_V_copy[i+1], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                    CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&c_val), 
                                reinterpret_cast<cuDoubleComplex*>(d_w), 1));
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                                reinterpret_cast<cuDoubleComplex*>(&s_val), 
                                reinterpret_cast<cuDoubleComplex*>(d_V_copy[i]), 1, 
                                reinterpret_cast<cuDoubleComplex*>(d_w), 1));
                    
                    // Copy back
                    CUDA_CHECK(cudaMemcpy(d_V_copy[i], d_temp, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_V_copy[i+1], d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
                    
                    CUDA_CHECK(cudaFree(d_temp));
                }
                
                // Update alpha and beta
                alpha = alpha_temp;
                beta = beta_temp;
            }
            
            // Restore standard Lanczos form with new subspace
            for (int i = 0; i < num_eigs + 1; i++) {
                CUDA_CHECK(cudaMemcpy(d_V[i], d_V_copy[i], N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            }
            
            // Clean up temporary basis
            for (int i = 0; i < lanczos_size + 1; i++) {
                CUDA_CHECK(cudaFree(d_V_copy[i]));
            }
            
            // Use beta_save as the last component
            beta[num_eigs-1] = beta_save;
            
            // Truncate alpha and beta arrays to keep only num_eigs elements
            alpha.resize(num_eigs);
            beta.resize(num_eigs);
            
            // Update V_{num_eigs} with the new vector
            cuda_op.apply(d_V[num_eigs-1], d_w);
            
            // Orthogonalize against V[0...num_eigs-1]
            for (int i = 0; i < num_eigs; i++) {
                thrust_complex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(d_V[i]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                            reinterpret_cast<cuDoubleComplex*>(&dot)));
                
                thrust_complex neg_dot = -dot;
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&neg_dot), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[i]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            }
            
            // Normalize
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1, &norm));
            
            thrust_complex scale(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(&scale), 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            
            CUDA_CHECK(cudaMemcpy(d_V[num_eigs], d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            
            // Reset lanczos_size to the original value
            lanczos_size = std::min(lanczos_size, N);
            
            // Pad alpha and beta to full length
            alpha.resize(lanczos_size, 0.0);
            beta.resize(lanczos_size, 0.0);
        }
        
        // Free resources
        CUDA_CHECK(cudaFree(d_diag));
        CUDA_CHECK(cudaFree(d_offdiag));
        CUDA_CHECK(cudaFree(d_evals));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_Z));
        
        restart_count++;
    }
    
    // Extract final results
    double *d_diag, *d_offdiag, *d_evals;
    CUDA_CHECK(cudaMalloc(&d_diag, num_eigs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_offdiag, (num_eigs-1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evals, num_eigs * sizeof(double)));
    
    // Copy alpha and beta to device (use only the first num_eigs elements)
    CUDA_CHECK(cudaMemcpy(d_diag, alpha.data(), num_eigs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offdiag, beta.data(), (num_eigs-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Workspace for cuSOLVER
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDstebz_bufferSize(cusolver_handle, num_eigs, &lwork));
    
    double *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Solve final tridiagonal eigenvalue problem
    CUSOLVER_CHECK(cusolverDnDstebz(cusolver_handle, CUSOLVER_EIG_VALUE_SMALLEST_FIRST, 
                   num_eigs, d_diag, d_offdiag, lwork, d_work, d_evals, d_info));
    
    // Copy eigenvalues to host
    eigenvalues.resize(num_eigs);
    CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_evals, num_eigs * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Save eigenvalues to file
    std::string eigenvalue_file = evec_dir + "/eigenvalues.bin";
    std::ofstream eval_outfile(eigenvalue_file, std::ios::binary);
    if (eval_outfile) {
        size_t count = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<char*>(&count), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), eigenvalues.size() * sizeof(double));
        eval_outfile.close();
    }
    
    // Compute eigenvectors if requested
    if (compute_eigenvectors) {
        // Compute eigenvectors of tridiagonal matrix
        double *d_Z;
        CUDA_CHECK(cudaMalloc(&d_Z, num_eigs * num_eigs * sizeof(double)));
        
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                       CUBLAS_FILL_MODE_LOWER, num_eigs, d_diag, num_eigs, 
                       d_evals, d_work, lwork, d_info));
        
        CUDA_CHECK(cudaMemcpy(d_Z, d_diag, num_eigs * num_eigs * sizeof(double), cudaMemcpyDeviceToDevice));
        
        // Copy eigenvectors of tridiagonal matrix
        std::vector<double> Z(num_eigs * num_eigs);
        CUDA_CHECK(cudaMemcpy(Z.data(), d_Z, num_eigs * num_eigs * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Transform back to original basis
        for (int i = 0; i < num_eigs; i++) {
            // Allocate memory for eigenvector in original basis
            thrust_complex *d_evec;
            CUDA_CHECK(cudaMalloc(&d_evec, N * sizeof(thrust_complex)));
            CUDA_CHECK(cudaMemset(d_evec, 0, N * sizeof(thrust_complex)));
            
            // Form eigenvector in original basis
            for (int j = 0; j < num_eigs; j++) {
                thrust_complex z_ji(Z[j + i*num_eigs], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, 
                            reinterpret_cast<cuDoubleComplex*>(&z_ji), 
                            reinterpret_cast<cuDoubleComplex*>(d_V[j]), 1, 
                            reinterpret_cast<cuDoubleComplex*>(d_evec), 1));
            }
            
            // Copy eigenvector to host and save to file
            ComplexVector h_evec(N);
            CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            
            // Save to file
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            if (evec_outfile) {
                evec_outfile.write(reinterpret_cast<char*>(h_evec.data()), N * sizeof(Complex));
                evec_outfile.close();
            }
            
            CUDA_CHECK(cudaFree(d_evec));
        }
        
        CUDA_CHECK(cudaFree(d_Z));
    }
    
    // Clean up resources
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_offdiag));
    CUDA_CHECK(cudaFree(d_evals));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_w));
    
    for (int i = 0; i < lanczos_size + 1; i++) {
        CUDA_CHECK(cudaFree(d_V[i]));
    }
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    std::cout << "CUDA Implicitly Restarted Lanczos: Completed after " << restart_count 
              << " restarts, converged = " << converged << std::endl;
}

// Finite Temperature Lanczos Method in CUDA
void finite_temperature_lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, 
                                    int max_iter, int num_samples, double T_min, double T_max, 
                                    int num_temps, std::vector<double>& temperatures, 
                                    std::vector<double>& free_energy, std::vector<double>& energy,
                                    std::vector<double>& entropy, std::vector<double>& specific_heat, 
                                    std::string dir = "", bool save_observables = false) {
    
    std::cout << "CUDA Finite Temperature Lanczos: Starting for matrix of dimension " << N << std::endl;
    std::cout << "Temperature range: " << T_min << " to " << T_max << " K, " << num_temps << " points" << std::endl;
    std::cout << "Using " << num_samples << " random samples, " << max_iter << " Lanczos iterations per sample" << std::endl;
    
    // Initialize CUDA resources
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Setup output directory
    std::string ftlm_dir = dir + "/finite_temperature_lanczos_cuda";
    system(("mkdir -p " + ftlm_dir).c_str());
    
    // Create temperature array (can be linear or logarithmic spacing)
    temperatures.resize(num_temps);
    if (num_temps == 1) {
        temperatures[0] = T_min;
    } else {
        double T_step = (T_max - T_min) / (num_temps - 1);
        for (int i = 0; i < num_temps; i++) {
            temperatures[i] = T_min + i * T_step;
        }
    }
    
    // Boltzmann constant in appropriate units (adjust as needed)
    const double kB = 1.0; // Using natural units where kB = 1
    
    // Initialize arrays for thermodynamic quantities
    free_energy.resize(num_temps, 0.0);
    energy.resize(num_temps, 0.0);
    entropy.resize(num_temps, 0.0);
    specific_heat.resize(num_temps, 0.0);
    
    // Storage for partition function and energy accumulators
    std::vector<double> Z(num_temps, 0.0);
    std::vector<double> E_accum(num_temps, 0.0);
    std::vector<double> E2_accum(num_temps, 0.0);
    
    // Create operator wrapper
    CudaOperator cuda_op(H, N);
    
    // Random number generator
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate device memory
    thrust_complex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(thrust_complex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(thrust_complex)));
    
    // Main sampling loop
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "CUDA FTLM: Processing sample " << sample + 1 << " of " << num_samples << std::endl;
        
        // Generate random initial state
        generateRandomVectorCUDA(d_v_current, N, gen, dist, cublas_handle);
        
        // Initialize v_prev to zeros
        CUDA_CHECK(cudaMemset(d_v_prev, 0, N * sizeof(thrust_complex)));
        
        // Initialize alpha and beta vectors for tridiagonal matrix
        std::vector<double> alpha;
        std::vector<double> beta;
        beta.push_back(0.0);
        
        // Store Lanczos basis vectors for later reconstruction
        std::vector<ComplexVector> basis_vectors;
        
        // Get initial vector
        ComplexVector v_initial(N);
        CUDA_CHECK(cudaMemcpy(v_initial.data(), d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
        basis_vectors.push_back(v_initial);
        
        // Lanczos iteration
        for (int j = 0; j < max_iter; j++) {
            // w = H*v_j
            cuda_op.apply(d_v_current, d_w);
            
            // w = w - beta_j * v_{j-1}
            if (j > 0) {
                thrust_complex neg_beta(-beta[j], 0.0);
                CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_beta), 
                             reinterpret_cast<cuDoubleComplex*>(d_v_prev), 1,
                             reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            }
            
            // alpha_j = <v_j, w>
            thrust_complex dot_product;
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, 
                        reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                        reinterpret_cast<cuDoubleComplex*>(d_w), 1, 
                        reinterpret_cast<cuDoubleComplex*>(&dot_product)));
            
            alpha.push_back(dot_product.real());
            
            // w = w - alpha_j * v_j
            thrust_complex neg_alpha(-alpha[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&neg_alpha), 
                         reinterpret_cast<cuDoubleComplex*>(d_v_current), 1, 
                         reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            
            // beta_{j+1} = ||w||
            double beta_next;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(d_w), 1, &beta_next));
            beta.push_back(beta_next);
            
            // Check for invariant subspace
            if (beta_next < 1e-10) {
                std::cout << "CUDA FTLM: Invariant subspace found at iteration " << j+1 << std::endl;
                break;
            }
            
            // v_{j+1} = w / beta_{j+1}
            thrust_complex scale(1.0/beta_next, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, reinterpret_cast<cuDoubleComplex*>(&scale), 
                         reinterpret_cast<cuDoubleComplex*>(d_w), 1));
            
            // Store basis vector
            ComplexVector v_next(N);
            CUDA_CHECK(cudaMemcpy(v_next.data(), d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToHost));
            basis_vectors.push_back(v_next);
            
            // v_{j-1} = v_j
            CUDA_CHECK(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
            
            // v_j = v_{j+1}
            CUDA_CHECK(cudaMemcpy(d_v_current, d_w, N * sizeof(thrust_complex), cudaMemcpyDeviceToDevice));
        }
        
        // Size of the Lanczos basis
        int m = alpha.size();
        
        // Construct and solve tridiagonal matrix
        std::cout << "CUDA FTLM: Solving tridiagonal matrix of size " << m << std::endl;
        
        // Allocate arrays for solver
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m-1);
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = beta[i+1];
        }
        
        // Allocate device memory for solver
        double *d_diag, *d_offdiag, *d_evals;
        CUDA_CHECK(cudaMalloc(&d_diag, m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_offdiag, (m-1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), m * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
        
        // Workspace for cuSOLVER
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                      CUBLAS_FILL_MODE_LOWER, m, d_diag, m, d_evals, &lwork));
        
        double *d_work;
        int *d_info;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        
        // Solve eigenvalue problem
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, 
                      CUBLAS_FILL_MODE_LOWER, m, d_diag, m, d_evals, d_work, lwork, d_info));
        
        // Copy results back to host
        std::vector<double> eigenvalues(m);
        std::vector<double> eigenvectors(m*m);
        CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_evals, m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(eigenvectors.data(), d_diag, m*m * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Calculate overlap with initial state: |<r|s>|^2
        // In Lanczos basis, initial state is |v_1> = (1,0,0,...), so we need just the first component
        std::vector<double> overlaps(m);
        for (int i = 0; i < m; i++) {
            overlaps[i] = eigenvectors[i*m] * eigenvectors[i*m]; // Square of first component
        }
        
        // Compute thermal quantities for each temperature
        for (int t = 0; t < num_temps; t++) {
            double T = temperatures[t];
            
            // Skip zero temperature to avoid division by zero
            if (T < 1e-10) continue;
            
            double beta_T = 1.0 / (kB * T);
            
            // Contributions to partition function and energy
            double Z_r = 0.0;
            double E_r = 0.0;
            double E2_r = 0.0;
            
            for (int i = 0; i < m; i++) {
                double E_i = eigenvalues[i];
                double overlap = overlaps[i];
                double boltzmann_factor = std::exp(-beta_T * E_i);
                
                Z_r += overlap * boltzmann_factor;
                E_r += overlap * E_i * boltzmann_factor;
                E2_r += overlap * E_i * E_i * boltzmann_factor;
            }
            
            // Accumulate contributions (will be normalized later)
            Z[t] += Z_r;
            E_accum[t] += E_r;
            E2_accum[t] += E2_r;
        }
        
        // Free device memory for this iteration
        CUDA_CHECK(cudaFree(d_diag));
        CUDA_CHECK(cudaFree(d_offdiag));
        CUDA_CHECK(cudaFree(d_evals));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
    }
    
    // Normalize and compute thermodynamic quantities
    for (int t = 0; t < num_temps; t++) {
        double T = temperatures[t];
        
        // Skip zero temperature
        if (T < 1e-10) continue;
        
        // Normalize by partition function
        double Z_t = Z[t] / num_samples;
        double E_t = E_accum[t] / Z[t];
        double E2_t = E2_accum[t] / Z[t];
        double beta_T = 1.0 / (kB * T);
        
        // Compute thermodynamic quantities
        // Free energy: F = -kB*T*ln(Z)
        free_energy[t] = -kB * T * std::log(Z_t / num_samples);
        
        // Average energy: E = <H> = Sum(E_i * e^(-beta*E_i)) / Z
        energy[t] = E_t;
        
        // Entropy: S = (E - F) / T
        entropy[t] = (energy[t] - free_energy[t]) / T;
        
        // Specific heat: C = (d<E>/dT) = kB*beta^2 * (<E^2> - <E>^2)
        specific_heat[t] = kB * beta_T * beta_T * (E2_t - E_t*E_t);
    }
    
    // Save results to file
    if (save_observables) {
        std::string results_file = ftlm_dir + "/thermodynamic_quantities.txt";
        std::ofstream outfile(results_file);
        
        if (outfile) {
            outfile << "# Temperature | Free Energy | Energy | Entropy | Specific Heat\n";
            for (int t = 0; t < num_temps; t++) {
                outfile << temperatures[t] << " " 
                        << free_energy[t] << " " 
                        << energy[t] << " " 
                        << entropy[t] << " " 
                        << specific_heat[t] << "\n";
            }
            outfile.close();
            std::cout << "CUDA FTLM: Saved thermodynamic quantities to " << results_file << std::endl;
        }
    }
    
    // Clean up CUDA resources
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    std::cout << "CUDA Finite Temperature Lanczos: Completed successfully" << std::endl;
}
