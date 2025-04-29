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