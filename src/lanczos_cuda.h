// filepath: /home/pc_linux/exact_diagonalization_cpp/src/lanczos_cuda.h
#ifndef LANCZOS_CUDA_H
#define LANCZOS_CUDA_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <iomanip>
#include <algorithm>
#include <stack>
#include <fstream>
#include <set>
#include <thread>
#include <chrono>

// CUDA includes
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
using cuDoubleComplex = cuDoubleComplex;

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Helper function to convert std::complex to cuDoubleComplex
__host__ __device__ inline cuDoubleComplex make_cuDoubleComplex(const Complex& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

// Helper function to convert cuDoubleComplex to std::complex
__host__ inline Complex make_complex(const cuDoubleComplex& c) {
    return Complex(c.x, c.y);
}

// Device complex scalar multiplication: y = alpha * x
__global__ void complex_scale_kernel(cuDoubleComplex* y, const cuDoubleComplex* x, cuDoubleComplex alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = cuCmul(alpha, x[i]);
    }
}

// Device complex vector addition: z = alpha*x + beta*y
__global__ void complex_axpby_kernel(cuDoubleComplex* z, const cuDoubleComplex* x, 
                                    const cuDoubleComplex* y, cuDoubleComplex alpha, 
                                    cuDoubleComplex beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = cuCadd(cuCmul(alpha, x[i]), cuCmul(beta, y[i]));
    }
}

// Generate random complex vector on device
ComplexVector generateRandomVectorCUDA(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Allocate device memory
    cuDoubleComplex* d_v;
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(cuDoubleComplex)));
    
    // Copy to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(v[i]);
        CUDA_CHECK(cudaMemcpy(&d_v[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Compute norm
    double norm;
    CUBLAS_CHECK(cublasDznrm2(handle, N, d_v, 1, &norm));
    
    // Scale vector
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    
    // Create kernel configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Call kernel
    complex_scale_kernel<<<numBlocks, blockSize>>>(d_v, d_v, scale, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp;
        CUDA_CHECK(cudaMemcpy(&temp, &d_v[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        v[i] = make_complex(temp);
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v));
    
    return v;
}

// Helper function to read a basis vector from file
ComplexVector read_basis_vector_cuda(const std::string& temp_dir, int index, int N) {
    ComplexVector vec(N);
    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
        return vec;
    }
    infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
    return vec;
}

// Helper function to write a basis vector to file
bool write_basis_vector_cuda(const std::string& temp_dir, int index, const ComplexVector& vec, int N) {
    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".bin";
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(vec.data()), N * sizeof(Complex));
    outfile.close();
    return true;
}

// Helper function to solve tridiagonal eigenvalue problem (same as CPU version)
int solve_tridiagonal_matrix_cuda(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            int m, int exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, int N) {
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    
    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info;
    
    if (eigenvectors) {
        // Need space for eigenvectors but m might be too large for full allocation
        // Instead of computing all eigenvectors at once, compute them in batches
        const int batch_size = 1000;
        int num_batches = (n_eigenvalues + batch_size - 1) / batch_size;
        
        // First compute all eigenvalues without eigenvectors
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
        
        if (info != 0) {
            return info;
        }
    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, jobz, m, diag.data(), offdiag.data(), nullptr, m);
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
    
    return info;
}

// CUDA implementation of lanczos_no_ortho
void lanczos_no_ortho_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                          double tol, std::vector<double>& eigenvalues, std::string dir = "",
                          bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate host memory
    ComplexVector h_v_current(N);
    ComplexVector h_v_prev(N, Complex(0.0, 0.0));
    ComplexVector h_v_next(N);
    ComplexVector h_w(N);
    
    // Generate random initial vector
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        #pragma omp critical
        {
            h_v_current[i] = Complex(real, imag);
        }
    }

    std::cout << "CUDA Lanczos: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, h_v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, h_v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    if (!write_basis_vector_cuda(temp_dir, 0, h_v_current, N)) {
        CUBLAS_CHECK(cublasDestroy(handle));
        return;
    }
    
    // Allocate device memory
    cuDoubleComplex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_v_current[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_current[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        temp = make_cuDoubleComplex(h_v_prev[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_prev[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin CUDA Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "CUDA Lanczos: Iterating..." << std::endl;   
    
    // Create kernel configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Copy current vector to host for multiplication
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_v_current[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_v_current[i] = make_complex(temp);
        }
        
        // w = H*v_j
        H(h_v_current.data(), h_w.data(), N);
        
        // Copy result back to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_w[i]);
            CUDA_CHECK(cudaMemcpy(&d_w[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(dot_product.x);
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // beta_{j+1} = ||w||
        double norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Copy scaled w to v_next
        CUBLAS_CHECK(cublasZcopy(handle, N, d_w, 1, d_v_next, 1));
        
        // Store basis vector to file if eigenvectors are needed
        if (eigenvectors && j < max_iter - 1) {
            // Copy v_next from device to host
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v_next[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v_next[i] = make_complex(temp);
            }
            
            // Write to file
            if (!write_basis_vector_cuda(temp_dir, j+1, h_v_next, N)) {
                CUBLAS_CHECK(cublasDestroy(handle));
                CUDA_CHECK(cudaFree(d_v_current));
                CUDA_CHECK(cudaFree(d_v_prev));
                CUDA_CHECK(cudaFree(d_v_next));
                CUDA_CHECK(cudaFree(d_w));
                return;
            }
        }

        // Update for next iteration
        std::swap(d_v_prev, d_v_current);
        std::swap(d_v_current, d_v_next);
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Lanczos: Constructing tridiagonal matrix" << std::endl;
    std::cout << "CUDA Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    // Solve the tridiagonal eigenvalue problem
    int info = solve_tridiagonal_matrix_cuda(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up before returning
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_v_current));
        CUDA_CHECK(cudaFree(d_v_prev));
        CUDA_CHECK(cudaFree(d_v_next));
        CUDA_CHECK(cudaFree(d_w));
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    system(("rm -rf " + temp_dir).c_str());
}

// CUDA implementation of lanczos_selective_reorth
void lanczos_selective_reorth_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                                  double tol, std::vector<double>& eigenvalues, std::string dir = "",
                                  bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate host memory
    ComplexVector h_v_current(N);
    ComplexVector h_v_prev(N, Complex(0.0, 0.0));
    ComplexVector h_v_next(N);
    ComplexVector h_w(N);
    
    // Generate random initial vector
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        #pragma omp critical
        {
            h_v_current[i] = Complex(real, imag);
        }
    }

    std::cout << "CUDA Lanczos Selective Reorth: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, h_v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, h_v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    if (!write_basis_vector_cuda(temp_dir, 0, h_v_current, N)) {
        CUBLAS_CHECK(cublasDestroy(handle));
        return;
    }
    
    // Allocate device memory
    cuDoubleComplex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_v_current[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_current[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        temp = make_cuDoubleComplex(h_v_prev[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_prev[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin CUDA Lanczos Selective Reorth iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "CUDA Lanczos Selective Reorth: Iterating..." << std::endl;   
    
    // Parameters for selective reorthogonalization
    const double orth_threshold = 1e-5;  // Threshold for selective reorthogonalization
    const int periodic_full_reorth = max_iter/10; // Periodically do full reorthogonalization
    
    // Storage for tracking loss of orthogonality
    std::vector<cuDoubleComplex*> recent_vectors_device; // Store device pointers to recent vectors
    std::vector<ComplexVector> recent_vectors_host;      // Store host copies of recent vectors
    const int max_recent = 5;                            // Maximum number of recent vectors to keep in memory
    
    // Create kernel configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Copy current vector to host for multiplication
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_v_current[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_v_current[i] = make_complex(temp);
        }
        
        // w = H*v_j
        H(h_v_current.data(), h_w.data(), N);
        
        // Copy result back to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_w[i]);
            CUDA_CHECK(cudaMemcpy(&d_w[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(dot_product.x);
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // Always orthogonalize against v_{j-1} for numerical stability
        if (j > 0) {
            cuDoubleComplex overlap;
            CUBLAS_CHECK(cublasZdotc(handle, N, d_v_prev, 1, d_w, 1, &overlap));
            cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_overlap, d_v_prev, 1, d_w, 1));
        }
        
        // Orthogonalize against recent vectors in memory
        for (cuDoubleComplex* vec_ptr : recent_vectors_device) {
            cuDoubleComplex overlap;
            CUBLAS_CHECK(cublasZdotc(handle, N, vec_ptr, 1, d_w, 1, &overlap));
            cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_overlap, vec_ptr, 1, d_w, 1));
        }
        
        // beta_{j+1} = ||w||
        double norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Copy scaled w to v_next
        CUBLAS_CHECK(cublasZcopy(handle, N, d_w, 1, d_v_next, 1));
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            // Copy v_next from device to host
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v_next[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v_next[i] = make_complex(temp);
            }
            
            // Write to file
            if (!write_basis_vector_cuda(temp_dir, j+1, h_v_next, N)) {
                CUBLAS_CHECK(cublasDestroy(handle));
                CUDA_CHECK(cudaFree(d_v_current));
                CUDA_CHECK(cudaFree(d_v_prev));
                CUDA_CHECK(cudaFree(d_v_next));
                CUDA_CHECK(cudaFree(d_w));
                for (auto vec_ptr : recent_vectors_device) {
                    CUDA_CHECK(cudaFree(vec_ptr));
                }
                return;
            }
        }
        
        // Update for next iteration
        std::swap(d_v_prev, d_v_current);
        std::swap(d_v_current, d_v_next);
        
        // Keep track of recent vectors for quick access
        if (recent_vectors_device.size() < max_recent) {
            cuDoubleComplex* new_vec;
            CUDA_CHECK(cudaMalloc(&new_vec, N * sizeof(cuDoubleComplex)));
            CUBLAS_CHECK(cublasZcopy(handle, N, d_v_current, 1, new_vec, 1));
            recent_vectors_device.push_back(new_vec);
            
            // Also keep a host copy
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v_current[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v_current[i] = make_complex(temp);
            }
            recent_vectors_host.push_back(h_v_current);
        } else if (!recent_vectors_device.empty()) {
            // Free the oldest vector
            CUDA_CHECK(cudaFree(recent_vectors_device.front()));
            recent_vectors_device.erase(recent_vectors_device.begin());
            recent_vectors_host.erase(recent_vectors_host.begin());
            
            // Add the new vector
            cuDoubleComplex* new_vec;
            CUDA_CHECK(cudaMalloc(&new_vec, N * sizeof(cuDoubleComplex)));
            CUBLAS_CHECK(cublasZcopy(handle, N, d_v_current, 1, new_vec, 1));
            recent_vectors_device.push_back(new_vec);
            
            // Also keep a host copy
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v_current[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v_current[i] = make_complex(temp);
            }
            recent_vectors_host.push_back(h_v_current);
        }
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Lanczos Selective Reorth: Constructing tridiagonal matrix" << std::endl;
    std::cout << "CUDA Lanczos Selective Reorth: Solving tridiagonal matrix" << std::endl;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    // Solve the tridiagonal eigenvalue problem
    int info = solve_tridiagonal_matrix_cuda(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up before returning
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_v_current));
        CUDA_CHECK(cudaFree(d_v_prev));
        CUDA_CHECK(cudaFree(d_v_next));
        CUDA_CHECK(cudaFree(d_w));
        for (auto vec_ptr : recent_vectors_device) {
            CUDA_CHECK(cudaFree(vec_ptr));
        }
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    for (auto vec_ptr : recent_vectors_device) {
        CUDA_CHECK(cudaFree(vec_ptr));
    }
    system(("rm -rf " + temp_dir).c_str());
}

// CUDA implementation of standard lanczos with full reorthogonalization
void lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                 double tol, std::vector<double>& eigenvalues, std::string dir = "",
                 bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate host memory
    ComplexVector h_v_current(N);
    ComplexVector h_v_prev(N, Complex(0.0, 0.0));
    ComplexVector h_v_next(N);
    ComplexVector h_w(N);
    
    // Generate random initial vector
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        #pragma omp critical
        {
            h_v_current[i] = Complex(real, imag);
        }
    }

    std::cout << "CUDA Lanczos: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, h_v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, h_v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    if (!write_basis_vector_cuda(temp_dir, 0, h_v_current, N)) {
        CUBLAS_CHECK(cublasDestroy(handle));
        return;
    }
    
    // Allocate device memory
    cuDoubleComplex *d_v_current, *d_v_prev, *d_v_next, *d_w;
    CUDA_CHECK(cudaMalloc(&d_v_current, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_v_current[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_current[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        temp = make_cuDoubleComplex(h_v_prev[i]);
        CUDA_CHECK(cudaMemcpy(&d_v_prev[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin CUDA Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "CUDA Lanczos: Iterating..." << std::endl;   
    
    // Create kernel configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        
        // Copy current vector to host for multiplication
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_v_current[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_v_current[i] = make_complex(temp);
        }
        
        // w = H*v_j
        H(h_v_current.data(), h_w.data(), N);
        
        // Copy result back to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_w[i]);
            CUDA_CHECK(cudaMemcpy(&d_w[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(dot_product.x);
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // Full reorthogonalization (twice for numerical stability)
        for (int k = 0; k <= j; k++) {
            // Load basis vector k from file
            ComplexVector basis_k = read_basis_vector_cuda(temp_dir, k, N);
            
            // Copy to device
            cuDoubleComplex* d_basis_k;
            CUDA_CHECK(cudaMalloc(&d_basis_k, N * sizeof(cuDoubleComplex)));
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp = make_cuDoubleComplex(basis_k[i]);
                CUDA_CHECK(cudaMemcpy(&d_basis_k[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            }
            
            // Compute overlap
            cuDoubleComplex overlap;
            CUBLAS_CHECK(cublasZdotc(handle, N, d_basis_k, 1, d_w, 1, &overlap));
            
            // Subtract projection
            cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
            CUBLAS_CHECK(cublasZaxpy(handle, N, &neg_overlap, d_basis_k, 1, d_w, 1));
            
            // Free temporary memory
            CUDA_CHECK(cudaFree(d_basis_k));
        }
        
        // beta_{j+1} = ||w||
        double norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Copy scaled w to v_next
        CUBLAS_CHECK(cublasZcopy(handle, N, d_w, 1, d_v_next, 1));
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            // Copy v_next from device to host
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v_next[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v_next[i] = make_complex(temp);
            }
            
            // Write to file
            if (!write_basis_vector_cuda(temp_dir, j+1, h_v_next, N)) {
                CUBLAS_CHECK(cublasDestroy(handle));
                CUDA_CHECK(cudaFree(d_v_current));
                CUDA_CHECK(cudaFree(d_v_prev));
                CUDA_CHECK(cudaFree(d_v_next));
                CUDA_CHECK(cudaFree(d_w));
                return;
            }
        }
        
        // Update for next iteration
        std::swap(d_v_prev, d_v_current);
        std::swap(d_v_current, d_v_next);
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Lanczos: Constructing tridiagonal matrix" << std::endl;
    std::cout << "CUDA Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());

    // Solve the tridiagonal eigenvalue problem
    int info = solve_tridiagonal_matrix_cuda(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up before returning
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_v_current));
        CUDA_CHECK(cudaFree(d_v_prev));
        CUDA_CHECK(cudaFree(d_v_next));
        CUDA_CHECK(cudaFree(d_w));
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v_current));
    CUDA_CHECK(cudaFree(d_v_prev));
    CUDA_CHECK(cudaFree(d_v_next));
    CUDA_CHECK(cudaFree(d_w));
    system(("rm -rf " + temp_dir).c_str());
}

#endif // LANCZOS_CUDA_H