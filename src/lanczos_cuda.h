// filepath: /home/zb/exact_diagonalization_cpp/src/lanczos_cuda.h
#ifndef LANCZOS_CUDA_H
#define LANCZOS_CUDA_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <set>
#include <stack>
#include <chrono>
#include <thread>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// CUDA error checking macro
#define CHECK_CUDA(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << \
        cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUBLAS error checking macro
#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " << \
        status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Convert std::complex to cuComplex
inline cuDoubleComplex toCuComplex(const Complex& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

// Convert cuComplex to std::complex
inline Complex fromCuComplex(const cuDoubleComplex& z) {
    return Complex(cuCreal(z), cuCimag(z));
}

// Helper function to allocate GPU memory for complex vectors
cuDoubleComplex* allocateDeviceComplex(int size) {
    cuDoubleComplex* d_ptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, size * sizeof(cuDoubleComplex)));
    return d_ptr;
}

// Helper function to copy host vector to device
void copyToDevice(const ComplexVector& h_vec, cuDoubleComplex* d_vec, int size) {
    std::vector<cuDoubleComplex> temp(size);
    for (int i = 0; i < size; i++) {
        temp[i] = toCuComplex(h_vec[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_vec, temp.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

// Helper function to copy device vector to host
void copyToHost(cuDoubleComplex* d_vec, ComplexVector& h_vec, int size) {
    std::vector<cuDoubleComplex> temp(size);
    CHECK_CUDA(cudaMemcpy(temp.data(), d_vec, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
        h_vec[i] = fromCuComplex(temp[i]);
    }
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

// Helper function to solve tridiagonal eigenvalue problem using cuSOLVER
int solve_tridiagonal_matrix_cuda(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            int m, int exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, int N) {
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    
    // Allocate arrays for diagonalization
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Allocate device memory
    double *d_diag, *d_offdiag, *d_eigvals, *d_eigvecs = nullptr;
    int *d_info;
    
    CHECK_CUDA(cudaMalloc(&d_diag, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_offdiag, (m-1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_eigvals, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    
    if (eigenvectors) {
        CHECK_CUDA(cudaMalloc(&d_eigvecs, m * m * sizeof(double)));
    }
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_diag, diag.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    
    // Query working space
    int lwork = 0;
    if (eigenvectors) {
        cusolverDnDstevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, m, d_diag, d_offdiag, d_eigvals, d_eigvecs, m, &lwork);
    } else {
        cusolverDnDstevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, m, d_diag, d_offdiag, d_eigvals, nullptr, m, &lwork);
    }
    
    double *d_work;
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
    
    // Solve eigenvalue problem
    if (eigenvectors) {
        cusolverDnDstevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, m, d_diag, d_offdiag, d_eigvals, d_eigvecs, m, d_work, lwork, d_info);
    } else {
        cusolverDnDstevd(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, m, d_diag, d_offdiag, d_eigvals, nullptr, m, d_work, lwork, d_info);
    }
    
    // Check for errors
    int info_h = 0;
    CHECK_CUDA(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (info_h != 0) {
        std::cerr << "cuSolverDn Dstevd failed with error code " << info_h << std::endl;
        
        // Clean up
        cusolverDnDestroy(cusolverH);
        CHECK_CUDA(cudaFree(d_diag));
        CHECK_CUDA(cudaFree(d_offdiag));
        CHECK_CUDA(cudaFree(d_eigvals));
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(d_info));
        if (eigenvectors) CHECK_CUDA(cudaFree(d_eigvecs));
        
        return info_h;
    }
    
    // Copy eigenvalues back to host
    std::vector<double> all_eigenvalues(m);
    CHECK_CUDA(cudaMemcpy(all_eigenvalues.data(), d_eigvals, m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy the required number of eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(all_eigenvalues.begin(), all_eigenvalues.begin() + n_eigenvalues, eigenvalues.begin());
    
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
    
    // If eigenvectors are requested, save them to files
    if (eigenvectors) {
        std::vector<double> all_eigvecs(m * m);
        CHECK_CUDA(cudaMemcpy(all_eigvecs.data(), d_eigvecs, m * m * sizeof(double), cudaMemcpyDeviceToHost));
        
        // TODO: Add code to handle eigenvectors construction from basis vectors
    }
    
    // Clean up
    cusolverDnDestroy(cusolverH);
    CHECK_CUDA(cudaFree(d_diag));
    CHECK_CUDA(cudaFree(d_offdiag));
    CHECK_CUDA(cudaFree(d_eigvals));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    if (eigenvectors) CHECK_CUDA(cudaFree(d_eigvecs));
    
    return 0;
}

// Generate a random complex vector on GPU
ComplexVector generateRandomVector_cuda(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector h_v(N);
    
    // Generate random values on host
    for (int i = 0; i < N; i++) {
        h_v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_v = allocateDeviceComplex(N);
    
    // Copy to device
    copyToDevice(h_v, d_v, N);
    
    // Compute norm on device
    double norm;
    CHECK_CUBLAS(cublasDznrm2(handle, N, d_v, 1, &norm));
    
    // Scale the vector
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_v, 1));
    
    // Copy result back to host
    copyToHost(d_v, h_v, N);
    
    // Clean up
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return h_v;
}

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector_cuda(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector result(N);
    
    // Generate a random vector
    result = generateRandomVector_cuda(N, gen, dist);
    
    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_result = allocateDeviceComplex(N);
    cuDoubleComplex* d_v = allocateDeviceComplex(N);
    
    // Copy result to device
    copyToDevice(result, d_result, N);
    
    // Orthogonalize against all provided vectors using Gram-Schmidt
    for (const auto& v : vectors) {
        // Copy v to device
        copyToDevice(v, d_v, N);
        
        // Calculate projection: <v, result>
        cuDoubleComplex projection;
        CHECK_CUBLAS(cublasZdotc(handle, N, d_v, 1, d_result, 1, &projection));
        
        // Subtract projection: result -= projection * v
        cuDoubleComplex neg_projection = make_cuDoubleComplex(-cuCreal(projection), -cuCimag(projection));
        CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_projection, d_v, 1, d_result, 1));
    }
    
    // Get normalized result
    double norm;
    CHECK_CUBLAS(cublasDznrm2(handle, N, d_result, 1, &norm));
    
    // Scale the vector
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_result, 1));
    
    // Copy result back to host
    copyToHost(d_result, result, N);
    
    // Clean up
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return result;
}

// Lanczos algorithm implementation with CUDA, without reorthogonalization
void lanczos_no_ortho_cuda(std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)> H_cuda,
                         int N, int max_iter, int exct, double tol, std::vector<double>& eigenvalues, 
                         std::string dir = "", bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v_current[i] = Complex(real, imag);
    }

    std::cout << "CUDA Lanczos: Initial vector generated" << std::endl;
    
    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_v_current = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_prev = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_next = allocateDeviceComplex(N);
    cuDoubleComplex* d_w = allocateDeviceComplex(N);
    
    // Initialize d_v_prev with zeros
    CHECK_CUDA(cudaMemset(d_v_prev, 0, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device and normalize
    copyToDevice(v_current, d_v_current, N);
    
    double norm;
    CHECK_CUBLAS(cublasDznrm2(handle, N, d_v_current, 1, &norm));
    
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_v_current, 1));
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_cuda_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    copyToHost(d_v_current, v_current, N);
    if (!write_basis_vector_cuda(temp_dir, 0, v_current, N)) {
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_v_current));
        CHECK_CUDA(cudaFree(d_v_prev));
        CHECK_CUDA(cudaFree(d_v_next));
        CHECK_CUDA(cudaFree(d_w));
        return;
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
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H_cuda(d_v_current, d_w, N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CHECK_CUBLAS(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(cuCreal(dot_product));
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // beta_{j+1} = ||w||
        CHECK_CUBLAS(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Store basis vector to file if eigenvectors are needed
        if (eigenvectors && j < max_iter - 1) {
            copyToHost(d_w, v_current, N);
            if (!write_basis_vector_cuda(temp_dir, j+1, v_current, N)) {
                break;
            }
        }

        // Update for next iteration: v_prev = v_current, v_current = v_next
        CHECK_CUDA(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_v_current, d_w, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
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
        // Clean up temporary files before returning
        system(("rm -rf " + temp_dir).c_str());
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_v_current));
        CHECK_CUDA(cudaFree(d_v_prev));
        CHECK_CUDA(cudaFree(d_v_next));
        CHECK_CUDA(cudaFree(d_w));
        return;
    }
    
    // Clean up temporary files and CUDA resources
    system(("rm -rf " + temp_dir).c_str());
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_v_current));
    CHECK_CUDA(cudaFree(d_v_prev));
    CHECK_CUDA(cudaFree(d_v_next));
    CHECK_CUDA(cudaFree(d_w));
}

// Lanczos algorithm with selective reorthogonalization using CUDA
void lanczos_selective_reorth_cuda(std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)> H_cuda,
                                 int N, int max_iter, int exct, double tol, std::vector<double>& eigenvalues, 
                                 std::string dir = "", bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v_current[i] = Complex(real, imag);
    }
    
    std::cout << "CUDA Lanczos Selective Reorth: Initial vector generated" << std::endl;
    
    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_v_current = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_prev = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_next = allocateDeviceComplex(N);
    cuDoubleComplex* d_w = allocateDeviceComplex(N);
    
    // Initialize d_v_prev with zeros
    CHECK_CUDA(cudaMemset(d_v_prev, 0, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device and normalize
    copyToDevice(v_current, d_v_current, N);
    
    double norm;
    CHECK_CUBLAS(cublasDznrm2(handle, N, d_v_current, 1, &norm));
    
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_v_current, 1));
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_cuda_selective_reorth";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());
    
    // Write the first basis vector to file
    copyToHost(d_v_current, v_current, N);
    if (!write_basis_vector_cuda(temp_dir, 0, v_current, N)) {
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_v_current));
        CHECK_CUDA(cudaFree(d_v_prev));
        CHECK_CUDA(cudaFree(d_v_next));
        CHECK_CUDA(cudaFree(d_w));
        return;
    }
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin CUDA Lanczos Selective Reorth iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    
    // Parameters for selective reorthogonalization
    const double orth_threshold = 1e-5;  // Threshold for selective reorthogonalization
    const int periodic_full_reorth = max_iter/10; // Periodically do full reorthogonalization
    
    // Storage for tracking recent vectors directly on GPU
    const int max_recent = 5;
    std::vector<cuDoubleComplex*> recent_vectors;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H_cuda(d_v_current, d_w, N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CHECK_CUBLAS(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(cuCreal(dot_product));
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // Always orthogonalize against v_{j-1} for numerical stability
        if (j > 0) {
            cuDoubleComplex overlap;
            CHECK_CUBLAS(cublasZdotc(handle, N, d_v_prev, 1, d_w, 1, &overlap));
            
            cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
            CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_overlap, d_v_prev, 1, d_w, 1));
        }
        
        // Orthogonalize against recent vectors stored in GPU memory
        for (auto d_vec : recent_vectors) {
            cuDoubleComplex overlap;
            CHECK_CUBLAS(cublasZdotc(handle, N, d_vec, 1, d_w, 1, &overlap));
            
            if (std::abs(fromCuComplex(overlap)) > orth_threshold) {
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
                CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_overlap, d_vec, 1, d_w, 1));
            }
        }
        
        // beta_{j+1} = ||w||
        CHECK_CUBLAS(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            copyToHost(d_w, v_current, N);
            if (!write_basis_vector_cuda(temp_dir, j+1, v_current, N)) {
                break;
            }
        }
        
        // Update for next iteration
        CHECK_CUDA(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_v_current, d_w, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        
        // Keep track of recent vectors for quick access
        if (recent_vectors.size() < max_recent) {
            // Allocate a new GPU vector and copy current vector to it
            cuDoubleComplex* d_recent = allocateDeviceComplex(N);
            CHECK_CUDA(cudaMemcpy(d_recent, d_v_current, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
            recent_vectors.push_back(d_recent);
        } else if (recent_vectors.size() > 0) {
            // Recycle the oldest vector
            cuDoubleComplex* d_oldest = recent_vectors.front();
            recent_vectors.erase(recent_vectors.begin());
            CHECK_CUDA(cudaMemcpy(d_oldest, d_v_current, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
            recent_vectors.push_back(d_oldest);
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
    }
    
    // Clean up GPU memory for recent vectors
    for (auto d_vec : recent_vectors) {
        CHECK_CUDA(cudaFree(d_vec));
    }
    
    // Clean up temporary files and CUDA resources
    system(("rm -rf " + temp_dir).c_str());
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_v_current));
    CHECK_CUDA(cudaFree(d_v_prev));
    CHECK_CUDA(cudaFree(d_v_next));
    CHECK_CUDA(cudaFree(d_w));
}

// Full Lanczos algorithm with complete reorthogonalization using CUDA
void lanczos_cuda(std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)> H_cuda, 
                int N, int max_iter, int exct, double tol, std::vector<double>& eigenvalues, 
                std::string dir = "", bool eigenvectors = false) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v_current[i] = Complex(real, imag);
    }
    
    std::cout << "CUDA Lanczos Full Reorth: Initial vector generated" << std::endl;
    
    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_v_current = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_prev = allocateDeviceComplex(N);
    cuDoubleComplex* d_v_next = allocateDeviceComplex(N);
    cuDoubleComplex* d_w = allocateDeviceComplex(N);
    
    // Initialize d_v_prev with zeros
    CHECK_CUDA(cudaMemset(d_v_prev, 0, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device and normalize
    copyToDevice(v_current, d_v_current, N);
    
    double norm;
    CHECK_CUBLAS(cublasDznrm2(handle, N, d_v_current, 1, &norm));
    
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_v_current, 1));
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir+"/lanczos_cuda_full_reorth";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());
    
    // Write the first basis vector to file
    copyToHost(d_v_current, v_current, N);
    if (!write_basis_vector_cuda(temp_dir, 0, v_current, N)) {
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_v_current));
        CHECK_CUDA(cudaFree(d_v_prev));
        CHECK_CUDA(cudaFree(d_v_next));
        CHECK_CUDA(cudaFree(d_w));
        return;
    }
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin CUDA Lanczos Full Reorth iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "CUDA Lanczos Full Reorth: Iterating..." << std::endl;
    
    // Vector to store previous basis vectors on the GPU
    std::vector<cuDoubleComplex*> basis_vectors;
    cuDoubleComplex* d_basis_k = allocateDeviceComplex(N);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H_cuda(d_v_current, d_w, N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_beta, d_v_prev, 1, d_w, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CHECK_CUBLAS(cublasZdotc(handle, N, d_v_current, 1, d_w, 1, &dot_product));
        alpha.push_back(cuCreal(dot_product));
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_alpha, d_v_current, 1, d_w, 1));
        
        // Full reorthogonalization (twice for numerical stability)
        for (int iter = 0; iter < 2; iter++) {
            for (int k = 0; k <= j; k++) {
                // Read basis vector k from file
                ComplexVector basis_k = read_basis_vector_cuda(temp_dir, k, N);
                copyToDevice(basis_k, d_basis_k, N);
                
                // Calculate <v_k, w>
                cuDoubleComplex overlap;
                CHECK_CUBLAS(cublasZdotc(handle, N, d_basis_k, 1, d_w, 1, &overlap));
                
                // w = w - overlap * v_k
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
                CHECK_CUBLAS(cublasZaxpy(handle, N, &neg_overlap, d_basis_k, 1, d_w, 1));
            }
        }
        
        // beta_{j+1} = ||w||
        CHECK_CUBLAS(cublasDznrm2(handle, N, d_w, 1, &norm));
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        scale = make_cuDoubleComplex(1.0/beta[j+1], 0.0);
        CHECK_CUBLAS(cublasZscal(handle, N, &scale, d_w, 1));
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            copyToHost(d_w, v_current, N);
            if (!write_basis_vector_cuda(temp_dir, j+1, v_current, N)) {
                break;
            }
        }
        
        // Update for next iteration
        CHECK_CUDA(cudaMemcpy(d_v_prev, d_v_current, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_v_current, d_w, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "CUDA Lanczos Full Reorth: Constructing tridiagonal matrix" << std::endl;
    std::cout << "CUDA Lanczos Full Reorth: Solving tridiagonal matrix" << std::endl;
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd_mkdir = "mkdir -p " + evec_dir;
    system(cmd_mkdir.c_str());
    
    // Solve the tridiagonal eigenvalue problem
    int info = solve_tridiagonal_matrix_cuda(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
    }
    
    // Clean up temporary files and CUDA resources
    system(("rm -rf " + temp_dir).c_str());
    CHECK_CUDA(cudaFree(d_basis_k));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_v_current));
    CHECK_CUDA(cudaFree(d_v_prev));
    CHECK_CUDA(cudaFree(d_v_next));
    CHECK_CUDA(cudaFree(d_w));
}

#endif // LANCZOS_CUDA_H