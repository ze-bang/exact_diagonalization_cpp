// test_lanczos_cuda.cu - Test script for CUDA-accelerated Lanczos algorithms
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "lanczos.h"
#include "lanczos_cuda.h"

// Test matrices sizes
const std::vector<int> TEST_SIZES = {100, 500, 1000, 2000};

// Number of eigenvalues to compute
const int NUM_EIGENVALUES = 5;

// Maximum Lanczos iterations
const int MAX_ITER = 200;

// Convergence tolerance
const double TOLERANCE = 1e-10;

// Test directory for storing basis vectors
const std::string TEST_DIR = "./test_vectors";

// Simple random Hermitian Hamiltonian generator
void generate_random_hamiltonian(std::vector<std::vector<Complex>>& H, int N) {
    // Initialize with random values
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    H.resize(N, std::vector<Complex>(N));
    
    for (int i = 0; i < N; i++) {
        // Diagonal elements are real
        H[i][i] = Complex(dist(gen), 0.0);
        
        // Off-diagonal elements ensure Hermitian property: H[j][i] = conj(H[i][j])
        for (int j = i + 1; j < N; j++) {
            H[i][j] = Complex(dist(gen), dist(gen));
            H[j][i] = std::conj(H[i][j]);
        }
    }
}

// Apply Hamiltonian matrix-vector product H|v⟩ -> |w⟩
void apply_hamiltonian_cpu(const Complex* v, Complex* w, int N, const std::vector<std::vector<Complex>>& H) {
    for (int i = 0; i < N; i++) {
        w[i] = Complex(0.0, 0.0);
        for (int j = 0; j < N; j++) {
            w[i] += H[i][j] * v[j];
        }
    }
}

// CUDA kernel for Hamiltonian-vector product
__global__ void hamiltonian_mvp_kernel(const cuDoubleComplex* H, const cuDoubleComplex* v, 
                                       cuDoubleComplex* w, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int j = 0; j < N; j++) {
            sum = cuCadd(sum, cuCmul(H[i * N + j], v[j]));
        }
        w[i] = sum;
    }
}

// Apply Hamiltonian matrix-vector product on GPU
void apply_hamiltonian_gpu(const cuDoubleComplex* v, cuDoubleComplex* w, int N, cuDoubleComplex* d_H) {
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hamiltonian_mvp_kernel<<<numBlocks, blockSize>>>(d_H, v, w, N);
    
    // Check for kernel errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Compare eigenvalues and print results
void compare_eigenvalues(const std::vector<double>& exact_eigs, 
                        const std::vector<double>& approx_eigs, 
                        const std::string& method_name) {
    double max_error = 0.0;
    double avg_error = 0.0;
    
    int n = std::min(exact_eigs.size(), approx_eigs.size());
    
    std::cout << "\nComparison of " << method_name << " with exact diagonalization:" << std::endl;
    std::cout << std::setw(5) << "Index" << std::setw(20) << "Exact" 
              << std::setw(20) << method_name << std::setw(20) << "Error" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    for (int i = 0; i < n; i++) {
        double error = std::abs(exact_eigs[i] - approx_eigs[i]);
        avg_error += error;
        max_error = std::max(max_error, error);
        
        std::cout << std::setw(5) << i 
                  << std::setw(20) << std::scientific << exact_eigs[i]
                  << std::setw(20) << approx_eigs[i]
                  << std::setw(20) << error << std::endl;
    }
    
    avg_error /= n;
    
    std::cout << "\nSummary for " << method_name << ":" << std::endl;
    std::cout << "  Average error: " << std::scientific << avg_error << std::endl;
    std::cout << "  Maximum error: " << std::scientific << max_error << std::endl;
    std::cout << "  Passed: " << (max_error < TOLERANCE ? "YES" : "NO") << std::endl;
}

int main() {
    std::cout << "Starting test of CUDA-accelerated Lanczos algorithms..." << std::endl;
    
    // Test each matrix size
    for (int N : TEST_SIZES) {
        std::cout << "\n===================================================" << std::endl;
        std::cout << "Testing with matrix size N = " << N << std::endl;
        std::cout << "===================================================\n" << std::endl;
        
        // Generate a random Hermitian matrix
        std::vector<std::vector<Complex>> H_matrix;
        generate_random_hamiltonian(H_matrix, N);
        
        // Create CPU Hamiltonian apply function
        auto H_apply_cpu = [&H_matrix](const Complex* v, Complex* w, int size) {
            apply_hamiltonian_cpu(v, w, size, H_matrix);
        };
        
        // Prepare GPU Hamiltonian matrix
        cuDoubleComplex* d_H;
        CHECK_CUDA(cudaMalloc(&d_H, N * N * sizeof(cuDoubleComplex)));
        
        // Copy H_matrix to device in flattened form
        std::vector<cuDoubleComplex> h_H_flat(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                h_H_flat[i * N + j] = toCuComplex(H_matrix[i][j]);
            }
        }
        CHECK_CUDA(cudaMemcpy(d_H, h_H_flat.data(), N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // Create GPU Hamiltonian apply function
        auto H_apply_gpu = [d_H](const cuDoubleComplex* v, cuDoubleComplex* w, int size) {
            apply_hamiltonian_gpu(v, w, size, d_H);
        };
        
        // Vectors to store results
        std::vector<double> exact_eigenvalues;
        std::vector<double> lanczos_eigenvalues;
        std::vector<double> lanczos_no_ortho_eigenvalues;
        std::vector<double> lanczos_selective_eigenvalues;
        
        // 1. Run full diagonalization as reference
        std::cout << "Running full diagonalization..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        full_diagonalization(H_apply_cpu, N, exact_eigenvalues, TEST_DIR, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Full diagonalization completed in " << duration << " ms" << std::endl;
        
        // Keep only the first NUM_EIGENVALUES eigenvalues
        if (exact_eigenvalues.size() > NUM_EIGENVALUES) {
            exact_eigenvalues.resize(NUM_EIGENVALUES);
        }
        
        // 2. Run Lanczos with full reorthogonalization (CPU)
        std::cout << "\nRunning CPU Lanczos with full reorthogonalization..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        lanczos(H_apply_cpu, N, MAX_ITER, NUM_EIGENVALUES, TOLERANCE, lanczos_eigenvalues, TEST_DIR, false);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "CPU Lanczos completed in " << duration << " ms" << std::endl;
        
        // 3. Run CUDA Lanczos with full reorthogonalization
        std::cout << "\nRunning CUDA Lanczos with full reorthogonalization..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        lanczos_cuda(H_apply_gpu, N, MAX_ITER, NUM_EIGENVALUES, TOLERANCE, lanczos_no_ortho_eigenvalues, TEST_DIR, false);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "CUDA Lanczos completed in " << duration << " ms" << std::endl;
        
        // 4. Run CUDA Lanczos with selective reorthogonalization
        std::cout << "\nRunning CUDA Lanczos with selective reorthogonalization..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        lanczos_selective_reorth_cuda(H_apply_gpu, N, MAX_ITER, NUM_EIGENVALUES, TOLERANCE, lanczos_selective_eigenvalues, TEST_DIR, false);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "CUDA Lanczos with selective reorthogonalization completed in " << duration << " ms" << std::endl;
        
        // Compare results
        compare_eigenvalues(exact_eigenvalues, lanczos_eigenvalues, "CPU Lanczos");
        compare_eigenvalues(exact_eigenvalues, lanczos_no_ortho_eigenvalues, "CUDA Lanczos");
        compare_eigenvalues(exact_eigenvalues, lanczos_selective_eigenvalues, "CUDA Lanczos Selective");
        
        // Clean up GPU memory
        CHECK_CUDA(cudaFree(d_H));
    }
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}