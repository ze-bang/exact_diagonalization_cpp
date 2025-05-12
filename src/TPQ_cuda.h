// filepath: /home/pc_linux/exact_diagonalization_cpp/src/TPQ_cuda.h
// TPQ_cuda.h - CUDA implementation of Thermal Pure Quantum state

#ifndef TPQ_CUDA_H
#define TPQ_CUDA_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include "observables.h"
#include "construct_ham.h"
#include "lanczos_cuda.h"

// CUDA includes
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>


/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists_cuda(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        std::string cmd = "mkdir -p " + path;
        return system(cmd.c_str()) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Path exists and is a directory
        return true;
    } else {
        // Path exists but is not a directory
        return false;
    }
}

/**
 * Generate a random normalized vector for TPQ initial state using CUDA
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector_cuda(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate host memory
    ComplexVector h_v(N);
    
    // Generate random values on host
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        h_v[i] = Complex(real, imag);
    }
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex* d_v;
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(cuDoubleComplex)));
    
    // Copy to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_v[i]);
        CUDA_CHECK(cudaMemcpy(&d_v[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Compute norm
    double norm;
    CUBLAS_CHECK(cublasDznrm2(handle, N, d_v, 1, &norm));
    
    // Scale vector
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_v, 1));
    
    // Copy result back to host
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp;
        CUDA_CHECK(cudaMemcpy(&temp, &d_v[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        h_v[i] = make_complex(temp);
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v));
    
    return h_v;
}

/**
 * Calculate energy and variance for a TPQ state using CUDA
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance_cuda(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    int N
) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex *d_v, *d_Hv, *d_H2v;
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Hv, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_H2v, N * sizeof(cuDoubleComplex)));
    
    // Copy v to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(v[i]);
        CUDA_CHECK(cudaMemcpy(&d_v[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Calculate H|v⟩ on host (assuming H is a host function)
    ComplexVector h_Hv(N);
    H(v.data(), h_Hv.data(), N);
    
    // Copy H|v⟩ to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_Hv[i]);
        CUDA_CHECK(cudaMemcpy(&d_Hv[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Calculate energy = ⟨v|H|v⟩ using cuBLAS dot product
    cuDoubleComplex energy_complex;
    CUBLAS_CHECK(cublasZdotc(handle, N, d_v, 1, d_Hv, 1, &energy_complex));
    double energy = energy_complex.x; // Real part of the complex dot product
    
    // Calculate H²|v⟩ on host
    ComplexVector h_H2v(N);
    H(h_Hv.data(), h_H2v.data(), N);
    
    // Copy H²|v⟩ to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_H2v[i]);
        CUDA_CHECK(cudaMemcpy(&d_H2v[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    cuDoubleComplex h2_complex;
    CUBLAS_CHECK(cublasZdotc(handle, N, d_v, 1, d_H2v, 1, &h2_complex));
    double variance = h2_complex.x - energy * energy;
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_Hv));
    CUDA_CHECK(cudaFree(d_H2v));
    
    return {energy, variance};
}

/**
 * CUDA kernel for Taylor expansion terms in time evolution
 */
__global__ void time_evolution_taylor_term_kernel(cuDoubleComplex* result, const cuDoubleComplex* term, 
                                                 double coefficient, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        cuDoubleComplex coef = make_cuDoubleComplex(coefficient, 0.0);
        result[idx] = cuCadd(result[idx], cuCmul(coef, term[idx]));
    }
}

/**
 * Time evolve TPQ state using Taylor expansion of exp(-iH*delta_t) on CUDA
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_state_cuda(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int n_max = 10,
    bool normalize = true
) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex *d_state, *d_result, *d_term, *d_Hterm;
    CUDA_CHECK(cudaMalloc(&d_state, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_term, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Hterm, N * sizeof(cuDoubleComplex)));
    
    // Copy state to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(tpq_state[i]);
        CUDA_CHECK(cudaMemcpy(&d_state[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Copy initial state to term and result for the first term in Taylor series
    CUBLAS_CHECK(cublasZcopy(handle, N, d_state, 1, d_term, 1));
    CUBLAS_CHECK(cublasZcopy(handle, N, d_state, 1, d_result, 1));
    
    // Precompute coefficients for each term in the Taylor series
    std::vector<Complex> coefficients(n_max + 1);
    coefficients[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        coefficients[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    // Configure kernel execution
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Host memory for term calculations
    ComplexVector h_term(N);
    ComplexVector h_Hterm(N);
    
    // Apply Taylor expansion terms
    for (int order = 1; order <= n_max; order++) {
        // Copy current term to host for multiplication
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_term[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_term[i] = make_complex(temp);
        }
        
        // Apply H to the previous term
        H(h_term.data(), h_Hterm.data(), N);
        
        // Copy result back to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_Hterm[i]);
            CUDA_CHECK(cudaMemcpy(&d_Hterm[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Update term = Hterm
        CUBLAS_CHECK(cublasZcopy(handle, N, d_Hterm, 1, d_term, 1));
        
        // Add this term to the result
        double coef_real = coefficients[order].real();
        double coef_imag = coefficients[order].imag();
        cuDoubleComplex coef = make_cuDoubleComplex(coef_real, coef_imag);
        
        CUBLAS_CHECK(cublasZaxpy(handle, N, &coef, d_term, 1, d_result, 1));
    }
    
    // Copy result back to host
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp;
        CUDA_CHECK(cudaMemcpy(&temp, &d_result[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        tpq_state[i] = make_complex(temp);
    }
    
    // Normalize if requested
    if (normalize) {
        double norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_result, 1, &norm));
        
        if (norm > 1e-10) {  // Avoid division by zero
            cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_result, 1));
            
            // Copy normalized result back to host
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_result[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                tpq_state[i] = make_complex(temp);
            }
        }
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_term));
    CUDA_CHECK(cudaFree(d_Hterm));
}

/**
 * Save the current TPQ state to a file
 * 
 * @param tpq_state TPQ state vector to save
 * @param filename Name of the file to save to
 * @return True if successful
 */
bool save_tpq_state_cuda(const ComplexVector& tpq_state, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    size_t size = tpq_state.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tpq_state.data()), size * sizeof(Complex));
    
    out.close();
    return true;
}

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state_cuda(ComplexVector& tpq_state, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    tpq_state.resize(size);
    in.read(reinterpret_cast<char*>(tpq_state.data()), size * sizeof(Complex));
    
    in.close();
    return true;
}

/**
 * Write TPQ data to file
 */
void writeTPQData_cuda(const std::string& filename, double inv_temp, double energy, 
                      double variance, double norm, int step) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(16) << inv_temp << " " << energy << " " 
             << variance << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        file.close();
    }
}

/**
 * Read TPQ data from file
 */
bool readTPQData_cuda(const std::string& filename, int step, double& energy, 
                     double& temp, double& specificHeat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    double inv_temp, e, var, n, doublon;
    int s;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> inv_temp >> e >> var >> n >> doublon >> s)) {
            continue;
        }
        
        if (s == step) {
            energy = e;
            temp = 1.0/inv_temp;
            return true;
        }
    }
    
    return false;
}

/**
 * Calculate Sz and Sz^2 expectations for TPQ state using CUDA
 */
std::pair<Complex, Complex> calculateSzandSz2_cuda(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length
) {
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    Complex Sz_exps = Complex(0.0, 0.0);
    Complex Sz2_exps = Complex(0.0, 0.0);

    // Create S operators for each site
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.emplace_back(num_sites, spin_length, 2, site);
    }
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate device memory
    cuDoubleComplex *d_state, *d_Sz_psi, *d_Sz2_psi;
    CUDA_CHECK(cudaMalloc(&d_state, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Sz_psi, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Sz2_psi, N * sizeof(cuDoubleComplex)));
    
    // Copy state to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(tpq_state[i]);
        CUDA_CHECK(cudaMemcpy(&d_state[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators on host (assuming SingleSiteOperator is host-only)
        std::vector<Complex> Sz_psi = Sz_ops[site].apply(std::vector<Complex>(tpq_state.begin(), tpq_state.end()));
        
        // Copy result to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(Sz_psi[i]);
            CUDA_CHECK(cudaMemcpy(&d_Sz_psi[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Calculate expectation value Sz
        cuDoubleComplex Sz_exp;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_state, 1, d_Sz_psi, 1, &Sz_exp));
        Sz_exps += Complex(Sz_exp.x, Sz_exp.y);
        
        // Calculate Sz^2|psi⟩
        std::vector<Complex> Sz2_psi = Sz_ops[site].apply(Sz_psi);
        
        // Copy to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(Sz2_psi[i]);
            CUDA_CHECK(cudaMemcpy(&d_Sz2_psi[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Calculate expectation value Sz^2
        cuDoubleComplex Sz2_exp;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_state, 1, d_Sz2_psi, 1, &Sz2_exp));
        Sz2_exps += Complex(Sz2_exp.x, Sz2_exp.y);
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_Sz_psi));
    CUDA_CHECK(cudaFree(d_Sz2_psi));
    
    return {Sz_exps/double(num_sites), Sz2_exps/double(num_sites)};
}

/**
 * Estimate the largest eigenvalue (Emax) of the Hamiltonian using power iteration on CUDA
 */
double estimateLargestEigenvalue_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter = 100) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Start with a random vector
    ComplexVector h_v = generateTPQVector_cuda(N, 42); // Fixed seed for reproducibility
    
    // Allocate device memory
    cuDoubleComplex *d_v, *d_Hv;
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Hv, N * sizeof(cuDoubleComplex)));
    
    // Copy initial vector to device
    for (int i = 0; i < N; i++) {
        cuDoubleComplex temp = make_cuDoubleComplex(h_v[i]);
        CUDA_CHECK(cudaMemcpy(&d_v[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    // Power iteration
    double eigenvalue_estimate = 0.0;
    ComplexVector h_Hv(N);
    
    for (int i = 0; i < max_iter; i++) {
        // Copy current vector to host for multiplication
        for (int j = 0; j < N; j++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_v[j], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_v[j] = make_complex(temp);
        }
        
        // Compute Hv = H|v⟩ on host
        H(h_v.data(), h_Hv.data(), N);
        
        // Copy result back to device
        for (int j = 0; j < N; j++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_Hv[j]);
            CUDA_CHECK(cudaMemcpy(&d_Hv[j], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Calculate Rayleigh quotient: v†Hv / v†v
        cuDoubleComplex dot_product;
        CUBLAS_CHECK(cublasZdotc(handle, N, d_v, 1, d_Hv, 1, &dot_product));
        eigenvalue_estimate = dot_product.x; // Real part
        
        // Update v = Hv
        CUBLAS_CHECK(cublasZcopy(handle, N, d_Hv, 1, d_v, 1));
        
        // Normalize v
        double norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_v, 1, &norm));
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
        CUBLAS_CHECK(cublasZscal(handle, N, &scale, d_v, 1));
    }
    
    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_Hv));
    
    return eigenvalue_estimate;
}

/**
 * Standard TPQ (microcanonical) implementation using CUDA
 */
void microcanonical_tpq_cuda(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_spectrum = false,
    double LargeValue = 1e5,
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.1,
    float spin_length = 0.5,
    bool measure_sz = false
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists_cuda(dir);
    }
    
    eigenvalues.clear();
    
    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
        std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
        
        // Initialize output files
        {
            std::ofstream ss_out(ss_file);
            ss_out << "# inv_temp energy variance num doublon step" << std::endl;
            
            std::ofstream norm_out(norm_file);
            norm_out << "# inv_temp norm first_norm step" << std::endl;
            
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag) step" << std::endl;
        }
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector h_v1 = generateTPQVector_cuda(N, seed);
        
        // Allocate device memory
        cuDoubleComplex *d_v1, *d_v0, *d_w;
        CUDA_CHECK(cudaMalloc(&d_v1, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_v0, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(cuDoubleComplex)));
        
        // Copy to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_v1[i]);
            CUDA_CHECK(cudaMemcpy(&d_v1[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Apply hamiltonian to get v0 = H|v1⟩
        ComplexVector h_v0(N);
        H(h_v1.data(), h_v0.data(), N);
        
        // Copy to device
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp = make_cuDoubleComplex(h_v0[i]);
            CUDA_CHECK(cudaMemcpy(&d_v0[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // For each element, compute v0 = LargeValue*v1 - v0
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        
        // Launch custom kernel or calculate on CPU and transfer back
        for (int i = 0; i < N; i++) {
            cuDoubleComplex v1_val, v0_val;
            CUDA_CHECK(cudaMemcpy(&v1_val, &d_v1[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&v0_val, &d_v0[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            
            // v0 = LargeValue*v1 - v0
            cuDoubleComplex result = make_cuDoubleComplex(
                LargeValue * v1_val.x - v0_val.x,
                LargeValue * v1_val.y - v0_val.y
            );
            
            CUDA_CHECK(cudaMemcpy(&d_v0[i], &result, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
        
        // Calculate initial energy and norm
        auto [energy, variance] = calculateEnergyAndVariance_cuda(H, h_v1, N);
        
        // Calculate norm of v1
        double first_norm;
        CUBLAS_CHECK(cublasDznrm2(handle, N, d_v1, 1, &first_norm));
        double current_norm = first_norm;
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        writeTPQData_cuda(ss_file, inv_temp, energy, variance, current_norm, 0);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << 0 << std::endl;
        }
        
        // Step 1: Calculate v0 = H|v1⟩
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance_cuda(H, h_v1, N);
        inv_temp = (2.0) / (LargeValue - energy1);
        
        writeTPQData_cuda(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
            
            if (measure_sz) {
                auto [sz_exp, sz2_exp] = calculateSzandSz2_cuda(h_v1, N, spin_length);
                
                std::ofstream flct_out(flct_file, std::ios::app);
                flct_out << std::setprecision(16) << inv_temp << " " 
                         << sz_exp.real() << " " << sz_exp.imag() << " " 
                         << sz2_exp.real() << " " << sz2_exp.imag() << " " << step << std::endl;
            } else {
                std::ofstream flct_out(flct_file, std::ios::app);
                flct_out << std::setprecision(16) << inv_temp << " " 
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                         << 0.0 << " " << step << std::endl;
            }
        }
        
        // Copy v0 back to host for saving and the next iteration
        ComplexVector h_v0_updated(N);
        for (int i = 0; i < N; i++) {
            cuDoubleComplex temp;
            CUDA_CHECK(cudaMemcpy(&temp, &d_v0[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            h_v0_updated[i] = make_complex(temp);
        }
        
        // Save TPQ state
        save_tpq_state_cuda(h_v1, dir + "/tpq_state_" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat");
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Set v1 = v0 from previous iteration
            std::swap(h_v1, h_v0_updated);
            
            // Copy updated v1 to device
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp = make_cuDoubleComplex(h_v1[i]);
                CUDA_CHECK(cudaMemcpy(&d_v1[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            }
            
            // Calculate H|v1⟩
            H(h_v1.data(), h_v0.data(), N);
            
            // Copy back to device
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp = make_cuDoubleComplex(h_v0[i]);
                CUDA_CHECK(cudaMemcpy(&d_v0[i], &temp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            }
            
            // Calculate v0 = LargeValue*v1 - v0
            for (int i = 0; i < N; i++) {
                cuDoubleComplex v1_val, v0_val;
                CUDA_CHECK(cudaMemcpy(&v1_val, &d_v1[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&v0_val, &d_v0[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                
                cuDoubleComplex result = make_cuDoubleComplex(
                    LargeValue * v1_val.x - v0_val.x,
                    LargeValue * v1_val.y - v0_val.y
                );
                
                CUDA_CHECK(cudaMemcpy(&d_v0[i], &result, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            }
            
            // Copy back to host
            for (int i = 0; i < N; i++) {
                cuDoubleComplex temp;
                CUDA_CHECK(cudaMemcpy(&temp, &d_v0[i], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                h_v0_updated[i] = make_complex(temp);
            }
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance_cuda(H, h_v1, N);
            
            // Calculate norm of v1
            CUBLAS_CHECK(cublasDznrm2(handle, N, d_v1, 1, &current_norm));
            
            // Update inverse temperature
            inv_temp = (2.0 * step) / (LargeValue - energy_step);
            
            // Periodically save data and state
            if (step % temp_interval == 0 || step == max_iter) {
                writeTPQData_cuda(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
                
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
                
                if (measure_sz) {
                    auto [sz_exp, sz2_exp] = calculateSzandSz2_cuda(h_v1, N, spin_length);
                    
                    std::ofstream flct_out(flct_file, std::ios::app);
                    flct_out << std::setprecision(16) << inv_temp << " " 
                             << sz_exp.real() << " " << sz_exp.imag() << " " 
                             << sz2_exp.real() << " " << sz2_exp.imag() << " " << step << std::endl;
                } else {
                    std::ofstream flct_out(flct_file, std::ios::app);
                    flct_out << std::setprecision(16) << inv_temp << " " 
                             << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                             << 0.0 << " " << step << std::endl;
                }
                
                // Save TPQ state
                save_tpq_state_cuda(h_v1, dir + "/tpq_state_" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat");
            }
        }
        
        // Clean up device memory
        CUDA_CHECK(cudaFree(d_v1));
        CUDA_CHECK(cudaFree(d_v0));
        CUDA_CHECK(cudaFree(d_w));
        
        // Store final energy for this sample
        eigenvalues.push_back(energy);
    }
    
    // Clean up cuBLAS
    CUBLAS_CHECK(cublasDestroy(handle));
}

/**
 * Canonical TPQ implementation (imaginary time evolution) using CUDA
 */
void canonical_tpq_cuda(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double delta_tau = 0.0, // Default 0 means use 1/LargeValue
    bool compute_spectrum = false,
    int n_max = 10, // Order of Taylor expansion
    bool compute_observables = false,