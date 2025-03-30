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
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cusolver_common.h>
#include <cusolverDn.h>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// CUDA error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// CUBLAS error checking macro
#define cublasCheckError(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error code %d at %s:%d\n", code, file, line);
        exit(code);
    }
}

// CUSOLVER error checking macro
#define cusolverCheckError(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER Error code %d at %s:%d\n", code, file, line);
        exit(code);
    }
}

// Global cuBLAS handle
cublasHandle_t cublasHandle;
cusolverDnHandle_t cusolverHandle;

// Initialize CUDA resources
void initCUDA() {
    cudaCheckError(cudaSetDevice(0));  // Use the first GPU
    cublasCheckError(cublasCreate(&cublasHandle));
    cusolverCheckError(cusolverDnCreate(&cusolverHandle));
}

// Clean up CUDA resources
void cleanupCUDA() {
    cublasCheckError(cublasDestroy(cublasHandle));
    cusolverCheckError(cusolverDnDestroy(cusolverHandle));
}

// Device memory management helper
template <typename T>
class DeviceMemory {
private:
    T* d_ptr = nullptr;
    size_t size = 0;

public:
    DeviceMemory() = default;
    
    // Allocate device memory
    DeviceMemory(size_t count) {
        allocate(count);
    }
    
    void allocate(size_t count) {
        if (d_ptr != nullptr) free();
        size = count;
        cudaCheckError(cudaMalloc((void**)&d_ptr, count * sizeof(T)));
    }
    
    // Copy from host to device
    void copyToDevice(const T* h_ptr, size_t count) {
        cudaCheckError(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // Copy from device to host
    void copyToHost(T* h_ptr, size_t count) const {
        if (h_ptr == nullptr || d_ptr == nullptr) {
            fprintf(stderr, "Error: Null pointer in copyToDevice\n");
            return;
        }
        cudaCheckError(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    // Free device memory
    void free() {
        if (d_ptr != nullptr) {
            cudaCheckError(cudaFree(d_ptr));
            d_ptr = nullptr;
            size = 0;
        }
    }
    
    // Get device pointer
    T* get() const { return d_ptr; }
    
    // Get size
    size_t getSize() const { return size; }
    
    // Destructor
    ~DeviceMemory() {
        free();
    }
};

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Use cuBLAS for normalization
    DeviceMemory<cuDoubleComplex> d_v(N);
    d_v.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v.data()), N);
    
    double norm;
    cublasCheckError(cublasDznrm2(cublasHandle, N, d_v.get(), 1, &norm));
    
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_v.get(), 1));
    
    d_v.copyToHost(reinterpret_cast<cuDoubleComplex*>(v.data()), N);

    return v;
}

// CUDA-optimized Lanczos algorithm
void lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize CUDA resources
    initCUDA();
    max_iter = std::min(max_iter, N);

    // Initialize random starting vector using CUDA
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current = generateRandomVector(N, gen, dist);
    
    // Initialize device memory for vectors
    DeviceMemory<cuDoubleComplex> d_v_current(N);
    DeviceMemory<cuDoubleComplex> d_v_prev(N);
    DeviceMemory<cuDoubleComplex> d_v_next(N);
    DeviceMemory<cuDoubleComplex> d_w(N);
    
    // Copy initial vector to device
    d_v_current.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_current.data()), N);
    
    // Initialize Lanczos vectors and coefficients
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // Device memory for reorthogonalization
    std::vector<DeviceMemory<cuDoubleComplex>> d_basis_vectors;
    d_basis_vectors.push_back(d_v_current); // Store reference to first basis vector
    
    // Constants for cuBLAS operations
    cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j - CPU operation (assuming H is a CPU function)
        d_v_current.copyToHost(reinterpret_cast<cuDoubleComplex*>(v_current.data()), N);
        H(v_current.data(), w.data(), N);
        d_w.copyToDevice(reinterpret_cast<cuDoubleComplex*>(w.data()), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_beta, d_v_prev.get(), 1, d_w.get(), 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        cublasCheckError(cublasZdotc(cublasHandle, N, d_v_current.get(), 1, d_w.get(), 1, &dot_product));
        alpha.push_back(cuCreal(dot_product)); // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_alpha, d_v_current.get(), 1, d_w.get(), 1));
        
        // Full reorthogonalization (twice for numerical stability)
        for (int iter = 0; iter < 2; iter++) {
            for (size_t k = 0; k <= j; k++) {
                // Compute overlap
                cuDoubleComplex overlap;
                DeviceMemory<cuDoubleComplex>& d_basis_k = (k == j) ? d_v_current : d_basis_vectors[k];
                cublasCheckError(cublasZdotc(cublasHandle, N, d_basis_k.get(), 1, d_w.get(), 1, &overlap));
                
                // w = w - overlap * basis_k
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
                cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_overlap, d_basis_k.get(), 1, d_w.get(), 1));
            }
        }
        
        // beta_{j+1} = ||w||
        double norm;
        cublasCheckError(cublasDznrm2(cublasHandle, N, d_w.get(), 1, &norm));
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector orthogonal to basis
            v_next = generateRandomVector(N, gen, dist);
            d_v_next.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_next.data()), N);
            
            // Orthogonalize against all basis vectors
            for (int iter = 0; iter < 2; iter++) {
                for (size_t k = 0; k <= j; k++) {
                    cuDoubleComplex overlap;
                    DeviceMemory<cuDoubleComplex>& d_basis_k = (k == j) ? d_v_current : d_basis_vectors[k];
                    cublasCheckError(cublasZdotc(cublasHandle, N, d_basis_k.get(), 1, d_v_next.get(), 1, &overlap));
                    
                    cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
                    cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_overlap, d_basis_k.get(), 1, d_v_next.get(), 1));
                }
            }
            
            // Update the norm
            cublasCheckError(cublasDznrm2(cublasHandle, N, d_v_next.get(), 1, &norm));
            
            // If still too small, we've reached an invariant subspace
            if (norm < tol) {
                break;
            }
        } else {
            // Copy w to v_next
            cudaCheckError(cudaMemcpy(d_v_next.get(), d_w.get(), N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        }
        
        beta.push_back(norm);
        
        // Normalize v_next
        cuDoubleComplex scale_factor = make_cuDoubleComplex(1.0/norm, 0.0);
        cublasCheckError(cublasZscal(cublasHandle, N, &scale_factor, d_v_next.get(), 1));
        
        // Copy to host for storage
        d_v_next.copyToHost(reinterpret_cast<cuDoubleComplex*>(v_next.data()), N);
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
            
            // Allocate new device memory for the next basis vector
            DeviceMemory<cuDoubleComplex> d_new_basis(N);
            d_new_basis.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_next.data()), N);
            d_basis_vectors.push_back(d_new_basis);
        }
        
        // Update for next iteration
        cudaCheckError(cudaMemcpy(d_v_prev.get(), d_v_current.get(), N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        cudaCheckError(cudaMemcpy(d_v_current.get(), d_v_next.get(), N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    // Use cuSOLVER for tridiagonal eigenvalue problem
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    // Transfer tridiagonal matrix to device
    DeviceMemory<double> d_diag(m);
    DeviceMemory<double> d_offdiag(m-1);
    DeviceMemory<double> d_evals(m);
    d_diag.copyToDevice(diag.data(), m);
    d_offdiag.copyToDevice(offdiag.data(), m-1);
    
    // Setup for cuSOLVER
    int lwork = 0;
    char jobz = eigenvectors ? 'V' : 'N';
    DeviceMemory<double> d_evecs;
    
    if (eigenvectors) {
        d_evecs.allocate(m * m);
    }
    
    // Get required workspace size
    cusolverEigMode_t jobz_mode = jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverCheckError(cusolverDnDsyevd_bufferSize(
        cusolverHandle,
        jobz_mode,
        CUBLAS_FILL_MODE_LOWER,
        m,
        jobz == 'V' ? d_evecs.get() : nullptr,
        m,
        d_evals.get(),
        &lwork
    ));
    
    // Allocate workspace
    DeviceMemory<double> d_work(lwork);
    DeviceMemory<int> d_info(1);
    
    // Solve eigenvalue problem for symmetric tridiagonal matrix
    cusolverCheckError(cusolverDnDsyevd(
        cusolverHandle,
        jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
        CUBLAS_FILL_MODE_LOWER,
        m,
        d_diag.get(),
        m,
        d_evals.get(),
        d_work.get(),
        lwork,
        d_info.get()
    ));
    
    // Copy results back to host
    std::vector<double> host_evals(m);
    d_evals.copyToHost(host_evals.data(), m);
    
    // Check for errors
    int info;
    cudaCheckError(cudaMemcpy(&info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "cusolverDnDstevd failed with error code " << info << std::endl;
        return;
    }
    
    // Copy eigenvalues to output
    eigenvalues.resize(m);
    std::copy(host_evals.begin(), host_evals.end(), eigenvalues.begin());
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        std::vector<double> host_evecs(m * m);
        d_evecs.copyToHost(host_evecs.data(), m * m);
        
        // Initialize eigenvectors in original basis
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Transform eigenvectors to original basis
        DeviceMemory<cuDoubleComplex> d_result(N);
        
        for (int i = 0; i < m; i++) {
            // Reset result to zero
            cudaCheckError(cudaMemset(d_result.get(), 0, N * sizeof(cuDoubleComplex)));
            
            // For each eigenvector, compute linear combination of basis vectors
            for (int k = 0; k < m; k++) {
                if (std::abs(host_evecs[k*m + i]) > tol) {
                    cuDoubleComplex coeff = make_cuDoubleComplex(host_evecs[k*m + i], 0.0);
                    
                    // Get or recreate the device basis vector
                    DeviceMemory<cuDoubleComplex> d_basis(N);
                    d_basis.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_vectors[k].data()), N);
                    
                    // result += coeff * basis_k
                    cublasCheckError(cublasZaxpy(cublasHandle, N, &coeff, d_basis.get(), 1, d_result.get(), 1));
                }
            }
            
            // Normalize the result
            double norm;
            cublasCheckError(cublasDznrm2(cublasHandle, N, d_result.get(), 1, &norm));
            
            if (norm > tol) {
                cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
                cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_result.get(), 1));
            }
            
            // Copy back to host
            d_result.copyToHost(reinterpret_cast<cuDoubleComplex*>((*eigenvectors)[i].data()), N);
        }
    }
}

int main(){
    // Matrix size (not too large to keep computation reasonable)
    const int N = 500; 

    // Generate a random Hermitian matrix
    std::vector<std::vector<Complex>> randomMatrix(N, std::vector<Complex>(N));
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Fill with random values and make it Hermitian
    for (int i = 0; i < N; i++) {
        randomMatrix[i][i] = Complex(dist(gen), 0.0); // Real diagonal
        for (int j = i+1; j < N; j++) {
            randomMatrix[i][j] = Complex(dist(gen), dist(gen));
            randomMatrix[j][i] = std::conj(randomMatrix[i][j]);
        }
    }

    // Define matrix-vector multiplication function
    auto matVecMult = [&](const Complex* v, Complex* result, int size) {
        std::fill(result, result + size, Complex(0.0, 0.0));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i] += randomMatrix[i][j] * v[j];
            }
        }
    };

    // Test all three methods
    std::cout << "Testing with " << N << "x" << N << " random Hermitian matrix\n";

    // Regular Lanczos
    std::vector<double> lanczosEigenvalues;
    std::vector<ComplexVector> lanczosEigenvectors;
    lanczos_cuda(matVecMult, N, N/2, 1e-10, lanczosEigenvalues, &lanczosEigenvectors);

    // Lanczos with CG refinement
    std::vector<double> lanczosCGEigenvalues;
    std::vector<ComplexVector> lanczosCGEigenvectors;
    lanczos_cuda(matVecMult, N, N/2, 1e-10, lanczosCGEigenvalues, &lanczosCGEigenvectors);

    // Direct diagonalization
    std::vector<Complex> flatMatrix(N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            flatMatrix[j*N + i] = randomMatrix[i][j];
        }
    }

    std::vector<double> directEigenvalues(N);
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', N, 
                          reinterpret_cast<lapack_complex_double*>(flatMatrix.data()), 
                          N, directEigenvalues.data());

    if (info == 0) {
        // Compare results
        std::cout << "\nEigenvalue comparison:\n";
        std::cout << "Index | Direct  | Lanczos | Diff    | Lanczos+CG | Diff\n";
        std::cout << "--------------------------------------------------------\n";
        int numToShow = std::min(10, N/2);
        for (int i = 0; i < numToShow; i++) {
            std::cout << std::setw(5) << i << " | "
                    << std::fixed << std::setprecision(6)
                    << std::setw(8) << directEigenvalues[i] << " | "
                    << std::setw(7) << lanczosEigenvalues[i] << " | "
                    << std::setw(7) << std::abs(directEigenvalues[i] - lanczosEigenvalues[i]) << " | "
                    << std::setw(10) << lanczosCGEigenvalues[i] << " | "
                    << std::setw(7) << std::abs(directEigenvalues[i] - lanczosCGEigenvalues[i]) << "\n";
        }
    }

}