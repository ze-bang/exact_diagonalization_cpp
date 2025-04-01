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
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

void lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize CUDA
    initCUDA();

    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current = generateRandomVector(N, gen, dist);
    
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
    
    // GPU memory allocation for vectors
    DeviceMemory<cuDoubleComplex> d_v_current(N);
    DeviceMemory<cuDoubleComplex> d_v_prev(N);
    DeviceMemory<cuDoubleComplex> d_v_next(N);
    DeviceMemory<cuDoubleComplex> d_w(N);
    
    d_v_current.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_current.data()), N);
    d_v_prev.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_prev.data()), N);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // Copy current vector to host for matrix-vector product
        d_v_current.copyToHost(reinterpret_cast<cuDoubleComplex*>(v_current.data()), N);
        
        // w = H*v_j (using provided CPU function)
        H(v_current.data(), w.data(), N);
        
        // Copy result back to device
        d_w.copyToDevice(reinterpret_cast<cuDoubleComplex*>(w.data()), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_beta, d_v_prev.get(), 1, d_w.get(), 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        cublasCheckError(cublasZdotc(cublasHandle, N, d_v_current.get(), 1, d_w.get(), 1, &dot_product));
        alpha.push_back(cuCreal(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_alpha, d_v_current.get(), 1, d_w.get(), 1));
        
        // Full reorthogonalization (twice for numerical stability)
        for (int iter = 0; iter < 2; iter++) {
            for (size_t k = 0; k <= j; k++) {
                // Get basis vector k
                DeviceMemory<cuDoubleComplex> d_basis_k(N);
                d_basis_k.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_vectors[k].data()), N);
                
                // Compute overlap
                cuDoubleComplex overlap;
                cublasCheckError(cublasZdotc(cublasHandle, N, d_basis_k.get(), 1, d_w.get(), 1, &overlap));
                
                // Orthogonalize
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
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    DeviceMemory<cuDoubleComplex> d_basis_k(N);
                    d_basis_k.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_vectors[k].data()), N);
                    
                    cuDoubleComplex overlap;
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
        
        // Copy v_next back to host for storage in basis_vectors
        d_v_next.copyToHost(reinterpret_cast<cuDoubleComplex*>(v_next.data()), N);
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
        }
        
        // Update for next iteration
        cudaCheckError(cudaMemcpy(d_v_prev.get(), d_v_current.get(), N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        cudaCheckError(cudaMemcpy(d_v_current.get(), d_v_next.get(), N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    // Allocate arrays for cuSOLVER
    DeviceMemory<double> d_diag(m);
    DeviceMemory<double> d_offdiag(m-1);
    d_diag.copyToDevice(alpha.data(), m);
    
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    d_offdiag.copyToDevice(offdiag.data(), m-1);
    
    // Create output arrays for eigenvalues
    DeviceMemory<double> d_evals(m);
    DeviceMemory<double> d_evecs;
    
    // Query working space requirements
    int lwork = 0;
    char jobz = eigenvectors ? 'V' : 'N';
    cusolverEigMode_t jobz_cusolver = eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    
    // Convert tridiagonal to full symmetric matrix for use with Dsyevd
    std::vector<double> symMatrix(m * m, 0.0);
    for (int i = 0; i < m; i++) {
        symMatrix[i * m + i] = alpha[i]; // Diagonal
        if (i < m - 1) {
            symMatrix[i * m + (i+1)] = offdiag[i]; // Upper diagonal
            symMatrix[(i+1) * m + i] = offdiag[i]; // Lower diagonal
        }
    }
    
    // Allocate device memory for full matrix
    DeviceMemory<double> d_symMatrix(m * m);
    d_symMatrix.copyToDevice(symMatrix.data(), m * m);
    
    int* d_info;
    cudaCheckError(cudaMalloc(&d_info, sizeof(int)));
    
    // First query the optimal workspace size
    cusolverCheckError(cusolverDnDsyevd_bufferSize(
        cusolverHandle, jobz_cusolver, CUBLAS_FILL_MODE_LOWER, m,
        d_symMatrix.get(), m, d_evals.get(), &lwork));
        
    // Then allocate working space with the correct size
    DeviceMemory<double> d_work(lwork);

    // Solve eigenvalue problem using the full symmetric matrix
    cusolverCheckError(cusolverDnDsyevd(
        cusolverHandle, jobz_cusolver, CUBLAS_FILL_MODE_LOWER, m, 
        d_symMatrix.get(), m, d_evals.get(), d_work.get(), lwork, d_info));
    
    // Extract eigenvectors from the full matrix solution if needed
    if (eigenvectors) {
        DeviceMemory<double> d_temp_evecs(m * m);
        cudaCheckError(cudaMemcpy(d_temp_evecs.get(), d_symMatrix.get(), m * m * sizeof(double), cudaMemcpyDeviceToDevice));
        d_evecs.allocate(m * m);
        cudaCheckError(cudaMemcpy(d_evecs.get(), d_temp_evecs.get(), m * m * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    
    // Allocate memory for eigenvectors if needed
    if (eigenvectors) {
        d_evecs.allocate(m*m);
    }
    
    // Check for errors
    int info;
    cudaCheckError(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (info != 0) {
        std::cerr << "cusolverDnDstevd failed with error code " << info << std::endl;
        cudaFree(d_info);
        return;
    }
    cudaFree(d_info);
    
    // Copy eigenvalues to host
    eigenvalues.resize(m);
    d_evals.copyToHost(eigenvalues.data(), m);
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        // Bring eigenvectors back to host
        std::vector<double> evecs(m*m);
        d_evecs.copyToHost(evecs.data(), m*m);
        
        // Identify clusters of degenerate eigenvalues
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < m; i++) {
            bool added_to_cluster = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(eigenvalues[i] - eigenvalues[cluster[0]]) < degen_tol) {
                    cluster.push_back(i);
                    added_to_cluster = true;
                    break;
                }
            }
            if (!added_to_cluster) {
                degen_clusters.push_back({i});
            }
        }
        
        // Transform to original basis and handle degeneracy
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Process each cluster - this can be done in parallel with CUDA streams
        for (size_t cl = 0; cl < degen_clusters.size(); cl++) {
            const auto& cluster = degen_clusters[cl];
            
            if (cluster.size() == 1) {
                // Non-degenerate case - standard treatment
                int idx = cluster[0];
                ComplexVector evec(N, Complex(0.0, 0.0));
                
                // Transform on GPU: evec = sum_k z(k,idx) * basis_vectors[k]
                DeviceMemory<cuDoubleComplex> d_evec(N);
                cudaCheckError(cudaMemset(d_evec.get(), 0, N * sizeof(cuDoubleComplex)));
                
                for (int k = 0; k < m; k++) {
                    DeviceMemory<cuDoubleComplex> d_basis_k(N);
                    d_basis_k.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_vectors[k].data()), N);
                    
                    cuDoubleComplex coeff = make_cuDoubleComplex(evecs[k*m + idx], 0.0);
                    cublasCheckError(cublasZaxpy(cublasHandle, N, &coeff, d_basis_k.get(), 1, d_evec.get(), 1));
                }
                
                // Normalize
                double norm;
                cublasCheckError(cublasDznrm2(cublasHandle, N, d_evec.get(), 1, &norm));
                cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
                cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_evec.get(), 1));
                
                // Copy result back to host
                d_evec.copyToHost(reinterpret_cast<cuDoubleComplex*>(evec.data()), N);
                (*eigenvectors)[idx] = evec;
            } else {
                // Degenerate case - special handling
                int subspace_dim = cluster.size();
                std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N, Complex(0.0, 0.0)));
                std::vector<DeviceMemory<cuDoubleComplex>> d_subspace_vectors(subspace_dim);
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    d_subspace_vectors[c].allocate(N);
                    cudaCheckError(cudaMemset(d_subspace_vectors[c].get(), 0, N * sizeof(cuDoubleComplex)));
                    
                    int idx = cluster[c];
                    for (int k = 0; k < m; k++) {
                        DeviceMemory<cuDoubleComplex> d_basis_k(N);
                        d_basis_k.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_vectors[k].data()), N);
                        
                        cuDoubleComplex coeff = make_cuDoubleComplex(evecs[k*m + idx], 0.0);
                        cublasCheckError(cublasZaxpy(cublasHandle, N, &coeff, d_basis_k.get(), 1, d_subspace_vectors[c].get(), 1));
                    }
                }
                
                // Re-orthogonalize within degenerate subspace
                for (int c = 0; c < subspace_dim; c++) {
                    // Normalize current vector
                    double norm;
                    cublasCheckError(cublasDznrm2(cublasHandle, N, d_subspace_vectors[c].get(), 1, &norm));
                    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
                    cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_subspace_vectors[c].get(), 1));
                    
                    // Orthogonalize against previous vectors
                    for (int prev = 0; prev < c; prev++) {
                        cuDoubleComplex overlap;
                        cublasCheckError(cublasZdotc(cublasHandle, N, d_subspace_vectors[prev].get(), 1, d_subspace_vectors[c].get(), 1, &overlap));
                        
                        cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(overlap), -cuCimag(overlap));
                        cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_overlap, d_subspace_vectors[prev].get(), 1, d_subspace_vectors[c].get(), 1));
                    }
                    
                    // Renormalize if necessary
                    cublasCheckError(cublasDznrm2(cublasHandle, N, d_subspace_vectors[c].get(), 1, &norm));
                    if (norm > tol) {
                        scale = make_cuDoubleComplex(1.0/norm, 0.0);
                        cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_subspace_vectors[c].get(), 1));
                    }
                    
                    // Copy back to host and store
                    int idx = cluster[c];
                    d_subspace_vectors[c].copyToHost(reinterpret_cast<cuDoubleComplex*>(subspace_vectors[c].data()), N);
                    (*eigenvectors)[idx] = subspace_vectors[c];
                }
            }
        }
        
        // Optionally refine eigenvectors using conjugate gradient
        // This would need a CUDA implementation of refine_eigenvector_with_cg
    }
}


int main(){
    int num_site = 16;
    Operator op(num_site);
    op.loadFromFile("./ED_test/Trans.def");
    op.loadFromInterAllFile("./ED_test/InterAll.def");
    std::vector<double> eigenvalues;
    // std::vector<ComplexVector> eigenvectors;
    lanczos_cuda([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, (1<<num_site), 1000, 1e-10, eigenvalues);

    std::vector<double> eigenvalues_lanczos;
    lanczos_cuda([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        result = op.apply(vec);
        std::copy(result.begin(), result.end(), Hv);
    }, (1<<num_site), 1000, 1e-10, eigenvalues_lanczos);

    // Print the results
    // std::cout << "Eigenvalues:" << std::endl;
    // for (size_t i = 0; i < 20; i++) {
    //     std::cout << "Eigenvalue " << i << " Chebyshev Filtered Lanczos: " << eigenvalues[i] << " Lanczos: " << eigenvalues_lanczos[i] << std::endl;
    // }
    // Run full diagonalization for comparison
    // std::vector<double> full_eigenvalues;
    // full_diagonalization([&](const Complex* v, Complex* Hv, int N) {
    //     std::vector<Complex> vec(v, v + N);
    //     std::vector<Complex> result(N, Complex(0.0, 0.0));
    //     result = op.apply(vec);
    //     std::copy(result.begin(), result.end(), Hv);
    // }, 1<<num_site, full_eigenvalues);

    // // Sort both sets of eigenvalues for comparison
    // std::sort(eigenvalues.begin(), eigenvalues.end());
    // std::sort(full_eigenvalues.begin(), full_eigenvalues.end());

    // // Compare and print results
    // std::cout << "\nComparison between Lanczos and Full Diagonalization:" << std::endl;
    // std::cout << "Index | Lanczos        | Full          | Difference" << std::endl;
    // std::cout << "------------------------------------------------------" << std::endl;

    // int num_to_compare = std::min(eigenvalues.size(), full_eigenvalues.size());
    // num_to_compare = std::min(num_to_compare, 20);  // Limit to first 20 eigenvalues

    // for (int i = 0; i < num_to_compare; i++) {
    //     double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
    //     std::cout << std::setw(5) << i << " | " 
    //               << std::setw(14) << std::fixed << std::setprecision(10) << eigenvalues[i] << " | "
    //               << std::setw(14) << std::fixed << std::setprecision(10) << full_eigenvalues[i] << " | "
    //               << std::setw(10) << std::scientific << std::setprecision(3) << diff << std::endl;
    // }

    // // Calculate and print overall accuracy statistics
    // if (num_to_compare > 0) {
    //     double max_diff = 0.0;
    //     double sum_diff = 0.0;
    //     for (int i = 0; i < num_to_compare; i++) {
    //         double diff = std::abs(eigenvalues[i] - full_eigenvalues[i]);
    //         max_diff = std::max(max_diff, diff);
    //         sum_diff += diff;
    //     }
    //     double avg_diff = sum_diff / num_to_compare;
        
    //     std::cout << "\nAccuracy statistics:" << std::endl;
    //     std::cout << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
    //     std::cout << "Average difference: " << std::scientific << std::setprecision(3) << avg_diff << std::endl;
        
    //     // Special focus on ground state and first excited state
    //     if (full_eigenvalues.size() > 0 && eigenvalues.size() > 0) {
    //         double ground_diff = std::abs(eigenvalues[0] - full_eigenvalues[0]);
    //         std::cout << "Ground state error: " << std::scientific << std::setprecision(3) << ground_diff << std::endl;
            
    //         if (full_eigenvalues.size() > 1 && eigenvalues.size() > 1) {
    //             double excited_diff = std::abs(eigenvalues[1] - full_eigenvalues[1]);
    //             std::cout << "First excited state error: " << std::scientific << std::setprecision(3) << excited_diff << std::endl;
    //         }
    //     }
    // }

    return 0;
}


// int main(){
//     // Matrix size (not too large to keep computation reasonable)
//     const int N = 500; 

//     // Generate a random Hermitian matrix
//     std::vector<std::vector<Complex>> randomMatrix(N, std::vector<Complex>(N));
//     std::mt19937 gen(42); // Fixed seed for reproducibility
//     std::uniform_real_distribution<double> dist(-1.0, 1.0);

//     // Fill with random values and make it Hermitian
//     for (int i = 0; i < N; i++) {
//         randomMatrix[i][i] = Complex(dist(gen), 0.0); // Real diagonal
//         for (int j = i+1; j < N; j++) {
//             randomMatrix[i][j] = Complex(dist(gen), dist(gen));
//             randomMatrix[j][i] = std::conj(randomMatrix[i][j]);
//         }
//     }

//     // Define matrix-vector multiplication function
//     auto matVecMult = [&](const Complex* v, Complex* result, int size) {
//         std::fill(result, result + size, Complex(0.0, 0.0));
//         for (int i = 0; i < size; i++) {
//             for (int j = 0; j < size; j++) {
//                 result[i] += randomMatrix[i][j] * v[j];
//             }
//         }
//     };

//     // Test all three methods
//     std::cout << "Testing with " << N << "x" << N << " random Hermitian matrix\n";

//     // Regular Lanczos
//     std::vector<double> lanczosEigenvalues;
//     std::vector<ComplexVector> lanczosEigenvectors;
//     lanczos_cuda(matVecMult, N, N/2, 1e-10, lanczosEigenvalues, &lanczosEigenvectors);

//     // Lanczos with CG refinement
//     std::vector<double> lanczosCGEigenvalues;
//     std::vector<ComplexVector> lanczosCGEigenvectors;
//     lanczos_cuda(matVecMult, N, N/2, 1e-10, lanczosCGEigenvalues, &lanczosCGEigenvectors);

//     // Direct diagonalization
//     std::vector<Complex> flatMatrix(N * N);
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             flatMatrix[j*N + i] = randomMatrix[i][j];
//         }
//     }

//     std::vector<double> directEigenvalues(N);
//     int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', N, 
//                           reinterpret_cast<lapack_complex_double*>(flatMatrix.data()), 
//                           N, directEigenvalues.data());

//     if (info == 0) {
//         // Compare results
//         std::cout << "\nEigenvalue comparison:\n";
//         std::cout << "Index | Direct  | Lanczos | Diff    | Lanczos+CG | Diff\n";
//         std::cout << "--------------------------------------------------------\n";
//         int numToShow = std::min(10, N/2);
//         for (int i = 0; i < numToShow; i++) {
//             std::cout << std::setw(5) << i << " | "
//                     << std::fixed << std::setprecision(6)
//                     << std::setw(8) << directEigenvalues[i] << " | "
//                     << std::setw(7) << lanczosEigenvalues[i] << " | "
//                     << std::setw(7) << std::abs(directEigenvalues[i] - lanczosEigenvalues[i]) << " | "
//                     << std::setw(10) << lanczosCGEigenvalues[i] << " | "
//                     << std::setw(7) << std::abs(directEigenvalues[i] - lanczosCGEigenvalues[i]) << "\n";
//         }
//     }

// }
