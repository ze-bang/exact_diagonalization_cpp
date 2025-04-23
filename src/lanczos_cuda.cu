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

void lanczos_cuda(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
                 double tol, std::vector<double>& eigenvalues, std::string dir = "",
                 bool eigenvectors = false) {
    
    // Initialize CUDA
    initCUDA();
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp critical
        {
            v_current[i] = Complex(dist(gen), dist(gen));
        }
    }

    std::cout << "Lanczos CUDA: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = dir + "/lanczos_cuda_basis_vectors";
    std::string cmd = "mkdir -p " + temp_dir;
    system(cmd.c_str());

    // Write the first basis vector to file
    std::string basis_file = temp_dir + "/basis_0.bin";
    std::ofstream outfile(basis_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << basis_file << " for writing" << std::endl;
        cleanupCUDA();
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
    
    std::cout << "Begin Lanczos CUDA iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "Lanczos CUDA: Iterating..." << std::endl;
    
    // GPU memory allocation for vectors
    DeviceMemory<cuDoubleComplex> d_v_current(N);
    DeviceMemory<cuDoubleComplex> d_v_prev(N);
    DeviceMemory<cuDoubleComplex> d_v_next(N);
    DeviceMemory<cuDoubleComplex> d_w(N);
    
    // Helper function to read basis vector from file
    auto read_basis_vector = [&temp_dir](int index, int N) -> ComplexVector {
        std::string basis_file = temp_dir + "/basis_" + std::to_string(index) + ".bin";
        std::ifstream infile(basis_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open file " << basis_file << " for reading" << std::endl;
            return ComplexVector(N);
        }
        ComplexVector v(N);
        infile.read(reinterpret_cast<char*>(v.data()), N * sizeof(Complex));
        infile.close();
        return v;
    };
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // Load vectors from disk if necessary
        if (j > 0) {
            v_prev = read_basis_vector(j-1, N);
            v_current = read_basis_vector(j, N);
        }
        
        // Copy vectors to device
        d_v_current.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_current.data()), N);
        d_v_prev.copyToDevice(reinterpret_cast<cuDoubleComplex*>(v_prev.data()), N);
        
        // w = H*v_j (using provided CPU function)
        H(v_current.data(), w.data(), N);
        
        // Copy result to device
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
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha.back(), 0.0);
        cublasCheckError(cublasZaxpy(cublasHandle, N, &neg_alpha, d_v_current.get(), 1, d_w.get(), 1));
        
        // Calculate ||w||
        double norm;
        cublasCheckError(cublasDznrm2(cublasHandle, N, d_w.get(), 1, &norm));
        
        // Check for invariant subspace
        if (norm < tol) {
            std::cout << "Lanczos CUDA: Found invariant subspace at iteration " << j << std::endl;
            break;
        }
        
        beta.push_back(norm);
        
        // Normalize w to get v_{j+1}
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
        cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_w.get(), 1));
        
        // Copy v_{j+1} back to host
        d_w.copyToHost(reinterpret_cast<cuDoubleComplex*>(v_next.data()), N);
        
        // Write v_{j+1} to disk if needed for next iteration
        if (j < max_iter - 1) {
            std::string next_basis_file = temp_dir + "/basis_" + std::to_string(j+1) + ".bin";
            std::ofstream outfile(next_basis_file, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error: Cannot open file " << next_basis_file << " for writing" << std::endl;
                cleanupCUDA();
                system(("rm -rf " + temp_dir).c_str());
                return;
            }
            outfile.write(reinterpret_cast<char*>(v_next.data()), N * sizeof(Complex));
            outfile.close();
        }
    }
    
    // Construct and solve tridiagonal matrix
    int m = alpha.size();
    
    std::cout << "Lanczos CUDA: Constructing tridiagonal matrix" << std::endl;

    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    #pragma omp parallel for
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }

    std::cout << "Lanczos CUDA: Solving tridiagonal matrix" << std::endl;
    
    // Save only the first exct eigenvalues, or all of them if m < exct
    int n_eigenvalues = std::min(exct, m);
    
    // Device memory for eigenvalue computation
    DeviceMemory<double> d_diag(m);
    DeviceMemory<double> d_offdiag(m-1);
    DeviceMemory<double> d_evals(m);
    DeviceMemory<double> d_evecs;
    
    d_diag.copyToDevice(diag.data(), m);
    d_offdiag.copyToDevice(offdiag.data(), m-1);
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    
    // Query working space requirements
    int lwork = 0;
    cusolverEigMode_t jobz_cusolver = eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    
    // Convert to full symmetric matrix for cuSOLVER
    std::vector<double> symMatrix(m * m, 0.0);
    for (int i = 0; i < m; i++) {
        symMatrix[i * m + i] = diag[i]; // Diagonal
        if (i < m - 1) {
            symMatrix[i * m + (i+1)] = offdiag[i]; // Upper diagonal
            symMatrix[(i+1) * m + i] = offdiag[i]; // Lower diagonal
        }
    }
    
    // Allocate device memory for full matrix
    DeviceMemory<double> d_symMatrix(m * m);
    d_symMatrix.copyToDevice(symMatrix.data(), m * m);
    
    // Allocate info variable
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
    
    // Check for errors
    int info;
    cudaCheckError(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_info);
    
    if (info != 0) {
        std::cerr << "cusolverDnDsyevd failed with error code " << info << std::endl;
        cleanupCUDA();
        system(("rm -rf " + temp_dir).c_str());
        return;
    }
    
    // Copy eigenvalues to host
    eigenvalues.resize(n_eigenvalues);
    d_evals.copyToHost(eigenvalues.data(), n_eigenvalues);
    
    // Write eigenvalues and eigenvectors to files
    std::string evec_dir = dir + "/lanczos_cuda_eigenvectors";
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
        std::cout << "Lanczos CUDA: Transforming eigenvectors back to original basis" << std::endl;
        
        // Process in batches to save memory
        const int batch_size = 10;
        for (int start_idx = 0; start_idx < n_eigenvalues; start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, n_eigenvalues);
            int batch_n = end_idx - start_idx;
            
            std::vector<ComplexVector> batch_evecs(batch_n, ComplexVector(N));
            
            // Extract Lanczos basis eigenvectors
            std::vector<double> lanczos_evecs(m * batch_n);
            for (int i = 0; i < batch_n; i++) {
                for (int j = 0; j < m; j++) {
                    lanczos_evecs[i*m + j] = static_cast<double>(d_symMatrix.get()[j*m + (start_idx + i)]);
                }
            }
            
            // Transform to original basis in parallel
            #pragma omp parallel for
            for (int i = 0; i < batch_n; i++) {
                ComplexVector& evec = batch_evecs[i];
                std::fill(evec.begin(), evec.end(), Complex(0.0, 0.0));
                
                DeviceMemory<cuDoubleComplex> d_evec(N);
                cudaCheckError(cudaMemset(d_evec.get(), 0, N * sizeof(cuDoubleComplex)));
                
                for (int j = 0; j < m; j++) {
                    double coef = lanczos_evecs[i*m + j];
                    ComplexVector basis_j = read_basis_vector(j, N);
                    
                    DeviceMemory<cuDoubleComplex> d_basis_j(N);
                    d_basis_j.copyToDevice(reinterpret_cast<cuDoubleComplex*>(basis_j.data()), N);
                    
                    cuDoubleComplex cuda_coef = make_cuDoubleComplex(coef, 0.0);
                    cublasCheckError(cublasZaxpy(cublasHandle, N, &cuda_coef, d_basis_j.get(), 1, d_evec.get(), 1));
                }
                
                // Normalize
                double norm;
                cublasCheckError(cublasDznrm2(cublasHandle, N, d_evec.get(), 1, &norm));
                if (norm > 1e-10) {
                    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
                    cublasCheckError(cublasZscal(cublasHandle, N, &scale, d_evec.get(), 1));
                }
                
                // Copy back to host
                d_evec.copyToHost(reinterpret_cast<cuDoubleComplex*>(evec.data()), N);
                
                // Save eigenvector to file
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(start_idx + i) + ".bin";
                std::ofstream outfile(evec_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(evec.data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
        }
    }
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    // Clean up CUDA resources
    cleanupCUDA();
    
    std::cout << "Lanczos CUDA: Completed successfully" << std::endl;
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
