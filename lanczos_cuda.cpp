#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <ezarpack/arpack_solver.hpp>
#include <ezarpack/storages/eigen.hpp>
#include <ezarpack/version.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <omp.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

// Error checking macros
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t stat = call; \
    if (stat != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error in %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t stat = call; \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error in %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
}

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// CUDA helper class to manage handles and contexts
class CudaHelper {
private:
    static cublasHandle_t cublas_handle;
    static cusparseHandle_t cusparse_handle;
    static cusolverDnHandle_t cusolver_handle;
    static bool initialized;

public:
    static void initialize() {
        if (!initialized) {
            CHECK_CUBLAS(cublasCreate(&cublas_handle));
            CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
            CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));
            initialized = true;
        }
    }

    static void destroy() {
        if (initialized) {
            CHECK_CUBLAS(cublasDestroy(cublas_handle));
            CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
            CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
            initialized = false;
        }
    }

    static cublasHandle_t& getCublasHandle() {
        initialize();
        return cublas_handle;
    }

    static cusparseHandle_t& getCusparseHandle() {
        initialize();
        return cusparse_handle;
    }

    static cusolverDnHandle_t& getCusolverHandle() {
        initialize();
        return cusolver_handle;
    }
};

// Initialize static members
cublasHandle_t CudaHelper::cublas_handle = nullptr;
cusparseHandle_t CudaHelper::cusparse_handle = nullptr;
cusolverDnHandle_t CudaHelper::cusolver_handle = nullptr;
bool CudaHelper::initialized = false;

// GPU memory management helper functions
template<typename T>
T* allocDeviceMem(size_t size) {
    T* d_ptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, size * sizeof(T)));
    return d_ptr;
}

template<typename T>
void copyToDevice(T* d_dest, const T* h_src, size_t size) {
    CHECK_CUDA(cudaMemcpy(d_dest, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copyToHost(T* h_dest, const T* d_src, size_t size) {
    CHECK_CUDA(cudaMemcpy(h_dest, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void freeDeviceMem(T* d_ptr) {
    if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
}

// Helper struct for device vectors
struct DeviceVector {
    cuDoubleComplex* data;
    int size;

    DeviceVector(int n) : size(n) {
        data = allocDeviceMem<cuDoubleComplex>(n);
    }

    DeviceVector(const ComplexVector& host_vec) : size(host_vec.size()) {
        data = allocDeviceMem<cuDoubleComplex>(size);
        copyToDevice(data, reinterpret_cast<const cuDoubleComplex*>(host_vec.data()), size);
    }

    ~DeviceVector() {
        freeDeviceMem(data);
    }

    // Copy to host
    ComplexVector toHost() const {
        ComplexVector host_vec(size);
        copyToHost(reinterpret_cast<cuDoubleComplex*>(host_vec.data()), data, size);
        return host_vec;
    }
};

// CUDA-accelerated version of generateRandomVector
ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    // Generate random vector on host
    ComplexVector h_v(N);
    for (int i = 0; i < N; i++) {
        h_v[i] = Complex(dist(gen), dist(gen));
    }
    
    // Transfer to device
    DeviceVector d_v(h_v);
    
    // Compute norm using cuBLAS
    double norm;
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_v.data, 1, &norm));
    
    // Scale the vector
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_v.data, 1));
    
    // Copy back to host
    return d_v.toHost();
}

// CUDA-accelerated version of generateOrthogonalVector
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, 
                                     std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    // Generate a random vector
    ComplexVector result = generateRandomVector(N, gen, dist);
    DeviceVector d_result(result);
    
    // Orthogonalize against all provided vectors using GPU
    for (const auto& v : vectors) {
        DeviceVector d_v(v);
        
        // Calculate projection: <v, result>
        cuDoubleComplex projection;
        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_v.data, 1, d_result.data, 1, &projection));
        
        // Subtract projection: result -= projection * v
        cuDoubleComplex neg_projection = make_cuDoubleComplex(-projection.x, -projection.y);
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_projection, d_v.data, 1, d_result.data, 1));
    }
    
    // Check if the resulting vector has sufficient magnitude
    double norm;
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_result.data, 1, &norm));
        
    // Normalize
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_result.data, 1));
    
    return d_result.toHost();
}

// CUDA-accelerated version of refine_eigenvector_with_cg
void refine_eigenvector_with_cg(std::function<void(const Complex*, Complex*, int)> H,
                               ComplexVector& v, double& lambda, int N, double tol) {
    // Normalize initial vector
    DeviceVector d_v(v);
    double norm;
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_v.data, 1, &norm));
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_v.data, 1));
    
    // Allocate device memory for vectors
    DeviceVector d_r(N);
    DeviceVector d_p(N);
    DeviceVector d_Hp(N);
    DeviceVector d_Hv(N);
    
    // Copy v back to host for H application
    v = d_v.toHost();
    
    // Apply H to v: Hv = H*v
    ComplexVector h_Hv(N);
    H(v.data(), h_Hv.data(), N);
    copyToDevice(d_Hv.data, reinterpret_cast<const cuDoubleComplex*>(h_Hv.data()), N);
    
    // Initial residual: r = Hv - λv
    CHECK_CUDA(cudaMemcpy(d_r.data, d_Hv.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    cuDoubleComplex neg_lambda = make_cuDoubleComplex(-lambda, 0.0);
    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_lambda, d_v.data, 1, d_r.data, 1));
    
    // Initial search direction
    CHECK_CUDA(cudaMemcpy(d_p.data, d_r.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    
    // CG iteration
    const int max_cg_iter = 50;
    const double cg_tol = tol * 0.1;
    double res_norm;
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_r.data, 1, &res_norm));
    
    for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol; iter++) {
        // Copy p back to host for H application
        ComplexVector h_p = DeviceVector(d_p).toHost();
        
        // Apply (H - λI) to p
        ComplexVector h_Hp(N);
        H(h_p.data(), h_Hp.data(), N);
        copyToDevice(d_Hp.data, reinterpret_cast<const cuDoubleComplex*>(h_Hp.data()), N);
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_lambda, d_p.data, 1, d_Hp.data, 1));
        
        // α = (r·r) / (p·(H-λI)p)
        cuDoubleComplex r_dot_r, p_dot_Hp;
        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_r.data, 1, d_r.data, 1, &r_dot_r));
        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_p.data, 1, d_Hp.data, 1, &p_dot_Hp));
        
        cuDoubleComplex alpha = make_cuDoubleComplex(
            (r_dot_r.x * p_dot_Hp.x + r_dot_r.y * p_dot_Hp.y) / 
            (p_dot_Hp.x * p_dot_Hp.x + p_dot_Hp.y * p_dot_Hp.y),
            
            (r_dot_r.y * p_dot_Hp.x - r_dot_r.x * p_dot_Hp.y) / 
            (p_dot_Hp.x * p_dot_Hp.x + p_dot_Hp.y * p_dot_Hp.y)
        );
        
        // v = v + α*p
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &alpha, d_p.data, 1, d_v.data, 1));
        
        // Store old r·r
        cuDoubleComplex r_dot_r_old = r_dot_r;
        
        // r = r - α*(H-λI)p
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha.x, -alpha.y);
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_alpha, d_Hp.data, 1, d_r.data, 1));
        
        // Check convergence
        CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_r.data, 1, &res_norm));
        
        // β = (r_new·r_new) / (r_old·r_old)
        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_r.data, 1, d_r.data, 1, &r_dot_r));
        cuDoubleComplex beta = make_cuDoubleComplex(
            (r_dot_r.x * r_dot_r_old.x + r_dot_r.y * r_dot_r_old.y) / 
            (r_dot_r_old.x * r_dot_r_old.x + r_dot_r_old.y * r_dot_r_old.y),
            
            (r_dot_r.y * r_dot_r_old.x - r_dot_r.x * r_dot_r_old.y) / 
            (r_dot_r_old.x * r_dot_r_old.x + r_dot_r_old.y * r_dot_r_old.y)
        );
        
        // Custom kernel or temporary solution for p = r + β*p
        ComplexVector h_r = DeviceVector(d_r).toHost();
        ComplexVector h_p = DeviceVector(d_p).toHost();
        Complex beta_host(beta.x, beta.y);
        
        for (int j = 0; j < N; j++) {
            h_p[j] = h_r[j] + beta_host * h_p[j];
        }
        
        copyToDevice(d_p.data, reinterpret_cast<const cuDoubleComplex*>(h_p.data()), N);
    }
    
    // Normalize final eigenvector
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_v.data, 1, &norm));
    scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_v.data, 1));
    
    // Copy result back to host
    v = d_v.toHost();
}

// CUDA-accelerated Lanczos algorithm
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current = generateRandomVector(N, gen, dist);
    
    // Initialize device vectors
    DeviceVector d_v_current(v_current);
    DeviceVector d_v_prev(N);
    DeviceVector d_v_next(N);
    DeviceVector d_w(N);
    
    // Initialize basis vectors storage
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // Apply H to v_j (need to transfer to host and back)
        ComplexVector h_v_current = d_v_current.toHost();
        ComplexVector h_w(N);
        H(h_v_current.data(), h_w.data(), N);
        copyToDevice(d_w.data, reinterpret_cast<const cuDoubleComplex*>(h_w.data()), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_beta, d_v_prev.data, 1, d_w.data, 1));
        }
        
        // alpha_j = <v_j, w>
        cuDoubleComplex dot_product;
        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_v_current.data, 1, d_w.data, 1, &dot_product));
        alpha.push_back(dot_product.x);  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_alpha, d_v_current.data, 1, d_w.data, 1));
        
        // Full reorthogonalization
        for (int iter = 0; iter < 2; iter++) {
            // Copy w to host for reorthogonalization
            ComplexVector h_w = DeviceVector(d_w).toHost();
            
            for (int k = 0; k <= j; k++) {
                DeviceVector d_basis_k(basis_vectors[k]);
                cuDoubleComplex overlap;
                CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, d_basis_k.data, 1, d_w.data, 1, &overlap));
                
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
                CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_overlap, d_basis_k.data, 1, d_w.data, 1));
            }
        }
        
        // beta_{j+1} = ||w||
        double norm;
        CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_w.data, 1, &norm));
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector orthogonal to basis
            ComplexVector v_next = generateOrthogonalVector(N, basis_vectors, gen, dist);
            copyToDevice(d_v_next.data, reinterpret_cast<const cuDoubleComplex*>(v_next.data()), N);
            
            // Update the norm
            CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_v_next.data, 1, &norm));
            
            // If still too small, we've reached an invariant subspace
            if (norm < tol) {
                break;
            }
        } else {
            // Copy w to v_next
            CHECK_CUDA(cudaMemcpy(d_v_next.data, d_w.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        }
        
        beta.push_back(norm);
        
        // Normalize v_next
        cuDoubleComplex scale_factor = make_cuDoubleComplex(1.0/norm, 0.0);
        CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale_factor, d_v_next.data, 1));
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(d_v_next.toHost());
        }
        
        // Update for next iteration
        cudaMemcpy(d_v_prev.data, d_v_current.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v_current.data, d_v_next.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
    
    // Construct and solve tridiagonal matrix using cuSOLVER
    int m = alpha.size();
    
    // Allocate arrays for cuSOLVER
    double* d_diag = nullptr;
    double* d_offdiag = nullptr;
    double* d_evals = nullptr;
    double* d_evecs = nullptr;
    
    CHECK_CUDA(cudaMalloc((void**)&d_diag, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_offdiag, (m-1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_evals, m * sizeof(double)));
    
    if (eigenvectors) {
        CHECK_CUDA(cudaMalloc((void**)&d_evecs, m * m * sizeof(double)));
    }
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_diag, alpha.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    
    std::vector<double> offdiag(m-1);
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    CHECK_CUDA(cudaMemcpy(d_offdiag, offdiag.data(), (m-1) * sizeof(double), cudaMemcpyHostToDevice));
    
    // Workspace parameters
    char jobz = eigenvectors ? 'V' : 'N';  // Compute eigenvectors?
    int info = 0;
    int lwork = 0;
    double* d_work = nullptr;
    
    // Get workspace size
    CHECK_CUSOLVER(cusolverDnDstev_bufferSize(
        CudaHelper::getCusolverHandle(),
        jobz, m, d_diag, d_offdiag, d_evals, 
        eigenvectors ? d_evecs : nullptr, m, &lwork
    ));
    
    // Allocate workspace
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));
    
    // Compute eigenvalues and eigenvectors
    int* d_info;
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));
    
    CHECK_CUSOLVER(cusolverDnDstev(
        CudaHelper::getCusolverHandle(),
        jobz, m, d_diag, d_offdiag, d_evals, 
        eigenvectors ? d_evecs : nullptr, m, 
        d_work, lwork, d_info
    ));
    
    CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (info != 0) {
        std::cerr << "cusolverDnDstev failed with error code " << info << std::endl;
        
        // Free device memory
        CHECK_CUDA(cudaFree(d_diag));
        CHECK_CUDA(cudaFree(d_offdiag));
        CHECK_CUDA(cudaFree(d_evals));
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(d_info));
        if (eigenvectors) {
            CHECK_CUDA(cudaFree(d_evecs));
        }
        
        return;
    }
    
    // Copy eigenvalues to host
    eigenvalues.resize(m);
    CHECK_CUDA(cudaMemcpy(eigenvalues.data(), d_evals, m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        std::vector<double> h_evecs(m * m);
        CHECK_CUDA(cudaMemcpy(h_evecs.data(), d_evecs, m * m * sizeof(double), cudaMemcpyDeviceToHost));
        
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        // For each eigenvector
        for (int i = 0; i < m; i++) {
            // Transform: evec = sum_k z(k,idx) * basis_vectors[k]
            DeviceVector d_result(N);
            
            for (int k = 0; k < m; k++) {
                if (k < basis_vectors.size()) {
                    DeviceVector d_basis_k(basis_vectors[k]);
                    cuDoubleComplex coeff = make_cuDoubleComplex(h_evecs[k*m + i], 0.0);
                    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &coeff, d_basis_k.data, 1, d_result.data, 1));
                }
            }
            
            // Normalize
            double norm;
            CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_result.data, 1, &norm));
            cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
            CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_result.data, 1));
            
            (*eigenvectors)[i] = d_result.toHost();
        }
    }
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_diag));
    CHECK_CUDA(cudaFree(d_offdiag));
    CHECK_CUDA(cudaFree(d_evals));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    if (eigenvectors) {
        CHECK_CUDA(cudaFree(d_evecs));
    }
}
