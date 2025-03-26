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


void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                  double tol, std::vector<double>& eigenvalues, 
                  std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Block size for handling degenerate eigenvalues
    const int block_size = 4;
    
    // Initialize random starting vectors with thread-local generators
    unsigned int seed = std::random_device{}();
    
    // Generate orthonormal set of starting vectors
    std::vector<ComplexVector> basis_vectors;
    std::vector<ComplexVector> curr_block(block_size);
    
    #pragma omp parallel
    {
        // Thread local random generators
        std::mt19937 local_gen(seed + omp_get_thread_num());
        std::uniform_real_distribution<double> local_dist(-1.0, 1.0);
        
        #pragma omp single
        {
            curr_block[0] = generateRandomVector(N, local_gen, local_dist);
            basis_vectors.push_back(curr_block[0]);
            
            for (int i = 1; i < block_size; i++) {
                curr_block[i] = generateOrthogonalVector(N, 
                                                      std::vector<ComplexVector>(basis_vectors.begin(), basis_vectors.end()), 
                                                      local_gen, local_dist);
                basis_vectors.push_back(curr_block[i]);
            }
        }
    }
    
    std::vector<ComplexVector> prev_block(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    std::vector<ComplexVector> next_block(block_size, ComplexVector(N));
    std::vector<ComplexVector> work_block(block_size, ComplexVector(N));
    
    // Tridiagonal matrix elements
    std::vector<std::vector<std::vector<Complex>>> alpha;  // Diagonal blocks
    std::vector<std::vector<std::vector<Complex>>> beta;   // Off-diagonal blocks
    
    // First empty beta block (not used)
    beta.push_back(std::vector<std::vector<Complex>>(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0))));
    
    // Limit iterations to either max_iter or N/block_size, whichever is smaller
    int num_steps = std::min(max_iter / block_size, N / block_size);
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        // Current alpha block
        std::vector<std::vector<Complex>> curr_alpha(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        // Apply H to each vector in current block
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < block_size; b++) {
            H(curr_block[b].data(), work_block[b].data(), N);
        }
        
        // Subtract contribution from previous block: w = w - beta_j * prev_block
        if (j > 0) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    Complex neg_beta = -beta[j][i][k];
                    cblas_zaxpy(N, &neg_beta, prev_block[k].data(), 1, work_block[i].data(), 1);
                }
            }
        }
        
        // Compute alpha_j block and update residuals
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    Complex dot;
                    cblas_zdotc_sub(N, curr_block[k].data(), 1, work_block[i].data(), 1, &dot);
                    
                    #pragma omp critical
                    {
                        curr_alpha[i][k] = dot;
                    }
                    
                    // Subtract from work vector: work -= dot * curr
                    Complex neg_dot = -dot;
                    cblas_zaxpy(N, &neg_dot, curr_block[k].data(), 1, work_block[i].data(), 1);
                }
            }
        }
        
        alpha.push_back(curr_alpha);
        
        // Full reorthogonalization against all previous basis vectors
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < block_size; b++) {
            for (int iter = 0; iter < 2; iter++) {  // Double orthogonalization for stability
                // Store original work vector for reduction
                ComplexVector work_orig = work_block[b];
                ComplexVector work_correction(N, Complex(0.0, 0.0));
                
                // Calculate all corrections
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, basis_vectors[k].data(), 1, work_orig.data(), 1, &overlap);
                    Complex neg_overlap = -overlap;
                    
                    // Add correction
                    for (int i = 0; i < N; i++) {
                        work_correction[i] += neg_overlap * basis_vectors[k][i];
                    }
                }
                
                // Apply all corrections at once
                #pragma omp critical
                {
                    for (int i = 0; i < N; i++) {
                        work_block[b][i] = work_orig[i] + work_correction[i];
                    }
                }
            }
        }
        
        // QR factorization of work block
        std::vector<std::vector<Complex>> next_beta(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        for (int i = 0; i < block_size; i++) {
            // Compute the norm
            double norm = cblas_dznrm2(N, work_block[i].data(), 1);
            
            // Handle invariant subspace or linear dependence
            if (norm < tol) {
                #pragma omp parallel
                {
                    std::mt19937 local_gen(seed + omp_get_thread_num());
                    std::uniform_real_distribution<double> local_dist(-1.0, 1.0);
                    
                    #pragma omp single
                    {
                        next_block[i] = generateOrthogonalVector(N, basis_vectors, local_gen, local_dist);
                    }
                }
                
                // Re-orthogonalize
                ComplexVector vec_orig = next_block[i];
                ComplexVector vec_correction(N, Complex(0.0, 0.0));
                
                #pragma omp parallel
                {
                    // Thread-local correction vector
                    ComplexVector local_correction(N, Complex(0.0, 0.0));
                    
                    #pragma omp for nowait
                    for (size_t k = 0; k < basis_vectors.size(); k++) {
                        Complex overlap;
                        cblas_zdotc_sub(N, basis_vectors[k].data(), 1, vec_orig.data(), 1, &overlap);
                        Complex neg_overlap = -overlap;
                        
                        for (int idx = 0; idx < N; idx++) {
                            local_correction[idx] += neg_overlap * basis_vectors[k][idx];
                        }
                    }
                    
                    // Combine corrections
                    #pragma omp critical
                    {
                        for (int idx = 0; idx < N; idx++) {
                            vec_correction[idx] += local_correction[idx];
                        }
                    }
                }
                
                // Apply all corrections
                for (int idx = 0; idx < N; idx++) {
                    next_block[i][idx] = vec_orig[idx] + vec_correction[idx];
                }
                
                norm = cblas_dznrm2(N, next_block[i].data(), 1);
            } else {
                // Copy work to next
                next_block[i] = work_block[i];
            }
            
            // Set the diagonal beta element
            next_beta[i][i] = Complex(norm, 0.0);
            
            // Normalize
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, next_block[i].data(), 1);
            
            // Orthogonalize remaining work vectors against this one
            #pragma omp parallel for schedule(static)
            for (int l = i + 1; l < block_size; l++) {
                Complex overlap;
                cblas_zdotc_sub(N, next_block[i].data(), 1, work_block[l].data(), 1, &overlap);
                
                #pragma omp critical
                {
                    next_beta[l][i] = overlap;
                }
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, next_block[i].data(), 1, work_block[l].data(), 1);
            }
        }
        
        beta.push_back(next_beta);
        
        // Store basis vectors for next iteration
        if (j < num_steps - 1) {
            for (int i = 0; i < block_size; i++) {
                basis_vectors.push_back(next_block[i]);
            }
        }
        
        // Update for next iteration
        prev_block = curr_block;
        curr_block = next_block;
    }
    
    // Convert block tridiagonal matrix to regular format
    int total_dim = basis_vectors.size();
    std::vector<Complex> block_matrix(total_dim * total_dim, Complex(0.0, 0.0));
    
    // Fill diagonal blocks (alpha)
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t j = 0; j < alpha.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = j * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix[col * total_dim + row] = alpha[j][r][c];
                }
            }
        }
    }
    
    // Fill off-diagonal blocks (beta)
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t j = 1; j < beta.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = (j-1) * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    block_matrix[col * total_dim + row] = beta[j][r][c];
                    block_matrix[row * total_dim + col] = std::conj(beta[j][r][c]);
                }
            }
        }
    }
    
    // Diagonalize the block tridiagonal matrix
    std::vector<double> evals(total_dim);
    
    int info;
    if (eigenvectors) {
        // Need to copy matrix as LAPACK destroys input
        std::vector<Complex> evecs = block_matrix;
        info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', total_dim, 
                           reinterpret_cast<lapack_complex_double*>(evecs.data()), 
                           total_dim, evals.data());
                           
        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
            return;
        }
        
        // Transform eigenvectors back to original basis
        eigenvectors->resize(total_dim, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Group eigenvalues into degenerate clusters
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < total_dim; i++) {
            bool added = false;
            for (auto& cluster : degen_clusters) {
                if (std::abs(evals[i] - evals[cluster[0]]) < degen_tol) {
                    cluster.push_back(i);
                    added = true;
                    break;
                }
            }
            if (!added) {
                degen_clusters.push_back({i});
            }
        }
        
        // Process each cluster in parallel using tasks
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (const auto& cluster : degen_clusters) {
                    #pragma omp task
                    {
                        if (cluster.size() == 1) {
                            // Non-degenerate case
                            int idx = cluster[0];
                            ComplexVector& eigenvector = (*eigenvectors)[idx];
                            std::fill(eigenvector.begin(), eigenvector.end(), Complex(0.0, 0.0));
                            
                            // Transform eigenvector using basis vectors
                            for (int i = 0; i < N; i++) {
                                for (size_t k = 0; k < basis_vectors.size(); k++) {
                                    eigenvector[i] += evecs[k*total_dim + idx] * basis_vectors[k][i];
                                }
                            }
                            
                            // Normalize
                            double norm = cblas_dznrm2(N, eigenvector.data(), 1);
                            Complex scale = Complex(1.0/norm, 0.0);
                            cblas_zscal(N, &scale, eigenvector.data(), 1);
                        } else {
                            // Handle degenerate case
                            std::vector<ComplexVector> subspace(cluster.size(), ComplexVector(N, Complex(0.0, 0.0)));
                            
                            // Compute raw vectors
                            #pragma omp parallel for collapse(2) if(N > 1000) schedule(static)
                            for (size_t c = 0; c < cluster.size(); c++) {
                                for (int i = 0; i < N; i++) {
                                    for (size_t k = 0; k < basis_vectors.size(); k++) {
                                        subspace[c][i] += evecs[k*total_dim + cluster[c]] * basis_vectors[k][i];
                                    }
                                }
                            }
                            
                            // Re-orthogonalize
                            refine_degenerate_eigenvectors(H, subspace, evals[cluster[0]], N, tol);
                            
                            // Copy back
                            for (size_t c = 0; c < cluster.size(); c++) {
                                (*eigenvectors)[cluster[c]] = subspace[c];
                            }
                        }
                    }
                }
                
                #pragma omp taskwait
                
                // Optional refinement in parallel
                for (int i = 0; i < total_dim; i++) {
                    #pragma omp task
                    {
                        refine_eigenvector_with_cg(H, (*eigenvectors)[i], evals[i], N, tol);
                    }
                }
            }
        }
    } else {
        // Just compute eigenvalues
        info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', total_dim, 
                           reinterpret_cast<lapack_complex_double*>(block_matrix.data()), 
                           total_dim, evals.data());
                           
        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
            return;
        }
    }
    
    // Store the eigenvalues
    eigenvalues = evals;
}



// Automatically estimate spectral bounds and optimal parameters for Chebyshev filtering
struct ChebysehvFilterParams {
    double a;          // Lower bound of interval
    double b;          // Upper bound of interval
    int filter_degree; // Optimal filter degree
    int lanczos_iter;  // Recommended Lanczos iterations
};

ChebysehvFilterParams estimate_filter_parameters(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,               // Matrix dimension
    int num_eigenvalues, // Number of desired eigenvalues
    bool lowest = true,  // Whether to target lowest (true) or highest (false) eigenvalues
    int sample_iter = 30 // Number of Lanczos iterations for estimation
) {
    // 1. Run quick Lanczos to estimate spectral range
    std::vector<double> sample_eigenvalues;
    sample_iter = std::min(N, sample_iter);
    lanczos(H, N, sample_iter, 1e-10, sample_eigenvalues);
    
    // Sort eigenvalues
    std::sort(sample_eigenvalues.begin(), sample_eigenvalues.end());
    
    // 2. Estimate the full spectrum bounds
    double min_eig = sample_eigenvalues.front();
    double max_eig = sample_eigenvalues.back();
    
    // Add some margin to ensure we cover the full spectrum
    double buffer = (max_eig - min_eig) * 0.1;
    double global_min = min_eig - buffer;
    double global_max = max_eig + buffer;
    
    // 3. Define the target interval [a, b] based on which eigenvalues are desired
    double a, b;
    if (lowest) {
        // For lowest eigenvalues, set [a,b] to lower portion of spectrum
        a = global_min;
        // Set b to cover a bit more than the desired eigenvalue range
        int idx = std::min<int>(static_cast<int>(sample_eigenvalues.size() * 0.8), 
                              static_cast<int>(num_eigenvalues * 2));
        if (idx < sample_eigenvalues.size()) {
            b = sample_eigenvalues[idx];
        } else {
            b = global_max * 0.5;
        }
    } else {
        // For highest eigenvalues, set [a,b] to upper portion of spectrum
        b = global_max;
        int idx = std::max(0, static_cast<int>(sample_eigenvalues.size()) - 
                               static_cast<int>(num_eigenvalues * 2));
        if (idx < sample_eigenvalues.size()) {
            a = sample_eigenvalues[idx];
        } else {
            a = global_min * 0.5;
        }
    }
    
    // 4. Calculate filter degree based on spectrum width and desired accuracy
    double spectrum_width = global_max - global_min;
    double target_width = b - a;
    int filter_degree = static_cast<int>(15 * std::sqrt(spectrum_width / target_width));
    // Clamp to reasonable values
    filter_degree = std::min(std::max(filter_degree, 5), 50);
    
    // 5. Recommend Lanczos iterations - typically 2-3× the number of desired eigenvalues
    int lanczos_iter = std::min(N, std::max(2 * num_eigenvalues, 30));
    std::cout << "Estimated spectral bounds: [" << a << ", " << b << "]" << std::endl;
    std::cout << "Estimated filter degree: " << filter_degree << std::endl;
    return {a, b, filter_degree, lanczos_iter};
}

// Apply Chebyshev polynomial filter to a vector using CUDA
void chebyshev_filter(std::function<void(const Complex*, Complex*, int)> H,
                     const ComplexVector& v, ComplexVector& result,
                     int N, double a, double b, int degree) {    

    // Scale and shift parameters for mapping [a, b] to [-1, 1]
    double e = (b - a) / 2;    // Half-width of interval
    double c = (b + a) / 2;    // Center of interval
    
    // Create device vectors
    DeviceVector d_v_prev(N);
    DeviceVector d_v_curr(v);
    DeviceVector d_v_next(N);
    DeviceVector d_temp(N);
    
    // T_0(x) = 1, so v_curr = v (already done during initialization)
    
    // T_1(x) = x, so v_next = (H-c*I)*v / e
    ComplexVector h_v = d_v_curr.toHost();
    ComplexVector h_temp(N);
    H(h_v.data(), h_temp.data(), N);
    
    copyToDevice(d_temp.data, reinterpret_cast<const cuDoubleComplex*>(h_temp.data()), N);
    
    // Compute v_next = (temp - c*v)/e
    cuDoubleComplex neg_c = make_cuDoubleComplex(-c, 0);
    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_c, d_v_curr.data, 1, d_temp.data, 1));
    
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/e, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_temp.data, 1));
    
    // Copy result to v_next
    CHECK_CUDA(cudaMemcpy(d_v_next.data, d_temp.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    
    // Apply Chebyshev recurrence: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
    for (int k = 1; k < degree; k++) {
        // Store current as previous
        CHECK_CUDA(cudaMemcpy(d_v_prev.data, d_v_curr.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_v_curr.data, d_v_next.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        
        // Apply H to v_curr
        h_v = d_v_curr.toHost();
        H(h_v.data(), h_temp.data(), N);
        copyToDevice(d_temp.data, reinterpret_cast<const cuDoubleComplex*>(h_temp.data()), N);
        
        // Compute temp = temp - c*v_curr
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_c, d_v_curr.data, 1, d_temp.data, 1));
        
        // Scale by 2/e
        cuDoubleComplex scale2 = make_cuDoubleComplex(2.0/e, 0.0);
        CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale2, d_temp.data, 1));
        
        // v_next = temp - v_prev
        CHECK_CUDA(cudaMemcpy(d_v_next.data, d_temp.data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, &neg_one, d_v_prev.data, 1, d_v_next.data, 1));
    }
    
    // Normalize the result
    double norm;
    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_v_next.data, 1, &norm));
    
    cuDoubleComplex norm_scale = make_cuDoubleComplex(1.0/norm, 0.0);
    CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &norm_scale, d_v_next.data, 1));
    
    // Copy back to host
    result = d_v_next.toHost();
}

// Block Chebyshev filtered Lanczos algorithm using CUDA
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, 
                               int N, int max_iter, double tol, 
                               std::vector<double>& eigenvalues,
                               std::vector<ComplexVector>* eigenvectors = nullptr, 
                               double a = 0.0, double b = 0.0, int filter_degree = 0) {
    
    // Block size for handling degenerate eigenvalues
    const int block_size = 4;
    
    // Initialize CUDA
    CudaHelper::initialize();
    
    // Initialize random starting vectors
    std::vector<ComplexVector> block_vectors(block_size, ComplexVector(N));
    std::vector<DeviceVector> d_block_vectors;
    
    // Initialize thread-local random generators
    unsigned int seed = std::random_device{}();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Generate first random vector
    block_vectors[0] = generateRandomVector(N, gen, dist);
    
    // Generate orthogonal vectors
    for (int i = 1; i < block_size; i++) {
        block_vectors[i] = generateOrthogonalVector(N, 
                                                std::vector<ComplexVector>(block_vectors.begin(), 
                                                                        block_vectors.begin() + i), 
                                                gen, dist);
    }
    
    // Transfer vectors to device
    for (int i = 0; i < block_size; i++) {
        d_block_vectors.emplace_back(block_vectors[i]);
    }
    
    // Get filter parameters if not provided
    if (a == 0.0 && b == 0.0 && filter_degree == 0) {
        ChebysehvFilterParams params = estimate_filter_parameters(H, N, max_iter, true);
        a = params.a;
        b = params.b;
        filter_degree = params.filter_degree;
    }
    
    // Apply initial Chebyshev filter to each starting vector
    for (int i = 0; i < block_size; i++) {
        chebyshev_filter(H, block_vectors[i], block_vectors[i], N, a, b, filter_degree);
        // Update device vectors
        copyToDevice(d_block_vectors[i].data, reinterpret_cast<const cuDoubleComplex*>(block_vectors[i].data()), N);
    }
    
    // Re-orthonormalize after filtering
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < i; j++) {
            cuDoubleComplex overlap;
            CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, 
                                   d_block_vectors[j].data, 1, 
                                   d_block_vectors[i].data, 1, 
                                   &overlap));
            
            cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
            CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                   &neg_overlap, 
                                   d_block_vectors[j].data, 1, 
                                   d_block_vectors[i].data, 1));
        }
        
        double norm;
        CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_block_vectors[i].data, 1, &norm));
        
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
        CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_block_vectors[i].data, 1));
        
        // Update host copy
        block_vectors[i] = d_block_vectors[i].toHost();
    }
    
    // Initialize Lanczos basis vectors storage
    std::vector<ComplexVector> basis_vectors;
    std::vector<DeviceVector> d_basis_vectors;
    
    for (int i = 0; i < block_size; i++) {
        basis_vectors.push_back(block_vectors[i]);
        d_basis_vectors.push_back(d_block_vectors[i]);
    }
    
    // Create device vectors for the block Lanczos algorithm
    std::vector<DeviceVector> d_prev_block = d_block_vectors;
    std::vector<DeviceVector> d_curr_block = d_block_vectors;
    std::vector<DeviceVector> d_next_block;
    std::vector<DeviceVector> d_work_block;
    
    for (int i = 0; i < block_size; i++) {
        d_next_block.emplace_back(N);
        d_work_block.emplace_back(N);
    }
    
    // Block tridiagonal matrix elements
    std::vector<std::vector<std::vector<Complex>>> alpha;  // Diagonal blocks
    std::vector<std::vector<std::vector<Complex>>> beta;   // Off-diagonal blocks
    
    // First empty beta block
    beta.push_back(std::vector<std::vector<Complex>>(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0))));
    
    // Number of Lanczos steps
    int num_steps = max_iter / block_size;
    
    // Block Lanczos iteration
    for (int j = 0; j < num_steps; j++) {
        // Current alpha block
        std::vector<std::vector<Complex>> curr_alpha(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        // Apply H to each vector in the current block
        for (int b = 0; b < block_size; b++) {
            ComplexVector h_v = d_curr_block[b].toHost();
            ComplexVector h_work(N);
            H(h_v.data(), h_work.data(), N);
            copyToDevice(d_work_block[b].data, reinterpret_cast<const cuDoubleComplex*>(h_work.data()), N);
        }
        
        // Subtract beta_j * prev_block
        if (j > 0) {
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    Complex beta_jik = beta[j][i][k];
                    cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta_jik.real(), -beta_jik.imag());
                    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                           &neg_beta, 
                                           d_prev_block[k].data, 1, 
                                           d_work_block[i].data, 1));
                }
            }
        }
        
        // Compute alpha_j block and update residuals
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                cuDoubleComplex dot;
                CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, 
                                       d_curr_block[k].data, 1, 
                                       d_work_block[i].data, 1, 
                                       &dot));
                
                curr_alpha[i][k] = Complex(dot.x, dot.y);
                
                cuDoubleComplex neg_dot = make_cuDoubleComplex(-dot.x, -dot.y);
                CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                       &neg_dot, 
                                       d_curr_block[k].data, 1, 
                                       d_work_block[i].data, 1));
            }
        }
        
        alpha.push_back(curr_alpha);
        
        // Full reorthogonalization against all previous basis vectors
        for (int b = 0; b < block_size; b++) {
            for (int iter = 0; iter < 2; iter++) {  // Twice for stability
                // Create temporary copy of the work vector for consistent orthogonalization
                DeviceVector d_work_orig(N);
                CHECK_CUDA(cudaMemcpy(d_work_orig.data, d_work_block[b].data, 
                                     N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
                
                // Orthogonalize against all basis vectors
                for (size_t k = 0; k < d_basis_vectors.size(); k++) {
                    cuDoubleComplex overlap;
                    CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, 
                                           d_basis_vectors[k].data, 1, 
                                           d_work_orig.data, 1, 
                                           &overlap));
                    
                    cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
                    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                           &neg_overlap, 
                                           d_basis_vectors[k].data, 1, 
                                           d_work_block[b].data, 1));
                }
            }
        }
        
        // QR factorization of work block to get next orthonormal block
        std::vector<std::vector<Complex>> next_beta(block_size, std::vector<Complex>(block_size, Complex(0.0, 0.0)));
        
        // Process each vector
        for (int i = 0; i < block_size; i++) {
            // Compute norm
            double norm;
            CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_work_block[i].data, 1, &norm));
            
            // Handle invariant subspace or linear dependence
            if (norm < tol) {
                // Generate a new random vector orthogonal to basis
                ComplexVector new_vec = generateOrthogonalVector(N, basis_vectors, gen, dist);
                chebyshev_filter(H, new_vec, new_vec, N, a, b, filter_degree);
                copyToDevice(d_next_block[i].data, reinterpret_cast<const cuDoubleComplex*>(new_vec.data()), N);
                
                // Re-orthogonalize against basis vectors
                for (size_t k = 0; k < d_basis_vectors.size(); k++) {
                    cuDoubleComplex overlap;
                    CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, 
                                           d_basis_vectors[k].data, 1, 
                                           d_next_block[i].data, 1, 
                                           &overlap));
                    
                    cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
                    CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                           &neg_overlap, 
                                           d_basis_vectors[k].data, 1, 
                                           d_next_block[i].data, 1));
                }
                
                CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_next_block[i].data, 1, &norm));
            } else {
                // Copy work to next
                CHECK_CUDA(cudaMemcpy(d_next_block[i].data, d_work_block[i].data, 
                                     N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
            }
            
            // Set the diagonal beta element
            next_beta[i][i] = Complex(norm, 0.0);
            
            // Normalize
            cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
            CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_next_block[i].data, 1));
            
            // Orthogonalize remaining work vectors against this one
            for (int l = i + 1; l < block_size; l++) {
                cuDoubleComplex overlap;
                CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N, 
                                       d_next_block[i].data, 1, 
                                       d_work_block[l].data, 1, 
                                       &overlap));
                
                next_beta[l][i] = Complex(overlap.x, overlap.y);
                
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
                CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N, 
                                       &neg_overlap, 
                                       d_next_block[i].data, 1, 
                                       d_work_block[l].data, 1));
            }
        }
        
        beta.push_back(next_beta);
        
        // Store new basis vectors
        if (j < num_steps - 1) {
            for (int i = 0; i < block_size; i++) {
                ComplexVector host_vec = d_next_block[i].toHost();
                basis_vectors.push_back(host_vec);
                d_basis_vectors.push_back(d_next_block[i]);
            }
        }
        
        // Update for next iteration
        d_prev_block = d_curr_block;
        d_curr_block = d_next_block;
    }
    
    // Convert block tridiagonal matrix to regular format
    int total_dim = basis_vectors.size();
    
    // Allocate device memory for matrix and results
    double* d_alpha = allocDeviceMem<double>(total_dim);
    double* d_beta = allocDeviceMem<double>(total_dim-1);
    double* d_eigenvalues = allocDeviceMem<double>(total_dim);
    cuDoubleComplex* d_matrix = allocDeviceMem<cuDoubleComplex>(total_dim * total_dim);
    cuDoubleComplex* d_eigenvectors = nullptr;
    
    // Host matrix for constructing the block tridiagonal
    std::vector<Complex> h_matrix(total_dim * total_dim, Complex(0.0, 0.0));
    
    // Fill diagonal blocks (alpha)
    for (size_t j = 0; j < alpha.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = j * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    h_matrix[col * total_dim + row] = alpha[j][r][c];
                }
            }
        }
    }
    
    // Fill off-diagonal blocks (beta)
    for (size_t j = 1; j < beta.size(); j++) {
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                int row = (j-1) * block_size + r;
                int col = j * block_size + c;
                if (row < total_dim && col < total_dim) {
                    h_matrix[col * total_dim + row] = beta[j][r][c];
                    h_matrix[row * total_dim + col] = std::conj(beta[j][r][c]);
                }
            }
        }
    }
    
    // Copy matrix to device
    copyToDevice(d_matrix, reinterpret_cast<const cuDoubleComplex*>(h_matrix.data()), total_dim * total_dim);
    
    // Allocate workspace for cuSOLVER
    int lwork = 0;
    cuDoubleComplex* d_work = nullptr;
    int* d_info = allocDeviceMem<int>(1);
    
    // Whether to compute eigenvectors
    char jobz = eigenvectors ? 'V' : 'N';
    
    // Get required workspace size
    if (eigenvectors) {
        d_eigenvectors = allocDeviceMem<cuDoubleComplex>(total_dim * total_dim);
        CHECK_CUDA(cudaMemcpy(d_eigenvectors, d_matrix, 
                             total_dim * total_dim * sizeof(cuDoubleComplex), 
                             cudaMemcpyDeviceToDevice));
    }
    
    // Get workspace size
    CHECK_CUSOLVER(cusolverDnZheevd_bufferSize(
        CudaHelper::getCusolverHandle(),
        jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
        CUBLAS_FILL_MODE_UPPER,
        total_dim,
        d_eigenvectors ? d_eigenvectors : d_matrix,
        total_dim,
        d_eigenvalues,
        &lwork
    ));
    
    // Allocate workspace
    d_work = allocDeviceMem<cuDoubleComplex>(lwork);
    
    // Solve eigenvalue problem
    CHECK_CUSOLVER(cusolverDnZheevd(
        CudaHelper::getCusolverHandle(),
        jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
        CUBLAS_FILL_MODE_UPPER,
        total_dim,
        d_eigenvectors ? d_eigenvectors : d_matrix,
        total_dim,
        d_eigenvalues,
        d_work,
        lwork,
        d_info
    ));
    
    // Check for errors
    int info = 0;
    copyToHost(&info, d_info, 1);
    
    if (info != 0) {
        std::cerr << "cusolverDnZheevd failed with error code " << info << std::endl;
        // Clean up and return
        freeDeviceMem(d_alpha);
        freeDeviceMem(d_beta);
        freeDeviceMem(d_eigenvalues);
        freeDeviceMem(d_matrix);
        if (d_eigenvectors) freeDeviceMem(d_eigenvectors);
        freeDeviceMem(d_work);
        freeDeviceMem(d_info);
        return;
    }
    
    // Copy eigenvalues to host
    eigenvalues.resize(total_dim);
    copyToHost(eigenvalues.data(), d_eigenvalues, total_dim);
    
    // Transform eigenvectors back to original basis if requested
    if (eigenvectors && d_eigenvectors) {
        // Copy eigenvectors to host
        std::vector<Complex> h_eigenvectors(total_dim * total_dim);
        copyToHost(reinterpret_cast<cuDoubleComplex*>(h_eigenvectors.data()), 
                  d_eigenvectors, total_dim * total_dim);
        
        eigenvectors->clear();
        eigenvectors->resize(total_dim, ComplexVector(N, Complex(0.0, 0.0)));
        
        // Group eigenvalues into degenerate clusters
        const double degen_tol = 1e-10;
        std::vector<std::vector<int>> degen_clusters;
        
        for (int i = 0; i < total_dim; i++) {
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
        
        // Process each cluster
        for (const auto& cluster : degen_clusters) {
            if (cluster.size() == 1) {
                // Non-degenerate case
                int idx = cluster[0];
                ComplexVector& eigenvector = (*eigenvectors)[idx];
                std::fill(eigenvector.begin(), eigenvector.end(), Complex(0.0, 0.0));
                
                // Transform eigenvector using basis vectors
                for (size_t k = 0; k < basis_vectors.size(); k++) {
                    Complex coef = h_eigenvectors[k*total_dim + idx];
                    for (int i = 0; i < N; i++) {
                        eigenvector[i] += coef * basis_vectors[k][i];
                    }
                }
                
                // Normalize
                double norm = 0.0;
                for (int i = 0; i < N; i++) {
                    norm += std::norm(eigenvector[i]);
                }
                norm = std::sqrt(norm);
                
                if (norm > tol) {
                    for (int i = 0; i < N; i++) {
                        eigenvector[i] /= norm;
                    }
                }
            } else {
                // Degenerate case
                int subspace_dim = cluster.size();
                std::vector<ComplexVector> subspace_vectors(subspace_dim, ComplexVector(N, Complex(0.0, 0.0)));
                
                // Compute raw eigenvectors in original basis
                for (int c = 0; c < subspace_dim; c++) {
                    for (int i = 0; i < N; i++) {
                        for (size_t k = 0; k < basis_vectors.size(); k++) {
                            subspace_vectors[c][i] += h_eigenvectors[k*total_dim + cluster[c]] * basis_vectors[k][i];
                        }
                    }
                }
                
                // Orthogonalize within degenerate subspace
                for (int c = 0; c < subspace_dim; c++) {
                    // Create device vector
                    DeviceVector d_vec(subspace_vectors[c]);
                    
                    // Orthogonalize against previous vectors
                    for (int prev = 0; prev < c; prev++) {
                        DeviceVector d_prev(subspace_vectors[prev]);
                        
                        cuDoubleComplex overlap;
                        CHECK_CUBLAS(cublasZdotc(CudaHelper::getCublasHandle(), N,
                                               d_prev.data, 1,
                                               d_vec.data, 1,
                                               &overlap));
                        
                        cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.x, -overlap.y);
                        CHECK_CUBLAS(cublasZaxpy(CudaHelper::getCublasHandle(), N,
                                               &neg_overlap,
                                               d_prev.data, 1,
                                               d_vec.data, 1));
                    }
                    
                    // Normalize
                    double norm;
                    CHECK_CUBLAS(cublasDznrm2(CudaHelper::getCublasHandle(), N, d_vec.data, 1, &norm));
                    
                    if (norm > tol) {
                        cuDoubleComplex scale = make_cuDoubleComplex(1.0/norm, 0.0);
                        CHECK_CUBLAS(cublasZscal(CudaHelper::getCublasHandle(), N, &scale, d_vec.data, 1));
                    }
                    
                    // Copy back to host
                    subspace_vectors[c] = d_vec.toHost();
                    
                    // Store the refined eigenvector
                    int idx = cluster[c];
                    (*eigenvectors)[idx] = subspace_vectors[c];
                }
            }
        }
        
        // Optional: Refine eigenvectors
        for (int i = 0; i < std::min(20, total_dim); i++) {
            refine_eigenvector_with_cg(H, (*eigenvectors)[i], eigenvalues[i], N, tol);
        }
    }
    
    // Free device memory
    freeDeviceMem(d_alpha);
    freeDeviceMem(d_beta);
    freeDeviceMem(d_eigenvalues);
    freeDeviceMem(d_matrix);
    if (d_eigenvectors) freeDeviceMem(d_eigenvectors);
    freeDeviceMem(d_work);
    freeDeviceMem(d_info);
}
