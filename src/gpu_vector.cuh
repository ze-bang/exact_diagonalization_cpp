// gpu_vector.cuh - GPU vector class with BLAS operations
#ifndef GPU_VECTOR_CUH
#define GPU_VECTOR_CUH

#include "gpu_utils.cuh"
#include <vector>
#include <complex>
#include <memory>

namespace gpu {

/**
 * GPU Vector class for complex double vectors
 * Provides BLAS operations using cuBLAS
 */
class GPUVector {
private:
    cuDoubleComplex* d_data_;
    size_t size_;
    cublasHandle_t cublas_handle_;
    bool owns_handle_;
    
public:
    /**
     * Constructor: allocate GPU memory
     */
    GPUVector(size_t size, cublasHandle_t handle = nullptr) 
        : size_(size), owns_handle_(handle == nullptr) {
        
        if (owns_handle_) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        } else {
            cublas_handle_ = handle;
        }
        
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(cuDoubleComplex)));
    }
    
    /**
     * Destructor: free GPU memory
     */
    ~GPUVector() {
        if (d_data_) {
            cudaFree(d_data_);
        }
        if (owns_handle_ && cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
    }
    
    // Disable copy constructor (use clone() for explicit copy)
    GPUVector(const GPUVector&) = delete;
    GPUVector& operator=(const GPUVector&) = delete;
    
    // Enable move constructor
    GPUVector(GPUVector&& other) noexcept 
        : d_data_(other.d_data_), size_(other.size_), 
          cublas_handle_(other.cublas_handle_), owns_handle_(other.owns_handle_) {
        other.d_data_ = nullptr;
        other.cublas_handle_ = nullptr;
        other.owns_handle_ = false;
    }
    
    /**
     * Get size
     */
    size_t size() const { return size_; }
    
    /**
     * Get raw device pointer (read-only)
     */
    const cuDoubleComplex* data() const { return d_data_; }
    
    /**
     * Get raw device pointer (read-write)
     */
    cuDoubleComplex* data() { return d_data_; }
    
    /**
     * Get cuBLAS handle
     */
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
    
    /**
     * Upload data from CPU to GPU
     */
    void upload(const std::vector<std::complex<double>>& host_vec) {
        if (host_vec.size() != size_) {
            throw std::runtime_error("Vector size mismatch in upload");
        }
        CUDA_CHECK(cudaMemcpy(d_data_, host_vec.data(), 
                             size_ * sizeof(cuDoubleComplex), 
                             cudaMemcpyHostToDevice));
    }
    
    /**
     * Download data from GPU to CPU
     */
    void download(std::vector<std::complex<double>>& host_vec) const {
        host_vec.resize(size_);
        CUDA_CHECK(cudaMemcpy(host_vec.data(), d_data_, 
                             size_ * sizeof(cuDoubleComplex), 
                             cudaMemcpyDeviceToHost));
    }
    
    /**
     * Set all elements to zero
     */
    void zero() {
        CUDA_CHECK(cudaMemset(d_data_, 0, size_ * sizeof(cuDoubleComplex)));
    }
    
    /**
     * Fill with random values (uniform distribution)
     */
    void randomize(unsigned int seed = 0) {
        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        if (seed != 0) {
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        }
        
        // Generate random real and imaginary parts
        CURAND_CHECK(curandGenerateUniformDouble(gen, 
                                                 reinterpret_cast<double*>(d_data_), 
                                                 2 * size_));
        CURAND_CHECK(curandDestroyGenerator(gen));
        
        // Shift from [0,1] to [-1,1]
        double alpha = 2.0;
        double beta = -1.0;
        CUBLAS_CHECK(cublasDscal(cublas_handle_, 2 * size_, &alpha, 
                                reinterpret_cast<double*>(d_data_), 1));
        
        // Note: This treats the complex vector as a real vector of length 2*size
        // A more sophisticated approach would use a custom kernel
    }
    
    /**
     * Compute L2 norm: ||v||_2
     */
    double norm() const {
        double result;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle_, size_, d_data_, 1, &result));
        return result;
    }
    
    /**
     * Normalize the vector: v = v / ||v||
     */
    void normalize() {
        double n = norm();
        if (n < 1e-14) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        cuDoubleComplex scale = make_cuDoubleComplex(1.0/n, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle_, size_, &scale, d_data_, 1));
    }
    
    /**
     * Compute dot product: <this|other>
     * Note: cuBLAS computes conj(this) . other
     */
    std::complex<double> dot(const GPUVector& other) const {
        if (size_ != other.size_) {
            throw std::runtime_error("Vector size mismatch in dot product");
        }
        
        cuDoubleComplex result;
        CUBLAS_CHECK(cublasZdotc(cublas_handle_, size_, d_data_, 1, 
                                other.d_data_, 1, &result));
        
        return std::complex<double>(cuCreal(result), cuCimag(result));
    }
    
    /**
     * AXPY operation: this = alpha * x + this
     */
    void axpy(std::complex<double> alpha, const GPUVector& x) {
        if (size_ != x.size_) {
            throw std::runtime_error("Vector size mismatch in axpy");
        }
        
        cuDoubleComplex cu_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
        CUBLAS_CHECK(cublasZaxpy(cublas_handle_, size_, &cu_alpha, 
                                x.d_data_, 1, d_data_, 1));
    }
    
    /**
     * Scale operation: this = alpha * this
     */
    void scale(std::complex<double> alpha) {
        cuDoubleComplex cu_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
        CUBLAS_CHECK(cublasZscal(cublas_handle_, size_, &cu_alpha, d_data_, 1));
    }
    
    /**
     * Copy operation: this = other
     */
    void copy_from(const GPUVector& other) {
        if (size_ != other.size_) {
            throw std::runtime_error("Vector size mismatch in copy");
        }
        
        CUBLAS_CHECK(cublasZcopy(cublas_handle_, size_, other.d_data_, 1, d_data_, 1));
    }
    
    /**
     * Create a clone on GPU
     */
    std::unique_ptr<GPUVector> clone() const {
        auto result = std::make_unique<GPUVector>(size_, cublas_handle_);
        result->copy_from(*this);
        return result;
    }
    
    /**
     * Swap with another vector (efficient pointer swap)
     */
    void swap(GPUVector& other) {
        if (size_ != other.size_) {
            throw std::runtime_error("Cannot swap vectors of different sizes");
        }
        std::swap(d_data_, other.d_data_);
    }
    
    /**
     * Linear combination: this = alpha * x + beta * y
     */
    void linear_combination(std::complex<double> alpha, const GPUVector& x,
                           std::complex<double> beta, const GPUVector& y) {
        if (size_ != x.size_ || size_ != y.size_) {
            throw std::runtime_error("Vector size mismatch in linear combination");
        }
        
        // this = alpha * x
        cuDoubleComplex cu_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
        CUBLAS_CHECK(cublasZcopy(cublas_handle_, size_, x.d_data_, 1, d_data_, 1));
        CUBLAS_CHECK(cublasZscal(cublas_handle_, size_, &cu_alpha, d_data_, 1));
        
        // this += beta * y
        cuDoubleComplex cu_beta = make_cuDoubleComplex(beta.real(), beta.imag());
        CUBLAS_CHECK(cublasZaxpy(cublas_handle_, size_, &cu_beta, 
                                y.d_data_, 1, d_data_, 1));
    }
    
    /**
     * Orthogonalize against another vector: this = this - <other|this> * other
     */
    void orthogonalize_against(const GPUVector& other) {
        std::complex<double> projection = other.dot(*this);
        this->axpy(-projection, other);
    }
    
    /**
     * Print statistics (for debugging)
     */
    void print_stats(const std::string& name = "") const {
        std::vector<std::complex<double>> host_vec;
        download(host_vec);
        
        double max_abs = 0.0;
        size_t max_idx = 0;
        double sum_abs = 0.0;
        
        for (size_t i = 0; i < size_; i++) {
            double abs_val = std::abs(host_vec[i]);
            sum_abs += abs_val;
            if (abs_val > max_abs) {
                max_abs = abs_val;
                max_idx = i;
            }
        }
        
        std::cout << "Vector stats";
        if (!name.empty()) std::cout << " (" << name << ")";
        std::cout << ":" << std::endl;
        std::cout << "  Size: " << size_ << std::endl;
        std::cout << "  Norm: " << norm() << std::endl;
        std::cout << "  Max |element|: " << max_abs << " at index " << max_idx << std::endl;
        std::cout << "  Mean |element|: " << sum_abs / size_ << std::endl;
        std::cout << "  Max element: " << host_vec[max_idx] << std::endl;
    }
};

} // namespace gpu

#endif // GPU_VECTOR_CUH
