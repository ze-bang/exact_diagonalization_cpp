// TPQ_cuda.cuh - CUDA implementation of Thermal Pure Quantum state

#ifndef TPQ_CUDA_CUH
#define TPQ_CUDA_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include "observables.h"
#include "construct_ham.h"

// Type definitions for CUDA
using ComplexCuda = thrust::complex<double>;
using ComplexVectorDevice = thrust::device_vector<ComplexCuda>;
using ComplexVectorHost = std::vector<Complex>;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ \
                     << " code=" << status << std::endl; \
            exit(1); \
        } \
    } while(0)

// Convert between host and device complex types
__host__ __device__ inline ComplexCuda toDeviceComplex(const Complex& c) {
    return ComplexCuda(c.real(), c.imag());
}

__host__ __device__ inline Complex toHostComplex(const ComplexCuda& c) {
    return Complex(c.real(), c.imag());
}

// CUDA kernel for applying sparse Hermitian matrix
__global__ void sparseHermitianMatVecKernel(
    const int* rowPtr,
    const int* colIdx,
    const ComplexCuda* values,
    const ComplexCuda* x,
    ComplexCuda* y,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    ComplexCuda sum = ComplexCuda(0.0, 0.0);
    
    for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
        int col = colIdx[idx];
        sum += values[idx] * x[col];
    }
    
    y[row] = sum;
}

// CUDA kernel for complex vector operations
__global__ void complexAxpyKernel(
    ComplexCuda alpha,
    const ComplexCuda* x,
    ComplexCuda* y,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// CUDA kernel for element-wise complex multiplication and reduction
__global__ void complexDotProductKernel(
    const ComplexCuda* x,
    const ComplexCuda* y,
    ComplexCuda* partial_sums,
    int N
) {
    extern __shared__ ComplexCuda sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    ComplexCuda sum = ComplexCuda(0.0, 0.0);
    
    // Grid-stride loop
    while (idx < N) {
        sum += thrust::conj(x[idx]) * y[idx];
        idx += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Sparse matrix wrapper for CUDA
class SparseMatrixCuda {
private:
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matDescr;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer;
    size_t bufferSize;
    
public:
    thrust::device_vector<int> rowPtr;
    thrust::device_vector<int> colIdx;
    thrust::device_vector<ComplexCuda> values;
    int N;
    int nnz;
    
    SparseMatrixCuda(int n) : N(n), nnz(0), dBuffer(nullptr), bufferSize(0) {
        CUSPARSE_CHECK(cusparseCreate(&handle));
    }
    
    ~SparseMatrixCuda() {
        if (matDescr) cusparseDestroySpMat(matDescr);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (dBuffer) cudaFree(dBuffer);
        cusparseDestroy(handle);
    }
    
    void setFromCSR(const int* h_rowPtr, const int* h_colIdx, const ComplexCuda* h_values, int h_nnz) {
        nnz = h_nnz;
        rowPtr.resize(N + 1);
        colIdx.resize(nnz);
        values.resize(nnz);
        
        thrust::copy(h_rowPtr, h_rowPtr + N + 1, rowPtr.begin());
        thrust::copy(h_colIdx, h_colIdx + nnz, colIdx.begin());
        thrust::copy(h_values, h_values + nnz, values.begin());
        
        // Create sparse matrix descriptor
        CUSPARSE_CHECK(cusparseCreateCsr(&matDescr, N, N, nnz,
            thrust::raw_pointer_cast(rowPtr.data()),
            thrust::raw_pointer_cast(colIdx.data()),
            thrust::raw_pointer_cast(values.data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    }
    
    void apply(const ComplexCuda* d_x, ComplexCuda* d_y) {
        const ComplexCuda alpha = ComplexCuda(1.0, 0.0);
        const ComplexCuda beta = ComplexCuda(0.0, 0.0);
        
        // Create dense vector descriptors
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, (void*)d_x, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, N, (void*)d_y, CUDA_C_64F));
        
        // Allocate buffer if needed
        size_t newBufferSize = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescr, vecX, &beta, vecY, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &newBufferSize));
            
        if (newBufferSize > bufferSize) {
            if (dBuffer) cudaFree(dBuffer);
            CUDA_CHECK(cudaMalloc(&dBuffer, newBufferSize));
            bufferSize = newBufferSize;
        }
        
        // Perform SpMV
        CUSPARSE_CHECK(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescr, vecX, &beta, vecY, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
            
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    }
};

// CUDA version of generateTPQVector
ComplexVectorDevice generateTPQVectorCuda(int N, unsigned int seed) {
    ComplexVectorDevice v(N);
    
    // Create cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    // Generate random numbers
    thrust::device_vector<double> random_data(2 * N);
    curandGenerateUniformDouble(gen, thrust::raw_pointer_cast(random_data.data()), 2 * N);
    
    // Transform to complex numbers in range [-1, 1]
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        v.begin(),
        [raw_ptr = thrust::raw_pointer_cast(random_data.data())] __device__ (int i) {
            return ComplexCuda(2.0 * raw_ptr[2*i] - 1.0, 2.0 * raw_ptr[2*i+1] - 1.0);
        }
    );
    
    // Normalize
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    
    double norm;
    CUBLAS_CHECK(cublasDznrm2(cublasHandle, N, 
        reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v.data())), 1, &norm));
    
    ComplexCuda scale_factor(1.0/norm, 0.0);
    CUBLAS_CHECK(cublasZscal(cublasHandle, N, 
        reinterpret_cast<cuDoubleComplex*>(&scale_factor),
        reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v.data())), 1));
    
    cublasDestroy(cublasHandle);
    curandDestroyGenerator(gen);
    
    return v;
}

// CUDA version of calculateEnergyAndVariance
std::pair<double, double> calculateEnergyAndVarianceCuda(
    SparseMatrixCuda& H,
    const ComplexVectorDevice& v,
    int N
) {
    ComplexVectorDevice Hv(N);
    ComplexVectorDevice H2v(N);
    
    // Calculate H|v⟩
    H.apply(thrust::raw_pointer_cast(v.data()), thrust::raw_pointer_cast(Hv.data()));
    
    // Calculate H²|v⟩
    H.apply(thrust::raw_pointer_cast(Hv.data()), thrust::raw_pointer_cast(H2v.data()));
    
    // Calculate energy = ⟨v|H|v⟩
    ComplexCuda energy_complex = thrust::inner_product(
        v.begin(), v.end(), Hv.begin(), 
        ComplexCuda(0.0, 0.0),
        thrust::plus<ComplexCuda>(),
        [] __device__ (const ComplexCuda& a, const ComplexCuda& b) {
            return thrust::conj(a) * b;
        }
    );
    
    // Calculate ⟨v|H²|v⟩
    ComplexCuda h2_complex = thrust::inner_product(
        v.begin(), v.end(), H2v.begin(),
        ComplexCuda(0.0, 0.0),
        thrust::plus<ComplexCuda>(),
        [] __device__ (const ComplexCuda& a, const ComplexCuda& b) {
            return thrust::conj(a) * b;
        }
    );
    
    double energy = energy_complex.real();
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

// CUDA kernel for time evolution using Taylor expansion
__global__ void timeEvolveKernel(
    const ComplexCuda* coefficients,
    const ComplexCuda* current_term,
    ComplexCuda* result,
    int order,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] += coefficients[order] * current_term[idx];
    }
}

// CUDA version of time evolution
void time_evolve_tpq_state_cuda(
    SparseMatrixCuda& H,
    ComplexVectorDevice& tpq_state,
    int N,
    double delta_t,
    int n_max = 100,
    bool normalize = true
) {
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    
    ComplexVectorDevice result(N);
    ComplexVectorDevice term(N);
    ComplexVectorDevice Hterm(N);
    
    // Copy initial state
    thrust::copy(tpq_state.begin(), tpq_state.end(), term.begin());
    thrust::copy(tpq_state.begin(), tpq_state.end(), result.begin());
    
    // Precompute coefficients
    thrust::device_vector<ComplexCuda> coefficients(n_max + 1);
    thrust::host_vector<ComplexCuda> h_coefficients(n_max + 1);
    
    h_coefficients[0] = ComplexCuda(1.0, 0.0);
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        ComplexCuda coef = thrust::pow(ComplexCuda(0.0, -1.0), order);
        h_coefficients[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    coefficients = h_coefficients;
    
    // Apply Taylor expansion terms
    for (int order = 1; order <= n_max; order++) {
        // Apply H to the previous term
        H.apply(thrust::raw_pointer_cast(term.data()), thrust::raw_pointer_cast(Hterm.data()));
        thrust::swap(term, Hterm);
        
        // Add this term to the result
        ComplexCuda coef = h_coefficients[order];
        CUBLAS_CHECK(cublasZaxpy(cublasHandle, N,
            reinterpret_cast<cuDoubleComplex*>(&coef),
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(term.data())), 1,
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(result.data())), 1));
    }
    
    thrust::swap(tpq_state, result);
    
    // Normalize if requested
    if (normalize) {
        double norm;
        CUBLAS_CHECK(cublasDznrm2(cublasHandle, N,
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(tpq_state.data())), 1, &norm));
        
        ComplexCuda scale_factor(1.0/norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublasHandle, N,
            reinterpret_cast<cuDoubleComplex*>(&scale_factor),
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(tpq_state.data())), 1));
    }
    
    cublasDestroy(cublasHandle);
}

// CUDA wrapper for microcanonical TPQ with same interface as CPU version
void microcanonical_tpq_cuda(
    std::function<void(const Complex*, Complex*, int)> H_cpu,
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
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    int num_points = 10000,
    double t_end = 100.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    int sublattice_size = 1
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    int num_sites = static_cast<int>(std::log2(N));
    eigenvalues.clear();
    
    // Convert Hamiltonian to sparse matrix format on CPU first
    // This is a placeholder - you'll need to implement the actual conversion
    // based on your Hamiltonian structure
    std::vector<int> h_rowPtr(N + 1);
    std::vector<int> h_colIdx;
    std::vector<ComplexCuda> h_values;
    
    // Build sparse matrix (this is problem-specific)
    // For now, we'll create a dummy sparse matrix structure
    // You need to replace this with actual Hamiltonian construction
    
    // Create CUDA sparse matrix
    SparseMatrixCuda H_cuda(N);
    // H_cuda.setFromCSR(h_rowPtr.data(), h_colIdx.data(), h_values.data(), h_values.size());
    
    // Initialize cuBLAS
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "TPQ CUDA sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state on GPU
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVectorDevice v1 = generateTPQVectorCuda(N, seed);
        
        // Apply Hamiltonian to get v0 = H|v1⟩
        ComplexVectorDevice v0(N);
        H_cuda.apply(thrust::raw_pointer_cast(v1.data()), thrust::raw_pointer_cast(v0.data()));
        
        // Compute v0 = (L-H)|v1⟩
        thrust::transform(
            v0.begin(), v0.end(), v1.begin(), v0.begin(),
            [L=LargeValue] __device__ (const ComplexCuda& h_val, const ComplexCuda& v_val) {
                return L * v_val - h_val;
            }
        );
        
        // Calculate energy and variance
        auto [energy1, variance1] = calculateEnergyAndVarianceCuda(H_cuda, v0, N);
        double inv_temp = 2.0 / (LargeValue * num_sites - energy1);
        
        // Normalize v0
        double first_norm;
        CUBLAS_CHECK(cublasDznrm2(cublasHandle, N,
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v0.data())), 1, &first_norm));
        
        ComplexCuda scale_factor(1.0/first_norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublasHandle, N,
            reinterpret_cast<cuDoubleComplex*>(&scale_factor),
            reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v0.data())), 1));
        
        double current_norm = first_norm;
        
        // Write initial data
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, 1);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " "
                     << current_norm << " " << first_norm << " " << 1 << std::endl;
        }
        
        // Main TPQ loop
        for (int step = 2; step <= max_iter; step++) {
            // Apply H again
            ComplexVectorDevice v_new(N);
            H_cuda.apply(thrust::raw_pointer_cast(v0.data()), thrust::raw_pointer_cast(v_new.data()));
            
            // Compute v_new = (L-H)|v0⟩
            thrust::transform(
                v_new.begin(), v_new.end(), v0.begin(), v_new.begin(),
                [L=LargeValue] __device__ (const ComplexCuda& h_val, const ComplexCuda& v_val) {
                    return L * v_val - h_val;
                }
            );
            
            // Normalize
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublasHandle, N,
                reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v_new.data())), 1, &norm));
            
            current_norm *= norm;
            
            ComplexCuda norm_factor(1.0/norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublasHandle, N,
                reinterpret_cast<cuDoubleComplex*>(&norm_factor),
                reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v_new.data())), 1));
            
            thrust::swap(v0, v_new);
            
            // Calculate observables periodically
            if (step % temp_interval == 0) {
                auto [energy, variance] = calculateEnergyAndVarianceCuda(H_cuda, v0, N);
                inv_temp = (2.0 * step) / (LargeValue * num_sites - energy);
                
                writeTPQData(ss_file, inv_temp, energy, variance, current_norm, step);
                
                {
                    std::ofstream norm_out(norm_file, std::ios::app);
                    norm_out << std::setprecision(16) << inv_temp << " "
                             << current_norm << " " << first_norm << " " << step << std::endl;
                }
                
                // Copy state to host if needed for measurements
                if (measure_sz || compute_observables) {
                    ComplexVectorHost h_state(N);
                    thrust::copy(v0.begin(), v0.end(), h_state.begin());
                    
                    // Perform measurements on host for now
                    // You can implement CUDA versions of these as well
                    if (measure_sz) {
                        // Call CPU version of writeFluctuationData
                        // This is a placeholder - implement CUDA version if needed
                    }
                    
                    if (compute_observables && 
                        (std::abs(inv_temp - 10.0) < 0.5 || 
                         std::abs(inv_temp - 100.0) < 5.0 || 
                         std::abs(inv_temp - 1000.0) < 50.0)) {
                        // Call CPU version of computeObservableDynamics
                        // This is a placeholder - implement CUDA version if needed
                    }
                }
            }
            
            // Check for convergence
            if (variance1 < 1e-14) {
                std::cout << "Converged at step " << step << std::endl;
                break;
            }
        }
        
        eigenvalues.push_back(energy1);
    }
    
    cublasDestroy(cublasHandle);
}

// Helper function to convert sparse matrix from host to device
class HamiltonianCuda {
private:
    SparseMatrixCuda sparseMatrix;
    
public:
    HamiltonianCuda(int N) : sparseMatrix(N) {}
    
    void constructFromFunction(std::function<void(const Complex*, Complex*, int)> H_cpu, int N) {
        // This is a placeholder implementation
        // You need to implement actual sparse matrix construction based on your Hamiltonian
        
        // For demonstration, create identity matrix
        std::vector<int> rowPtr(N + 1);
        std::vector<int> colIdx(N);
        std::vector<ComplexCuda> values(N);
        
        for (int i = 0; i <= N; i++) {
            rowPtr[i] = i;
        }
        for (int i = 0; i < N; i++) {
            colIdx[i] = i;
            values[i] = ComplexCuda(1.0, 0.0);
        }
        
        sparseMatrix.setFromCSR(rowPtr.data(), colIdx.data(), values.data(), N);
    }
    
    void apply(const ComplexCuda* x, ComplexCuda* y) {
        sparseMatrix.apply(x, y);
    }
    
    SparseMatrixCuda& getMatrix() { return sparseMatrix; }
};

#endif // TPQ_CUDA_CUH