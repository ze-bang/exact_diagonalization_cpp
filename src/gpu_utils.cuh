// gpu_utils.cuh - GPU utility functions and error checking
#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuComplex.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Simple pair structure for device code (avoids thrust::pair C++17 issues)
template<typename T1, typename T2>
struct device_pair {
    T1 first;
    T2 second;
    
    __device__ __host__ device_pair() : first(), second() {}
    __device__ __host__ device_pair(const T1& f, const T2& s) : first(f), second(s) {}
};

template<typename T1, typename T2>
__device__ __host__ inline device_pair<T1, T2> make_device_pair(const T1& f, const T2& s) {
    return device_pair<T1, T2>(f, s);
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

// cuBLAS error checking
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("cuBLAS error code: " + std::to_string(status)); \
        } \
    } while(0)

// cuRAND error checking
#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("cuRAND error code: " + std::to_string(status)); \
        } \
    } while(0)

namespace gpu {

/**
 * Get GPU device properties and display information
 */
inline void print_gpu_info(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " 
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
              << " GB/s" << std::endl;
    std::cout << "=======================" << std::endl;
}

/**
 * Check if sufficient GPU memory is available
 */
inline bool check_gpu_memory(size_t required_bytes, int device = 0) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    std::cout << "GPU Memory: " << free_mem / (1024*1024*1024.0) << " GB free / "
              << total_mem / (1024*1024*1024.0) << " GB total" << std::endl;
    std::cout << "Required: " << required_bytes / (1024*1024*1024.0) << " GB" << std::endl;
    
    return free_mem >= required_bytes;
}

/**
 * Get optimal thread block size for a given problem size
 */
inline dim3 get_optimal_block_size(size_t problem_size) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Use 256 threads per block as a good default for most GPUs
    int threads_per_block = 256;
    if (prop.maxThreadsPerBlock < threads_per_block) {
        threads_per_block = prop.maxThreadsPerBlock;
    }
    
    return dim3(threads_per_block);
}

/**
 * Get optimal grid size for a given problem size and block size
 */
inline dim3 get_optimal_grid_size(size_t problem_size, dim3 block_size) {
    size_t grid_size = (problem_size + block_size.x - 1) / block_size.x;
    return dim3(grid_size);
}

/**
 * Device function: Get Sz eigenvalue for a site in a basis state
 */
__device__ inline double get_sz_eigenvalue(uint64_t state, int site, float spin_length) {
    // Bit value 0 => spin up, bit value 1 => spin down
    bool is_down = (state >> site) & 1ULL;
    return is_down ? -spin_length : spin_length;
}

/**
 * Device function: Apply S+ operator at a site
 * Returns: {new_state, matrix_element}
 * If transition not allowed, returns {0, 0.0}
 */
__device__ inline device_pair<uint64_t, cuDoubleComplex> 
apply_splus(uint64_t state, int site, float spin_length) {
    // S+ flips spin from down (bit=1) to up (bit=0)
    bool is_down = (state >> site) & 1ULL;
    if (!is_down) {
        // Already up, S+|up⟩ = 0
        return make_device_pair(static_cast<uint64_t>(0), make_cuDoubleComplex(0.0, 0.0));
    }
    
    // Flip the bit: down (1) -> up (0)
    uint64_t new_state = state & ~(static_cast<uint64_t>(1) << site);
    
    // Matrix element for S+ is sqrt(s(s+1) - m(m+1)) = sqrt(2s) for spin-s
    double matrix_elem = sqrt(2.0 * spin_length);
    
    return make_device_pair(new_state, make_cuDoubleComplex(matrix_elem, 0.0));
}

/**
 * Device function: Apply S- operator at a site
 */
__device__ inline device_pair<uint64_t, cuDoubleComplex> 
apply_sminus(uint64_t state, int site, float spin_length) {
    // S- flips spin from up (bit=0) to down (bit=1)
    bool is_down = (state >> site) & 1ULL;
    if (is_down) {
        // Already down, S-|down⟩ = 0
        return make_device_pair(static_cast<uint64_t>(0), make_cuDoubleComplex(0.0, 0.0));
    }
    
    // Flip the bit: up (0) -> down (1)
    uint64_t new_state = state | (static_cast<uint64_t>(1) << site);
    
    // Matrix element for S- is sqrt(s(s+1) - m(m-1)) = sqrt(2s) for spin-s
    double matrix_elem = sqrt(2.0 * spin_length);
    
    return make_device_pair(new_state, make_cuDoubleComplex(matrix_elem, 0.0));
}

/**
 * Device function: Apply Sx = (S+ + S-)/2
 */
__device__ inline void apply_sx_contributions(
    uint64_t state, 
    int site, 
    float spin_length,
    uint64_t& state_plus,
    uint64_t& state_minus,
    cuDoubleComplex& elem_plus,
    cuDoubleComplex& elem_minus
) {
    device_pair<uint64_t, cuDoubleComplex> sp_result = apply_splus(state, site, spin_length);
    device_pair<uint64_t, cuDoubleComplex> sm_result = apply_sminus(state, site, spin_length);
    
    state_plus = sp_result.first;
    state_minus = sm_result.first;
    elem_plus = cuCmul(sp_result.second, make_cuDoubleComplex(0.5, 0.0));
    elem_minus = cuCmul(sm_result.second, make_cuDoubleComplex(0.5, 0.0));
}

/**
 * Device function: Apply Sy = (S+ - S-)/(2i)
 */
__device__ inline void apply_sy_contributions(
    uint64_t state, 
    int site, 
    float spin_length,
    uint64_t& state_plus,
    uint64_t& state_minus,
    cuDoubleComplex& elem_plus,
    cuDoubleComplex& elem_minus
) {
    device_pair<uint64_t, cuDoubleComplex> sp_result = apply_splus(state, site, spin_length);
    device_pair<uint64_t, cuDoubleComplex> sm_result = apply_sminus(state, site, spin_length);
    
    state_plus = sp_result.first;
    state_minus = sm_result.first;
    // 1/(2i) = -i/2
    elem_plus = cuCmul(sp_result.second, make_cuDoubleComplex(0.0, -0.5));
    elem_minus = cuCmul(sm_result.second, make_cuDoubleComplex(0.0, 0.5));
}

/**
 * Device function: Count number of bits set (popcount)
 */
__device__ __host__ inline int popcount_64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}

/**
 * Timing utilities
 */
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

} // namespace gpu

#endif // GPU_UTILS_CUH
