#pragma once
#include <complex>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Helper: safely convert a 64-bit Hilbert space dimension to BLAS-compatible int
// Use this in callers before passing sizes into CUDA BLAS routines that require int.
inline int to_blas_length(size_t n64){
    if (n64 > static_cast<size_t>(std::numeric_limits<int>::max())){
        throw std::runtime_error("Dimension exceeds cuBLAS int range; reduce size or refactor to block/distribute state.");
    }
    return static_cast<int>(n64);
}

struct CudaContext {
    void* cublas = nullptr;
    void* cusparse = nullptr;
    int device = 0;
    CudaContext();
    ~CudaContext();
};

// Device vector RAII wrapper (host-visible handle)
struct DeviceVector {
    Complex* ptr = nullptr;
    size_t n = 0;
    DeviceVector() = default;
    DeviceVector(size_t n);
    ~DeviceVector();
    void resize(size_t n);
};

// Copy helpers
void copyHostToDevice(DeviceVector& dst, const Complex* src, size_t n);
void copyDeviceToHost(Complex* dst, const DeviceVector& src, size_t n);

// GPU matvec interface: user provides a function that applies H on device memory
// Signature: H_dev(d_in, d_out, N, streamId)
using GpuMatvec = std::function<void(const Complex*, Complex*, int)>;

// Krylov time evolution on GPU using Lanczos in device memory
void time_evolve_tpq_krylov_cuda(
    GpuMatvec H_dev,
    Complex* d_state, // device pointer
    int N,
    double delta_t,
    int krylov_dim,
    bool normalize,
    CudaContext& ctx
);

inline void time_evolve_tpq_krylov_cuda(
    GpuMatvec H_dev,
    Complex* d_state, // device pointer
    size_t N,
    double delta_t,
    int krylov_dim,
    bool normalize,
    CudaContext& ctx
){
    time_evolve_tpq_krylov_cuda(H_dev, d_state, to_blas_length(N), delta_t, krylov_dim, normalize, ctx);
}

// Utility: compute 2-norm on device, scale vector, and BLAS-like ops
double dz_nrm2_device(const Complex* d_x, int n, CudaContext& ctx);
void dz_scal_device(int n, Complex alpha, Complex* d_x, CudaContext& ctx);
void dz_axpy_device(int n, Complex alpha, const Complex* d_x, Complex* d_y, CudaContext& ctx);
void dz_copy_device(int n, const Complex* d_x, Complex* d_y, CudaContext& ctx);
Complex dz_dotc_device(int n, const Complex* d_x, const Complex* d_y, CudaContext& ctx);

// size_t-safe overloads forwarding to int-based implementations
inline double dz_nrm2_device(const Complex* d_x, size_t n, CudaContext& ctx){
    return dz_nrm2_device(d_x, to_blas_length(n), ctx);
}
inline void dz_scal_device(size_t n, Complex alpha, Complex* d_x, CudaContext& ctx){
    dz_scal_device(to_blas_length(n), alpha, d_x, ctx);
}
inline void dz_axpy_device(size_t n, Complex alpha, const Complex* d_x, Complex* d_y, CudaContext& ctx){
    dz_axpy_device(to_blas_length(n), alpha, d_x, d_y, ctx);
}
inline void dz_copy_device(size_t n, const Complex* d_x, Complex* d_y, CudaContext& ctx){
    dz_copy_device(to_blas_length(n), d_x, d_y, ctx);
}
inline Complex dz_dotc_device(size_t n, const Complex* d_x, const Complex* d_y, CudaContext& ctx){
    return dz_dotc_device(to_blas_length(n), d_x, d_y, ctx);
}

// Convenience: wrap a host-side H(matvec) into a device callback by staging through host
// This is mainly for validation; for performance, provide a native GPU implementation.
#include <cuda_runtime.h>
inline GpuMatvec wrap_host_matvec(const std::function<void(const Complex*, Complex*, int)>& H){
    return [H](const Complex* d_in, Complex* d_out, int N){
        std::vector<Complex> h_in(N), h_out(N);
        cudaMemcpy(h_in.data(), d_in, N*sizeof(Complex), cudaMemcpyDeviceToHost);
        H(h_in.data(), h_out.data(), N);
        cudaMemcpy(d_out, h_out.data(), N*sizeof(Complex), cudaMemcpyHostToDevice);
    };
}
