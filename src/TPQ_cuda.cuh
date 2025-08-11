#pragma once
#include <complex>
#include <functional>
#include <vector>

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

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

// Utility: compute 2-norm on device, scale vector, and BLAS-like ops
double dz_nrm2_device(const Complex* d_x, int n, CudaContext& ctx);
void dz_scal_device(int n, Complex alpha, Complex* d_x, CudaContext& ctx);
void dz_axpy_device(int n, Complex alpha, const Complex* d_x, Complex* d_y, CudaContext& ctx);
void dz_copy_device(int n, const Complex* d_x, Complex* d_y, CudaContext& ctx);
Complex dz_dotc_device(int n, const Complex* d_x, const Complex* d_y, CudaContext& ctx);

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
