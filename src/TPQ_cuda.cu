#include "TPQ_cuda.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); } while(0)
#define CUSPARSE_CHECK(x) do { cusparseStatus_t _s = (x); if (_s != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cuSPARSE error"); } while(0)

static inline cublasHandle_t get_cublas(CudaContext& ctx) { return reinterpret_cast<cublasHandle_t>(ctx.cublas); }

CudaContext::CudaContext() {
    CUDA_CHECK(cudaGetDevice(&device));
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h)); cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    cublas = h;
}
CudaContext::~CudaContext() {
    if (cublas) { cublasDestroy(get_cublas(*this)); cublas = nullptr; }
}

DeviceVector::DeviceVector(size_t n_) { resize(n_); }
DeviceVector::~DeviceVector() { if (ptr) cudaFree(ptr); }
void DeviceVector::resize(size_t n_) { if (ptr) cudaFree(ptr); n = n_; if (n) CUDA_CHECK(cudaMalloc(&ptr, n*sizeof(Complex))); }

void copyHostToDevice(DeviceVector& dst, const Complex* src, size_t n) {
    if (dst.n < n) dst.resize(n);
    CUDA_CHECK(cudaMemcpy(dst.ptr, src, n*sizeof(Complex), cudaMemcpyHostToDevice));
}
void copyDeviceToHost(Complex* dst, const DeviceVector& src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src.ptr, n*sizeof(Complex), cudaMemcpyDeviceToHost));
}

// cuBLAS complex type aliases
static inline cuDoubleComplex* cptr(Complex* p){ return reinterpret_cast<cuDoubleComplex*>(p);} 
static inline const cuDoubleComplex* cptr(const Complex* p){ return reinterpret_cast<const cuDoubleComplex*>(p);} 

// BLAS-like ops on device

double dz_nrm2_device(const Complex* d_x, int n, CudaContext& ctx){
    double result = 0.0; CUBLAS_CHECK(cublasDznrm2(get_cublas(ctx), n, cptr(d_x), 1, &result)); return result;
}
void dz_scal_device(int n, Complex alpha, Complex* d_x, CudaContext& ctx){ CUBLAS_CHECK(cublasZscal(get_cublas(ctx), n, cptr(&alpha), cptr(d_x), 1)); }
void dz_axpy_device(int n, Complex alpha, const Complex* d_x, Complex* d_y, CudaContext& ctx){ CUBLAS_CHECK(cublasZaxpy(get_cublas(ctx), n, cptr(&alpha), cptr(d_x), 1, cptr(d_y), 1)); }
void dz_copy_device(int n, const Complex* d_x, Complex* d_y, CudaContext& ctx){ CUBLAS_CHECK(cublasZcopy(get_cublas(ctx), n, cptr(d_x), 1, cptr(d_y), 1)); }
Complex dz_dotc_device(int n, const Complex* d_x, const Complex* d_y, CudaContext& ctx){ cuDoubleComplex out; CUBLAS_CHECK(cublasZdotc(get_cublas(ctx), n, cptr(d_x), 1, cptr(d_y), 1, &out)); return *reinterpret_cast<Complex*>(&out);} 

// GPU Krylov time evolution: structure mirrors CPU version but keeps data on device
void time_evolve_tpq_krylov_cuda(
    GpuMatvec H_dev,
    Complex* d_state,
    int N,
    double delta_t,
    int krylov_dim,
    bool normalize,
    CudaContext& ctx
){
    krylov_dim = std::min(krylov_dim, N);
    if (krylov_dim <= 0) return;

    // Allocate device workspaces
    std::vector<DeviceVector> V(krylov_dim); for (auto& v : V) v.resize(N);
    DeviceVector d_w(N);

    // alpha, beta on host
    std::vector<double> alpha(krylov_dim, 0.0);
    std::vector<double> beta(std::max(0, krylov_dim-1), 0.0);

    // Initialize v0 = state / ||state||
    double nrm = dz_nrm2_device(d_state, N, ctx);
    if (nrm < 1e-14) return;
    dz_copy_device(N, d_state, V[0].ptr, ctx);
    dz_scal_device(N, Complex(1.0/nrm, 0.0), V[0].ptr, ctx);

    int eff = krylov_dim;
    const double ortho_thr = 1e-10;
    const double brk_thr = 1e-14;

    for (int j=0; j<krylov_dim-1; ++j){
        // w = H v_j
        H_dev(V[j].ptr, d_w.ptr, N);
        // alpha_j = Re(<v_j|w>)
        Complex a = dz_dotc_device(N, V[j].ptr, d_w.ptr, ctx);
        alpha[j] = a.real();
        // w -= alpha_j v_j
        dz_axpy_device(N, Complex(-alpha[j],0), V[j].ptr, d_w.ptr, ctx);
        // simple re-orthogonalization
        for (int i=0; i<=j; ++i){
            Complex ov = dz_dotc_device(N, V[i].ptr, d_w.ptr, ctx);
            if (std::abs(ov) > ortho_thr){ dz_axpy_device(N, -ov, V[i].ptr, d_w.ptr, ctx); }
        }
        beta[j] = dz_nrm2_device(d_w.ptr, N, ctx);
        if (beta[j] < brk_thr){ eff = j+1; break; }
        dz_copy_device(N, d_w.ptr, V[j+1].ptr, ctx);
        dz_scal_device(N, Complex(1.0/beta[j],0), V[j+1].ptr, ctx);
    }
    if (eff == krylov_dim){
        H_dev(V[eff-1].ptr, d_w.ptr, N);
        Complex a = dz_dotc_device(N, V[eff-1].ptr, d_w.ptr, ctx);
        alpha[eff-1] = a.real();
    }

    // Small eigensolve on host
    Eigen::MatrixXd Hk(eff, eff); Hk.setZero();
    for (int i=0;i<eff;i++){ Hk(i,i) = alpha[i]; }
    for (int i=0;i<eff-1;i++){ Hk(i,i+1) = beta[i]; Hk(i+1,i) = beta[i]; }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hk);
    const auto &eval = es.eigenvalues();
    const auto &evec = es.eigenvectors();

    // coeffs = exp(-i eval dt) * evec.col(0) * ||psi||
    std::vector<Complex> coeffs(eff);
    for (int i=0;i<eff;i++) coeffs[i] = std::exp(Complex(0, -eval(i)*delta_t)) * evec(i,0) * nrm;

    // Reconstruct state: psi = sum_i (sum_j evec(j,i)*coeffs[j]) * v_i
    // Accumulate on device
    CUDA_CHECK(cudaMemset(d_state, 0, N*sizeof(Complex)));
    for (int i=0;i<eff;i++){
        Complex ci(0,0);
        for (int j=0;j<eff;j++) ci += evec(j,i) * coeffs[j];
        if (std::abs(ci) != 0.0) dz_axpy_device(N, ci, V[i].ptr, d_state, ctx);
    }

    if (normalize){
        double fn = dz_nrm2_device(d_state, N, ctx);
        if (fn > 1e-14) dz_scal_device(N, Complex(1.0/fn,0), d_state, ctx);
    }
}
