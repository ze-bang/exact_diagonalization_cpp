// filepath: /home/pc_linux/exact_diagonalization_cpp/src/TPQ_cuda.h
// TPQ_cuda.h - CUDA implementation for Thermal Pure Quantum state methods

#ifndef TPQ_CUDA_H
#define TPQ_CUDA_H

#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "construct_ham.h"
#include "observables.h"
#include "TPQ.h" // Include for utility functions like file I/O, etc.

using Complex = std::complex<double>;
using SparseMatrix = Eigen::SparseMatrix<Complex>;

// Forward declaration for the helper function
SparseMatrix construct_sparse_matrix(
    const std::function<void(const std::complex<double>*, std::complex<double>*, int)>& H_func,
    int N);

// Helper function to extract data from Eigen::SparseMatrix
void extract_eigen_sparse_data(const SparseMatrix& matrix,
                               std::vector<cuDoubleComplex>& values,
                               std::vector<int>& row_ptr,
                               std::vector<int>& col_ind) {
    SparseMatrix compressed_matrix = matrix;
    compressed_matrix.makeCompressed(); // Ensure the matrix is in compressed format

    int nnz = compressed_matrix.nonZeros();
    values.resize(nnz);
    row_ptr.resize(compressed_matrix.outerSize() + 1);

    // Eigen's valuePtr(), innerIndexPtr(), and outerIndexPtr() give direct access
    const Complex* eigen_values = compressed_matrix.valuePtr();
    const int* eigen_col_ind = compressed_matrix.innerIndexPtr();
    const int* eigen_row_ptr = compressed_matrix.outerIndexPtr();

    // Copy data, converting std::complex<double> to cuDoubleComplex
    for (int i = 0; i < nnz; ++i) {
        values[i] = make_cuDoubleComplex(eigen_values[i].real(), eigen_values[i].imag());
    }
    std::copy(eigen_col_ind, eigen_col_ind + nnz, col_ind.begin());
    std::copy(eigen_row_ptr, eigen_row_ptr + compressed_matrix.outerSize() + 1, row_ptr.begin());
}

// It's a helper to construct a sparse matrix from a function.
SparseMatrix construct_sparse_matrix(
    const std::function<void(const std::complex<double>*, std::complex<double>*, int)>& H_func,
    int N) {
    SparseMatrix H_sparse(N, N);
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    std::vector<std::complex<double>> v(N, {0.0, 0.0});
    std::vector<std::complex<double>> Hv(N, {0.0, 0.0});

    for (int j = 0; j < N; ++j) {
        v[j] = {1.0, 0.0};
        H_func(v.data(), Hv.data(), N);
        for (int i = 0; i < N; ++i) {
            if (std::abs(Hv[i]) > 1e-12) {
                triplets.emplace_back(i, j, Hv[i]);
            }
        }
        v[j] = {0.0, 0.0};
    }
    H_sparse.setFromTriplets(triplets.begin(), triplets.end());
    return H_sparse;
}


// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error: %s:%d, %s\n", __FILE__, __LINE__, cusparseGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %s:%d, Code: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// Forward declaration of the CUDA-based TPQ function
void microcanonical_tpq_cuda(
    const SparseMatrix& H_sparse,
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
);


/**
 * @brief A wrapper class to manage cuSPARSE resources for SpMV.
 */
class CudaHamiltonian {
public:
    CudaHamiltonian(const SparseMatrix& H_cpu, int N) : n(N), nnz(H_cpu.nonZeros()) {
        // Create cuSPARSE handle
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

        // Extract data from Eigen sparse matrix
        std::vector<cuDoubleComplex> h_values;
        std::vector<int> h_row_ptr;
        std::vector<int> h_col_ind;
        extract_eigen_sparse_data(H_cpu, h_values, h_row_ptr, h_col_ind);

        // Allocate memory on device
        CUDA_CHECK(cudaMalloc(&d_csr_values, nnz * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_csr_values, h_values.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csr_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

        // Create matrix descriptor
        CUSPARSE_CHECK(cusparseCreateCsr(&mat_descr, n, n, nnz, d_csr_row_ptr, d_csr_col_ind, d_csr_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    }

    ~CudaHamiltonian() {
        if (mat_descr) cusparseDestroySpMat(mat_descr);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_csr_values) cudaFree(d_csr_values);
        if (d_csr_row_ptr) cudaFree(d_csr_row_ptr);
        if (d_csr_col_ind) cudaFree(d_csr_col_ind);
    }

    // Apply H|v_in> = |v_out>
    void apply(const cuDoubleComplex* d_v_in, cuDoubleComplex* d_v_out) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        void* buffer = nullptr; // cuSPARSE can manage buffer internally if passed nullptr

        // Create vector descriptors
        cusparseDnVecDescr_t vec_in_descr, vec_out_descr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_in_descr, n, (void*)d_v_in, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out_descr, n, (void*)d_v_out, CUDA_C_64F));

        // Perform SpMV: y = alpha * op(A) * x + beta * y
        CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, mat_descr, vec_in_descr, &beta, vec_out_descr,
                                    CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        // Clean up vector descriptors
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_in_descr));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out_descr));
    }

private:
    int n;
    int nnz;
    cusparseHandle_t cusparse_handle = nullptr;
    cusparseSpMatDescr_t mat_descr = nullptr;
    cuDoubleComplex* d_csr_values = nullptr;
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
};

/**
 * Calculate energy and variance for a TPQ state on the GPU.
 */
std::pair<double, double> calculateEnergyAndVariance_cuda(
    CudaHamiltonian& H_cuda,
    cublasHandle_t cublas_handle,
    const cuDoubleComplex* d_v,
    int N
) {
    cuDoubleComplex* d_Hv;
    cuDoubleComplex* d_H2v;
    CUDA_CHECK(cudaMalloc(&d_Hv, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_H2v, N * sizeof(cuDoubleComplex)));

    // Calculate H|v⟩
    H_cuda.apply(d_v, d_Hv);

    // Calculate energy = ⟨v|H|v⟩
    cuDoubleComplex energy_complex;
    CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_v, 1, d_Hv, 1, &energy_complex));
    double energy = cuCreal(energy_complex);

    // Calculate H²|v⟩ = H|Hv⟩
    H_cuda.apply(d_Hv, d_H2v);

    // Calculate ⟨v|H²|v⟩
    cuDoubleComplex h2_complex;
    CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_v, 1, d_H2v, 1, &h2_complex));
    double h2_exp = cuCreal(h2_complex);

    double variance = h2_exp - energy * energy;

    CUDA_CHECK(cudaFree(d_Hv));
    CUDA_CHECK(cudaFree(d_H2v));

    return {energy, variance};
}




/**
 * Wrapper function to match the CPU interface of microcanonical_tpq.
 * It converts the std::function Hamiltonian to a sparse matrix format
 * and then calls the main CUDA TPQ implementation.
 */
void microcanonical_tpq_cuda_wrapper(
    std::function<void(const Complex*, Complex*, int)> H_func,
    int N,
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_spectrum,
    double LargeValue,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    int num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    int sublattice_size
) {
    std::cout << "Constructing sparse Hamiltonian for CUDA..." << std::endl;
    SparseMatrix H_sparse = construct_sparse_matrix(H_func, N);
    std::cout << "Hamiltonian constructed. NNZ = " << H_sparse.nonZeros() << std::endl;

    microcanonical_tpq_cuda(
        H_sparse, N, max_iter, num_samples, temp_interval, eigenvalues,
        dir, compute_spectrum, LargeValue, compute_observables,
        observables, observable_names, omega_min, omega_max, num_points,
        t_end, dt, spin_length, measure_sz, sublattice_size
    );
}


/**
 * @brief A wrapper class to manage cuSPARSE resources for a generic sparse operator.
 * This is essentially the same as CudaHamiltonian but with a more generic name.
 */
class CudaOperator {
public:
    CudaOperator(const SparseMatrix& op_cpu, int N) : n(N), nnz(op_cpu.nonZeros()) {
        // Create cuSPARSE handle
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

        // Extract data from Eigen sparse matrix
        std::vector<cuDoubleComplex> h_values;
        std::vector<int> h_row_ptr;
        std::vector<int> h_col_ind;
        extract_eigen_sparse_data(op_cpu, h_values, h_row_ptr, h_col_ind);

        // Allocate memory on device
        CUDA_CHECK(cudaMalloc(&d_csr_values, nnz * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_csr_values, h_values.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csr_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

        // Create matrix descriptor
        CUSPARSE_CHECK(cusparseCreateCsr(&mat_descr, n, n, nnz, d_csr_row_ptr, d_csr_col_ind, d_csr_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    }

    ~CudaOperator() {
        if (mat_descr) cusparseDestroySpMat(mat_descr);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_csr_values) cudaFree(d_csr_values);
        if (d_csr_row_ptr) cudaFree(d_csr_row_ptr);
        if (d_csr_col_ind) cudaFree(d_csr_col_ind);
    }

    // Apply Op|v_in> = |v_out>
    void apply(const cuDoubleComplex* d_v_in, cuDoubleComplex* d_v_out) const {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        void* buffer = nullptr;

        cusparseDnVecDescr_t vec_in_descr, vec_out_descr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_in_descr, n, (void*)d_v_in, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out_descr, n, (void*)d_v_out, CUDA_C_64F));

        CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, mat_descr, vec_in_descr, &beta, vec_out_descr,
                                    CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_in_descr));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out_descr));
    }

private:
    int n;
    long long nnz;
    cusparseHandle_t cusparse_handle = nullptr;
    cusparseSpMatDescr_t mat_descr = nullptr;
    cuDoubleComplex* d_csr_values = nullptr;
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
};

/**
 * Create Sz operators on the GPU.
 */
std::vector<std::unique_ptr<CudaOperator>> createSzOperators_cuda(int num_sites, float spin_length) {
    std::vector<std::unique_ptr<CudaOperator>> Sz_ops_cuda;
    int N = 1 << num_sites;
    for (int site = 0; site < num_sites; ++site) {
        SingleSiteOperator op_cpu(num_sites, spin_length, 2, site); // op_type 2 for Sz
        SparseMatrix op_sparse = op_cpu.getSparseMatrix();
        Sz_ops_cuda.push_back(std::make_unique<CudaOperator>(op_sparse, N));
    }
    return Sz_ops_cuda;
}

/**
 * Create Szz and Spm operators on the GPU.
 */
std::pair<std::vector<std::unique_ptr<CudaOperator>>, std::vector<std::unique_ptr<CudaOperator>>>
createDoubleSiteOperators_cuda(int num_sites, float spin_length) {
    std::vector<std::unique_ptr<CudaOperator>> Szz_ops_cuda;
    std::vector<std::unique_ptr<CudaOperator>> Spm_ops_cuda;
    int N = 1 << num_sites;

    for (int site1 = 0; site1 < num_sites; ++site1) {
        for (int site2 = 0; site2 < num_sites; ++site2) {
            // Szz
            DoubleSiteOperator szz_op_cpu(num_sites, spin_length, 2, site1, 2, site2);
            Szz_ops_cuda.push_back(std::make_unique<CudaOperator>(szz_op_cpu.getSparseMatrix(), N));
            // Spm
            DoubleSiteOperator spm_op_cpu(num_sites, spin_length, 0, site1, 1, site2);
            Spm_ops_cuda.push_back(std::make_unique<CudaOperator>(spm_op_cpu.getSparseMatrix(), N));
        }
    }
    return {std::move(Szz_ops_cuda), std::move(Spm_ops_cuda)};
}

/**
 * Calculate <Sz> and <Sz^2> on the GPU.
 */
std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2_cuda(
    const cuDoubleComplex* d_tpq_state,
    int num_sites,
    const std::vector<std::unique_ptr<CudaOperator>>& Sz_ops,
    int sublattice_size,
    cublasHandle_t cublas_handle
) {
    int N = 1 << num_sites;
    ComplexVector Sz_exps(sublattice_size + 1, {0.0, 0.0});
    ComplexVector Sz2_exps(sublattice_size + 1, {0.0, 0.0});

    cuDoubleComplex *d_Op_psi, *d_Op2_psi;
    CUDA_CHECK(cudaMalloc(&d_Op_psi, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Op2_psi, N * sizeof(cuDoubleComplex)));

    for (int site = 0; site < num_sites; ++site) {
        int i = site % sublattice_size;
        Sz_ops[site]->apply(d_tpq_state, d_Op_psi);

        cuDoubleComplex sz_exp_gpu, sz2_exp_gpu;
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_tpq_state, 1, d_Op_psi, 1, &sz_exp_gpu));

        Sz_ops[site]->apply(d_Op_psi, d_Op2_psi);
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_tpq_state, 1, d_Op2_psi, 1, &sz2_exp_gpu));

        Sz_exps[i] += Complex(cuCreal(sz_exp_gpu), cuCimag(sz_exp_gpu));
        Sz2_exps[i] += Complex(cuCreal(sz2_exp_gpu), cuCimag(sz2_exp_gpu));
    }

    CUDA_CHECK(cudaFree(d_Op_psi));
    CUDA_CHECK(cudaFree(d_Op2_psi));

    for (int i = 0; i < sublattice_size; ++i) {
        Sz_exps[i] /= double(num_sites);
        Sz2_exps[i] /= double(num_sites);
        Sz_exps[sublattice_size] += Sz_exps[i];
        Sz2_exps[sublattice_size] += Sz2_exps[i];
    }

    return {Sz_exps, Sz2_exps};
}

/**
 * Calculate <Szz> and <Spm> on the GPU.
 */
std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm_cuda(
    const cuDoubleComplex* d_tpq_state,
    int num_sites,
    const std::pair<std::vector<std::unique_ptr<CudaOperator>>, std::vector<std::unique_ptr<CudaOperator>>>& double_site_ops,
    int sublattice_size,
    cublasHandle_t cublas_handle
) {
    int N = 1 << num_sites;
    ComplexVector Szz_exps(sublattice_size * sublattice_size + 1, {0.0, 0.0});
    ComplexVector Spm_exps(sublattice_size * sublattice_size + 1, {0.0, 0.0});

    const auto& Szz_ops = double_site_ops.first;
    const auto& Spm_ops = double_site_ops.second;

    cuDoubleComplex* d_Op_psi;
    CUDA_CHECK(cudaMalloc(&d_Op_psi, N * sizeof(cuDoubleComplex)));

    for (int site1 = 0; site1 < num_sites; ++site1) {
        for (int site2 = 0; site2 < num_sites; ++site2) {
            int op_idx = site1 * num_sites + site2;
            int i = (site1 % sublattice_size) * sublattice_size + (site2 % sublattice_size);

            cuDoubleComplex szz_exp_gpu, spm_exp_gpu;

            Szz_ops[op_idx]->apply(d_tpq_state, d_Op_psi);
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_tpq_state, 1, d_Op_psi, 1, &szz_exp_gpu));

            Spm_ops[op_idx]->apply(d_tpq_state, d_Op_psi);
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_tpq_state, 1, d_Op_psi, 1, &spm_exp_gpu));

            Szz_exps[i] += Complex(cuCreal(szz_exp_gpu), cuCimag(szz_exp_gpu));
            Spm_exps[i] += Complex(cuCreal(spm_exp_gpu), cuCimag(spm_exp_gpu));
        }
    }

    CUDA_CHECK(cudaFree(d_Op_psi));

    for (int i = 0; i < sublattice_size * sublattice_size; ++i) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size * sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size * sublattice_size] += Szz_exps[i];
    }

    return {Szz_exps, Spm_exps};
}

/**
 * Write fluctuation data to file during TPQ evolution (CUDA version).
 */
void writeFluctuationData_cuda(
    const std::string& flct_file,
    const std::string& spin_corr_file,
    double inv_temp,
    const cuDoubleComplex* d_tpq_state,
    int num_sites,
    const std::vector<std::unique_ptr<CudaOperator>>& Sz_ops,
    const std::pair<std::vector<std::unique_ptr<CudaOperator>>, std::vector<std::unique_ptr<CudaOperator>>>& double_site_ops,
    int sublattice_size,
    int step,
    cublasHandle_t cublas_handle
) {
    // Calculate Sz and Sz^2
    auto [Sz, Sz2] = calculateSzandSz2_cuda(d_tpq_state, num_sites, Sz_ops, sublattice_size, cublas_handle);

    std::ofstream flct_out(flct_file, std::ios::app);
    flct_out << std::setprecision(16) << inv_temp
             << " " << Sz[sublattice_size].real() << " " << Sz[sublattice_size].imag()
             << " " << Sz2[sublattice_size].real() << " " << Sz2[sublattice_size].imag();
    for (int i = 0; i < sublattice_size; ++i) {
        flct_out << " " << Sz[i].real() << " " << Sz[i].imag()
                 << " " << Sz2[i].real() << " " << Sz2[i].imag();
    }
    flct_out << " " << step << std::endl;

    // Calculate Szz and Spm
    auto [Szz, Spm] = calculateSzzSpm_cuda(d_tpq_state, num_sites, double_site_ops, sublattice_size, cublas_handle);

    std::ofstream spin_out(spin_corr_file, std::ios::app);
    spin_out << std::setprecision(16) << inv_temp
             << " " << Spm[sublattice_size * sublattice_size].real() << " " << Spm[sublattice_size * sublattice_size].imag()
             << " " << Szz[sublattice_size * sublattice_size].real() << " " << Szz[sublattice_size * sublattice_size].imag();
    for (int i = 0; i < sublattice_size * sublattice_size; ++i) {
        spin_out << " " << Spm[i].real() << " " << Spm[i].imag()
                 << " " << Szz[i].real() << " " << Szz[i].imag();
    }
    spin_out << " " << step << std::endl;
}

/**
 * @brief Creates a time evolution operator U(t) = exp(-iHt) for GPU.
 *
 * This function returns a lambda that applies the time evolution operator
 * to a device vector using a Taylor series expansion.
 *
 * @param H_cuda The Hamiltonian operator on the GPU.
 * @param cublas_handle A handle to the cuBLAS library context.
 * @param N The dimension of the Hilbert space.
 * @param delta_t The time step.
 * @param n_max The maximum order of the Taylor expansion.
 * @param normalize Whether to normalize the state after evolution.
 * @return A function that applies the time evolution operator on the device.
 */
std::function<void(const cuDoubleComplex*, cuDoubleComplex*)>
create_time_evolution_operator_cuda(
    CudaHamiltonian& H_cuda,
    cublasHandle_t cublas_handle,
    int N,
    double delta_t,
    int n_max = 10,
    bool normalize = true
) {
    // Precompute coefficients on the host
    auto coefficients = std::make_shared<std::vector<cuDoubleComplex>>(n_max + 1);
    (*coefficients)[0] = make_cuDoubleComplex(1.0, 0.0);
    double factorial = 1.0;

    for (int order = 1; order <= n_max; ++order) {
        factorial *= order;
        cuDoubleComplex i_pow_order = make_cuDoubleComplex(0.0, -1.0); // for -i*H*t
        for(int p=1; p<order; ++p) i_pow_order = cuCmul(i_pow_order, make_cuDoubleComplex(0.0, -1.0));

        double dt_pow_order = std::pow(delta_t, order);
        (*coefficients)[order] = make_cuDoubleComplex(cuCreal(i_pow_order) * dt_pow_order / factorial, cuCimag(i_pow_order) * dt_pow_order / factorial);
    }

    // Allocate persistent temporary buffers for the operator
    cuDoubleComplex *d_term, *d_Hterm;
    CUDA_CHECK(cudaMalloc(&d_term, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Hterm, N * sizeof(cuDoubleComplex)));

    // Return a function that applies the time evolution
    return [=, &H_cuda](const cuDoubleComplex* d_input, cuDoubleComplex* d_output) mutable {
        // Initialize result with the 0th order term (input vector)
        CUDA_CHECK(cudaMemcpy(d_output, d_input, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // Initialize term for the 1st order calculation
        CUDA_CHECK(cudaMemcpy(d_term, d_input, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

        // Apply Taylor expansion terms
        for (int order = 1; order <= n_max; ++order) {
            // Apply H to the previous term: Hterm = H * term
            H_cuda.apply(d_term, d_Hterm);
            std::swap(d_term, d_Hterm); // term is now H*term

            // Add this term to the result: result += coeff * term
            CUBLAS_CHECK(cublasZaxpy(cublas_handle, N, &(*coefficients)[order], d_term, 1, d_output, 1));
        }

        // Normalize if requested
        if (normalize) {
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, d_output, 1, &norm));
            cuDoubleComplex scale_factor = make_cuDoubleComplex(1.0 / norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, &scale_factor, d_output, 1));
        }
    };
}

/**
 * @brief Calculates time correlation functions C(t) = <ψ|O(t)† O(0)|ψ> on the GPU.
 *
 * This function computes the time correlation for multiple operators simultaneously.
 * It takes a time evolution operator U_t and a set of CUDA-based operators.
 * The correlation is C(t) = <ψ(t)|O† O|ψ(t)>, where |ψ(t)> = U(t)|ψ>.
 *
 * @param U_t A function that applies the time evolution operator U(t) = exp(-iHt) on a device vector.
 * @param operators A vector of unique pointers to CudaOperator objects.
 * @param d_tpq_state The initial TPQ state vector on the device.
 * @param N The dimension of the Hilbert space.
 * @param num_steps The number of time steps to evolve.
 * @param cublas_handle A handle to the cuBLAS library context.
 * @return A 2D vector of complex numbers containing the time correlations for each operator.
 */
std::vector<std::vector<Complex>> calculate_spectral_function_from_tpq_U_t_cuda(
    std::function<void(const cuDoubleComplex*, cuDoubleComplex*)> U_t,
    const std::vector<std::unique_ptr<CudaOperator>>& operators,
    const cuDoubleComplex* d_tpq_state,
    int N,
    const int num_steps,
    cublasHandle_t cublas_handle
) {
    int num_operators = operators.size();

    // Allocate device memory
    cuDoubleComplex *d_state, *d_state_next;
    CUDA_CHECK(cudaMalloc(&d_state, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_state_next, N * sizeof(cuDoubleComplex)));

    std::vector<cuDoubleComplex*> d_O_psi_vec(num_operators);
    std::vector<cuDoubleComplex*> d_O_psi_next_vec(num_operators);
    std::vector<cuDoubleComplex*> d_O_dag_state_vec(num_operators);
    for (int op = 0; op < num_operators; ++op) {
        CUDA_CHECK(cudaMalloc(&d_O_psi_vec[op], N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_O_psi_next_vec[op], N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_O_dag_state_vec[op], N * sizeof(cuDoubleComplex)));
    }

    // Initialize state
    CUDA_CHECK(cudaMemcpy(d_state, d_tpq_state, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // Calculate O|ψ> once for each operator
    for (int op = 0; op < num_operators; ++op) {
        operators[op]->apply(d_state, d_O_psi_vec[op]);
    }

    // Prepare for time evolution
    std::vector<std::vector<Complex>> time_correlations(num_operators, std::vector<Complex>(num_steps));

    // Calculate initial correlation C(0) = <ψ|O†O|ψ>
    for (int op = 0; op < num_operators; ++op) {
        operators[op]->apply(d_state, d_O_dag_state_vec[op]);
        cuDoubleComplex corr_gpu;
        CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_O_dag_state_vec[op], 1, d_O_psi_vec[op], 1, &corr_gpu));
        time_correlations[op][0] = Complex(cuCreal(corr_gpu), cuCimag(corr_gpu));
    }

    std::cout << "Starting real-time evolution for correlation function on GPU..." << std::endl;

    // Time evolution loop
    for (int step = 1; step < num_steps; ++step) {
        // Evolve state: |ψ(t)> = U_t|ψ(t-dt)>
        U_t(d_state, d_state_next);

        // For each operator
        for (int op = 0; op < num_operators; ++op) {
            // Evolve O|ψ>: O|ψ(t)> = U_t * O|ψ(t-dt)>
            U_t(d_O_psi_vec[op], d_O_psi_next_vec[op]);

            // Calculate O†|ψ(t)>
            operators[op]->apply(d_state_next, d_O_dag_state_vec[op]);

            // Calculate correlation C(t) = <ψ(t)|O† O|ψ(t)>
            cuDoubleComplex corr_gpu;
            CUBLAS_CHECK(cublasZdotc(cublas_handle, N, d_O_dag_state_vec[op], 1, d_O_psi_next_vec[op], 1, &corr_gpu));
            time_correlations[op][step] = Complex(cuCreal(corr_gpu), cuCimag(corr_gpu));

            // Swap O_psi for next iteration
            std::swap(d_O_psi_vec[op], d_O_psi_next_vec[op]);
        }

        // Swap state for next iteration
        std::swap(d_state, d_state_next);

        if (step % 100 == 0) {
            std::cout << "  Step " << step << "/" << num_steps << std::endl;
        }
    }

    // Cleanup device memory
    cudaFree(d_state);
    cudaFree(d_state_next);
    for (int op = 0; op < num_operators; ++op) {
        cudaFree(d_O_psi_vec[op]);
        cudaFree(d_O_psi_next_vec[op]);
        cudaFree(d_O_dag_state_vec[op]);
    }

    return time_correlations;
}

/**
 * @brief Computes observable dynamics using a pre-computed time evolution operator on the GPU.
 *
 * @param U_t The time evolution operator U(t) = exp(-iHt).
 * @param U_nt The time evolution operator for negative time U(-t) = exp(iHt).
 * @param d_tpq_state The initial TPQ state on the device.
 * @param observables A vector of CPU-based Operator objects.
 * @param observable_names The names of the observables.
 * @param N The dimension of the Hilbert space.
 * @param dir The output directory.
 * @param sample The current sample index.
 * @param inv_temp The inverse temperature.
 * @param t_end The maximum evolution time.
 * @param dt The time step.
 * @param cublas_handle A handle to the cuBLAS library context.
 */
void computeObservableDynamics_U_t_cuda(
    std::function<void(const cuDoubleComplex*, cuDoubleComplex*)> U_t,
    std::function<void(const cuDoubleComplex*, cuDoubleComplex*)> U_nt,
    const cuDoubleComplex* d_tpq_state,
    const std::vector<Operator>& observables,
    const std::vector<std::string>& observable_names,
    int N,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end,
    double dt,
    cublasHandle_t cublas_handle
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
    ComplexVector h_tpq_state(N);
    CUDA_CHECK(cudaMemcpy(h_tpq_state.data(), d_tpq_state, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    save_tpq_state(h_tpq_state, state_file);

    std::cout << "Computing dynamical susceptibility for sample " << sample
              << ", beta=" << inv_temp << ", for " << observables.size() << " observables on GPU" << std::endl;

    // Create CudaOperator for each observable
    std::vector<std::unique_ptr<CudaOperator>> cuda_ops;
    for(const auto& obs : observables) {
        SparseMatrix op_sparse = construct_sparse_matrix([&](const Complex* in, Complex* out, int size){ obs.apply(in, out, size); }, N);
        cuda_ops.push_back(std::make_unique<CudaOperator>(op_sparse, N));
    }

    int num_steps = static_cast<int>(t_end / dt) + 1;

    // --- Positive time evolution ---
    auto time_correlations = calculate_spectral_function_from_tpq_U_t_cuda(
        U_t, cuda_ops, d_tpq_state, N, num_steps, cublas_handle);

    // --- Negative time evolution ---
    auto negative_time_correlations = calculate_spectral_function_from_tpq_U_t_cuda(
        U_nt, cuda_ops, d_tpq_state, N, num_steps, cublas_handle);

    // Process and save results for each observable
    for (size_t i = 0; i < observables.size(); ++i) {
        std::string time_corr_file = dir + "/time_corr_rand" + std::to_string(sample) + "_"
                                   + observable_names[i] + "_beta=" + std::to_string(inv_temp) + ".dat";

        std::vector<double> time_points(num_steps);
        for (int j = 0; j < num_steps; ++j) {
            time_points[j] = j * dt;
        }

        // Combine negative and positive time correlations
        std::vector<Complex> combined_time_correlation;
        std::vector<double> combined_time_points;
        combined_time_correlation.reserve(2 * num_steps - 1);
        combined_time_points.reserve(2 * num_steps - 1);

        // Add negative time points (in reverse, skipping t=0)
        for (int j = num_steps - 1; j > 0; --j) {
            combined_time_points.push_back(-time_points[j]);
            combined_time_correlation.push_back(negative_time_correlations[i][j]);
        }
        // Add positive time points
        for (int j = 0; j < num_steps; ++j) {
            combined_time_points.push_back(time_points[j]);
            combined_time_correlation.push_back(time_correlations[i][j]);
        }

        // Write time correlation to file
        std::ofstream time_corr_out(time_corr_file);
        if (time_corr_out.is_open()) {
            time_corr_out << "# time C(t)_real C(t)_imag" << std::endl;
            for (size_t j = 0; j < combined_time_correlation.size(); ++j) {
                time_corr_out << std::setprecision(16) << combined_time_points[j] << " "
                              << combined_time_correlation[j].real() << " "
                              << combined_time_correlation[j].imag() << std::endl;
            }
        }
    }
}

/**
 * CUDA version of microcanonical TPQ.
 * This function performs the main TPQ loop on the GPU.
 */
void microcanonical_tpq_cuda(
    const SparseMatrix& H_sparse,
    int N,
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_spectrum,
    double LargeValue,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    int num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    int sublattice_size
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Initialize Hamiltonian on GPU
    CudaHamiltonian H_cuda(H_sparse, N);

    // Allocate device memory for state vectors
    cuDoubleComplex *d_v, *d_v_next;
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_next, N * sizeof(cuDoubleComplex)));

    // Host vector for intermediate operations
    ComplexVector h_v(N);

    eigenvalues.clear();

    // --- NOTE: Observables are computed on CPU for simplicity. ---
    // For full GPU acceleration, observable operators would also need to be
    // moved to the GPU and applied with cuSPARSE.
    int num_sites = static_cast<int>(std::log2(N));
    std::vector<std::unique_ptr<CudaOperator>> Sz_ops;
    std::pair<std::vector<std::unique_ptr<CudaOperator>>, std::vector<std::unique_ptr<CudaOperator>>> double_site_ops;
    std::function<void(const cuDoubleComplex*, cuDoubleComplex*)> U_t, U_nt;
    if (measure_sz) {
        Sz_ops = createSzOperators_cuda(num_sites, spin_length);
        double_site_ops = createDoubleSiteOperators_cuda(num_sites, spin_length);
    }
    U_t = create_time_evolution_operator_cuda(
        H_cuda, cublas_handle, N, dt, 10, true
    );
    U_nt = create_time_evolution_operator_cuda(
        H_cuda, cublas_handle, N, -dt, 10, true
    );
    // ---
    const int num_temp_points = 50;
    std::array<double, num_temp_points> measure_inv_temp;
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 100
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    std::cout << "Starting CUDA microcanonical TPQ..." << std::endl;

    for (int sample = 0; sample < num_samples; ++sample) {
        auto start_time = std::chrono::high_resolution_clock::now();
        unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + sample;
        double inv_temp = 0;
        int step = 1;
        // Generate initial random vector on host
        h_v = generateTPQVector(N, seed);
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        auto [ss_file, norm_file, flct_file, spin_corr_file] = initializeTPQFiles(dir, sample, sublattice_size);

        for (int step = 0; step < max_iter; ++step) {
            // Evolve state: v_next = H * v
            H_cuda.apply(d_v, d_v_next);

            // Normalize on GPU using cuBLAS
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle, N, d_v_next, 1, &norm));
            cuDoubleComplex scale_factor = make_cuDoubleComplex(1.0 / norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle, N, &scale_factor, d_v_next, 1));

            // Swap pointers for next iteration
            std::swap(d_v, d_v_next);

            if (step % temp_interval == 0) {
                auto [energy, variance] = calculateEnergyAndVariance_cuda(H_cuda, cublas_handle, d_v, N);
                inv_temp = (2.0*step) / (LargeValue * num_sites - energy);

                // Copy current state back to host for measurements
                CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

                writeTPQData(ss_file, inv_temp, energy, variance, norm, step);

                if (measure_sz) {
                    writeFluctuationData_cuda(flct_file, spin_corr_file, inv_temp, d_v, num_sites, Sz_ops,
                                              double_site_ops, sublattice_size, step, cublas_handle);
                }
                // If inv_temp is at one of the specified inverse temperature points, compute observables
                if (compute_observables) {
                    for (auto temp : measure_inv_temp) {
                        if (std::abs(inv_temp - temp) < 8e-3) {
                            std::cout << "Computing observables at inv_temp = " << inv_temp << std::endl;
                            computeObservableDynamics_U_t_cuda(
                                U_t, U_nt, d_v, observables, observable_names, N,
                                dir, sample, inv_temp, t_end, dt, cublas_handle
                            );
                        }
                    }
                }
            }
        }

        // Final energy calculation
        auto [final_energy, final_variance] = calculateEnergyAndVariance_cuda(H_cuda, cublas_handle, d_v, N);
        eigenvalues.push_back(final_energy);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Sample " << sample << " finished. Final Energy: " << final_energy
                  << ", Time: " << elapsed.count() << "s" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_v_next));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}


#endif // TPQ_CUDA_H