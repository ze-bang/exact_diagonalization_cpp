#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <cblas.h>
#include <lapacke.h>
#include <cstring>

template<size_t N, size_t M>
class LanczosAlgorithm {
public:
    using complex_t = std::complex<double>;
    
    // Constructor takes a Hermitian matrix A
    LanczosAlgorithm(const complex_t* A) : A_(A) {}

    // Perform the Lanczos algorithm and return eigenvalues and eigenvectors
    void compute(double* eigenvalues, complex_t* eigenvectors) const {
        // Initialize random v_1
        complex_t V[N * (M + 1)];
        complex_t w[N];
        double alpha[M];
        double beta[M];
        
        // Create random initial vector
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (size_t i = 0; i < N; ++i) {
            V[i] = complex_t(dist(gen), dist(gen));
        }
        
        // Normalize v_1
        double norm = cblas_dznrm2(N, V, 1);
        cblas_zdscal(N, 1.0 / norm, V, 1);
        
        // Lanczos iteration
        for (size_t j = 0; j < M; ++j) {
            complex_t* v_j = &V[j * N];
            complex_t* v_jp1 = &V[(j + 1) * N];
            
            // w = A * v_j
            // Create local variables for alpha and beta to avoid rvalue issues
            complex_t complex_1(1.0, 0.0);
            complex_t complex_0(0.0, 0.0);
            cblas_zhemv(CblasColMajor, CblasUpper, N, &complex_1, A_, N, 
                        v_j, 1, &complex_0, w, 1);
            
            // Subtract beta_{j-1} * v_{j-1} if j > 0
            if (j > 0) {
                complex_t beta_jm1(-beta[j-1], 0.0);
                cblas_zaxpy(N, &beta_jm1, &V[(j-1) * N], 1, w, 1);
            }
            
            // Calculate alpha_j = <w, v_j>
            complex_t dot_product;
            cblas_zdotc_sub(N, w, 1, v_j, 1, &dot_product);
            alpha[j] = dot_product.real();  // For Hermitian matrices, alpha is real
            
            // w = w - alpha_j * v_j
            complex_t alpha_jm1(-alpha[j], 0.0);
            cblas_zaxpy(N, &alpha_jm1, v_j, 1, w, 1);
            
            // Reorthogonalize (for numerical stability)
            for (size_t k = 0; k <= j; ++k) {
                complex_t proj;
                cblas_zdotc_sub(N, w, 1, &V[k * N], 1, &proj);
                cblas_zaxpy(N, &proj, &V[k * N], 1, w, 1);
            }
            
            // Calculate beta_j = ||w||
            beta[j] = cblas_dznrm2(N, w, 1);
            
            // Check for convergence or linear dependency
            if (beta[j] < 1e-12) {
                std::cout << "Converged or found linear dependency at iteration " << j << std::endl;
                beta[j] = 0.0;
                break;
            }
            
            // v_{j+1} = w / beta_j
            cblas_zcopy(N, w, 1, v_jp1, 1);
            cblas_zdscal(N, 1.0 / beta[j], v_jp1, 1);
        }
        
        // Construct the tridiagonal matrix
        double d[M];  // Diagonal elements
        double e[M-1];  // Off-diagonal elements
        
        std::memcpy(d, alpha, M * sizeof(double));
        std::memcpy(e, beta, (M - 1) * sizeof(double));
        
        // Workspace for LAPACK
        double z[M * M];
        
        // Solve the eigenvalue problem for the tridiagonal matrix
        int info = LAPACKE_dsteqr(LAPACK_COL_MAJOR, 'I', M, d, e, z, M);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dsteqr failed with error code " << info << std::endl;
            return;
        }
        
        // Copy eigenvalues
        std::memcpy(eigenvalues, d, M * sizeof(double));
        
        // Transform eigenvectors back to original space: eigenvectors = V * z
        for (size_t j = 0; j < M; ++j) {
            for (size_t i = 0; i < N; ++i) {
                eigenvectors[j * N + i] = 0.0;
            }
            
            for (size_t k = 0; k < M; ++k) {
                if (beta[k] == 0.0) break;  // Stop at convergence
                complex_t z_jk(z[j * M + k], 0.0);
                cblas_zaxpy(N, &z_jk, &V[k * N], 1, &eigenvectors[j * N], 1);
            }
        }
    }
    
private:
    const complex_t* A_;  // The input matrix
};

int main() {
    constexpr size_t N = 100;
    constexpr size_t M = 20;  // Number of Lanczos vectors to use

    // Create a sparse Hermitian matrix (100x100)
    // This matrix has the following structure:
    // 1. Real values on the diagonal
    // 2. A tridiagonal structure (common in many physical systems)
    // 3. A few random off-diagonal elements (~5% of total elements)
    // 4. Hermitian property: A[i][j] = conj(A[j][i])
    std::complex<double> A[N * N] = {};  // Initialize all elements to zero

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> diag_dist(1.0, 10.0);    // For diagonal elements
    std::uniform_real_distribution<double> real_dist(-1.0, 1.0);    // For real parts
    std::uniform_real_distribution<double> imag_dist(-1.0, 1.0);    // For imaginary parts
    std::uniform_int_distribution<int> index_dist(0, N-1);          // For random indices

    // Fill diagonal with real values (Hermitian matrices have real diagonals)
    for (size_t i = 0; i < N; ++i) {
        A[i * N + i] = std::complex<double>(diag_dist(gen), 0.0);
    }

    // Add a tridiagonal structure (common in physical systems)
    for (size_t i = 0; i < N - 1; ++i) {
        std::complex<double> val(real_dist(gen), imag_dist(gen));
        A[i * N + (i + 1)] = val;
        A[(i + 1) * N + i] = std::conj(val);  // Ensure Hermitian property
    }

    // Add some random off-diagonal elements to make the matrix more interesting
    // while keeping it sparse (about 5% non-zero elements)
    int num_random_elements = N * 5;
    for (int k = 0; k < num_random_elements; ++k) {
        int i = index_dist(gen);
        int j = index_dist(gen);
        
        // Skip diagonal and adjacent elements as they're already filled
        if (i == j || i == j + 1 || i == j - 1) continue;
        
        // Ensure i < j to avoid setting the same element twice
        if (i > j) std::swap(i, j);
        
        std::complex<double> val(real_dist(gen), imag_dist(gen));
        A[i * N + j] = val;
        A[j * N + i] = std::conj(val);  // Ensure Hermitian property
    }
    
    LanczosAlgorithm<N, M> lanczos(A);
    
    double eigenvalues[M];
    std::complex<double> eigenvectors[N * M];
    
    lanczos.compute(eigenvalues, eigenvectors);
    
    // Display results
    std::cout << "Eigenvalues:" << std::endl;
    for (size_t i = 0; i < M; ++i) {
        std::cout << eigenvalues[i] << std::endl;
    }
    
    std::cout << "\nEigenvectors:" << std::endl;
    for (size_t j = 0; j < M; ++j) {
        std::cout << "Eigenvector " << j << ":" << std::endl;
        for (size_t i = 0; i < N; ++i) {
            std::cout << eigenvectors[j * N + i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compare with direct diagonalization using LAPACK
    std::cout << "\nComparing with direct diagonalization:" << std::endl;

    // Create a copy of A for LAPACK (ZHEEV overwrites the input)
    std::complex<double> A_copy[N * N];
    std::memcpy(A_copy, A, N * N * sizeof(std::complex<double>));

    // Workspace for LAPACK
    double w_direct[N];

    // Call ZHEEV to compute all eigenvalues and eigenvectors
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', N, (lapack_complex_double*)A_copy, N, w_direct);
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        return 1;
    }

    // Display direct diagonalization results
    std::cout << "Direct eigenvalues:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << w_direct[i] << std::endl;
    }

    // Find closest matches between Lanczos and direct eigenvalues
    std::cout << "\nEigenvalue comparison:" << std::endl;
    for (size_t i = 0; i < M; ++i) {
        // Find the closest eigenvalue from direct method
        size_t closest_idx = 0;
        double min_diff = std::abs(eigenvalues[i] - w_direct[0]);
        
        for (size_t j = 1; j < N; ++j) {
            double diff = std::abs(eigenvalues[i] - w_direct[j]);
            if (diff < min_diff) {
                min_diff = diff;
                closest_idx = j;
            }
        }
        
        double rel_error = std::abs(eigenvalues[i] - w_direct[closest_idx]) / 
                          std::abs(w_direct[closest_idx]);
        
        std::cout << "Lanczos λ" << i << " = " << eigenvalues[i] 
                  << " closest to direct λ" << closest_idx << " = " << w_direct[closest_idx]
                  << " (relative error: " << rel_error << ")" << std::endl;
    }

    // Calculate residual norms to verify eigenpairs quality
    std::cout << "\nResidual norms for Lanczos eigenpairs:" << std::endl;
    for (size_t i = 0; i < M; ++i) {
        std::complex<double> Av[N];
        
        // Calculate A*v
        std::complex<double> one(1.0, 0.0);
        std::complex<double> zero(0.0, 0.0);
        cblas_zhemv(CblasColMajor, CblasUpper, N, &one, A, N, 
                    &eigenvectors[i * N], 1, &zero, Av, 1);
        
        // Calculate A*v - λ*v
        for (size_t j = 0; j < N; ++j) {
            Av[j] -= eigenvalues[i] * eigenvectors[i * N + j];
        }
        
        double residual = cblas_dznrm2(N, Av, 1);
        std::cout << "Residual for eigenpair " << i << ": " << residual << std::endl;
    }

    return 0;
}