#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <cblas.h>
#include <lapacke.h>
#include <cstring>
#include <functional>
#include "construct_ham.h"
template<size_t N, size_t M>
class LanczosAlgorithm {
public:
    using complex_t = std::complex<double>;
    using MatrixVectorFunction = std::function<void(const complex_t*, complex_t*)>;
    using VectorApplyFunction = std::function<std::vector<complex_t>(const std::vector<complex_t>&)>;
    
    // Constructor for direct matrix operations
    LanczosAlgorithm(const complex_t* matrix) {
        matvec_func_ = [matrix](const complex_t* v, complex_t* result) {
            std::complex<double> one(1.0, 0.0);
            std::complex<double> zero(0.0, 0.0);
            cblas_zhemv(CblasColMajor, CblasUpper, N, &one, matrix, N, v, 1, &zero, result, 1);
        };
    }
    
    // Constructor for vector-based apply function (compatible with Operator::apply)
    LanczosAlgorithm(VectorApplyFunction vector_apply_func) {
        matvec_func_ = [vector_apply_func](const complex_t* v, complex_t* result) {
            // Convert raw array to vector
            std::vector<complex_t> vec_in(v, v + N);
            
            // Apply the function
            std::vector<complex_t> vec_out = vector_apply_func(vec_in);
            
            // Copy result back to raw array
            std::copy(vec_out.begin(), vec_out.end(), result);
        };
        std::cout << "LanczosAlgorithm initialized" << std::endl;
    }

    // Perform the Lanczos algorithm and return eigenvalues and eigenvectors
    void compute(double* eigenvalues) const {
        // Initialize random v_1
        complex_t V[N * (M + 1)];
        complex_t w[N];
        double alpha[M];
        double beta[M];
        
        // Create random initial vector
        std::cout << "Generating random initial vector" << std::endl;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (size_t i = 0; i < N; ++i) {
            V[i] = complex_t(dist(gen), dist(gen));
        }

        
        // Normalize v_1
        double norm = cblas_dznrm2(N, V, 1);
        cblas_zdscal(N, 1.0 / norm, V, 1);

        std::cout << "Random initial vector generated" << std::endl;
        for (size_t i = 0; i < N; ++i) {
            std::cout << V[i] << " ";
        }
        std::cout << std::endl;
        // Lanczos iteration
        for (size_t j = 0; j < M; ++j) {
            std::cout << "Lanczos step " << j << std::endl;
            complex_t* v_j = &V[j * N];
            complex_t* v_jp1 = &V[(j + 1) * N];
            
            // w = A * v_j using the function
            matvec_func_(v_j, w);
            std::cout << "Computing w " << std::endl;
            for (size_t i = 0; i < N; ++i) {
                std::cout << w[i] << " ";
            }
            std::cout << std::endl;
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
                complex_t neg_proj = -proj;
                cblas_zaxpy(N, &neg_proj, &V[k * N], 1, w, 1);
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
        // for (size_t j = 0; j < M; ++j) {
        //     for (size_t i = 0; i < N; ++i) {
        //         eigenvectors[j * N + i] = 0.0;
        //     }
            
        //     for (size_t k = 0; k < M; ++k) {
        //         if (beta[k] == 0.0 && k > 0) break;  // Stop at convergence
        //         complex_t z_jk(z[j * M + k], 0.0);
        //         cblas_zaxpy(N, &z_jk, &V[k * N], 1, &eigenvectors[j * N], 1);
        //     }
        // }
    }
    
private:
    MatrixVectorFunction matvec_func_;  // Function that computes A*v
};



// Example of how to use the Operator class
int main() {
    // Create an operator for a 3-bit system
    int n_bits = 4;
    Operator op(n_bits);

    op.loadFromFile("./ED_test/Trans.def");
    // op.loadFromInterAllFile("./ED_test/InterAll.def");

    printMatrix(op.generateMatrix());
    LanczosAlgorithm<1 << 4, 16> lanczos([&op](const std::vector<std::complex<double>>& vec) {
        return op.apply(vec);
    });
    double eigenvalues[16];
    // std::complex<double> eigenvectors[100 * (1 << 16)];
    lanczos.compute(eigenvalues);
    for (int i = 0; i < 16; ++i) {
        std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << std::endl;
    }
    
    return 0;
}
