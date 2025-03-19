#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include "construct_ham.h"

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist
                                    ) {
    ComplexVector result(N);
    
    // Generate a random vector
    result = generateRandomVector(N, gen, dist);
    
    // Orthogonalize against all provided vectors using Gram-Schmidt
    for (const auto& v : vectors) {
        // Calculate projection: <v, result>
        Complex projection;
        cblas_zdotc_sub(N, v.data(), 1, result.data(), 1, &projection);
        
        // Subtract projection: result -= projection * v
        Complex neg_projection = -projection;
        cblas_zaxpy(N, &neg_projection, v.data(), 1, result.data(), 1);
    }
    
    // Check if the resulting vector has sufficient magnitude
    double norm = cblas_dznrm2(N, result.data(), 1);
        // Normalize
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, result.data(), 1);
    return result;
    
    // If all attempts failed, throw an exception
}

// Lanczos algorithm implementation
// H: Function representing the Hamiltonian operator (H|v⟩)
// N: Dimension of the Hilbert space
// max_iter: Maximum number of Lanczos iterations
// tol: Tolerance for convergence and detecting invariant subspaces
// eigenvalues: Output vector to store the eigenvalues
// eigenvectors: Output matrix to store the eigenvectors (optional)
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
             double tol, std::vector<double>& eigenvalues, 
             std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // Initialize random starting vector
    ComplexVector v_current(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    v_current = generateRandomVector(N, gen, dist);
    
    // Initialize Lanczos vectors and coefficients
    std::vector<ComplexVector> basis_vectors;
    basis_vectors.push_back(v_current);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    double norm;
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex alpha(-beta[j], 0.0);  // -beta[j] as complex number
            cblas_zaxpy(N, &alpha, v_prev.data(), 1, w.data(), 1);
        }

        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Reorthogonalize (for numerical stability)
        for (int k = 0; k <= j; k++) {
            // Calculate overlap = <basis_vectors[k], w>
            Complex overlap;
            cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
            
            // Subtract projection: w -= overlap * basis_vectors[k]
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        
        // Check for invariant subspace
        if (norm < tol) {
            // Generate a random vector instead of breaking
            v_next = generateOrthogonalVector(N, basis_vectors, gen, dist);
            
            // Update the norm for use in scaling
            norm = cblas_dznrm2(N, v_next.data(), 1);
            
            // If after orthogonalization the norm is still too small, we should break
            if (norm < tol) {
                break;
            }
        }else{
            cblas_zcopy(N, w.data(), 1, v_next.data(), 1);
        }
        
        beta.push_back(norm);
        
        // v_{j+1} = w / beta_{j+1}
        // Copy w to v_next

        // Check for invariant subspace
        // Scale v_next by 1/norm
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, v_next.data(), 1);
        
        // Store basis vector
        if (j < max_iter - 1) {
            basis_vectors.push_back(v_next);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct tridiagonal matrix
    int m = alpha.size();
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(beta.begin() + 1, beta.end()); // Skip β_0
    
    // Allocate arrays for eigenvalues and eigenvectors
    std::vector<double> evals(m);
    std::vector<double> z(m * m, 0.0);
    
    // Diagonalize tridiagonal matrix using LAPACK
    int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, eigenvectors ? 'V' : 'N', m, 
                           diag.data(), offdiag.data(), z.data(), m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstev failed with error code " << info << std::endl;
        return;
    }
    
    // Copy eigenvalues to output
    eigenvalues = diag;
    
    // If eigenvectors requested, transform back to original basis
    if (eigenvectors) {
        eigenvectors->clear();
        eigenvectors->resize(m, ComplexVector(N, Complex(0.0, 0.0)));
        
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < m; k++) {
                    (*eigenvectors)[j][i] += z[k*m + j] * basis_vectors[k][i];
                }
            }
        }
    }
}

// Lanczos algorithm with Conjugate Gradient refinement for eigenvectors
void lanczos_with_cg(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                    double tol, std::vector<double>& eigenvalues, 
                    std::vector<ComplexVector>* eigenvectors = nullptr) {
    
    // First, run the standard Lanczos algorithm to get initial approximations
    std::vector<ComplexVector> initial_eigenvectors;
    lanczos(H, N, max_iter, tol, eigenvalues, eigenvectors ? &initial_eigenvectors : nullptr);
    
    // If eigenvectors are not requested, we're done
    if (!eigenvectors) return;
    
    // Initialize output eigenvectors
    eigenvectors->clear();
    eigenvectors->resize(initial_eigenvectors.size(), ComplexVector(N, Complex(0.0, 0.0)));
    
    // For each eigenvector, apply CG refinement
    for (size_t i = 0; i < initial_eigenvectors.size(); i++) {
        double lambda = eigenvalues[i];
        ComplexVector v = initial_eigenvectors[i];
        
        // Normalize initial vector
        double norm = cblas_dznrm2(N, v.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v.data(), 1);
        
        ComplexVector r(N), p(N), Hp(N), Hv(N);
        
        // Apply H to v: Hv = H*v
        H(v.data(), Hv.data(), N);
        
        // Initial residual: r = Hv - λv
        std::copy(Hv.begin(), Hv.end(), r.begin());
        Complex neg_lambda(-lambda, 0.0);
        cblas_zaxpy(N, &neg_lambda, v.data(), 1, r.data(), 1);
        
        // Initial search direction
        std::copy(r.begin(), r.end(), p.begin());
        
        // CG iteration
        const int max_cg_iter = 50;
        const double cg_tol = tol * 0.1;
        double res_norm = cblas_dznrm2(N, r.data(), 1);
        
        for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol; iter++) {
            // Apply (H - λI) to p
            H(p.data(), Hp.data(), N);
            cblas_zaxpy(N, &neg_lambda, p.data(), 1, Hp.data(), 1);
            
            // α = (r·r) / (p·(H-λI)p)
            Complex r_dot_r, p_dot_Hp;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &p_dot_Hp);
            
            Complex alpha = r_dot_r / p_dot_Hp;
            
            // v = v + α*p
            cblas_zaxpy(N, &alpha, p.data(), 1, v.data(), 1);
            
            // Store old r·r
            Complex r_dot_r_old = r_dot_r;
            
            // r = r - α*(H-λI)p
            Complex neg_alpha = -alpha;
            cblas_zaxpy(N, &neg_alpha, Hp.data(), 1, r.data(), 1);
            
            // Check convergence
            res_norm = cblas_dznrm2(N, r.data(), 1);
            
            // β = (r_new·r_new) / (r_old·r_old)
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            Complex beta = r_dot_r / r_dot_r_old;
            
            // p = r + β*p
            for (int j = 0; j < N; j++) {
                p[j] = r[j] + beta * p[j];
            }
        }
        
        // Normalize final eigenvector
        norm = cblas_dznrm2(N, v.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v.data(), 1);
        
        // Update eigenvalue using Rayleigh quotient
        H(v.data(), Hv.data(), N);
        Complex lambda_new;
        cblas_zdotc_sub(N, v.data(), 1, Hv.data(), 1, &lambda_new);
        eigenvalues[i] = std::real(lambda_new);
        
        // Store the refined eigenvector
        (*eigenvectors)[i] = v;
    }
}

int main(){
    Operator op(4);
    op.readTrans("./ED_test/Trans.def");
    // op.readInterAll("./ED_test/InterAll.def");
    // printMatrixRepresentation(op);
    std::vector<double> eigenvalues;

    lanczos([&](const Complex* v, Complex* Hv, int N) {
        std::vector<Complex> vec(v, v + N);
        std::vector<Complex> result(N, Complex(0.0, 0.0));
        op.transform(vec, result);
        std::copy(result.begin(), result.end(), Hv);
    }, 16, 16, 1e-6, eigenvalues);
    
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << std::endl;
    } 
    return 0;
}
