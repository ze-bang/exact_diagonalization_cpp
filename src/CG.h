#ifndef CG_H
#define CG_H

#pragma once

#include <vector>
#include <complex>
#include <functional>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cblas.h>
#include <lapacke.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <string>

// Define types for convenience
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Standard CG method for finding the lowest eigenvalue and eigenvector of a Hermitian matrix
void cg_lowest_eigenvalue(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    double tol,                                            // Tolerance for convergence
    double& eigenvalue,                                    // Output eigenvalue
    ComplexVector& eigenvector                             // Output eigenvector
) {
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    eigenvector.resize(N);
    for (int i = 0; i < N; i++) {
        eigenvector[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, eigenvector.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, eigenvector.data(), 1);
    
    // Workspace
    ComplexVector Hv(N);
    ComplexVector r(N);  // Residual
    ComplexVector p(N);  // Search direction
    ComplexVector Hp(N); // H*p
    
    // Apply H to initial vector
    H(eigenvector.data(), Hv.data(), N);
    
    // Calculate initial Rayleigh quotient (eigenvalue estimate)
    Complex rayleigh;
    cblas_zdotc_sub(N, eigenvector.data(), 1, Hv.data(), 1, &rayleigh);
    eigenvalue = std::real(rayleigh);
    
    // Initial residual: r = Hv - λv
    for (int i = 0; i < N; i++) {
        r[i] = Hv[i] - eigenvalue * eigenvector[i];
    }
    
    // Initial search direction is the residual
    p = r;
    
    std::cout << "CG: Starting iterations..." << std::endl;
    std::cout << "Initial eigenvalue estimate: " << eigenvalue << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Apply H to search direction
        H(p.data(), Hp.data(), N);
        
        // Calculate step size
        Complex numerator, denominator;
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &numerator);
        cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &denominator);
        
        Complex alpha = numerator / denominator;
        
        // Update eigenvector: v = v + α*p
        cblas_zaxpy(N, &alpha, p.data(), 1, eigenvector.data(), 1);
        
        // Normalize the eigenvector
        norm = cblas_dznrm2(N, eigenvector.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, eigenvector.data(), 1);
        
        // Apply H to updated eigenvector
        H(eigenvector.data(), Hv.data(), N);
        
        // Update Rayleigh quotient
        cblas_zdotc_sub(N, eigenvector.data(), 1, Hv.data(), 1, &rayleigh);
        double new_eigenvalue = std::real(rayleigh);
        
        // Calculate new residual
        ComplexVector r_new(N);
        for (int i = 0; i < N; i++) {
            r_new[i] = Hv[i] - new_eigenvalue * eigenvector[i];
        }
        
        // Check convergence
        double res_norm = cblas_dznrm2(N, r_new.data(), 1);
        double eig_change = std::abs(new_eigenvalue - eigenvalue);
        
        if (iter % 10 == 0 || res_norm < tol) {
            std::cout << "Iteration " << iter << ": eigenvalue = " << new_eigenvalue 
                      << ", residual = " << res_norm 
                      << ", change = " << eig_change << std::endl;
        }
        
        if (res_norm < tol && eig_change < tol) {
            std::cout << "CG converged after " << iter+1 << " iterations." << std::endl;
            eigenvalue = new_eigenvalue;
            break;
        }
        
        // Update eigenvalue
        eigenvalue = new_eigenvalue;
        
        // Calculate beta for CG update
        Complex r_new_dot_r_new;
        cblas_zdotc_sub(N, r_new.data(), 1, r_new.data(), 1, &r_new_dot_r_new);
        
        Complex r_dot_r;
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
        
        Complex beta = r_new_dot_r_new / r_dot_r;
        
        // Update search direction: p = r_new + beta*p
        for (int i = 0; i < N; i++) {
            p[i] = r_new[i] + beta * p[i];
        }
        
        // Update residual
        r = r_new;
    }
}

// CG with deflation for finding multiple eigenvalues
void cg_multiple_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations per eigenvalue
    int num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors               // Output eigenvectors
) {
    eigenvalues.clear();
    eigenvectors.clear();
    
    // Function to apply deflated operator: H' = H - sum_i λ_i |ψ_i⟩⟨ψ_i|
    auto apply_deflated_operator = [&](const Complex* v, Complex* result, int size) {
        // First apply original operator H
        H(v, result, size);
        
        // Then subtract projections onto already found eigenvectors
        for (size_t i = 0; i < eigenvectors.size(); i++) {
            Complex projection;
            cblas_zdotc_sub(size, eigenvectors[i].data(), 1, v, 1, &projection);
            
            // result -= projection * eigenvalues[i] * eigenvectors[i]
            Complex scale = -eigenvalues[i] * projection;
            cblas_zaxpy(size, &scale, eigenvectors[i].data(), 1, result, 1);
        }
    };
    
    std::cout << "CG: Finding " << num_eigenvalues << " eigenvalues..." << std::endl;
    
    // Find eigenvalues one by one using deflation
    for (int k = 0; k < num_eigenvalues; k++) {
        std::cout << "Finding eigenvalue " << k+1 << " of " << num_eigenvalues << std::endl;
        
        double eigenvalue;
        ComplexVector eigenvector;
        
        // Initialize random starting vector
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        eigenvector.resize(N);
        for (int i = 0; i < N; i++) {
            eigenvector[i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against already found eigenvectors
        for (const auto& v : eigenvectors) {
            Complex projection;
            cblas_zdotc_sub(N, v.data(), 1, eigenvector.data(), 1, &projection);
            
            // eigenvector -= projection * v
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, v.data(), 1, eigenvector.data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, eigenvector.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, eigenvector.data(), 1);
        
        // Workspace
        ComplexVector Hv(N);
        ComplexVector r(N);  // Residual
        ComplexVector p(N);  // Search direction
        ComplexVector Hp(N); // H*p
        
        // Apply deflated operator to initial vector
        apply_deflated_operator(eigenvector.data(), Hv.data(), N);
        
        // Calculate initial Rayleigh quotient (eigenvalue estimate)
        Complex rayleigh;
        cblas_zdotc_sub(N, eigenvector.data(), 1, Hv.data(), 1, &rayleigh);
        eigenvalue = std::real(rayleigh);
        
        // Initial residual: r = Hv - λv
        for (int i = 0; i < N; i++) {
            r[i] = Hv[i] - eigenvalue * eigenvector[i];
        }
        
        // Initial search direction is the residual
        p = r;
        
        for (int iter = 0; iter < max_iter; iter++) {
            // Apply deflated operator to search direction
            apply_deflated_operator(p.data(), Hp.data(), N);
            
            // Calculate step size
            Complex numerator, denominator;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &numerator);
            cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &denominator);
            
            Complex alpha = numerator / denominator;
            
            // Update eigenvector: v = v + α*p
            cblas_zaxpy(N, &alpha, p.data(), 1, eigenvector.data(), 1);
            
            // Normalize the eigenvector
            norm = cblas_dznrm2(N, eigenvector.data(), 1);
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, eigenvector.data(), 1);
            
            // Orthogonalize against already found eigenvectors for stability
            for (const auto& v : eigenvectors) {
                Complex projection;
                cblas_zdotc_sub(N, v.data(), 1, eigenvector.data(), 1, &projection);
                
                // eigenvector -= projection * v
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, v.data(), 1, eigenvector.data(), 1);
            }
            
            // Renormalize
            norm = cblas_dznrm2(N, eigenvector.data(), 1);
            if (norm > 1e-10) {
                scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, eigenvector.data(), 1);
            }
            
            // Apply deflated operator to updated eigenvector
            apply_deflated_operator(eigenvector.data(), Hv.data(), N);
            
            // Update Rayleigh quotient with original operator
            Complex orig_rayleigh;
            H(eigenvector.data(), Hp.data(), N);
            cblas_zdotc_sub(N, eigenvector.data(), 1, Hp.data(), 1, &orig_rayleigh);
            double new_eigenvalue = std::real(orig_rayleigh);
            
            // Calculate new residual with deflated operator
            ComplexVector r_new(N);
            for (int i = 0; i < N; i++) {
                r_new[i] = Hv[i] - new_eigenvalue * eigenvector[i];
            }
            
            // Check convergence
            double res_norm = cblas_dznrm2(N, r_new.data(), 1);
            double eig_change = std::abs(new_eigenvalue - eigenvalue);
            
            if (iter % 10 == 0 || res_norm < tol) {
                std::cout << "Iteration " << iter << ": eigenvalue = " << new_eigenvalue 
                          << ", residual = " << res_norm 
                          << ", change = " << eig_change << std::endl;
            }
            
            if (res_norm < tol && eig_change < tol) {
                std::cout << "CG converged after " << iter+1 << " iterations." << std::endl;
                eigenvalue = new_eigenvalue;
                break;
            }
            
            // Update eigenvalue
            eigenvalue = new_eigenvalue;
            
            // Calculate beta for CG update
            Complex r_new_dot_r_new;
            cblas_zdotc_sub(N, r_new.data(), 1, r_new.data(), 1, &r_new_dot_r_new);
            
            Complex r_dot_r;
            cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
            
            Complex beta = r_new_dot_r_new / r_dot_r;
            
            // Update search direction: p = r_new + beta*p
            for (int i = 0; i < N; i++) {
                p[i] = r_new[i] + beta * p[i];
            }
            
            // Update residual
            r = r_new;
        }
        
        // Store the computed eigenvalue and eigenvector
        eigenvalues.push_back(eigenvalue);
        eigenvectors.push_back(eigenvector);
    }
}

// Block CG method for finding multiple eigenvalues simultaneously
void block_cg(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    int block_size,                                        // Block size (number of vectors)
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir = ""                                   // Directory for temporary files
) {
    // Create a directory for temporary files if needed
    std::string temp_dir = dir + "/block_cg_temp";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Initialize block of random vectors
    eigenvectors.clear();
    eigenvectors.resize(block_size, ComplexVector(N));
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int b = 0; b < block_size; b++) {
        for (int i = 0; i < N; i++) {
            eigenvectors[b][i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors
        for (int prev = 0; prev < b; prev++) {
            Complex projection;
            cblas_zdotc_sub(N, eigenvectors[prev].data(), 1, eigenvectors[b].data(), 1, &projection);
            
            // Subtract projection
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, eigenvectors[prev].data(), 1, eigenvectors[b].data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, eigenvectors[b].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, eigenvectors[b].data(), 1);
    }
    
    // Initialize block workspace
    std::vector<ComplexVector> HV(block_size, ComplexVector(N));  // H applied to each vector
    std::vector<ComplexVector> R(block_size, ComplexVector(N));   // Residuals
    std::vector<ComplexVector> P(block_size, ComplexVector(N));   // Search directions
    std::vector<ComplexVector> HP(block_size, ComplexVector(N));  // H applied to search directions
    
    // Apply H to initial block
    for (int b = 0; b < block_size; b++) {
        H(eigenvectors[b].data(), HV[b].data(), N);
    }
    
    // Calculate Rayleigh quotient matrix
    std::vector<std::vector<Complex>> rayleigh_matrix(block_size, std::vector<Complex>(block_size));
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            cblas_zdotc_sub(N, eigenvectors[i].data(), 1, HV[j].data(), 1, &rayleigh_matrix[i][j]);
        }
    }
    
    // Diagonalize the Rayleigh quotient matrix
    Eigen::MatrixXcd eigen_rayleigh(block_size, block_size);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            eigen_rayleigh(i, j) = rayleigh_matrix[i][j];
        }
    }
    
    // Ensure the matrix is Hermitian
    eigen_rayleigh = (eigen_rayleigh + eigen_rayleigh.adjoint()) / 2.0;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(eigen_rayleigh);
    Eigen::VectorXd rq_eigenvals = solver.eigenvalues();
    Eigen::MatrixXcd rq_eigenvecs = solver.eigenvectors();
    
    // Initialize eigenvalues
    eigenvalues.resize(block_size);
    for (int i = 0; i < block_size; i++) {
        eigenvalues[i] = rq_eigenvals[i];
    }
    
    // Rotate the block to align with eigenvectors of the Rayleigh quotient matrix
    std::vector<ComplexVector> rotated_block(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    std::vector<ComplexVector> rotated_HV(block_size, ComplexVector(N, Complex(0.0, 0.0)));
    
    for (int b = 0; b < block_size; b++) {
        for (int j = 0; j < block_size; j++) {
            Complex coef = rq_eigenvecs(j, b);
            for (int i = 0; i < N; i++) {
                rotated_block[b][i] += coef * eigenvectors[j][i];
                rotated_HV[b][i] += coef * HV[j][i];
            }
        }
    }
    
    eigenvectors = rotated_block;
    HV = rotated_HV;
    
    // Calculate initial residuals
    for (int b = 0; b < block_size; b++) {
        for (int i = 0; i < N; i++) {
            R[b][i] = HV[b][i] - eigenvalues[b] * eigenvectors[b][i];
        }
        
        // Initial search direction is the residual
        P[b] = R[b];
    }
    
    std::cout << "Block CG: Starting iterations with block size " << block_size << "..." << std::endl;
    std::cout << "Initial eigenvalues: ";
    for (int b = 0; b < block_size; b++) {
        std::cout << eigenvalues[b] << " ";
    }
    std::cout << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Apply H to each search direction
        for (int b = 0; b < block_size; b++) {
            H(P[b].data(), HP[b].data(), N);
        }
        
        // Calculate step sizes
        std::vector<Complex> alpha(block_size);
        for (int b = 0; b < block_size; b++) {
            Complex numerator, denominator;
            cblas_zdotc_sub(N, R[b].data(), 1, R[b].data(), 1, &numerator);
            cblas_zdotc_sub(N, P[b].data(), 1, HP[b].data(), 1, &denominator);
            
            alpha[b] = numerator / denominator;
        }
        
        // Update eigenvectors
        for (int b = 0; b < block_size; b++) {
            cblas_zaxpy(N, &alpha[b], P[b].data(), 1, eigenvectors[b].data(), 1);
        }
        
        // Orthogonalize the block using Gram-Schmidt
        for (int b = 0; b < block_size; b++) {
            // Orthogonalize against previous vectors in block
            for (int prev = 0; prev < b; prev++) {
                Complex projection;
                cblas_zdotc_sub(N, eigenvectors[prev].data(), 1, eigenvectors[b].data(), 1, &projection);
                
                // Subtract projection
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, eigenvectors[prev].data(), 1, eigenvectors[b].data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, eigenvectors[b].data(), 1);
            if (norm > 1e-10) {
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, eigenvectors[b].data(), 1);
            }
        }
        
        // Apply H to updated eigenvectors
        for (int b = 0; b < block_size; b++) {
            H(eigenvectors[b].data(), HV[b].data(), N);
        }
        
        // Calculate new Rayleigh quotient matrix
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                cblas_zdotc_sub(N, eigenvectors[i].data(), 1, HV[j].data(), 1, &rayleigh_matrix[i][j]);
            }
        }
        
        // Diagonalize the new Rayleigh quotient matrix
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                eigen_rayleigh(i, j) = rayleigh_matrix[i][j];
            }
        }
        
        // Ensure the matrix is Hermitian
        eigen_rayleigh = (eigen_rayleigh + eigen_rayleigh.adjoint()) / 2.0;
        
        solver.compute(eigen_rayleigh);
        Eigen::VectorXd new_eigenvals = solver.eigenvalues();
        Eigen::MatrixXcd new_eigenvecs = solver.eigenvectors();
        
        // Rotate the block to align with new eigenvectors
        for (int b = 0; b < block_size; b++) {
            std::fill(rotated_block[b].begin(), rotated_block[b].end(), Complex(0.0, 0.0));
            std::fill(rotated_HV[b].begin(), rotated_HV[b].end(), Complex(0.0, 0.0));
            
            for (int j = 0; j < block_size; j++) {
                Complex coef = new_eigenvecs(j, b);
                for (int i = 0; i < N; i++) {
                    rotated_block[b][i] += coef * eigenvectors[j][i];
                    rotated_HV[b][i] += coef * HV[j][i];
                }
            }
        }
        
        // Check convergence
        double max_res_norm = 0.0;
        double max_eig_change = 0.0;
        
        for (int b = 0; b < block_size; b++) {
            double eig_change = std::abs(new_eigenvals[b] - eigenvalues[b]);
            max_eig_change = std::max(max_eig_change, eig_change);
            
            // Update eigenvalues and vectors
            eigenvalues[b] = new_eigenvals[b];
            eigenvectors[b] = rotated_block[b];
            HV[b] = rotated_HV[b];
            
            // Calculate residuals
            for (int i = 0; i < N; i++) {
                R[b][i] = HV[b][i] - eigenvalues[b] * eigenvectors[b][i];
            }
            
            double res_norm = cblas_dznrm2(N, R[b].data(), 1);
            max_res_norm = std::max(max_res_norm, res_norm);
        }
        
        if (iter % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << iter << ": max residual = " << max_res_norm 
                      << ", max eigenvalue change = " << max_eig_change << std::endl;
            std::cout << "Eigenvalues: ";
            for (int b = 0; b < block_size; b++) {
                std::cout << eigenvalues[b] << " ";
            }
            std::cout << std::endl;
        }
        
        if (max_res_norm < tol && max_eig_change < tol) {
            std::cout << "Block CG converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
        
        // Calculate beta for CG update
        std::vector<Complex> beta(block_size);
        for (int b = 0; b < block_size; b++) {
            Complex r_new_dot_r_new;
            cblas_zdotc_sub(N, R[b].data(), 1, R[b].data(), 1, &r_new_dot_r_new);
            
            Complex r_old_dot_r_old;
            if (iter > 0) {
                ComplexVector r_old(N);
                for (int i = 0; i < N; i++) {
                    r_old[i] = HV[b][i] - eigenvalues[b] * eigenvectors[b][i];
                }
                cblas_zdotc_sub(N, r_old.data(), 1, r_old.data(), 1, &r_old_dot_r_old);
            } else {
                r_old_dot_r_old = r_new_dot_r_new;
            }
            
            beta[b] = r_new_dot_r_new / r_old_dot_r_old;
        }
        
        // Update search directions: P = R + beta*P
        for (int b = 0; b < block_size; b++) {
            for (int i = 0; i < N; i++) {
                P[b][i] = R[b][i] + beta[b] * P[b][i];
            }
        }
    }
    
    // Clean up temporary directory if created
    if (!dir.empty()) {
        std::string cmd = "rm -rf " + temp_dir;
        system(cmd.c_str());
    }
    
    // Sort eigenvalues and eigenvectors
    std::vector<std::pair<double, int>> eig_pairs(block_size);
    for (int i = 0; i < block_size; i++) {
        eig_pairs[i] = {eigenvalues[i], i};
    }
    std::sort(eig_pairs.begin(), eig_pairs.end());
    
    std::vector<double> sorted_eigenvalues(block_size);
    std::vector<ComplexVector> sorted_eigenvectors(block_size, ComplexVector(N));
    
    for (int i = 0; i < block_size; i++) {
        sorted_eigenvalues[i] = eig_pairs[i].first;
        sorted_eigenvectors[i] = eigenvectors[eig_pairs[i].second];
    }
    
    eigenvalues = sorted_eigenvalues;
    eigenvectors = sorted_eigenvectors;
}

// Davidson method for finding lowest eigenvalues
void davidson_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    int max_subspace,                                      // Maximum subspace size
    int num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir = ""                                   // Directory for temporary files
) {
    // Create directory for temporary files if needed
    std::string temp_dir = dir + "/davidson_temp";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Initialize subspace with random vectors
    std::vector<ComplexVector> V;  // Basis vectors of the subspace
    std::vector<ComplexVector> HV; // H applied to basis vectors
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Start with num_eigenvalues initial vectors
    for (int i = 0; i < num_eigenvalues; i++) {
        ComplexVector v(N);
        for (int j = 0; j < N; j++) {
            v[j] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors
        for (size_t j = 0; j < V.size(); j++) {
            Complex projection;
            cblas_zdotc_sub(N, V[j].data(), 1, v.data(), 1, &projection);
            
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, V[j].data(), 1, v.data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v.data(), 1);
        
        // Apply H to the vector
        ComplexVector hv(N);
        H(v.data(), hv.data(), N);
        
        // Add to subspace
        V.push_back(v);
        HV.push_back(hv);
    }
    
    // Initialize eigenvalues and residuals
    eigenvalues.resize(num_eigenvalues);
    std::vector<ComplexVector> residuals(num_eigenvalues, ComplexVector(N));
    
    std::cout << "Davidson: Starting iterations..." << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        int subspace_size = V.size();
        
        // Construct the projected Hamiltonian matrix H_proj = V^H * H * V
        std::vector<std::vector<Complex>> H_proj(subspace_size, std::vector<Complex>(subspace_size));
        
        for (int i = 0; i < subspace_size; i++) {
            for (int j = 0; j < subspace_size; j++) {
                cblas_zdotc_sub(N, V[i].data(), 1, HV[j].data(), 1, &H_proj[i][j]);
            }
        }
        
        // Solve the eigenvalue problem in the subspace
        Eigen::MatrixXcd eigen_H_proj(subspace_size, subspace_size);
        for (int i = 0; i < subspace_size; i++) {
            for (int j = 0; j < subspace_size; j++) {
                eigen_H_proj(i, j) = H_proj[i][j];
            }
        }
        
        // Ensure the matrix is Hermitian
        eigen_H_proj = (eigen_H_proj + eigen_H_proj.adjoint()) / 2.0;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(eigen_H_proj);
        Eigen::VectorXd proj_eigenvals = solver.eigenvalues();
        Eigen::MatrixXcd proj_eigenvecs = solver.eigenvectors();
        
        // Update eigenvalues and compute eigenvectors in the original space
        for (int i = 0; i < num_eigenvalues; i++) {
            eigenvalues[i] = proj_eigenvals[i];
            
            // Initialize Ritz vector in the original space
            ComplexVector ritz_vector(N, Complex(0.0, 0.0));
            for (int j = 0; j < subspace_size; j++) {
                Complex coef = proj_eigenvecs(j, i);
                cblas_zaxpy(N, &coef, V[j].data(), 1, ritz_vector.data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, ritz_vector.data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, ritz_vector.data(), 1);
            
            // Apply H to get residual
            ComplexVector Hritz(N);
            H(ritz_vector.data(), Hritz.data(), N);
            
            // Compute residual: r = Hv - λv
            for (int j = 0; j < N; j++) {
                residuals[i][j] = Hritz[j] - eigenvalues[i] * ritz_vector[j];
            }
            
            // Store eigenvector if this is the last iteration or if converged
            if (i < eigenvectors.size()) {
                eigenvectors[i] = ritz_vector;
            } else {
                eigenvectors.push_back(ritz_vector);
            }
        }
        
        // Check convergence
        double max_res_norm = 0.0;
        for (int i = 0; i < num_eigenvalues; i++) {
            double res_norm = cblas_dznrm2(N, residuals[i].data(), 1);
            max_res_norm = std::max(max_res_norm, res_norm);
        }
        
        if (iter % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << iter << ": max residual = " << max_res_norm << std::endl;
            std::cout << "Eigenvalues: ";
            for (int i = 0; i < num_eigenvalues; i++) {
                std::cout << eigenvalues[i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (max_res_norm < tol) {
            std::cout << "Davidson converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
        
        // Check if subspace is getting too large
        if (subspace_size >= max_subspace) {
            std::cout << "Subspace size limit reached. Restarting with current Ritz vectors." << std::endl;
            
            // Restart with the current Ritz vectors
            std::vector<ComplexVector> new_V;
            std::vector<ComplexVector> new_HV;
            
            for (int i = 0; i < num_eigenvalues; i++) {
                ComplexVector Hritz(N);
                H(eigenvectors[i].data(), Hritz.data(), N);
                
                new_V.push_back(eigenvectors[i]);
                new_HV.push_back(Hritz);
            }
            
            V = new_V;
            HV = new_HV;
            continue;
        }
        
        // Add correction vectors to the subspace
        for (int i = 0; i < num_eigenvalues; i++) {
            // Compute preconditioner: diagonal approximation (D-λI)^-1
            ComplexVector correction(N);
            
            for (int j = 0; j < N; j++) {
                // Simple diagonal preconditioner
                // In practice, you might need a better approximation of the diagonal
                double diag_approx = 1.0;  // Placeholder - ideally get diagonal element of H
                
                if (std::abs(diag_approx - eigenvalues[i]) > 1e-10) {
                    correction[j] = residuals[i][j] / (diag_approx - eigenvalues[i]);
                } else {
                    correction[j] = residuals[i][j];
                }
            }
            
            // Orthogonalize against all existing basis vectors
            for (size_t j = 0; j < V.size(); j++) {
                Complex projection;
                cblas_zdotc_sub(N, V[j].data(), 1, correction.data(), 1, &projection);
                
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, V[j].data(), 1, correction.data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, correction.data(), 1);
            if (norm > 1e-10) {
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, correction.data(), 1);
                
                // Apply H to the correction vector
                ComplexVector hv(N);
                H(correction.data(), hv.data(), N);
                
                // Add to subspace
                V.push_back(correction);
                HV.push_back(hv);
            }
        }
    }
    
    // Clean up temporary directory if created
    if (!dir.empty()) {
        std::string cmd = "rm -rf " + temp_dir;
        system(cmd.c_str());
    }
}

// BiCG method for finding eigenvalues from CalcSpectrumByBiCG.c
void bicg_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir = ""                                   // Directory for temporary files
) {
    // Create directory for temporary files if needed
    std::string temp_dir = dir + "/bicg_temp";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Initialize random vectors
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v(N);  // Current vector
    ComplexVector w(N);  // Shadow vector
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
        w[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double v_norm = cblas_dznrm2(N, v.data(), 1);
    Complex v_scale = Complex(1.0/v_norm, 0.0);
    cblas_zscal(N, &v_scale, v.data(), 1);
    
    double w_norm = cblas_dznrm2(N, w.data(), 1);
    Complex w_scale = Complex(1.0/w_norm, 0.0);
    cblas_zscal(N, &w_scale, w.data(), 1);
    
    // Workspace
    ComplexVector v_old(N);  // Previous v
    ComplexVector w_old(N);  // Previous w
    ComplexVector Hv(N);     // H*v
    ComplexVector Hw(N);     // H*w
    
    // Initialize alpha and beta for tridiagonal matrix
    std::vector<Complex> alpha;  // Diagonal elements
    std::vector<Complex> beta;   // Off-diagonal elements
    beta.push_back(Complex(0.0, 0.0));  // β₀ not used
    
    // Temporary vectors for BiCG without look-ahead
    std::vector<Complex> gamma;  // Additional coefficients
    
    std::cout << "BiCG: Starting iterations..." << std::endl;
    
    for (int j = 0; j < max_iter; j++) {
        // Save old vectors
        v_old = v;
        w_old = w;
        
        // Apply H to v and w
        H(v.data(), Hv.data(), N);
        H(w.data(), Hw.data(), N);
        
        // Compute α_j = <w, Hv> / <w, v>
        Complex wHv, wv;
        cblas_zdotc_sub(N, w.data(), 1, Hv.data(), 1, &wHv);
        cblas_zdotc_sub(N, w.data(), 1, v.data(), 1, &wv);
        
        Complex alpha_j;
        if (std::abs(wv) > 1e-10) {
            alpha_j = wHv / wv;
        } else {
            alpha_j = Complex(0.0, 0.0);
        }
        alpha.push_back(alpha_j);
        
        // Compute residuals
        ComplexVector r(N), s(N);
        for (int i = 0; i < N; i++) {
            r[i] = Hv[i] - alpha_j * v[i];
            if (j > 0) {
                r[i] -= beta[j] * v_old[i];
            }
            
            s[i] = Hw[i] - std::conj(alpha_j) * w[i];
            if (j > 0) {
                s[i] -= std::conj(beta[j]) * w_old[i];
            }
        }
        
        // Compute β_{j+1} = <s, r> / <w, v>
        Complex sr;
        cblas_zdotc_sub(N, s.data(), 1, r.data(), 1, &sr);
        
        Complex beta_jp1;
        if (std::abs(wv) > 1e-10) {
            beta_jp1 = sr / wv;
        } else {
            beta_jp1 = Complex(0.0, 0.0);
        }
        beta.push_back(beta_jp1);
        
        // Normalize for the next iteration
        if (std::abs(beta_jp1) > 1e-10) {
            Complex scale_r = Complex(1.0/std::sqrt(std::abs(beta_jp1)), 0.0);
            Complex scale_s = std::conj(scale_r);
            
            for (int i = 0; i < N; i++) {
                v[i] = r[i] * scale_r;
                w[i] = s[i] * scale_s;
            }
        } else {
            // If β_{j+1} is too small, generate new random vectors
            for (int i = 0; i < N; i++) {
                v[i] = Complex(dist(gen), dist(gen));
                w[i] = Complex(dist(gen), dist(gen));
            }
            
            // Normalize
            v_norm = cblas_dznrm2(N, v.data(), 1);
            v_scale = Complex(1.0/v_norm, 0.0);
            cblas_zscal(N, &v_scale, v.data(), 1);
            
            w_norm = cblas_dznrm2(N, w.data(), 1);
            w_scale = Complex(1.0/w_norm, 0.0);
            cblas_zscal(N, &w_scale, w.data(), 1);
        }
        
        // Check convergence by computing the tridiagonal matrix eigenvalues
        if ((j+1) % 10 == 0 || j == max_iter-1) {
            int m = alpha.size();
            std::vector<double> diag(m);
            std::vector<double> offdiag(m-1);
            
            for (int i = 0; i < m; i++) {
                diag[i] = std::real(alpha[i]);
            }
            
            for (int i = 0; i < m-1; i++) {
                offdiag[i] = std::real(beta[i+1]);
            }
            
            // Solve the tridiagonal eigenvalue problem
            std::vector<double> evals(m);
            int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', m, diag.data(), 
                                    offdiag.data(), nullptr, m);
            
            if (info == 0) {
                eigenvalues = diag;
                std::sort(eigenvalues.begin(), eigenvalues.end());
                
                std::cout << "Iteration " << j+1 << " eigenvalues: ";
                for (int i = 0; i < std::min(5, (int)eigenvalues.size()); i++) {
                    std::cout << eigenvalues[i] << " ";
                }
                std::cout << "..." << std::endl;
                
                // Check for convergence
                if (j > 10 && std::abs(beta.back()) < tol) {
                    std::cout << "BiCG converged after " << j+1 << " iterations." << std::endl;
                    break;
                }
            } else {
                std::cerr << "Warning: LAPACKE_dstev failed with error code " << info << std::endl;
            }
        }
    }
    
    // Calculate eigenvectors if needed
    if (!eigenvectors.empty() || eigenvectors.size() == 0) {
        // Construct the full tridiagonal matrix
        int m = alpha.size();
        std::vector<double> diag(m);
        std::vector<double> offdiag(m-1);
        
        for (int i = 0; i < m; i++) {
            diag[i] = std::real(alpha[i]);
        }
        
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = std::real(beta[i+1]);
        }
        
        // Solve the tridiagonal eigenvalue problem with eigenvectors
        std::vector<double> evecs(m * m);
        int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', m, diag.data(), 
                                offdiag.data(), evecs.data(), m);
        
        if (info == 0) {
            // Update eigenvalues
            eigenvalues = diag;
            
            // Sort indices
            std::vector<std::pair<double, int>> eig_pairs(m);
            for (int i = 0; i < m; i++) {
                eig_pairs[i] = {eigenvalues[i], i};
            }
            std::sort(eig_pairs.begin(), eig_pairs.end());
            
            // Determine how many eigenvectors to compute
            int num_vecs = std::min(10, m);  // Default to 10 or fewer
            if (!eigenvectors.empty()) {
                num_vecs = std::min(num_vecs, (int)eigenvectors.size());
            }
            
            // Resize eigenvectors if needed
            if (eigenvectors.size() < num_vecs) {
                eigenvectors.resize(num_vecs, ComplexVector(N));
            }
            
            // Compute transformed eigenvectors in original space
            for (int k = 0; k < num_vecs; k++) {
                int idx = eig_pairs[k].second;  // Get sorted index
                
                // Initialize eigenvector
                std::fill(eigenvectors[k].begin(), eigenvectors[k].end(), Complex(0.0, 0.0));
                
                // Read each BiCG vector from disk or rebuild them
                ComplexVector v_j(N), v_prev(N, Complex(0.0, 0.0));
                
                // Initialize with random vector (same seed as in the iteration)
                for (int i = 0; i < N; i++) {
                    v_j[i] = Complex(dist(gen), dist(gen));
                }
                
                double norm = cblas_dznrm2(N, v_j.data(), 1);
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, v_j.data(), 1);
                
                // Add contribution from first vector
                Complex coef = Complex(evecs[idx * m], 0.0);
                for (int i = 0; i < N; i++) {
                    eigenvectors[k][i] += coef * v_j[i];
                }
                
                // Rebuild BiCG vectors and accumulate
                for (int j = 1; j < m; j++) {
                    ComplexVector v_next(N);
                    ComplexVector Hv(N);
                    
                    // Apply H to v_j
                    H(v_j.data(), Hv.data(), N);
                    
                    // v_next = (Hv - alpha[j]*v_j - beta[j]*v_prev) / beta[j+1]
                    for (int i = 0; i < N; i++) {
                        v_next[i] = Hv[i] - alpha[j] * v_j[i];
                        if (j > 0) {
                            v_next[i] -= beta[j] * v_prev[i];
                        }
                    }
                    
                    // Normalize v_next using beta[j+1]
                    if (std::abs(beta[j+1]) > 1e-10) {
                        Complex scale_next = Complex(1.0/std::sqrt(std::abs(beta[j+1])), 0.0);
                        cblas_zscal(N, &scale_next, v_next.data(), 1);
                    } else {
                        // Fallback normalization
                        norm = cblas_dznrm2(N, v_next.data(), 1);
                        if (norm > 1e-10) {
                            scale = Complex(1.0/norm, 0.0);
                            cblas_zscal(N, &scale, v_next.data(), 1);
                        }
                    }
                    
                    // Add contribution to eigenvector
                    coef = Complex(evecs[idx * m + j], 0.0);
                    for (int i = 0; i < N; i++) {
                        eigenvectors[k][i] += coef * v_next[i];
                    }
                    
                    // Update for next iteration
                    v_prev = v_j;
                    v_j = v_next;
                }
                
                // Normalize final eigenvector
                norm = cblas_dznrm2(N, eigenvectors[k].data(), 1);
                scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, eigenvectors[k].data(), 1);
            }
        } else {
            std::cerr << "Warning: Failed to compute eigenvectors. LAPACKE_dstev returned " << info << std::endl;
        }
    }
    
    // Clean up temporary directory if created
    if (!dir.empty()) {
        std::string cmd = "rm -rf " + temp_dir;
        system(cmd.c_str());
    }
}

// Function with same interface as lanczos.cpp
void cg_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N, 
    int max_iter, 
    int exct, 
    double tol, 
    std::vector<double>& eigenvalues, 
    std::string dir = "",
    bool compute_eigenvectors = false
) {
    std::cout << "CG Diagonalization: Starting with max_iter = " << max_iter 
              << ", exct = " << exct << std::endl;
    
    // Choose the appropriate CG variant based on parameters
    if (exct == 1) {
        // Find single lowest eigenvalue
        double eigenvalue;
        ComplexVector eigenvector(N);
        
        cg_lowest_eigenvalue(H, N, max_iter, tol, eigenvalue, eigenvector);
        
        eigenvalues.clear();
        eigenvalues.push_back(eigenvalue);
        
        // Save eigenvector to file if requested
        if (compute_eigenvectors && !dir.empty()) {
            std::string evec_dir = dir + "/cg_eigenvectors";
            std::string cmd = "mkdir -p " + evec_dir;
            system(cmd.c_str());
            
            std::string evec_file = evec_dir + "/eigenvector_0.bin";
            std::ofstream outfile(evec_file, std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<char*>(eigenvector.data()), N * sizeof(Complex));
                outfile.close();
                std::cout << "Saved eigenvector to " << evec_file << std::endl;
            } else {
                std::cerr << "Error: Could not save eigenvector to " << evec_file << std::endl;
            }
            
            // Save eigenvalues to a file
            std::string eval_file = evec_dir + "/eigenvalues.bin";
            std::ofstream eval_outfile(eval_file, std::ios::binary);
            if (eval_outfile) {
                size_t n_evals = eigenvalues.size();
                eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
                eval_outfile.close();
                std::cout << "Saved eigenvalues to " << eval_file << std::endl;
            }
        }
    } 
    else if (exct <= 10) {
        // Use CG with deflation for a small number of eigenvalues
        std::vector<ComplexVector> eigenvectors;
        
        cg_multiple_eigenvalues(H, N, max_iter, exct, tol, eigenvalues, eigenvectors);
        
        // Save eigenvectors to files if requested
        if (compute_eigenvectors && !dir.empty()) {
            std::string evec_dir = dir + "/cg_eigenvectors";
            std::string cmd = "mkdir -p " + evec_dir;
            system(cmd.c_str());
            
            for (size_t i = 0; i < eigenvectors.size(); i++) {
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream outfile(evec_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(eigenvectors[i].data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
            
            // Save eigenvalues to a file
            std::string eval_file = evec_dir + "/eigenvalues.bin";
            std::ofstream eval_outfile(eval_file, std::ios::binary);
            if (eval_outfile) {
                size_t n_evals = eigenvalues.size();
                eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
                eval_outfile.close();
                std::cout << "Saved eigenvalues to " << eval_file << std::endl;
            }
        }
    } 
    else if (exct <= 100) {
        // Use block CG for a moderate number of eigenvalues
        std::vector<ComplexVector> eigenvectors;
        int block_size = std::min(exct, 20);  // Limit block size for stability
        
        block_cg(H, N, max_iter, block_size, tol, eigenvalues, eigenvectors, dir);
        
        // Limit the number of eigenvalues returned
        if (eigenvalues.size() > exct) {
            eigenvalues.resize(exct);
        }
        
        // Save eigenvectors to files if requested
        if (compute_eigenvectors && !dir.empty()) {
            std::string evec_dir = dir + "/cg_eigenvectors";
            std::string cmd = "mkdir -p " + evec_dir;
            system(cmd.c_str());
            
            int num_evecs = std::min(exct, (int)eigenvectors.size());
            for (int i = 0; i < num_evecs; i++) {
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream outfile(evec_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(eigenvectors[i].data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
            
            // Save eigenvalues to a file
            std::string eval_file = evec_dir + "/eigenvalues.bin";
            std::ofstream eval_outfile(eval_file, std::ios::binary);
            if (eval_outfile) {
                size_t n_evals = eigenvalues.size();
                eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
                eval_outfile.close();
                std::cout << "Saved eigenvalues to " << eval_file << std::endl;
            }
        }
    } 
    else {
        // Use Davidson method for a large number of eigenvalues
        std::vector<ComplexVector> eigenvectors;
        int max_subspace = std::min(exct * 2, N);  // Limit subspace size
        
        davidson_method(H, N, max_iter, max_subspace, exct, tol, eigenvalues, eigenvectors, dir);
        
        // Save eigenvectors to files if requested
        if (compute_eigenvectors && !dir.empty()) {
            std::string evec_dir = dir + "/cg_eigenvectors";
            std::string cmd = "mkdir -p " + evec_dir;
            system(cmd.c_str());
            
            int num_evecs = std::min(exct, (int)eigenvectors.size());
            for (int i = 0; i < num_evecs; i++) {
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
                std::ofstream outfile(evec_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(eigenvectors[i].data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
            
            // Save eigenvalues to a file
            std::string eval_file = evec_dir + "/eigenvalues.bin";
            std::ofstream eval_outfile(eval_file, std::ios::binary);
            if (eval_outfile) {
                size_t n_evals = eigenvalues.size();
                eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
                eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
                eval_outfile.close();
                std::cout << "Saved eigenvalues to " << eval_file << std::endl;
            }
        }
    }
}

// Locally Optimized Block Preconditioned Conjugate Gradient (LOBPCG) method
void lobpcg_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    int block_size,                                        // Number of eigenpairs to compute
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::function<void(const Complex*, Complex*, int)> M = nullptr  // Preconditioner (optional)
) {
    // Initialize block of vectors
    eigenvectors.clear();
    eigenvectors.resize(block_size, ComplexVector(N));
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize random vectors and orthonormalize them
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < N; i++) {
            eigenvectors[j][i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors
        for (int k = 0; k < j; k++) {
            Complex projection;
            cblas_zdotc_sub(N, eigenvectors[k].data(), 1, eigenvectors[j].data(), 1, &projection);
            
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, eigenvectors[k].data(), 1, eigenvectors[j].data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, eigenvectors[j].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, eigenvectors[j].data(), 1);
    }
    
    // Initialize workspace
    std::vector<ComplexVector> X = eigenvectors;  // Current approximations
    std::vector<ComplexVector> R(block_size, ComplexVector(N));  // Residuals
    std::vector<ComplexVector> P(block_size, ComplexVector(N));  // Previous direction
    std::vector<ComplexVector> HX(block_size, ComplexVector(N)); // H*X
    std::vector<ComplexVector> MR(block_size, ComplexVector(N)); // M^-1*R (preconditioned residuals)
    
    // Apply H to initial block
    for (int j = 0; j < block_size; j++) {
        H(X[j].data(), HX[j].data(), N);
    }
    
    // Initialize eigenvalues
    eigenvalues.resize(block_size);
    for (int j = 0; j < block_size; j++) {
        Complex rayleigh;
        cblas_zdotc_sub(N, X[j].data(), 1, HX[j].data(), 1, &rayleigh);
        eigenvalues[j] = std::real(rayleigh);
    }
    
    // Initialize residuals
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < N; i++) {
            R[j][i] = HX[j][i] - eigenvalues[j] * X[j][i];
        }
    }
    
    // Apply preconditioner if provided
    if (M) {
        for (int j = 0; j < block_size; j++) {
            M(R[j].data(), MR[j].data(), N);
        }
    } else {
        // Without preconditioner, MR = R
        MR = R;
    }
    
    // Initialize P = MR for first iteration
    P = MR;
    
    std::cout << "LOBPCG: Starting iterations with block size " << block_size << "..." << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Construct 3*blocksize x 3*blocksize projected problem matrices
        int subspace_size = 3 * block_size;
        if (iter == 0) {
            // No P yet on first iteration
            subspace_size = 2 * block_size;
        }
        
        // Projected operator matrices
        Eigen::MatrixXcd A_proj(subspace_size, subspace_size);
        Eigen::MatrixXcd B_proj(subspace_size, subspace_size);  // For generalized EVP with preconditioner
        A_proj.setZero();
        B_proj.setZero();
        
        // Apply H to P if not first iteration
        std::vector<ComplexVector> HP;
        if (iter > 0) {
            HP.resize(block_size, ComplexVector(N));
            for (int j = 0; j < block_size; j++) {
                H(P[j].data(), HP[j].data(), N);
            }
        }
        
        // Fill the projected operator matrices
        // [X, MR, P]^H * H * [X, MR, P]
        for (int i = 0; i < block_size; i++) {
            // X block
            for (int j = 0; j < block_size; j++) {
                Complex val;
                cblas_zdotc_sub(N, X[i].data(), 1, HX[j].data(), 1, &val);
                A_proj(i, j) = val;
                
                // B_proj for X block is just identity
                cblas_zdotc_sub(N, X[i].data(), 1, X[j].data(), 1, &val);
                B_proj(i, j) = val;
            }
            
            // X-MR and MR-X blocks
            for (int j = 0; j < block_size; j++) {
                Complex val;
                cblas_zdotc_sub(N, X[i].data(), 1, HX[j].data(), 1, &val);
                A_proj(i, block_size + j) = val;
                A_proj(block_size + j, i) = std::conj(val);
                
                // B_proj for X-MR block
                cblas_zdotc_sub(N, X[i].data(), 1, MR[j].data(), 1, &val);
                B_proj(i, block_size + j) = val;
                B_proj(block_size + j, i) = std::conj(val);
            }
            
            // MR-MR block
            for (int j = 0; j < block_size; j++) {
                Complex val;
                cblas_zdotc_sub(N, MR[i].data(), 1, HX[j].data(), 1, &val);
                A_proj(block_size + i, block_size + j) = val;
                
                // B_proj for MR-MR block
                cblas_zdotc_sub(N, MR[i].data(), 1, MR[j].data(), 1, &val);
                B_proj(block_size + i, block_size + j) = val;
            }
            
            if (iter > 0) {
                // X-P and P-X blocks
                for (int j = 0; j < block_size; j++) {
                    Complex val;
                    cblas_zdotc_sub(N, X[i].data(), 1, HP[j].data(), 1, &val);
                    A_proj(i, 2*block_size + j) = val;
                    A_proj(2*block_size + j, i) = std::conj(val);
                    
                    // B_proj for X-P block
                    cblas_zdotc_sub(N, X[i].data(), 1, P[j].data(), 1, &val);
                    B_proj(i, 2*block_size + j) = val;
                    B_proj(2*block_size + j, i) = std::conj(val);
                }
                
                // MR-P and P-MR blocks
                for (int j = 0; j < block_size; j++) {
                    Complex val;
                    cblas_zdotc_sub(N, MR[i].data(), 1, HP[j].data(), 1, &val);
                    A_proj(block_size + i, 2*block_size + j) = val;
                    A_proj(2*block_size + j, block_size + i) = std::conj(val);
                    
                    // B_proj for MR-P block
                    cblas_zdotc_sub(N, MR[i].data(), 1, P[j].data(), 1, &val);
                    B_proj(block_size + i, 2*block_size + j) = val;
                    B_proj(2*block_size + j, block_size + i) = std::conj(val);
                }
                
                // P-P block
                for (int j = 0; j < block_size; j++) {
                    Complex val;
                    cblas_zdotc_sub(N, P[i].data(), 1, HP[j].data(), 1, &val);
                    A_proj(2*block_size + i, 2*block_size + j) = val;
                    
                    // B_proj for P-P block
                    cblas_zdotc_sub(N, P[i].data(), 1, P[j].data(), 1, &val);
                    B_proj(2*block_size + i, 2*block_size + j) = val;
                }
            }
        }
        
        // Ensure matrices are Hermitian
        A_proj = (A_proj + A_proj.adjoint()) / 2.0;
        B_proj = (B_proj + B_proj.adjoint()) / 2.0;
        
        // Solve the generalized eigenvalue problem A_proj * c = lambda * B_proj * c
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> solver(A_proj, B_proj);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Eigenvalue solver failed!" << std::endl;
            break;
        }
        
        Eigen::VectorXd proj_eigenvals = solver.eigenvalues();
        Eigen::MatrixXcd proj_eigenvecs = solver.eigenvectors();
        
        // Sort eigenvalues and eigenvectors
        std::vector<std::pair<double, int>> eig_pairs(subspace_size);
        for (int i = 0; i < subspace_size; i++) {
            eig_pairs[i] = {proj_eigenvals(i), i};
        }
        std::sort(eig_pairs.begin(), eig_pairs.end());
        
        // Update eigenvalues
        std::vector<double> new_eigenvalues(block_size);
        for (int j = 0; j < block_size; j++) {
            new_eigenvalues[j] = eig_pairs[j].first;
        }
        
        // Compute new X, P, and R
        std::vector<ComplexVector> new_X(block_size, ComplexVector(N, Complex(0.0, 0.0)));
        std::vector<ComplexVector> new_HX(block_size, ComplexVector(N, Complex(0.0, 0.0)));
        
        for (int j = 0; j < block_size; j++) {
            int col_idx = eig_pairs[j].second;
            
            // Construct new X as linear combination
            for (int i = 0; i < block_size; i++) {
                Complex coef = proj_eigenvecs(i, col_idx);
                cblas_zaxpy(N, &coef, X[i].data(), 1, new_X[j].data(), 1);
                cblas_zaxpy(N, &coef, HX[i].data(), 1, new_HX[j].data(), 1);
            }
            
            for (int i = 0; i < block_size; i++) {
                Complex coef = proj_eigenvecs(block_size + i, col_idx);
                cblas_zaxpy(N, &coef, MR[i].data(), 1, new_X[j].data(), 1);
                
                // For new_HX, need to apply H to MR contribution
                ComplexVector HMR(N);
                H(MR[i].data(), HMR.data(), N);
                cblas_zaxpy(N, &coef, HMR.data(), 1, new_HX[j].data(), 1);
            }
            
            if (iter > 0) {
                for (int i = 0; i < block_size; i++) {
                    Complex coef = proj_eigenvecs(2*block_size + i, col_idx);
                    cblas_zaxpy(N, &coef, P[i].data(), 1, new_X[j].data(), 1);
                    cblas_zaxpy(N, &coef, HP[i].data(), 1, new_HX[j].data(), 1);
                }
            }
            
            // Normalize new vector
            double norm = cblas_dznrm2(N, new_X[j].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, new_X[j].data(), 1);
            cblas_zscal(N, &scale, new_HX[j].data(), 1);
        }
        
        // Save previous X for P update
        std::vector<ComplexVector> prev_X = X;
        
        // Update X and HX
        X = new_X;
        HX = new_HX;
        
        // Re-orthogonalize X block to ensure numerical stability
        for (int j = 0; j < block_size; j++) {
            // Orthogonalize against previous vectors
            for (int k = 0; k < j; k++) {
                Complex projection;
                cblas_zdotc_sub(N, X[k].data(), 1, X[j].data(), 1, &projection);
                
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, X[k].data(), 1, X[j].data(), 1);
                
                // Update HX accordingly
                cblas_zaxpy(N, &neg_projection, HX[k].data(), 1, HX[j].data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, X[j].data(), 1);
            if (norm > 1e-10) {
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, X[j].data(), 1);
                cblas_zscal(N, &scale, HX[j].data(), 1);
            }
        }
        
        // Update eigenvalues using Rayleigh quotient
        for (int j = 0; j < block_size; j++) {
            Complex rayleigh;
            cblas_zdotc_sub(N, X[j].data(), 1, HX[j].data(), 1, &rayleigh);
            eigenvalues[j] = std::real(rayleigh);
        }
        
        // Compute new residuals
        double max_res_norm = 0.0;
        for (int j = 0; j < block_size; j++) {
            for (int i = 0; i < N; i++) {
                R[j][i] = HX[j][i] - eigenvalues[j] * X[j][i];
            }
            
            double res_norm = cblas_dznrm2(N, R[j].data(), 1);
            max_res_norm = std::max(max_res_norm, res_norm);
        }
        
        // Output progress
        if (iter % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << iter << ": max residual = " << max_res_norm << std::endl;
            std::cout << "Eigenvalues: ";
            for (int j = 0; j < std::min(5, block_size); j++) {
                std::cout << eigenvalues[j] << " ";
            }
            if (block_size > 5) std::cout << "...";
            std::cout << std::endl;
        }
        
        // Check convergence
        if (max_res_norm < tol) {
            std::cout << "LOBPCG converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
        
        // Apply preconditioner to residuals
        if (M) {
            for (int j = 0; j < block_size; j++) {
                M(R[j].data(), MR[j].data(), N);
            }
        } else {
            // Without preconditioner, MR = R
            MR = R;
        }
        
        // Update P for next iteration
        P = MR;
        
        // Orthogonalize P against X
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                Complex projection;
                cblas_zdotc_sub(N, X[k].data(), 1, P[j].data(), 1, &projection);
                
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, X[k].data(), 1, P[j].data(), 1);
            }
        }
    }
    
    // Update eigenvectors with the final X
    eigenvectors = X;
}

// Wrapper function to match the interface of other diagonalization methods
void lobpcg_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    int num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::string dir = "",                                  // Directory for temporary files
    bool compute_eigenvectors = false                      // Whether to compute eigenvectors
) {
    std::cout << "LOBPCG: Starting with max_iter = " << max_iter 
              << ", num_eigenvalues = " << num_eigenvalues << std::endl;
    
    std::vector<ComplexVector> eigenvectors;
    
    // Call LOBPCG implementation
    lobpcg_method(H, N, max_iter, num_eigenvalues, tol, eigenvalues, eigenvectors);
    
    // Save eigenvectors to files if requested
    if (compute_eigenvectors && !dir.empty()) {
        std::string evec_dir = dir + "/lobpcg_eigenvectors";
        std::string cmd = "mkdir -p " + evec_dir;
        system(cmd.c_str());
        
        for (size_t i = 0; i < eigenvectors.size(); i++) {
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".bin";
            std::ofstream outfile(evec_file, std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<char*>(eigenvectors[i].data()), N * sizeof(Complex));
                outfile.close();
            }
        }
        
        // Save eigenvalues to a file
        std::string eval_file = evec_dir + "/eigenvalues.bin";
        std::ofstream eval_outfile(eval_file, std::ios::binary);
        if (eval_outfile) {
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved eigenvalues to " << eval_file << std::endl;
        }
    }
}

#endif  // CG_H