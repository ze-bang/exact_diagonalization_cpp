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
#include <algorithm>
#include <cstdlib>
#include "blas_lapack_wrapper.h"
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
);

// CG with deflation for finding multiple eigenvalues
void cg_multiple_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations per eigenvalue
    int num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors               // Output eigenvectors
);

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
);


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
);

// BiCG method for finding eigenvalues from CalcSpectrumByBiCG.c
void bicg_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir = ""                                   // Directory for temporary files
);

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
);

void lobpcg_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int max_iter,                                          // Maximum iterations
    int num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                           // Tolerance for convergence
    std::vector<double>& eigenvalues,                     // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,             // Output eigenvectors
    std::string dir = "",                                 // Directory for temporary files
    bool use_preconditioning = false                      // Whether to use preconditioning
);

// Function with same interface as cg_diagonalization
void lobpcg_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N, 
    int max_iter, 
    int exct, 
    double tol, 
    std::vector<double>& eigenvalues, 
    std::string dir = "",
    bool compute_eigenvectors = false,
    bool use_preconditioning = false
);

#endif  // CG_H