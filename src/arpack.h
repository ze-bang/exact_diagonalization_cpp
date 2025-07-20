#ifndef ARPACK_H
#define ARPACK_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <mkl.h>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <map>
#include <string>
#include <cstring>

// Type definitions
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// ARPACK-ng external function declarations for complex Hermitian problems
extern "C" {
    void znaupd_(int* ido, char* bmat, int* n, char* which, int* nev,
                 double* tol, std::complex<double>* resid, int* ncv,
                 std::complex<double>* v, int* ldv, int* iparam, int* ipntr,
                 std::complex<double>* workd, std::complex<double>* workl,
                 int* lworkl, double* rwork, int* info);
    
    void zneupd_(int* rvec, char* howmny, int* select, std::complex<double>* d,
                 std::complex<double>* z, int* ldz, std::complex<double>* sigma,
                 std::complex<double>* workev, char* bmat, int* n, char* which,
                 int* nev, double* tol, std::complex<double>* resid, int* ncv,
                 std::complex<double>* v, int* ldv, int* iparam, int* ipntr,
                 std::complex<double>* workd, std::complex<double>* workl,
                 int* lworkl, double* rwork, int* info);
}

// Helper function to save eigenvector to file
void save_arpack_eigenvector(const std::string& filename, const ComplexVector& vec) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    size_t size = vec.size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(Complex));
    outfile.close();
}

// Standard ARPACK solver for Hermitian matrices
void arpack_standard(std::function<void(const Complex*, Complex*, int)> H, int N, int nev, 
                    const std::string& which, double tol, std::vector<double>& eigenvalues,
                    std::string dir = "", bool compute_eigenvectors = false) {
    
    std::cout << "Starting ARPACK Standard Mode for Hermitian matrix" << std::endl;
    std::cout << "Matrix dimension: " << N << std::endl;
    std::cout << "Number of eigenvalues requested: " << nev << std::endl;
    std::cout << "Which eigenvalues: " << which << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // Create output directory
    std::string evec_dir = dir + "/eigenvectors";
    if (compute_eigenvectors && !dir.empty()) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // ARPACK parameters
    int ido = 0;
    char bmat = 'I';  // Standard eigenvalue problem
    int ncv = std::min(2 * nev + 1, N);  // Number of Arnoldi vectors
    
    // Allocate workspace
    int ldv = N;
    std::vector<Complex> v(ldv * ncv);
    std::vector<Complex> resid(N);
    std::vector<Complex> workd(3 * N);
    int lworkl = 3 * ncv * ncv + 5 * ncv;
    std::vector<Complex> workl(lworkl);
    std::vector<double> rwork(ncv);
    
    // ARPACK control parameters
    std::vector<int> iparam(11, 0);
    iparam[0] = 1;      // Shift strategy (1 = exact shifts)
    iparam[2] = 10*N;   // Maximum iterations
    iparam[3] = 1;      // Block size (must be 1 for complex)
    iparam[6] = 1;      // Mode 1: standard eigenvalue problem
    
    std::vector<int> ipntr(14);
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N; i++) {
        resid[i] = Complex(dist(gen), dist(gen));
    }
    
    int info = 1;  // Use random initial vector
    char which_str[3];
    std::strncpy(which_str, which.c_str(), 2);
    which_str[2] = '\0';
    
    // Main ARPACK reverse communication loop
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;
    
    do {
        znaupd_(&ido, &bmat, &N, which_str, &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
                workl.data(), &lworkl, rwork.data(), &info);
        
        if (ido == -1 || ido == 1) {
            // Perform matrix-vector multiplication: y = H*x
            H(workd.data() + ipntr[0] - 1, workd.data() + ipntr[1] - 1, N);
            iter++;
            
            if (iter % 100 == 0) {
                std::cout << "ARPACK iteration " << iter << std::endl;
            }
        }
    } while (ido != 99);
    
    auto mv_time = std::chrono::high_resolution_clock::now();
    
    if (info < 0) {
        std::cerr << "Error in ARPACK znaupd: info = " << info << std::endl;
        return;
    }
    
    std::cout << "ARPACK converged after " << iparam[8] << " matrix-vector multiplications" << std::endl;
    std::cout << "Number of converged Ritz values: " << iparam[4] << std::endl;
    
    // Extract eigenvalues and eigenvectors
    int rvec = compute_eigenvectors ? 1 : 0;
    char howmny = 'A';  // Compute all NEV eigenvectors
    std::vector<int> select(ncv);
    std::vector<Complex> d(nev + 1);
    std::vector<Complex> z(N * nev);
    std::vector<Complex> workev(2 * ncv);
    Complex sigma = Complex(0.0, 0.0);
    
    zneupd_(&rvec, &howmny, select.data(), d.data(), z.data(), &N, &sigma,
            workev.data(), &bmat, &N, which_str, &nev, &tol, resid.data(),
            &ncv, v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
            workl.data(), &lworkl, rwork.data(), &info);
    
    if (info != 0) {
        std::cerr << "Error in ARPACK zneupd: info = " << info << std::endl;
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Extract real parts of eigenvalues (they should be real for Hermitian matrices)
    eigenvalues.clear();
    eigenvalues.reserve(nev);
    for (int i = 0; i < nev; i++) {
        eigenvalues.push_back(d[i].real());
    }
    
    // Sort eigenvalues in ascending order
    std::vector<size_t> idx(nev);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&eigenvalues](size_t i, size_t j) {
        return eigenvalues[i] < eigenvalues[j];
    });
    
    std::vector<double> sorted_eigenvalues(nev);
    for (int i = 0; i < nev; i++) {
        sorted_eigenvalues[i] = eigenvalues[idx[i]];
    }
    eigenvalues = sorted_eigenvalues;
    
    // Save eigenvalues
    std::string eval_file = evec_dir + "/eigenvalues.dat";
    std::ofstream eval_outfile(eval_file, std::ios::binary);
    if (eval_outfile) {
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<const char*>(eigenvalues.data()), n_evals * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_evals << " eigenvalues to " << eval_file << std::endl;
    }
    
    // Save eigenvalues in text format
    std::string eval_text_file = evec_dir + "/eigenvalues.txt";
    std::ofstream eval_text_outfile(eval_text_file);
    if (eval_text_outfile) {
        eval_text_outfile << std::scientific << std::setprecision(15);
        eval_text_outfile << eigenvalues.size() << std::endl;
        for (const auto& eval : eigenvalues) {
            eval_text_outfile << eval << std::endl;
        }
        eval_text_outfile.close();
    }
    
    // Save eigenvectors if requested
    if (compute_eigenvectors) {
        std::cout << "Saving eigenvectors..." << std::endl;
        
        #pragma omp parallel for
        for (int i = 0; i < nev; i++) {
            ComplexVector evec(N);
            // Copy eigenvector in the correct sorted order
            std::copy(z.data() + idx[i] * N, z.data() + (idx[i] + 1) * N, evec.begin());
            
            // Normalize eigenvector
            double norm = cblas_dznrm2(N, evec.data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, evec.data(), 1);
            
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            save_arpack_eigenvector(evec_file, evec);
        }
        
        std::cout << "Saved " << nev << " eigenvectors to " << evec_dir << std::endl;
    }
    
    // Print timing information
    auto mv_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mv_time - start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nARPACK Performance Summary:" << std::endl;
    std::cout << "Matrix-vector multiplication time: " << mv_duration.count() << " ms" << std::endl;
    std::cout << "Total computation time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Average time per MV operation: " 
              << static_cast<double>(mv_duration.count()) / iparam[8] << " ms" << std::endl;
}

// Shift-Invert ARPACK solver
void arpack_shift_invert(std::function<void(const Complex*, Complex*, int)> H, int N, int nev,
                        double sigma, double tol, std::vector<double>& eigenvalues,
                        std::string dir = "", bool compute_eigenvectors = false) {
    
    std::cout << "Starting ARPACK Shift-Invert Mode for eigenvalues near sigma = " << sigma << std::endl;
    std::cout << "Matrix dimension: " << N << std::endl;
    std::cout << "Number of eigenvalues requested: " << nev << std::endl;
    
    // Create output directory
    std::string evec_dir = dir + "/eigenvectors";
    if (compute_eigenvectors && !dir.empty()) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // ARPACK parameters
    int ido = 0;
    char bmat = 'I';  // Standard eigenvalue problem
    char which_str[3] = "LM";  // Find largest magnitude eigenvalues of (H-sigma*I)^{-1}
    int ncv = std::min(3 * nev, N);
    
    // Allocate workspace
    int ldv = N;
    std::vector<Complex> v(ldv * ncv);
    std::vector<Complex> resid(N);
    std::vector<Complex> workd(3 * N);
    int lworkl = 3 * ncv * ncv + 5 * ncv;
    std::vector<Complex> workl(lworkl);
    std::vector<double> rwork(ncv);
    
    // ARPACK control parameters
    std::vector<int> iparam(11, 0);
    iparam[0] = 1;      // Shift strategy
    iparam[2] = 10*N;   // Maximum iterations
    iparam[3] = 1;      // Block size
    iparam[6] = 3;      // Mode 3: shift-invert mode
    
    std::vector<int> ipntr(14);
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N; i++) {
        resid[i] = Complex(dist(gen), dist(gen));
    }
    
    int info = 1;
    
    // CG solver parameters for shift-invert
    const int max_cg_iter = 100;
    const double cg_tol = tol * 0.01;
    
    // Workspace for CG solver
    ComplexVector r(N), p(N), Ap(N);
    
    // Define shifted operator
    auto H_shifted = [&H, sigma, N](const Complex* v, Complex* result, int size) {
        H(v, result, size);
        Complex shift = Complex(-sigma, 0.0);
        cblas_zaxpy(size, &shift, v, 1, result, 1);
    };
    
    // Statistics
    std::vector<int> cg_iterations;
    int total_cg_iter = 0;
    
    // Main ARPACK reverse communication loop
    auto start_time = std::chrono::high_resolution_clock::now();
    int mv_count = 0;
    
    do {
        znaupd_(&ido, &bmat, &N, which_str, &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
                workl.data(), &lworkl, rwork.data(), &info);
        
        if (ido == -1 || ido == 1) {
            // Solve (H - sigma*I) * y = x using CG
            Complex* x = workd.data() + ipntr[0] - 1;
            Complex* y = workd.data() + ipntr[1] - 1;
            
            // Initialize solution
            std::fill(y, y + N, Complex(0.0, 0.0));
            
            // Compute initial residual: r = x - (H-sigma*I)*y = x
            std::copy(x, x + N, r.begin());
            
            // Initial search direction
            std::copy(r.begin(), r.end(), p.begin());
            
            double r_norm = cblas_dznrm2(N, r.data(), 1);
            int cg_iter = 0;
            
            while (r_norm > cg_tol && cg_iter < max_cg_iter) {
                // Apply operator: Ap = (H-sigma*I)*p
                H_shifted(p.data(), Ap.data(), N);
                
                // alpha = (r·r) / (p·Ap)
                Complex r_dot_r, p_dot_Ap;
                cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
                cblas_zdotc_sub(N, p.data(), 1, Ap.data(), 1, &p_dot_Ap);
                
                Complex alpha = r_dot_r / p_dot_Ap;
                
                // y = y + alpha*p
                cblas_zaxpy(N, &alpha, p.data(), 1, y, 1);
                
                // r = r - alpha*Ap
                Complex neg_alpha = -alpha;
                cblas_zaxpy(N, &neg_alpha, Ap.data(), 1, r.data(), 1);
                
                r_norm = cblas_dznrm2(N, r.data(), 1);
                
                // beta = (r_new·r_new) / (r_old·r_old)
                Complex r_dot_r_new;
                cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r_new);
                Complex beta = r_dot_r_new / r_dot_r;
                
                // p = r + beta*p
                cblas_zscal(N, &beta, p.data(), 1);
                Complex one = Complex(1.0, 0.0);
                cblas_zaxpy(N, &one, r.data(), 1, p.data(), 1);
                
                cg_iter++;
            }
            
            cg_iterations.push_back(cg_iter);
            total_cg_iter += cg_iter;
            mv_count++;
            
            if (mv_count % 10 == 0) {
                std::cout << "ARPACK iteration " << mv_count 
                          << ", CG iterations: " << cg_iter << std::endl;
            }
        }
    } while (ido != 99);
    
    if (info < 0) {
        std::cerr << "Error in ARPACK znaupd: info = " << info << std::endl;
        return;
    }
    
    std::cout << "ARPACK converged after " << mv_count << " linear solves" << std::endl;
    std::cout << "Total CG iterations: " << total_cg_iter << std::endl;
    std::cout << "Average CG iterations per solve: " 
              << static_cast<double>(total_cg_iter) / mv_count << std::endl;
    
    // Extract eigenvalues and eigenvectors
    int rvec = compute_eigenvectors ? 1 : 0;
    char howmny = 'A';
    std::vector<int> select(ncv);
    std::vector<Complex> d(nev + 1);
    std::vector<Complex> z(N * nev);
    std::vector<Complex> workev(2 * ncv);
    Complex sigma_complex = Complex(sigma, 0.0);
    
    zneupd_(&rvec, &howmny, select.data(), d.data(), z.data(), &N, &sigma_complex,
            workev.data(), &bmat, &N, which_str, &nev, &tol, resid.data(),
            &ncv, v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
            workl.data(), &lworkl, rwork.data(), &info);
    
    if (info != 0) {
        std::cerr << "Error in ARPACK zneupd: info = " << info << std::endl;
        return;
    }
    
    // Extract real eigenvalues
    eigenvalues.clear();
    eigenvalues.reserve(nev);
    for (int i = 0; i < nev; i++) {
        eigenvalues.push_back(d[i].real());
    }
    
    // Sort by distance from sigma
    std::vector<size_t> idx(nev);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&eigenvalues, sigma](size_t i, size_t j) {
        return std::abs(eigenvalues[i] - sigma) < std::abs(eigenvalues[j] - sigma);
    });
    
    std::vector<double> sorted_eigenvalues(nev);
    for (int i = 0; i < nev; i++) {
        sorted_eigenvalues[i] = eigenvalues[idx[i]];
    }
    eigenvalues = sorted_eigenvalues;
    
    // Save results
    std::string eval_file = evec_dir + "/eigenvalues.dat";
    std::ofstream eval_outfile(eval_file, std::ios::binary);
    if (eval_outfile) {
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<const char*>(eigenvalues.data()), n_evals * sizeof(double));
        eval_outfile.close();
    }
    
    if (compute_eigenvectors) {
        #pragma omp parallel for
        for (int i = 0; i < nev; i++) {
            ComplexVector evec(N);
            std::copy(z.data() + idx[i] * N, z.data() + (idx[i] + 1) * N, evec.begin());
            
            double norm = cblas_dznrm2(N, evec.data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, evec.data(), 1);
            
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            save_arpack_eigenvector(evec_file, evec);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nARPACK Shift-Invert completed in " << duration.count() << " ms" << std::endl;
}

// Spectrum slicing ARPACK solver
void arpack_spectrum_slicing(std::function<void(const Complex*, Complex*, int)> H, int N,
                           double lower_bound, double upper_bound, double tol,
                           std::vector<double>& eigenvalues, std::string dir = "",
                           bool compute_eigenvectors = false) {
    
    std::cout << "Starting ARPACK Spectrum Slicing for interval [" 
              << lower_bound << ", " << upper_bound << "]" << std::endl;
    
    // Estimate number of eigenvalues in the interval
    int estimated_count = static_cast<int>((upper_bound - lower_bound) / 
                                          (upper_bound - lower_bound) * N * 0.1);
    estimated_count = std::max(10, std::min(estimated_count, N/2));
    
    // Use multiple shifts to cover the interval
    int num_shifts = std::max(1, estimated_count / 20);
    std::vector<double> shifts;
    
    for (int i = 0; i < num_shifts; i++) {
        double shift = lower_bound + (upper_bound - lower_bound) * (i + 0.5) / num_shifts;
        shifts.push_back(shift);
    }
    
    std::cout << "Using " << num_shifts << " shift points to cover the interval" << std::endl;
    
    // Collect eigenvalues from all shifts
    std::map<double, int> eigenvalue_map;  // Use map to avoid duplicates
    
    for (size_t i = 0; i < shifts.size(); i++) {
        std::cout << "\nProcessing shift " << i+1 << "/" << shifts.size() 
                  << " at sigma = " << shifts[i] << std::endl;
        
        std::vector<double> local_eigenvalues;
        int local_nev = std::min(30, N/10);  // Find 30 eigenvalues near each shift
        
        arpack_shift_invert(H, N, local_nev, shifts[i], tol, local_eigenvalues, "", false);
        
        // Filter eigenvalues within the desired interval
        for (const auto& eval : local_eigenvalues) {
            if (eval >= lower_bound && eval <= upper_bound) {
                eigenvalue_map[eval]++;
            }
        }
    }
    
    // Extract unique eigenvalues
    eigenvalues.clear();
    for (const auto& [eval, count] : eigenvalue_map) {
        eigenvalues.push_back(eval);
    }
    
    std::cout << "\nFound " << eigenvalues.size() << " eigenvalues in the interval ["
              << lower_bound << ", " << upper_bound << "]" << std::endl;
    
    // If eigenvectors are needed, compute them using shift-invert at optimal shifts
    if (compute_eigenvectors && !eigenvalues.empty()) {
        std::string evec_dir = dir + "/eigenvectors";
        system(("mkdir -p " + evec_dir).c_str());
        
        // Group eigenvalues by proximity for efficient computation
        std::vector<std::pair<double, std::vector<int>>> groups;
        const double group_tol = (upper_bound - lower_bound) / num_shifts;
        
        for (size_t i = 0; i < eigenvalues.size(); i++) {
            bool added = false;
            for (auto& [center, indices] : groups) {
                if (std::abs(eigenvalues[i] - center) < group_tol) {
                    indices.push_back(i);
                    added = true;
                    break;
                }
            }
            if (!added) {
                groups.push_back({eigenvalues[i], {static_cast<int>(i)}});
            }
        }
        
        std::cout << "Computing eigenvectors in " << groups.size() << " groups" << std::endl;
        
        // Compute eigenvectors for each group
        for (const auto& [center, indices] : groups) {
            std::vector<double> group_evals;
            arpack_shift_invert(H, N, indices.size(), center, tol * 0.1, 
                              group_evals, dir, true);
        }
    }
}

// Main ARPACK interface function with automatic method selection
void arpack_diagonalize(std::function<void(const Complex*, Complex*, int)> H, int N,
                       int num_eigenvalues, const std::string& target,
                       double tol, std::vector<double>& eigenvalues,
                       std::string dir = "", bool compute_eigenvectors = false,
                       double sigma = 0.0) {
    
    std::cout << "\n=== ARPACK Diagonalization ===" << std::endl;
    std::cout << "Matrix dimension: " << N << std::endl;
    std::cout << "Number of eigenvalues requested: " << num_eigenvalues << std::endl;
    std::cout << "Target: " << target << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // Validate input
    if (num_eigenvalues > N - 1) {
        std::cerr << "Warning: Cannot compute " << num_eigenvalues 
                  << " eigenvalues for matrix of size " << N << std::endl;
        num_eigenvalues = N - 1;
    }
    
    // Select appropriate method based on target
    if (target == "LA" || target == "SA" || target == "BE") {
        // Use standard mode for algebraically largest/smallest or both ends
        arpack_standard(H, N, num_eigenvalues, target, tol, eigenvalues, dir, compute_eigenvectors);
    }
    else if (target == "LM" || target == "SM") {
        // Use standard mode for largest/smallest magnitude
        arpack_standard(H, N, num_eigenvalues, target, tol, eigenvalues, dir, compute_eigenvectors);
    }
    else if (target.substr(0, 2) == "SI") {
        // Shift-invert mode for eigenvalues near a specific value
        arpack_shift_invert(H, N, num_eigenvalues, sigma, tol, eigenvalues, dir, compute_eigenvectors);
    }
    else if (target.substr(0, 2) == "LI" || target.substr(0, 2) == "SR") {
        // Eigenvalues with largest/smallest real part near shift
        std::cout << "Using shift-invert mode with sigma = " << sigma << std::endl;
        arpack_shift_invert(H, N, num_eigenvalues, sigma, tol, eigenvalues, dir, compute_eigenvectors);
    }
    else {
        std::cerr << "Error: Unknown target type '" << target << "'" << std::endl;
        std::cerr << "Valid options: LA, SA, LM, SM, BE, SI, LI, SR" << std::endl;
        return;
    }
    
    // Print summary
    std::cout << "\nARPACK Diagonalization Summary:" << std::endl;
    std::cout << "Eigenvalues found: " << eigenvalues.size() << std::endl;
    if (!eigenvalues.empty()) {
        std::cout << "Eigenvalue range: [" << *std::min_element(eigenvalues.begin(), eigenvalues.end())
                  << ", " << *std::max_element(eigenvalues.begin(), eigenvalues.end()) << "]" << std::endl;
    }
}

// Convenience wrapper for common use cases
void arpack_lowest_eigenvalues(std::function<void(const Complex*, Complex*, int)> H, int N,
                              int num_eigenvalues, double tol, std::vector<double>& eigenvalues,
                              std::string dir = "", bool compute_eigenvectors = false) {
    arpack_diagonalize(H, N, num_eigenvalues, "SA", tol, eigenvalues, dir, compute_eigenvectors);
}

void arpack_highest_eigenvalues(std::function<void(const Complex*, Complex*, int)> H, int N,
                               int num_eigenvalues, double tol, std::vector<double>& eigenvalues,
                               std::string dir = "", bool compute_eigenvectors = false) {
    arpack_diagonalize(H, N, num_eigenvalues, "LA", tol, eigenvalues, dir, compute_eigenvectors);
}

void arpack_eigenvalues_near(std::function<void(const Complex*, Complex*, int)> H, int N,
                            int num_eigenvalues, double target_value, double tol,
                            std::vector<double>& eigenvalues, std::string dir = "",
                            bool compute_eigenvectors = false) {
    arpack_diagonalize(H, N, num_eigenvalues, "SI", tol, eigenvalues, dir, 
                      compute_eigenvectors, target_value);
}

#endif // ARPACK_H