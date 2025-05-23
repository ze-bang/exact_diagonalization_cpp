#ifndef FINITE_TEMPERATURE_LANCZOS_H
#define FINITE_TEMPERATURE_LANCZOS_H

#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <mkl.h>
#include "lanczos.h"
#include "CG.h"
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <chrono>
#include <numeric>

// Type definitions
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Structure to hold thermal expectation values
struct ThermalExpectation {
    double partition_function;
    double energy;
    double heat_capacity;
    double entropy;
    double free_energy;
    std::vector<double> observable_values;
};

// Finite Temperature Lanczos Method (FTLM)
// Based on Jaklic & Prelovsek method with optimizations
class FiniteTemperatureLanczos {
private:
    std::function<void(const Complex*, Complex*, int)> H;
    int N;
    int num_random_vectors;
    int lanczos_steps;
    double tol;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
    
    // Cache for eigenvalues to avoid recomputation
    std::vector<std::vector<double>> cached_eigenvalues;
    std::vector<std::vector<double>> cached_z_coeffs;
    
public:
    FiniteTemperatureLanczos(
        std::function<void(const Complex*, Complex*, int)> H_op,
        int hilbert_dim,
        int num_random = 100,
        int lanczos_iter = 100,
        double tolerance = 1e-10
    ) : H(H_op), N(hilbert_dim), num_random_vectors(num_random), 
        lanczos_steps(lanczos_iter), tol(tolerance), 
        gen(std::random_device{}()), dist(-1.0, 1.0) {
        
        cached_eigenvalues.resize(num_random_vectors);
        cached_z_coeffs.resize(num_random_vectors);
    }
    
    // Main FTLM algorithm
    ThermalExpectation compute_thermal_expectation(
        double beta,
        std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {},
        bool use_importance_sampling = true
    ) {
        std::cout << "FTLM: Computing thermal expectation at β = " << beta << std::endl;
        std::cout << "Using " << num_random_vectors << " random vectors" << std::endl;
        
        ThermalExpectation result;
        result.partition_function = 0.0;
        result.energy = 0.0;
        result.observable_values.resize(observables.size(), 0.0);
        
        double Z_sum = 0.0;
        double E_sum = 0.0;
        double E2_sum = 0.0;
        std::vector<double> obs_sums(observables.size(), 0.0);
        
        // Parallel computation over random vectors
        #pragma omp parallel for reduction(+:Z_sum,E_sum,E2_sum) schedule(dynamic)
        for (int r = 0; r < num_random_vectors; r++) {
            // Generate random vector
            ComplexVector v0(N);
            generate_random_vector(v0, r);
            
            // Run Lanczos to get tridiagonal matrix
            std::vector<double> alpha, beta;
            std::vector<ComplexVector> lanczos_vectors;
            
            run_lanczos_iteration(v0, alpha, beta, lanczos_vectors);
            
            // Compute contribution from this random vector
            double Z_r, E_r, E2_r;
            std::vector<double> obs_r(observables.size());
            
            if (use_importance_sampling) {
                compute_contribution_importance_sampling(
                    alpha, beta, lanczos_vectors, beta, observables,
                    Z_r, E_r, E2_r, obs_r, r
                );
            } else {
                compute_contribution_standard(
                    alpha, beta, lanczos_vectors, beta, observables,
                    Z_r, E_r, E2_r, obs_r, r
                );
            }
            
            Z_sum += Z_r;
            E_sum += E_r;
            E2_sum += E2_r;
            
            #pragma omp critical
            {
                for (size_t i = 0; i < observables.size(); i++) {
                    obs_sums[i] += obs_r[i];
                }
            }
            
            if (r % 10 == 0) {
                std::cout << "FTLM: Processed " << r+1 << "/" << num_random_vectors 
                         << " random vectors" << std::endl;
            }
        }
        
        // Normalize results
        result.partition_function = Z_sum / num_random_vectors;
        result.energy = E_sum / Z_sum;
        double E_squared = E2_sum / Z_sum;
        result.heat_capacity = beta * beta * (E_squared - result.energy * result.energy);
        result.free_energy = -log(result.partition_function) / beta;
        result.entropy = beta * (result.energy - result.free_energy);
        
        for (size_t i = 0; i < observables.size(); i++) {
            result.observable_values[i] = obs_sums[i] / Z_sum;
        }
        
        return result;
    }
    
    // Compute thermal density of states
    std::vector<std::pair<double, double>> compute_thermal_dos(
        double beta,
        double E_min,
        double E_max,
        int num_bins = 1000
    ) {
        std::vector<double> dos(num_bins, 0.0);
        double dE = (E_max - E_min) / num_bins;
        
        #pragma omp parallel for reduction(+:dos[:num_bins])
        for (int r = 0; r < num_random_vectors; r++) {
            ComplexVector v0(N);
            generate_random_vector(v0, r);
            
            std::vector<double> alpha, beta_coeffs;
            std::vector<ComplexVector> lanczos_vectors;
            
            run_lanczos_iteration(v0, alpha, beta_coeffs, lanczos_vectors);
            
            // Compute spectral function using continued fraction
            for (int bin = 0; bin < num_bins; bin++) {
                double E = E_min + (bin + 0.5) * dE;
                double spectral = compute_spectral_function(alpha, beta_coeffs, E, 0.01);
                dos[bin] += spectral * exp(-beta * E);
            }
        }
        
        // Normalize and create output
        std::vector<std::pair<double, double>> result;
        double norm = 0.0;
        for (int bin = 0; bin < num_bins; bin++) {
            norm += dos[bin] * dE;
        }
        
        for (int bin = 0; bin < num_bins; bin++) {
            double E = E_min + (bin + 0.5) * dE;
            result.push_back({E, dos[bin] / norm});
        }
        
        return result;
    }
    
    // Microcanonical Lanczos Method (MCLM) for very low temperatures
    ThermalExpectation compute_microcanonical(
        double energy_target,
        double energy_window,
        std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {}
    ) {
        std::cout << "MCLM: Computing microcanonical expectation at E = " 
                  << energy_target << " ± " << energy_window << std::endl;
        
        ThermalExpectation result;
        result.energy = energy_target;
        result.observable_values.resize(observables.size(), 0.0);
        
        double weight_sum = 0.0;
        std::vector<double> obs_sums(observables.size(), 0.0);
        
        #pragma omp parallel for reduction(+:weight_sum) schedule(dynamic)
        for (int r = 0; r < num_random_vectors; r++) {
            ComplexVector v0(N);
            generate_random_vector(v0, r);
            
            // Apply energy filter
            ComplexVector filtered = apply_energy_filter(v0, energy_target, energy_window);
            
            double norm = cblas_dznrm2(N, filtered.data(), 1);
            if (norm < 1e-12) continue;
            
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, filtered.data(), 1);
            
            weight_sum += norm * norm;
            
            // Compute observables
            #pragma omp critical
            {
                for (size_t i = 0; i < observables.size(); i++) {
                    ComplexVector Ov(N);
                    observables[i](filtered.data(), Ov.data(), N);
                    
                    Complex expectation;
                    cblas_zdotc_sub(N, filtered.data(), 1, Ov.data(), 1, &expectation);
                    obs_sums[i] += std::real(expectation) * norm * norm;
                }
            }
        }
        
        // Normalize
        result.partition_function = weight_sum / num_random_vectors;
        for (size_t i = 0; i < observables.size(); i++) {
            result.observable_values[i] = obs_sums[i] / weight_sum;
        }
        
        return result;
    }
    
    // Low-temperature Lanczos Method (LTLM) with ground state projection
    ThermalExpectation compute_low_temperature(
        double beta,
        int num_low_states = 20,
        std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {}
    ) {
        std::cout << "LTLM: Computing low-temperature expectation at β = " << beta << std::endl;
        
        // Find lowest eigenvalues using Lanczos
        std::vector<double> low_eigenvalues;
        std::vector<ComplexVector> low_eigenvectors;
        
        // Use block Lanczos to find degenerate states
        block_cg(H, N, 200, num_low_states, tol, low_eigenvalues, low_eigenvectors);
        
        // Compute thermal averages using exact enumeration over low-energy states
        ThermalExpectation result;
        result.partition_function = 0.0;
        result.energy = 0.0;
        result.observable_values.resize(observables.size(), 0.0);
        
        double Z = 0.0;
        for (size_t i = 0; i < low_eigenvalues.size(); i++) {
            double boltzmann = exp(-beta * low_eigenvalues[i]);
            Z += boltzmann;
            result.energy += low_eigenvalues[i] * boltzmann;
            
            for (size_t j = 0; j < observables.size(); j++) {
                ComplexVector Ov(N);
                observables[j](low_eigenvectors[i].data(), Ov.data(), N);
                
                Complex expectation;
                cblas_zdotc_sub(N, low_eigenvectors[i].data(), 1, Ov.data(), 1, &expectation);
                result.observable_values[j] += std::real(expectation) * boltzmann;
            }
        }
        
        // Add contribution from high-energy states using FTLM
        ThermalExpectation high_T = compute_thermal_expectation(beta, observables, true);
        
        // Combine results with proper weighting
        double cutoff_energy = low_eigenvalues.back() + 5.0/beta;
        double high_weight = exp(-beta * cutoff_energy);
        
        result.partition_function = Z + high_weight * high_T.partition_function;
        result.energy = (result.energy + high_weight * high_T.energy * high_T.partition_function) 
                       / result.partition_function;
        
        for (size_t j = 0; j < observables.size(); j++) {
            result.observable_values[j] = (result.observable_values[j] + 
                high_weight * high_T.observable_values[j] * high_T.partition_function) 
                / result.partition_function;
        }
        
        result.free_energy = -log(result.partition_function) / beta;
        result.entropy = beta * (result.energy - result.free_energy);
        
        return result;
    }
    
private:
    void generate_random_vector(ComplexVector& v, int seed) {
        std::mt19937 local_gen(seed + gen());
        std::uniform_real_distribution<double> local_dist(-1.0, 1.0);
        
        for (int i = 0; i < N; i++) {
            v[i] = Complex(local_dist(local_gen), local_dist(local_gen));
        }
        
        double norm = cblas_dznrm2(N, v.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v.data(), 1);
    }
    
    void run_lanczos_iteration(
        const ComplexVector& v0,
        std::vector<double>& alpha,
        std::vector<double>& beta,
        std::vector<ComplexVector>& vectors
    ) {
        vectors.clear();
        vectors.push_back(v0);
        
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_current = v0;
        ComplexVector w(N);
        
        alpha.clear();
        beta.clear();
        beta.push_back(0.0);
        
        for (int j = 0; j < lanczos_steps && j < N; j++) {
            H(v_current.data(), w.data(), N);
            
            if (j > 0) {
                Complex neg_beta = Complex(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            Complex alpha_j;
            cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &alpha_j);
            alpha.push_back(std::real(alpha_j));
            
            Complex neg_alpha = Complex(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
            
            double beta_j = cblas_dznrm2(N, w.data(), 1);
            
            if (beta_j < tol) {
                break;
            }
            
            beta.push_back(beta_j);
            
            Complex scale = Complex(1.0/beta_j, 0.0);
            cblas_zscal(N, &scale, w.data(), 1);
            
            v_prev = v_current;
            v_current = w;
            vectors.push_back(v_current);
        }
    }
    
    void compute_contribution_standard(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        const std::vector<ComplexVector>& vectors,
        double temperature,
        const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables,
        double& Z_r,
        double& E_r,
        double& E2_r,
        std::vector<double>& obs_r,
        int r_index
    ) {
        int m = alpha.size();
        
        // Use cached eigenvalues if available
        std::vector<double> eigenvalues;
        if (r_index < cached_eigenvalues.size() && !cached_eigenvalues[r_index].empty()) {
            eigenvalues = cached_eigenvalues[r_index];
        } else {
            // Solve tridiagonal eigenvalue problem
            std::vector<double> diag = alpha;
            std::vector<double> offdiag(m-1);
            for (int i = 0; i < m-1; i++) {
                offdiag[i] = beta[i+1];
            }
            
            eigenvalues.resize(m);
            int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'N', m, 
                                    diag.data(), offdiag.data(), nullptr, m);
            
            if (info == 0) {
                eigenvalues = diag;
                if (r_index < cached_eigenvalues.size()) {
                    cached_eigenvalues[r_index] = eigenvalues;
                }
            }
        }
        
        // Compute thermal quantities
        Z_r = 0.0;
        E_r = 0.0;
        E2_r = 0.0;
        
        for (double E : eigenvalues) {
            double w = exp(-temperature * E);
            Z_r += w;
            E_r += E * w;
            E2_r += E * E * w;
        }
        
        // Compute observables using spectral representation
        obs_r.resize(observables.size(), 0.0);
        for (size_t i = 0; i < observables.size(); i++) {
            obs_r[i] = compute_observable_lanczos(
                vectors, alpha, beta, temperature, observables[i]
            );
        }
    }
    
    void compute_contribution_importance_sampling(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        const std::vector<ComplexVector>& vectors,
        double temperature,
        const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables,
        double& Z_r,
        double& E_r,
        double& E2_r,
        std::vector<double>& obs_r,
        int r_index
    ) {
        // Use importance sampling with Gaussian broadening
        const double sigma = 1.0 / sqrt(2.0 * temperature);
        const int num_samples = 100;
        
        std::normal_distribution<double> energy_dist(0.0, sigma);
        std::mt19937 local_gen(r_index);
        
        Z_r = 0.0;
        E_r = 0.0;
        E2_r = 0.0;
        obs_r.resize(observables.size(), 0.0);
        
        for (int sample = 0; sample < num_samples; sample++) {
            double E_sample = energy_dist(local_gen);
            
            // Compute spectral weight at this energy
            double weight = compute_spectral_function(alpha, beta, E_sample, 0.01);
            double boltzmann = exp(-temperature * E_sample);
            double contribution = weight * boltzmann;
            
            Z_r += contribution;
            E_r += E_sample * contribution;
            E2_r += E_sample * E_sample * contribution;
            
            // Observable contributions
            for (size_t i = 0; i < observables.size(); i++) {
                double obs_val = compute_observable_at_energy(
                    vectors, alpha, beta, E_sample, observables[i]
                );
                obs_r[i] += obs_val * contribution;
            }
        }
        
        Z_r /= num_samples;
        E_r /= num_samples;
        E2_r /= num_samples;
        for (size_t i = 0; i < observables.size(); i++) {
            obs_r[i] /= num_samples;
        }
    }
    
    double compute_spectral_function(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        double energy,
        double eta
    ) {
        // Continued fraction calculation
        int m = alpha.size();
        Complex z(energy, eta);
        Complex G = Complex(0.0, 0.0);
        
        for (int i = m-1; i >= 0; i--) {
            if (i == m-1) {
                G = 1.0 / (z - Complex(alpha[i], 0.0));
            } else {
                G = 1.0 / (z - Complex(alpha[i], 0.0) - 
                          Complex(beta[i+1] * beta[i+1], 0.0) * G);
            }
        }
        
        return -std::imag(G) / M_PI;
    }
    
    double compute_observable_lanczos(
        const std::vector<ComplexVector>& vectors,
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        double temperature,
        std::function<void(const Complex*, Complex*, int)> O
    ) {
        // Compute <O>_T using Lanczos representation
        int m = vectors.size();
        if (m == 0) return 0.0;
        
        // Apply operator to first Lanczos vector
        ComplexVector Ov0(N);
        O(vectors[0].data(), Ov0.data(), N);
        
        // Project onto Lanczos subspace
        std::vector<Complex> O_proj(m);
        for (int i = 0; i < m; i++) {
            cblas_zdotc_sub(N, vectors[i].data(), 1, Ov0.data(), 1, &O_proj[i]);
        }
        
        // Solve tridiagonal eigenvalue problem
        std::vector<double> eigenvalues;
        std::vector<double> eigenvectors_tri;
        solve_tridiagonal_with_vectors(alpha, beta, eigenvalues, eigenvectors_tri);
        
        // Compute thermal average
        double result = 0.0;
        double Z = 0.0;
        
        for (int i = 0; i < m; i++) {
            double boltzmann = exp(-temperature * eigenvalues[i]);
            Z += boltzmann;
            
            // Matrix element in eigenbasis
            Complex matrix_elem = 0.0;
            for (int j = 0; j < m; j++) {
                matrix_elem += O_proj[j] * eigenvectors_tri[j * m + i] * 
                              eigenvectors_tri[0 * m + i];
            }
            
            result += std::real(matrix_elem) * boltzmann;
        }
        
        return result / Z;
    }
    
    double compute_observable_at_energy(
        const std::vector<ComplexVector>& vectors,
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        double energy,
        std::function<void(const Complex*, Complex*, int)> O
    ) {
        // Compute O(E) using spectral representation
        ComplexVector v_E = compute_eigenstate_at_energy(vectors, alpha, beta, energy);
        
        ComplexVector Ov(N);
        O(v_E.data(), Ov.data(), N);
        
        Complex result;
        cblas_zdotc_sub(N, v_E.data(), 1, Ov.data(), 1, &result);
        
        return std::real(result);
    }
    
    ComplexVector compute_eigenstate_at_energy(
        const std::vector<ComplexVector>& vectors,
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        double energy
    ) {
        // Approximate eigenstate at given energy using Lanczos vectors
        int m = vectors.size();
        ComplexVector result(N, Complex(0.0, 0.0));
        
        // Solve (T - E*I)c = e1 for expansion coefficients
        std::vector<Complex> coeffs(m);
        solve_shifted_tridiagonal(alpha, beta, energy, coeffs);
        
        // Reconstruct eigenstate
        for (int i = 0; i < m; i++) {
            cblas_zaxpy(N, &coeffs[i], vectors[i].data(), 1, result.data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, result.data(), 1);
        if (norm > 1e-12) {
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, result.data(), 1);
        }
        
        return result;
    }
    
    void solve_tridiagonal_with_vectors(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        std::vector<double>& eigenvalues,
        std::vector<double>& eigenvectors
    ) {
        int m = alpha.size();
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m-1);
        
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = beta[i+1];
        }
        
        eigenvalues.resize(m);
        eigenvectors.resize(m * m);
        
        int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', m,
                                diag.data(), offdiag.data(), 
                                eigenvectors.data(), m);
        
        if (info == 0) {
            eigenvalues = diag;
        }
    }
    
    void solve_shifted_tridiagonal(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        double shift,
        std::vector<Complex>& solution
    ) {
        int m = alpha.size();
        solution.resize(m);
        
        // Thomas algorithm for (T - shift*I)x = e1
        std::vector<Complex> c(m), d(m);
        
        // Forward sweep
        c[0] = Complex(alpha[0] - shift, 0.0);
        d[0] = Complex(1.0, 0.0);
        
        for (int i = 1; i < m; i++) {
            Complex ratio = Complex(beta[i], 0.0) / c[i-1];
            c[i] = Complex(alpha[i] - shift, 0.0) - ratio * Complex(beta[i], 0.0);
            d[i] = -ratio * d[i-1];
        }
        
        // Back substitution
        solution[m-1] = d[m-1] / c[m-1];
        for (int i = m-2; i >= 0; i--) {
            solution[i] = (d[i] - Complex(beta[i+1], 0.0) * solution[i+1]) / c[i];
        }
    }
    
    ComplexVector apply_energy_filter(
        const ComplexVector& v,
        double E_target,
        double E_window
    ) {
        // Apply Gaussian filter centered at E_target
        ComplexVector result(N, Complex(0.0, 0.0));
        
        // Use Chebyshev expansion of Gaussian filter
        const int cheb_order = 50;
        std::vector<ComplexVector> T_n(cheb_order + 1, ComplexVector(N));
        
        // Estimate spectral bounds
        double E_min = E_target - 5 * E_window;
        double E_max = E_target + 5 * E_window;
        
        // Map to [-1, 1]
        double a = (E_max + E_min) / 2.0;
        double b = (E_max - E_min) / 2.0;
        
        // Chebyshev coefficients for Gaussian
        std::vector<double> c_n(cheb_order + 1);
        for (int n = 0; n <= cheb_order; n++) {
            c_n[n] = compute_gaussian_chebyshev_coeff(n, E_target, E_window, a, b);
        }
        
        // Apply Chebyshev expansion
        T_n[0] = v;
        
        ComplexVector Hv(N);
        H(v.data(), Hv.data(), N);
        
        // T_1 = (H - aI)/b * v
        for (int i = 0; i < N; i++) {
            T_n[1][i] = (Hv[i] - a * v[i]) / b;
        }
        
        // Accumulate result
        for (int i = 0; i < N; i++) {
            result[i] = c_n[0] * T_n[0][i] + c_n[1] * T_n[1][i];
        }
        
        // Recurrence for higher order terms
        for (int n = 2; n <= cheb_order; n++) {
            ComplexVector HT(N);
            H(T_n[n-1].data(), HT.data(), N);
            
            for (int i = 0; i < N; i++) {
                T_n[n][i] = 2.0 * (HT[i] - a * T_n[n-1][i]) / b - T_n[n-2][i];
                result[i] += c_n[n] * T_n[n][i];
            }
        }
        
        return result;
    }
    
    double compute_gaussian_chebyshev_coeff(int n, double center, double width, double a, double b) {
        // Compute n-th Chebyshev coefficient for Gaussian exp(-(x-center)^2/(2*width^2))
        const int num_points = 100;
        double coeff = 0.0;
        
        for (int k = 0; k < num_points; k++) {
            double theta = M_PI * (k + 0.5) / num_points;
            double x = cos(theta);
            double E = a + b * x;
            double gaussian = exp(-(E - center) * (E - center) / (2 * width * width));
            coeff += gaussian * cos(n * theta);
        }
        
        coeff *= 2.0 / num_points;
        if (n == 0) coeff /= 2.0;
        
        return coeff;
    }
};

// Convenience function for computing thermal averages
ThermalExpectation compute_thermal_average(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double temperature,
    std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {},
    int num_random_vectors = 100,
    int lanczos_steps = 100,
    double tol = 1e-10,
    bool use_low_T_method = true
) {
    double beta = 1.0 / temperature;
    
    FiniteTemperatureLanczos ftlm(H, N, num_random_vectors, lanczos_steps, tol);
    
    if (use_low_T_method && beta > 10.0) {
        // Use low-temperature method for β > 10
        return ftlm.compute_low_temperature(beta, 30, observables);
    } else {
        // Use standard FTLM
        return ftlm.compute_thermal_expectation(beta, observables, true);
    }
}

// Compute temperature-dependent spectral function
std::vector<std::pair<double, double>> compute_thermal_spectral_function(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double temperature,
    double E_min,
    double E_max,
    int num_points = 1000,
    int num_random_vectors = 100
) {
    double beta = 1.0 / temperature;
    
    FiniteTemperatureLanczos ftlm(H, N, num_random_vectors, 100, 1e-10);
    
    return ftlm.compute_thermal_dos(beta, E_min, E_max, num_points);
}





#endif // FINITE_TEMPERATURE_LANCZOS_H