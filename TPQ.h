#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include "construct_ham.h"
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>
#include <fstream>
#include <set>

#ifdef TPQ_H
#define TPQ_H
// Thermal Pure Quantum (TPQ) state methods for thermodynamic calculations

// Generate a canonical TPQ state by applying (H - E)^k to a random vector
ComplexVector generate_tpq_state(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    double E_offset,                                       // Energy offset
    int k = 1,                                             // Power of (H - E)
    double tol = 1e-10                                     // Tolerance
) {
    // Create a random initial state
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector tpq_state(N);
    for (int i = 0; i < N; i++) {
        tpq_state[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, tpq_state.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, tpq_state.data(), 1);
    
    // Apply (H - E)^k
    ComplexVector temp_state(N);
    
    for (int i = 0; i < k; i++) {
        // Apply H to current state
        H(tpq_state.data(), temp_state.data(), N);
        
        // Subtract E * current state
        for (int j = 0; j < N; j++) {
            temp_state[j] -= E_offset * tpq_state[j];
        }
        
        // Normalize
        norm = cblas_dznrm2(N, temp_state.data(), 1);
        if (norm < tol) {
            // If norm is too small, we've probably hit an eigenstate
            std::cout << "Warning: TPQ generation may have converged to an eigenstate at step " << i + 1 << std::endl;
            break;
        }
        
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, temp_state.data(), 1);
        
        // Update for next iteration
        tpq_state = temp_state;
    }
    
    return tpq_state;
}

// Calculate the effective inverse temperature (beta) of a TPQ state
double calculate_tpq_beta(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N,                                                 // Hilbert space dimension
    double E_ref = 0.0                                     // Energy reference (usually ground state)
) {
    // Calculate <H>
    ComplexVector H_tpq(N);
    H(tpq_state.data(), H_tpq.data(), N);
    
    Complex energy_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H_tpq.data(), 1, &energy_exp);
    double energy = std::real(energy_exp);
    
    // Calculate <H²>
    ComplexVector H2_tpq(N);
    H(H_tpq.data(), H2_tpq.data(), N);
    
    Complex energy2_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H2_tpq.data(), 1, &energy2_exp);
    double energy2 = std::real(energy2_exp);
    
    // Variance of H
    double var_H = energy2 - energy * energy;
    
    // Effective inverse temperature: β = 2*(⟨H⟩ - E_ref)/⟨(H-⟨H⟩)²⟩
    if (var_H < 1e-10) {
        return std::numeric_limits<double>::infinity(); // If variance is zero, we have an eigenstate
    }
    
    return 2.0 * (energy - E_ref) / var_H;
}

// Calculate thermodynamic quantities from a TPQ state
struct TPQThermodynamics {
    double beta;       // Inverse temperature
    double energy;     // Energy
    double specific_heat;
    double entropy;
    double free_energy;
};

TPQThermodynamics calculate_tpq_thermodynamics(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N,                                                 // Hilbert space dimension
    double E_ref = 0.0                                     // Energy reference (usually ground state)
) {
    // Calculate <H>
    ComplexVector H_tpq(N);
    H(tpq_state.data(), H_tpq.data(), N);
    
    Complex energy_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H_tpq.data(), 1, &energy_exp);
    double energy = std::real(energy_exp);
    
    // Calculate <H²>
    ComplexVector H2_tpq(N);
    H(H_tpq.data(), H2_tpq.data(), N);
    
    Complex energy2_exp;
    cblas_zdotc_sub(N, tpq_state.data(), 1, H2_tpq.data(), 1, &energy2_exp);
    double energy2 = std::real(energy2_exp);
    
    // Variance of H
    double var_H = energy2 - energy * energy;
    
    // Effective inverse temperature
    double beta = 2.0 * (energy - E_ref) / var_H;
    double temperature = (beta > 1e-10) ? 1.0 / beta : std::numeric_limits<double>::infinity();
    
    // Specific heat: C = beta² * var_H
    double specific_heat = beta * beta * var_H;
    
    // Entropy and free energy require additional approximations
    // For canonical TPQ states, entropy can be approximated as S ≈ ln(D) - β²*var_H/2
    // where D is the Hilbert space dimension
    double entropy = std::log(N) - beta * beta * var_H / 2.0;
    
    // Free energy: F = E - TS
    double free_energy = energy - temperature * entropy;
    
    return {beta, energy, specific_heat, entropy, free_energy};
}

// Calculate expectation value of an observable using a TPQ state
Complex calculate_tpq_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    const ComplexVector& tpq_state,                        // TPQ state
    int N                                                 // Hilbert space dimension
) {
    ComplexVector A_tpq(N);
    A(tpq_state.data(), A_tpq.data(), N);
    
    Complex expectation;
    cblas_zdotc_sub(N, tpq_state.data(), 1, A_tpq.data(), 1, &expectation);
    
    return expectation;
}

// Main TPQ implementation for temperature scanning
struct TPQResults {
    std::vector<double> betas;           // Inverse temperatures
    std::vector<double> temperatures;    // Temperatures
    std::vector<double> energies;        // Energies
    std::vector<double> specific_heats;  // Specific heats
    std::vector<double> entropies;       // Entropies
    std::vector<double> free_energies;   // Free energies
    std::vector<std::vector<Complex>> observables; // Observable expectation values
};

TPQResults perform_tpq_calculation(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    int num_samples = 20,                                 // Number of TPQ samples
    int max_k = 50,                                       // Maximum power for (H-E)^k
    double E_min = -1.0,                                  // Minimum energy offset 
    double E_max = 1.0,                                   // Maximum energy offset
    double E_ref = 0.0,                                   // Energy reference
    double beta_min = 0.01,                               // Minimum inverse temperature
    double beta_max = 100.0,                              // Maximum inverse temperature
    int num_beta_bins = 100,                             // Number of temperature bins
    std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {}  // Optional observables
) {
    TPQResults results;
    
    // Initialize result containers
    results.betas.resize(num_beta_bins, 0.0);
    results.temperatures.resize(num_beta_bins, 0.0);
    results.energies.resize(num_beta_bins, 0.0);
    results.specific_heats.resize(num_beta_bins, 0.0);
    results.entropies.resize(num_beta_bins, 0.0);
    results.free_energies.resize(num_beta_bins, 0.0);
    
    // Initialize counters for each bin
    std::vector<int> bin_counts(num_beta_bins, 0);
    
    // Initialize bins for beta values
    double log_beta_min = std::log(beta_min);
    double log_beta_max = std::log(beta_max);
    double log_beta_step = (log_beta_max - log_beta_min) / (num_beta_bins - 1);
    
    for (int i = 0; i < num_beta_bins; i++) {
        results.betas[i] = std::exp(log_beta_min + i * log_beta_step);
        results.temperatures[i] = 1.0 / results.betas[i];
    }
    
    // Initialize observable containers if provided
    if (!observables.empty()) {
        results.observables.resize(observables.size(), std::vector<Complex>(num_beta_bins, Complex(0.0, 0.0)));
    }
    
    std::cout << "TPQ: Starting calculations with " << num_samples << " samples" << std::endl;
    
    // Generate TPQ states with different energy offsets and powers
    for (int sample = 0; sample < num_samples; sample++) {
        // Randomly choose energy offset between E_min and E_max
        double energy_offset = E_min + (E_max - E_min) * static_cast<double>(sample) / num_samples;
        
        // Generate multiple states with different powers
        for (int k = 1; k <= max_k; k += 2) { // Increment by 2 for efficiency
            // Generate TPQ state
            ComplexVector tpq_state = generate_tpq_state(H, N, energy_offset, k);
            
            // Calculate thermodynamics
            auto thermo = calculate_tpq_thermodynamics(H, tpq_state, N, E_ref);
            
            // Determine which beta bin this state belongs to
            double log_beta = std::log(thermo.beta);
            int bin = static_cast<int>((log_beta - log_beta_min) / log_beta_step);
            
            // Skip if outside our temperature range
            if (bin < 0 || bin >= num_beta_bins) {
                continue;
            }
            
            // Accumulate thermodynamic data
            bin_counts[bin]++;
            results.energies[bin] += thermo.energy;
            results.specific_heats[bin] += thermo.specific_heat;
            results.entropies[bin] += thermo.entropy;
            results.free_energies[bin] += thermo.free_energy;
            
            // Calculate observables if provided
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                Complex exp_val = calculate_tpq_expectation(observables[obs_idx], tpq_state, N);
                results.observables[obs_idx][bin] += exp_val;
            }
            
            // Progress reporting
            if ((sample * max_k + k) % 10 == 0) {
                std::cout << "TPQ: Sample " << sample + 1 << "/" << num_samples 
                          << ", k = " << k << ", β = " << thermo.beta
                          << ", E = " << thermo.energy << std::endl;
            }
        }
    }
    
    // Average the results over the number of samples in each bin
    for (int i = 0; i < num_beta_bins; i++) {
        if (bin_counts[i] > 0) {
            results.energies[i] /= bin_counts[i];
            results.specific_heats[i] /= bin_counts[i];
            results.entropies[i] /= bin_counts[i];
            results.free_energies[i] /= bin_counts[i];
            
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                results.observables[obs_idx][i] /= bin_counts[i];
            }
            
            std::cout << "TPQ: Bin " << i << " (β = " << results.betas[i] 
                      << ") has " << bin_counts[i] << " samples" << std::endl;
        } else {
            std::cout << "TPQ: Warning - no samples in bin " << i 
                      << " (β = " << results.betas[i] << ")" << std::endl;
        }
    }
    
    return results;
}

// Generate microcanonical TPQ states for better control over temperature
ComplexVector generate_microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                                 // Hilbert space dimension
    double target_energy,                                  // Target energy
    double energy_window = 0.1,                           // Energy window width
    int max_iter = 100,                                   // Maximum iterations
    double tol = 1e-6                                     // Tolerance
) {
    // Create a random initial state
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector psi(N);
    for (int i = 0; i < N; i++) {
        psi[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, psi.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, psi.data(), 1);
    
    // Apply a filter to target the desired energy window
    // This is approximated by exp(-γ(H-E)²)
    ComplexVector H_psi(N);
    ComplexVector H2_psi(N);
    ComplexVector psi_new(N);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate current energy expectation
        H(psi.data(), H_psi.data(), N);
        Complex energy_exp;
        cblas_zdotc_sub(N, psi.data(), 1, H_psi.data(), 1, &energy_exp);
        double current_energy = std::real(energy_exp);
        
        // Calculate energy variance
        H(H_psi.data(), H2_psi.data(), N);
        Complex energy2_exp;
        cblas_zdotc_sub(N, psi.data(), 1, H2_psi.data(), 1, &energy2_exp);
        double energy_var = std::real(energy2_exp) - current_energy * current_energy;
        
        std::cout << "Microcanonical TPQ: iter " << iter << ", E = " << current_energy 
                  << ", var = " << energy_var << std::endl;
        
        // Check if we're close enough to target energy with small variance
        if (std::abs(current_energy - target_energy) < tol && energy_var < energy_window) {
            std::cout << "Microcanonical TPQ: Converged at iteration " << iter << std::endl;
            break;
        }
        
        // Adjust filtering parameter based on current energy
        double gamma = 1.0 / (2.0 * energy_window);
        if (std::abs(current_energy - target_energy) > energy_window) {
            gamma = 0.1 / energy_var; // Faster approach when far from target
        }
        
        // Apply filter exp(-γ(H-E)²) using Chebyshev approximation
        ComplexVector temp1(N), temp2(N);
        for (int i = 0; i < N; i++) {
            psi_new[i] = psi[i];
            temp1[i] = psi[i];
        }
        
        // Subtract target energy: (H-E)|ψ⟩
        for (int i = 0; i < N; i++) {
            H_psi[i] -= target_energy * psi[i];
        }
        
        // Apply approximation of exp(-γ(H-E)²) using series expansion
        Complex coef = Complex(1.0, 0.0);
        cblas_zaxpy(N, &coef, psi.data(), 1, psi_new.data(), 1);
        
        coef = Complex(-gamma, 0.0);
        for (int i = 0; i < N; i++) {
            temp2[i] = H_psi[i] * H_psi[i]; // (H-E)²|ψ⟩
        }
        cblas_zaxpy(N, &coef, temp2.data(), 1, psi_new.data(), 1);
        
        // Normalize the new state
        norm = cblas_dznrm2(N, psi_new.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, psi_new.data(), 1);
        
        // Update state for next iteration
        psi = psi_new;
    }
    
    return psi;
}

// Save TPQ results to file
void save_tpq_results(const TPQResults& results, const std::string& filename, 
                     int num_observables = 0) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    outfile << "# Beta Temperature Energy SpecificHeat Entropy FreeEnergy";
    for (int i = 0; i < num_observables; i++) {
        outfile << " Observable" << i << "_Real Observable" << i << "_Imag";
    }
    outfile << std::endl;
    
    for (size_t i = 0; i < results.betas.size(); i++) {
        outfile << results.betas[i] << " "
               << results.temperatures[i] << " "
               << results.energies[i] << " "
               << results.specific_heats[i] << " "
               << results.entropies[i] << " "
               << results.free_energies[i];
        
        for (int j = 0; j < num_observables; j++) {
            outfile << " " << results.observables[j][i].real()
                   << " " << results.observables[j][i].imag();
        }
        
        outfile << std::endl;
    }
    
    outfile.close();
    std::cout << "TPQ results saved to " << filename << std::endl;
}

#endif // TPQ_H