#ifndef OBSERVABLES_H
#define OBSERVABLES_H

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
// #include <ezarpack/arpack_solver.hpp>
// #include <ezarpack/storages/eigen.hpp>
// #include <ezarpack/version.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>
#include <fstream>
#include <set>
#include <thread>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Observables

// Calculate thermodynamic quantities directly from eigenvalues
struct ThermodynamicData {
    std::vector<double> temperatures;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
};

// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min = 0.01,        // Minimum temperature
    double T_max = 10.0,        // Maximum temperature
    int num_points = 100        // Number of temperature points
) {
    ThermodynamicData results;
    
    // Generate logarithmically spaced temperature points
    results.temperatures.resize(num_points);
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Resize other arrays
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Find ground state energy (useful for numerical stability)
    double E0 = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    
    // For each temperature
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        // Use log-sum-exp trick for numerical stability in calculating Z
        // Find the maximum value for normalization
        double max_exp = -beta * E0;  // Start with ground state
        
        // Calculate partition function Z and energy using log-sum-exp trick
        double sum_exp = 0.0;
        double sum_E_exp = 0.0;
        double sum_E2_exp = 0.0;
        
        for (double E : eigenvalues) {
            double delta_E = E - E0;
            double exp_term = std::exp(-beta * delta_E);
            
            sum_exp += exp_term;
            sum_E_exp += E * exp_term;
            sum_E2_exp += E * E * exp_term;
        }
        
        // Calculate log(Z) = log(sum_exp) + (-beta*E0)
        double log_Z = std::log(sum_exp) - beta * E0;
        
        // Free energy F = -T * log(Z)
        results.free_energy[i] = -T * log_Z;
        
        // Energy E = (1/Z) * sum_i E_i * exp(-beta*E_i)
        results.energy[i] = sum_E_exp / sum_exp;
        
        // Specific heat C_v = beta^2 * (⟨E^2⟩ - ⟨E⟩^2)
        double avg_E2 = sum_E2_exp / sum_exp;
        double avg_E_squared = results.energy[i] * results.energy[i];
        results.specific_heat[i] = beta * beta * (avg_E2 - avg_E_squared);
        
        // Entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
    }
    
    // Handle special case for T → 0 (avoid numerical issues)
    if (T_min < 1e-6) {
        // In the limit T → 0, only the ground state contributes
        // Energy → E0
        results.energy[0] = E0;
        
        // Specific heat → 0
        results.specific_heat[0] = 0.0;
        
        // Entropy → 0 (third law of thermodynamics) or ln(g) if g-fold degenerate
        int degeneracy = 0;
        for (double E : eigenvalues) {
            if (std::abs(E - E0) < 1e-10) degeneracy++;
        }
        results.entropy[0] = (degeneracy > 1) ? std::log(degeneracy) : 0.0;
        
        // Free energy → E0 - TS
        results.free_energy[0] = E0 - results.temperatures[0] * results.entropy[0];
    }
    
    return results;
}

// Calculate the expectation value <ψ_a|A|ψ_a> for the a-th eigenstate of H
Complex calculate_expectation_value(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                                // Dimension of Hilbert space
    int a = 0,                                           // Index of eigenstate (default: ground state)
    int max_iter = 100,                                  // Maximum iterations for eigenstate calculation
    double tol = 1e-10                                   // Tolerance
) {
    // First, calculate the a-th eigenstate of H
    std::vector<double> eigenvalues;
    std::vector<ComplexVector> eigenvectors;
    
    // Use Chebyshev filtered Lanczos for better accuracy
    chebyshev_filtered_lanczos(H, N, max_iter, max_iter, tol, eigenvalues, &eigenvectors);
    
    // Check if we have enough eigenstates
    if (a >= eigenvectors.size()) {
        std::cerr << "Error: Requested eigenstate index " << a 
                  << " but only " << eigenvectors.size() << " states computed." << std::endl;
        return Complex(0.0, 0.0);
    }
    
    // Get the a-th eigenstate
    const ComplexVector& psi = eigenvectors[a];
    ComplexVector A_psi(N);
    
    // Apply operator A to the eigenstate
    A(psi.data(), A_psi.data(), N);
    
    // Calculate <ψ_a|A|ψ_a>
    Complex expectation_value;
    cblas_zdotc_sub(N, psi.data(), 1, A_psi.data(), 1, &expectation_value);
    
    return expectation_value;
}


// Calculate thermal expectation value of operator A using eigenvalues and eigenvectors
// <A> = (1/Z) * ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
Complex calculate_thermal_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                               // Hilbert space dimension
    double beta,                                         // Inverse temperature β = 1/kT
    const std::string& eig_dir                           // Directory with eigenvector files
) {

    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.bin";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return Complex(0.0, 0.0);
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();

    // Using the log-sum-exp trick for numerical stability
    // Find the maximum value for normalization
    double max_val = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_val = std::max(max_val, -beta * eigenvalues[i]);
    }
    
    // Calculate the numerator <A> = ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
    Complex numerator(0.0, 0.0);
    double sum_exp = 0.0;
    
    // Temporary vector to store A|ψ_i⟩
    ComplexVector A_psi(N);
    ComplexVector psi_i(N);
    
    // Calculate both the numerator and Z in one loop
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        // Calculate the Boltzmann factor with numerical stability
        double boltzmann = std::exp(-beta * eigenvalues[i] - max_val);
        sum_exp += boltzmann;
        
        // Load eigenvector from file
        std::string evec_file = eig_dir + "/eigenvector_" + std::to_string(i) + ".bin";
        std::ifstream infile(evec_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        infile.read(reinterpret_cast<char*>(psi_i.data()), N * sizeof(Complex));
        infile.close();
        
        // Calculate <ψ_i|A|ψ_i>
        A(psi_i.data(), A_psi.data(), N);
        
        Complex expectation;
        cblas_zdotc_sub(N, psi_i.data(), 1, A_psi.data(), 1, &expectation);
        
        // Add contribution to numerator
        numerator += boltzmann * expectation;
    }
    
    // Return <A> = numerator/Z
    return numerator / sum_exp;
}


// Calculate matrix element <ψ₁|A|ψ₂> between two state vectors
Complex calculate_matrix_element(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    const ComplexVector& psi1,                           // First state vector |ψ₁⟩
    const ComplexVector& psi2,                           // Second state vector |ψ₂⟩
    int N                                               // Dimension of Hilbert space
) {
    // Check that dimensions match
    if (psi1.size() != N || psi2.size() != N) {
        std::cerr << "Error: State vector dimensions don't match Hilbert space dimension" << std::endl;
        return Complex(0.0, 0.0);
    }
    
    // Apply operator A to |ψ₂⟩: A|ψ₂⟩
    ComplexVector A_psi2(N);
    A(psi2.data(), A_psi2.data(), N);
    
    // Calculate <ψ₁|A|ψ₂>
    Complex matrix_element;
    cblas_zdotc_sub(N, psi1.data(), 1, A_psi2.data(), 1, &matrix_element);
    
    return matrix_element;
}

// Calculate spectral function A(ω) for operator O using all eigenstates
// A(ω) = Σ_n,m |<n|O|m>|^2 δ(ω - (E_n - E_m)) * weight(m)
// where δ is approximated by a broadening function (Gaussian or Lorentzian)
struct SpectralFunctionData {
    std::vector<double> frequencies;     // ω values
    std::vector<double> spectral_function;  // A(ω) values
};

SpectralFunctionData calculate_spectral_function(
    std::function<void(const Complex*, Complex*, int)> O,  // Operator O
    int N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min = -10.0,                            // Minimum frequency
    double omega_max = 10.0,                             // Maximum frequency
    int num_points = 1000,                               // Number of frequency points
    double eta = 0.1,                                    // Broadening parameter
    double temperature = 0.0,                            // Temperature (0 for ground state only)
    bool use_lorentzian = false                          // Use Lorentzian (true) or Gaussian (false) broadening
) {
    SpectralFunctionData result;
    
    // Generate frequency points
    result.frequencies.resize(num_points);
    double omega_step = (omega_max - omega_min) / (num_points - 1);
    for (int i = 0; i < num_points; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.spectral_function.resize(num_points, 0.0);
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.bin";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return result;
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();
    
    // Calculate weights based on temperature
    std::vector<double> weights(num_eigenvalues, 0.0);
    
    if (temperature <= 0.0) {
        // Zero temperature: Only ground state has weight 1.0
        weights[0] = 1.0;
    } else {
        // Finite temperature: Boltzmann weights
        double beta = 1.0 / temperature;
        double min_energy = *std::min_element(eigenvalues.begin(), eigenvalues.end());
        
        // Use log-sum-exp trick for numerical stability
        double max_exp = -beta * min_energy;
        double Z = 0.0;
        
        for (size_t i = 0; i < num_eigenvalues; i++) {
            double boltzmann = std::exp(-beta * (eigenvalues[i] - min_energy));
            weights[i] = boltzmann;
            Z += boltzmann;
        }
        
        // Normalize weights
        for (size_t i = 0; i < num_eigenvalues; i++) {
            weights[i] /= Z;
        }
    }
    
    // Process all transitions between eigenstates
    for (size_t m = 0; m < num_eigenvalues; m++) {
        // Skip states with negligible weight
        if (weights[m] < 1e-12) continue;
        
        // Load eigenstate |m⟩
        ComplexVector psi_m(N);
        std::string evec_file_m = eig_dir + "/eigenvector_" + std::to_string(m) + ".bin";
        std::ifstream infile_m(evec_file_m, std::ios::binary);
        if (!infile_m) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file_m << std::endl;
            continue;
        }
        infile_m.read(reinterpret_cast<char*>(psi_m.data()), N * sizeof(Complex));
        infile_m.close();
        
        // Apply operator O to state |m⟩
        ComplexVector O_psi_m(N);
        O(psi_m.data(), O_psi_m.data(), N);
        
        for (size_t n = 0; n < num_eigenvalues; n++) {
            // Load eigenstate |n⟩
            ComplexVector psi_n(N);
            std::string evec_file_n = eig_dir + "/eigenvector_" + std::to_string(n) + ".bin";
            std::ifstream infile_n(evec_file_n, std::ios::binary);
            if (!infile_n) {
                std::cerr << "Error: Cannot open eigenvector file " << evec_file_n << std::endl;
                continue;
            }
            infile_n.read(reinterpret_cast<char*>(psi_n.data()), N * sizeof(Complex));
            infile_n.close();
            
            // Calculate matrix element ⟨n|O|m⟩
            Complex matrix_element;
            cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_m.data(), 1, &matrix_element);
            
            // Compute |⟨n|O|m⟩|^2
            double intensity = std::norm(matrix_element);
            
            // Skip if intensity is negligible
            if (intensity < 1e-12) continue;
            
            // Compute energy difference
            double delta_E = eigenvalues[n] - eigenvalues[m];
            
            // Add contribution to spectral function
            for (int i = 0; i < num_points; i++) {
                double omega = result.frequencies[i];
                double delta_omega = omega - delta_E;
                
                double broadening;
                if (use_lorentzian) {
                    // Lorentzian broadening
                    broadening = (1.0/M_PI) * (eta / (delta_omega*delta_omega + eta*eta));
                } else {
                    // Gaussian broadening
                    broadening = (1.0/(eta*std::sqrt(2.0*M_PI))) * 
                                 std::exp(-(delta_omega*delta_omega)/(2.0*eta*eta));
                }
                
                // Add weighted contribution
                result.spectral_function[i] += weights[m] * intensity * broadening;
            }
        }
        
        // Progress reporting
        std::cout << "Processed state " << m+1 << " of " << num_eigenvalues << std::endl;
    }
    
    return result;
}


// Calculate dynamical susceptibility χ(ω) for operator A
// χ(ω) = ∑_{n,m} (p_m - p_n) * |<n|A|m>|^2 / (ω - (E_n - E_m) + iη)
struct DynamicalSusceptibilityData {
    std::vector<double> frequencies;         // ω values
    std::vector<std::complex<double>> chi;   // χ(ω) values (complex)
};

DynamicalSusceptibilityData calculate_dynamical_susceptibility(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    int N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min = -10.0,                            // Minimum frequency
    double omega_max = 10.0,                             // Maximum frequency
    int num_points = 1000,                               // Number of frequency points
    double eta = 0.1,                                    // Broadening parameter
    double temperature = 1.0                             // Temperature (in energy units)
) {
    DynamicalSusceptibilityData result;
    
    // Generate frequency points
    result.frequencies.resize(num_points);
    double omega_step = (omega_max - omega_min) / (num_points - 1);
    for (int i = 0; i < num_points; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.chi.resize(num_points, std::complex<double>(0.0, 0.0));
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.bin";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return result;
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();
    
    // Calculate Boltzmann probabilities
    double beta = 1.0 / temperature;
    std::vector<double> probabilities(num_eigenvalues);
    
    // Use log-sum-exp trick for numerical stability
    double min_energy = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    double Z = 0.0;
    
    for (size_t i = 0; i < num_eigenvalues; i++) {
        double exp_factor = std::exp(-beta * (eigenvalues[i] - min_energy));
        probabilities[i] = exp_factor; // Temporarily store unnormalized probabilities
        Z += exp_factor;
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < num_eigenvalues; i++) {
        probabilities[i] /= Z;
    }
    
    // Temporary vectors for eigenvectors and operator application
    ComplexVector psi_m(N);
    ComplexVector psi_n(N);
    ComplexVector A_psi_m(N);
    
    // Process all transitions between eigenstates
    for (size_t m = 0; m < num_eigenvalues; m++) {
        // Skip states with negligible weight
        if (probabilities[m] < 1e-12) continue;
        
        // Load eigenstate |m⟩
        std::string evec_file_m = eig_dir + "/eigenvector_" + std::to_string(m) + ".bin";
        std::ifstream infile_m(evec_file_m, std::ios::binary);
        if (!infile_m) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file_m << std::endl;
            continue;
        }
        infile_m.read(reinterpret_cast<char*>(psi_m.data()), N * sizeof(Complex));
        infile_m.close();
        
        // Apply operator A to state |m⟩
        A_psi_m.resize(N);
        A(psi_m.data(), A_psi_m.data(), N);
        
        for (size_t n = 0; n < num_eigenvalues; n++) {
            // Skip if probability difference is too small
            double prob_diff = probabilities[m] - probabilities[n];
            if (std::abs(prob_diff) < 1e-12) continue;
            
            // Load eigenstate |n⟩
            std::string evec_file_n = eig_dir + "/eigenvector_" + std::to_string(n) + ".bin";
            std::ifstream infile_n(evec_file_n, std::ios::binary);
            if (!infile_n) {
                std::cerr << "Error: Cannot open eigenvector file " << evec_file_n << std::endl;
                continue;
            }
            infile_n.read(reinterpret_cast<char*>(psi_n.data()), N * sizeof(Complex));
            infile_n.close();
            
            // Calculate matrix element ⟨n|A|m⟩
            Complex matrix_element;
            cblas_zdotc_sub(N, psi_n.data(), 1, A_psi_m.data(), 1, &matrix_element);
            
            // Compute |⟨n|A|m⟩|^2
            double intensity = std::norm(matrix_element);
            
            // Skip if intensity is negligible
            if (intensity < 1e-12) continue;
            
            // Compute energy difference
            double delta_E = eigenvalues[n] - eigenvalues[m];
            
            // Calculate contribution to susceptibility for each frequency
            for (int i = 0; i < num_points; i++) {
                double omega = result.frequencies[i];
                std::complex<double> denominator(omega - delta_E, eta);
                std::complex<double> contribution = prob_diff * intensity / denominator;
                result.chi[i] += contribution;
            }
        }
        
        // Progress reporting
        std::cout << "Processed state " << m+1 << " of " << num_eigenvalues << std::endl;
    }
    
    return result;
}

// Calculate quantum Fisher information for operator A at temperature T
double calculate_quantum_fisher_information(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    int N,                                               // Hilbert space dimension
    double temperature,                                  // Temperature (in energy units)
    const std::string& eig_dir                           // Directory with eigenvector files
) {
    // Calculate beta = 1/kT
    double beta = 1.0 / temperature;
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.bin";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return 0.0;
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();
    
    // Calculate partition function Z and Boltzmann probabilities
    // Using the log-sum-exp trick for numerical stability
    double max_val = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_val = std::max(max_val, -beta * eigenvalues[i]);
    }
    
    double Z = 0.0;
    std::vector<double> probabilities(num_eigenvalues);
    
    for (size_t i = 0; i < num_eigenvalues; i++) {
        double exp_factor = std::exp(-beta * eigenvalues[i] - max_val);
        Z += exp_factor;
        probabilities[i] = exp_factor; // Will normalize by Z later
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < num_eigenvalues; i++) {
        probabilities[i] /= Z;
    }
    
    // Calculate quantum Fisher information
    double qfi = 0.0;
    
    // Temporary vectors for eigenvectors and operator application
    ComplexVector psi_m(N);
    ComplexVector psi_n(N);
    ComplexVector A_psi_n(N);
    
    // For each pair of eigenstates
    for (size_t m = 0; m < num_eigenvalues; m++) {
        // Load eigenvector |m⟩
        std::string evec_file_m = eig_dir + "/eigenvector_" + std::to_string(m) + ".bin";
        std::ifstream infile_m(evec_file_m, std::ios::binary);
        if (!infile_m) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file_m << std::endl;
            continue;
        }
        infile_m.read(reinterpret_cast<char*>(psi_m.data()), N * sizeof(Complex));
        infile_m.close();
        
        for (size_t n = 0; n < num_eigenvalues; n++) {
            // Skip if p_m and p_n are both effectively zero
            if (probabilities[m] < 1e-14 && probabilities[n] < 1e-14) continue;
            
            // Skip diagonal elements (they contribute zero to QFI)
            if (m == n) continue;
            
            // Calculate coefficient (p_m - p_n)^2/(p_m + p_n)
            double p_m = probabilities[m];
            double p_n = probabilities[n];
            double p_sum = p_m + p_n;
            
            // Handle numerical stability
            if (p_sum < 1e-14) continue; // Skip if denominator is too small
            
            double coef = (p_m - p_n) * (p_m - p_n) / p_sum;
            
            // Load eigenvector |n⟩
            std::string evec_file_n = eig_dir + "/eigenvector_" + std::to_string(n) + ".bin";
            std::ifstream infile_n(evec_file_n, std::ios::binary);
            if (!infile_n) {
                std::cerr << "Error: Cannot open eigenvector file " << evec_file_n << std::endl;
                continue;
            }
            infile_n.read(reinterpret_cast<char*>(psi_n.data()), N * sizeof(Complex));
            infile_n.close();
            
            // Calculate matrix element ⟨m|A|n⟩
            A(psi_n.data(), A_psi_n.data(), N);
            Complex matrix_element;
            cblas_zdotc_sub(N, psi_m.data(), 1, A_psi_n.data(), 1, &matrix_element);
            
            // Add contribution to QFI
            double abs_squared = std::norm(matrix_element); // |⟨m|A|n⟩|^2
            qfi += 2.0 * coef * abs_squared;
        }
    }
    
    return qfi;
}

// Compute thermal expectation values of S^+, S^-, S^z operators at each site
void compute_spin_expectations(
    const std::string& eigdir,  // Directory with eigenvalues and eigenvectors
    const std::string output_dir, // Directory for output files
    int num_sites,              // Number of sites
    float spin_l,              // Spin length (e.g., 0.5 for spin-1/2)
    double temperature,         // Temperature T (in energy units)
    bool print_output = true    // Whether to print the results to console
) {
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    // Load eigenvalues
    std::vector<double> eigenvalues;
    std::ifstream eigenvalue_file(eigdir + "/eigenvalues.bin", std::ios::binary);
    if (!eigenvalue_file) {
        std::cerr << "Error: Cannot open eigenvalue file " << eigdir + "/eigenvalues.bin" << std::endl;
        return;
    }
    
    size_t num_eigenvalues;
    eigenvalue_file.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    eigenvalue_file.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    eigenvalue_file.close();
    
    std::cout << "Loaded " << num_eigenvalues << " eigenvalues from " << eigdir + "/eigenvalues.bin" << std::endl;
    
    // Calculate beta = 1/kT
    double beta = 1.0 / temperature;
    
    // Using the log-sum-exp trick for numerical stability
    double max_exp = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_exp = std::max(max_exp, -beta * eigenvalues[i]);
    }
    
    // Calculate partition function Z
    double Z = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        Z += std::exp(-beta * eigenvalues[i] - max_exp);
    }
    
    // Create S operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // Process each eigenvector
    for (size_t idx = 0; idx < num_eigenvalues; idx++) {
        // Load eigenvector
        std::string evec_file = eigdir + "/eigenvector_" + std::to_string(idx) + ".bin";
        std::ifstream evec_stream(evec_file, std::ios::binary);
        if (!evec_stream) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        
        ComplexVector psi(N);
        evec_stream.read(reinterpret_cast<char*>(psi.data()), N * sizeof(Complex));
        evec_stream.close();
        
        // Calculate Boltzmann factor
        double boltzmann = std::exp(-beta * eigenvalues[idx] - max_exp) / Z;
        
        // For each site, compute the expectation values
        for (int site = 0; site < num_sites; site++) {
            // Apply operators
            std::vector<Complex> Sp_psi = Sp_ops[site].apply(std::vector<Complex>(psi.begin(), psi.end()));
            std::vector<Complex> Sm_psi = Sm_ops[site].apply(std::vector<Complex>(psi.begin(), psi.end()));
            std::vector<Complex> Sz_psi = Sz_ops[site].apply(std::vector<Complex>(psi.begin(), psi.end()));
            
            // Calculate expectation values
            Complex Sp_exp = Complex(0.0, 0.0);
            Complex Sm_exp = Complex(0.0, 0.0);
            Complex Sz_exp = Complex(0.0, 0.0);
            
            for (int i = 0; i < N; i++) {
                Sp_exp += std::conj(psi[i]) * Sp_psi[i];
                Sm_exp += std::conj(psi[i]) * Sm_psi[i];
                Sz_exp += std::conj(psi[i]) * Sz_psi[i];
            }
            
            // Add weighted contribution to thermal average
            expectations[0][site] += boltzmann * Sp_exp;
            expectations[1][site] += boltzmann * Sm_exp;
            expectations[2][site] += boltzmann * Sz_exp;
        }
        
        // Progress reporting
        if ((idx + 1) % 10 == 0 || idx == num_eigenvalues - 1) {
            std::cout << "Processed " << idx + 1 << "/" << num_eigenvalues << " eigenvectors" << std::endl;
        }
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values at T = " << temperature << ":" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^+ (real)" 
                << std::setw(20) << "S^+ (imag)" 
                << std::setw(20) << "S^- (real)"
                << std::setw(20) << "S^- (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file
    std::string outfile = output_dir + "/spin_expectations_T" + std::to_string(temperature) + ".dat";
    std::ofstream out(outfile);
    if (out.is_open()) {
        out << "# Site S+_real S+_imag S-_real S-_imag Sz_real Sz_imag" << std::endl;
        for (int site = 0; site < num_sites; site++) {
            out << site << " "
                << std::setprecision(10) << expectations[0][site].real() << " "
                << std::setprecision(10) << expectations[0][site].imag() << " "
                << std::setprecision(10) << expectations[1][site].real() << " "
                << std::setprecision(10) << expectations[1][site].imag() << " "
                << std::setprecision(10) << expectations[2][site].real() << " "
                << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
        out.close();
        std::cout << "Spin expectations saved to " << outfile << std::endl;
    }
}


#endif // OBSERVABLES_H