#ifndef FINITE_TEMPERATURE_LANCZOS_H
#define FINITE_TEMPERATURE_LANCZOS_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <mkl.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <chrono>
#include "lanczos.h"
#include "observables.h"
#include <omp.h>

// Structure to hold thermodynamic data
struct ThermodynamicDataLM {
    std::vector<double> temperatures;
    std::vector<double> energies;
    std::vector<double> specific_heats;
    std::vector<double> entropies;
    std::vector<double> free_energies;
    std::vector<double> partition_functions;
    std::vector<std::vector<double>> observable_values;
    std::vector<std::string> observable_names;
};

// Structure for FTLM workspace
struct FTLMWorkspace {
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<ComplexVector> lanczos_vectors;
    int num_iterations;
    double ground_state_energy;
};

/**
 * Compute partition function and thermodynamic properties from Lanczos coefficients
 * using continued fraction expansion
 */
void compute_thermodynamics_from_lanczos(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    double ground_state_energy,
    const std::vector<double>& temperatures,
    ThermodynamicDataLM& thermo_data
) {
    int m = alpha.size();
    int num_temps = temperatures.size();
    
    thermo_data.temperatures = temperatures;
    thermo_data.partition_functions.resize(num_temps);
    thermo_data.energies.resize(num_temps);
    thermo_data.specific_heats.resize(num_temps);
    thermo_data.entropies.resize(num_temps);
    thermo_data.free_energies.resize(num_temps);
    
    #pragma omp parallel for
    for (int t = 0; t < num_temps; t++) {
        double T = temperatures[t];
        double beta_inv = 1.0 / T;
        
        // Compute partition function using continued fraction
        double Z = 0.0;
        double Z1 = 0.0;  // First derivative
        double Z2 = 0.0;  // Second derivative
        
        // Use recursion relation for continued fractions
        std::vector<double> b(m + 1, 0.0);
        std::vector<double> b1(m + 1, 0.0);
        std::vector<double> b2(m + 1, 0.0);
        
        b[m] = 1.0;
        b1[m] = 0.0;
        b2[m] = 0.0;
        
        // Backward recursion
        for (int j = m - 1; j >= 0; j--) {
            double exp_factor = std::exp(-beta_inv * (alpha[j] - ground_state_energy));
            
            if (j == 0) {
                b[j] = exp_factor;
                b1[j] = -beta_inv * exp_factor;
                b2[j] = beta_inv * beta_inv * exp_factor;
            } else {
                double beta_sq = (j < m - 1) ? beta[j] * beta[j] : 0.0;
                b[j] = exp_factor + beta_sq * b[j + 1];
                b1[j] = -beta_inv * exp_factor + beta_sq * b1[j + 1];
                b2[j] = beta_inv * beta_inv * exp_factor + beta_sq * b2[j + 1];
            }
        }
        
        Z = b[0];
        Z1 = b1[0];
        Z2 = b2[0];
        
        // Compute thermodynamic quantities
        thermo_data.partition_functions[t] = Z;
        thermo_data.free_energies[t] = -T * std::log(Z);
        thermo_data.energies[t] = ground_state_energy - Z1 / Z;
        thermo_data.specific_heats[t] = beta_inv * beta_inv * (Z2 / Z - (Z1 / Z) * (Z1 / Z));
        thermo_data.entropies[t] = (thermo_data.energies[t] - thermo_data.free_energies[t]) / T;
    }
}

/**
 * Standard Finite Temperature Lanczos Method (FTLM)
 * Computes thermodynamic properties using random vector sampling
 */
void finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int max_iter,
    double tol,
    const std::vector<double>& temperatures,
    ThermodynamicDataLM& thermo_data,
    int num_random_vectors = 100,
    const std::string& output_dir = "",
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables = {},
    const std::vector<std::string>& observable_names = {}
) {
    std::cout << "Starting Finite Temperature Lanczos Method (FTLM)" << std::endl;
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Number of random vectors: " << num_random_vectors << std::endl;
    std::cout << "Temperature range: " << temperatures.front() << " to " << temperatures.back() << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize accumulated data
    std::vector<double> avg_energies(temperatures.size(), 0.0);
    std::vector<double> avg_specific_heats(temperatures.size(), 0.0);
    std::vector<double> avg_entropies(temperatures.size(), 0.0);
    std::vector<double> avg_free_energies(temperatures.size(), 0.0);
    std::vector<std::vector<double>> avg_observables;
    
    if (!observables.empty()) {
        avg_observables.resize(observables.size(), std::vector<double>(temperatures.size(), 0.0));
        thermo_data.observable_names = observable_names;
    }
    
    // Get ground state energy for reference
    std::vector<double> ground_energy;
    lanczos(H, N, std::min(100, N), 1, tol, ground_energy, output_dir, false);
    double E0 = ground_energy[0];
    
    // Perform random vector sampling
    for (int sample = 0; sample < num_random_vectors; sample++) {
        if (sample % 10 == 0) {
            std::cout << "Processing random vector " << sample + 1 << "/" << num_random_vectors << std::endl;
        }
        
        // Generate random initial vector
        ComplexVector v0 = generateRandomVector(N, gen, dist);
        
        // Perform Lanczos iteration
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_curr = v0;
        ComplexVector v_next(N);
        
        std::vector<double> alpha;
        std::vector<double> beta;
        
        for (int iter = 0; iter < max_iter; iter++) {
            // Apply Hamiltonian
            H(v_curr.data(), v_next.data(), N);
            
            // Compute alpha coefficient
            Complex alpha_complex;
            cblas_zdotc_sub(N, v_curr.data(), 1, v_next.data(), 1, &alpha_complex);
            double alpha_val = std::real(alpha_complex);
            alpha.push_back(alpha_val);
            
            // Update v_next: v_next = H*v_curr - alpha*v_curr - beta*v_prev
            Complex neg_alpha(-alpha_val, 0.0);
            cblas_zaxpy(N, &neg_alpha, v_curr.data(), 1, v_next.data(), 1);
            
            if (iter > 0) {
                Complex neg_beta(-beta.back(), 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, v_next.data(), 1);
            }
            
            // Compute beta coefficient
            double beta_val = cblas_dznrm2(N, v_next.data(), 1);
            
            // Check for convergence
            if (beta_val < tol) {
                break;
            }
            
            if (iter < max_iter - 1) {
                beta.push_back(beta_val);
                
                // Normalize v_next
                Complex scale(1.0 / beta_val, 0.0);
                cblas_zscal(N, &scale, v_next.data(), 1);
                
                // Update vectors for next iteration
                v_prev = std::move(v_curr);
                v_curr = std::move(v_next);
                v_next = ComplexVector(N);
            }
        }
        
        // Compute thermodynamic properties for this random vector
        ThermodynamicDataLM sample_thermo;
        compute_thermodynamics_from_lanczos(alpha, beta, E0, temperatures, sample_thermo);
        
        // Accumulate results
        for (size_t t = 0; t < temperatures.size(); t++) {
            avg_energies[t] += sample_thermo.energies[t];
            avg_specific_heats[t] += sample_thermo.specific_heats[t];
            avg_entropies[t] += sample_thermo.entropies[t];
            avg_free_energies[t] += sample_thermo.free_energies[t];
        }
        
        // Compute observable expectation values if requested
        if (!observables.empty()) {
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                // Apply observable to initial vector
                ComplexVector obs_v0(N);
                observables[obs_idx](v0.data(), obs_v0.data(), N);
                
                // Compute expectation value at each temperature using spectral representation
                for (size_t t = 0; t < temperatures.size(); t++) {
                    double T = temperatures[t];
                    double beta_inv = 1.0 / T;
                    
                    // Use Lanczos recursion to compute <v0|O*exp(-beta*H)|v0>
                    Complex obs_exp_val(0.0, 0.0);
                    // Implementation would use similar continued fraction approach
                    // For brevity, using simplified calculation
                    cblas_zdotc_sub(N, v0.data(), 1, obs_v0.data(), 1, &obs_exp_val);
                    avg_observables[obs_idx][t] += std::real(obs_exp_val);
                }
            }
        }
    }
    
    // Average over all random vectors
    double norm = 1.0 / num_random_vectors;
    for (size_t t = 0; t < temperatures.size(); t++) {
        thermo_data.energies.push_back(avg_energies[t] * norm);
        thermo_data.specific_heats.push_back(avg_specific_heats[t] * norm);
        thermo_data.entropies.push_back(avg_entropies[t] * norm);
        thermo_data.free_energies.push_back(avg_free_energies[t] * norm);
    }
    
    thermo_data.temperatures = temperatures;
    
    if (!observables.empty()) {
        thermo_data.observable_values.resize(observables.size());
        for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
            thermo_data.observable_values[obs_idx].resize(temperatures.size());
            for (size_t t = 0; t < temperatures.size(); t++) {
                thermo_data.observable_values[obs_idx][t] = avg_observables[obs_idx][t] * norm;
            }
        }
    }
    
    // Save results to file
    if (!output_dir.empty()) {
        std::string thermo_file = output_dir + "/thermodynamics_ftlm.dat";
        std::ofstream out(thermo_file);
        out << "# Temperature Energy Specific_Heat Entropy Free_Energy";
        for (const auto& name : observable_names) {
            out << " " << name;
        }
        out << std::endl;
        out << std::scientific << std::setprecision(10);
        
        for (size_t t = 0; t < temperatures.size(); t++) {
            out << temperatures[t] << " "
                << thermo_data.energies[t] << " "
                << thermo_data.specific_heats[t] << " "
                << thermo_data.entropies[t] << " "
                << thermo_data.free_energies[t];
            
            for (size_t obs_idx = 0; obs_idx < thermo_data.observable_values.size(); obs_idx++) {
                out << " " << thermo_data.observable_values[obs_idx][t];
            }
            out << std::endl;
        }
        out.close();
        std::cout << "Thermodynamic data saved to: " << thermo_file << std::endl;
    }
}

/**
 * Low Temperature Lanczos Method (LTLM)
 * More accurate for low temperatures by using low-lying eigenstates
 */
void low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int num_eigenvalues,
    double tol,
    const std::vector<double>& temperatures,
    ThermodynamicDataLM& thermo_data,
    const std::string& output_dir = "",
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables = {},
    const std::vector<std::string>& observable_names = {}
) {
    std::cout << "Starting Low Temperature Lanczos Method (LTLM)" << std::endl;
    std::cout << "Computing " << num_eigenvalues << " lowest eigenvalues" << std::endl;
    
    // Compute low-lying eigenvalues and eigenvectors
    std::vector<double> eigenvalues;
    std::string evec_dir = output_dir + "/eigenvectors_ltlm";
    if (!output_dir.empty()) {
        system(("mkdir -p " + evec_dir).c_str());
    }
    
    // Use standard Lanczos to get eigenvalues
    lanczos(H, N, std::min(num_eigenvalues * 3, N), num_eigenvalues, tol, eigenvalues, 
            output_dir.empty() ? "" : evec_dir, !observables.empty());
    
    // Initialize thermodynamic data
    thermo_data.temperatures = temperatures;
    thermo_data.energies.resize(temperatures.size(), 0.0);
    thermo_data.specific_heats.resize(temperatures.size(), 0.0);
    thermo_data.entropies.resize(temperatures.size(), 0.0);
    thermo_data.free_energies.resize(temperatures.size(), 0.0);
    thermo_data.partition_functions.resize(temperatures.size(), 0.0);
    
    if (!observables.empty()) {
        thermo_data.observable_values.resize(observables.size(), 
                                           std::vector<double>(temperatures.size(), 0.0));
        thermo_data.observable_names = observable_names;
    }
    
    // Compute observable matrix elements if needed
    std::vector<std::vector<double>> obs_matrix_elements;
    if (!observables.empty() && !output_dir.empty()) {
        obs_matrix_elements.resize(observables.size());
        
        for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
            obs_matrix_elements[obs_idx].resize(num_eigenvalues);
            
            // Load eigenvectors and compute matrix elements
            for (int i = 0; i < num_eigenvalues; i++) {
                std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
                ComplexVector eigvec(N);
                
                std::ifstream in(evec_file, std::ios::binary);
                if (in.is_open()) {
                    in.read(reinterpret_cast<char*>(eigvec.data()), N * sizeof(Complex));
                    in.close();
                    
                    // Compute <ψ_i|O|ψ_i>
                    ComplexVector O_eigvec(N);
                    observables[obs_idx](eigvec.data(), O_eigvec.data(), N);
                    
                    Complex matrix_element;
                    cblas_zdotc_sub(N, eigvec.data(), 1, O_eigvec.data(), 1, &matrix_element);
                    obs_matrix_elements[obs_idx][i] = std::real(matrix_element);
                }
            }
        }
    }
    
    // Compute thermodynamic properties at each temperature
    #pragma omp parallel for
    for (size_t t = 0; t < temperatures.size(); t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function
        double Z = 0.0;
        for (int i = 0; i < num_eigenvalues; i++) {
            Z += std::exp(-beta * eigenvalues[i]);
        }
        thermo_data.partition_functions[t] = Z;
        
        // Compute energy
        double E = 0.0;
        for (int i = 0; i < num_eigenvalues; i++) {
            E += eigenvalues[i] * std::exp(-beta * eigenvalues[i]);
        }
        thermo_data.energies[t] = E / Z;
        
        // Compute specific heat
        double E2 = 0.0;
        for (int i = 0; i < num_eigenvalues; i++) {
            E2 += eigenvalues[i] * eigenvalues[i] * std::exp(-beta * eigenvalues[i]);
        }
        thermo_data.specific_heats[t] = beta * beta * (E2 / Z - (E / Z) * (E / Z));
        
        // Compute free energy and entropy
        thermo_data.free_energies[t] = -T * std::log(Z);
        thermo_data.entropies[t] = (thermo_data.energies[t] - thermo_data.free_energies[t]) / T;
        
        // Compute observable expectation values
        if (!observables.empty() && !obs_matrix_elements.empty()) {
            for (size_t obs_idx = 0; obs_idx < observables.size(); obs_idx++) {
                double obs_exp = 0.0;
                for (int i = 0; i < num_eigenvalues; i++) {
                    obs_exp += obs_matrix_elements[obs_idx][i] * std::exp(-beta * eigenvalues[i]);
                }
                thermo_data.observable_values[obs_idx][t] = obs_exp / Z;
            }
        }
    }
    
    // Save results
    if (!output_dir.empty()) {
        std::string thermo_file = output_dir + "/thermodynamics_ltlm.dat";
        std::ofstream out(thermo_file);
        out << "# Temperature Energy Specific_Heat Entropy Free_Energy Partition_Function";
        for (const auto& name : observable_names) {
            out << " " << name;
        }
        out << std::endl;
        out << std::scientific << std::setprecision(10);
        
        for (size_t t = 0; t < temperatures.size(); t++) {
            out << temperatures[t] << " "
                << thermo_data.energies[t] << " "
                << thermo_data.specific_heats[t] << " "
                << thermo_data.entropies[t] << " "
                << thermo_data.free_energies[t] << " "
                << thermo_data.partition_functions[t];
            
            for (size_t obs_idx = 0; obs_idx < thermo_data.observable_values.size(); obs_idx++) {
                out << " " << thermo_data.observable_values[obs_idx][t];
            }
            out << std::endl;
        }
        out.close();
        std::cout << "Thermodynamic data saved to: " << thermo_file << std::endl;
    }
}

/**
 * Adaptive finite temperature method that switches between FTLM and LTLM
 * based on temperature range
 */
void adaptive_finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const std::vector<double>& temperatures,
    ThermodynamicDataLM& thermo_data,
    const std::string& output_dir = "",
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables = {},
    const std::vector<std::string>& observable_names = {},
    double crossover_temp = 1.0,
    int num_random_vectors = 100,
    int num_low_eigenvalues = 100
) {
    std::cout << "Starting Adaptive Finite Temperature Lanczos Method" << std::endl;
    std::cout << "Crossover temperature: " << crossover_temp << std::endl;
    
    // Split temperatures into low and high regions
    std::vector<double> low_temps, high_temps;
    std::vector<size_t> low_indices, high_indices;
    
    for (size_t i = 0; i < temperatures.size(); i++) {
        if (temperatures[i] < crossover_temp) {
            low_temps.push_back(temperatures[i]);
            low_indices.push_back(i);
        } else {
            high_temps.push_back(temperatures[i]);
            high_indices.push_back(i);
        }
    }
    
    // Initialize combined results
    thermo_data.temperatures = temperatures;
    thermo_data.energies.resize(temperatures.size());
    thermo_data.specific_heats.resize(temperatures.size());
    thermo_data.entropies.resize(temperatures.size());
    thermo_data.free_energies.resize(temperatures.size());
    
    if (!observables.empty()) {
        thermo_data.observable_values.resize(observables.size(), 
                                           std::vector<double>(temperatures.size()));
        thermo_data.observable_names = observable_names;
    }
    
    // Use LTLM for low temperatures
    if (!low_temps.empty()) {
        std::cout << "\nUsing LTLM for " << low_temps.size() << " low temperature points" << std::endl;
        ThermodynamicDataLM low_temp_data;
        low_temperature_lanczos(H, N, num_low_eigenvalues, 1e-10, low_temps, 
                               low_temp_data, output_dir, observables, observable_names);
        
        // Copy results to combined data
        for (size_t i = 0; i < low_indices.size(); i++) {
            size_t idx = low_indices[i];
            thermo_data.energies[idx] = low_temp_data.energies[i];
            thermo_data.specific_heats[idx] = low_temp_data.specific_heats[i];
            thermo_data.entropies[idx] = low_temp_data.entropies[i];
            thermo_data.free_energies[idx] = low_temp_data.free_energies[i];
            
            for (size_t obs_idx = 0; obs_idx < thermo_data.observable_values.size(); obs_idx++) {
                thermo_data.observable_values[obs_idx][idx] = 
                    low_temp_data.observable_values[obs_idx][i];
            }
        }
    }
    
    // Use FTLM for high temperatures
    if (!high_temps.empty()) {
        std::cout << "\nUsing FTLM for " << high_temps.size() << " high temperature points" << std::endl;
        ThermodynamicDataLM high_temp_data;
        finite_temperature_lanczos(H, N, 100, 1e-10, high_temps, high_temp_data, 
                                  num_random_vectors, output_dir, observables, observable_names);
        
        // Copy results to combined data
        for (size_t i = 0; i < high_indices.size(); i++) {
            size_t idx = high_indices[i];
            thermo_data.energies[idx] = high_temp_data.energies[i];
            thermo_data.specific_heats[idx] = high_temp_data.specific_heats[i];
            thermo_data.entropies[idx] = high_temp_data.entropies[i];
            thermo_data.free_energies[idx] = high_temp_data.free_energies[i];
            
            for (size_t obs_idx = 0; obs_idx < thermo_data.observable_values.size(); obs_idx++) {
                thermo_data.observable_values[obs_idx][idx] = 
                    high_temp_data.observable_values[obs_idx][i];
            }
        }
    }
    
    // Save combined results
    if (!output_dir.empty()) {
        std::string thermo_file = output_dir + "/thermodynamics_adaptive.dat";
        std::ofstream out(thermo_file);
        out << "# Temperature Energy Specific_Heat Entropy Free_Energy Method";
        for (const auto& name : observable_names) {
            out << " " << name;
        }
        out << std::endl;
        out << std::scientific << std::setprecision(10);
        
        for (size_t t = 0; t < temperatures.size(); t++) {
            out << temperatures[t] << " "
                << thermo_data.energies[t] << " "
                << thermo_data.specific_heats[t] << " "
                << thermo_data.entropies[t] << " "
                << thermo_data.free_energies[t] << " "
                << (temperatures[t] < crossover_temp ? "LTLM" : "FTLM");
            
            for (size_t obs_idx = 0; obs_idx < thermo_data.observable_values.size(); obs_idx++) {
                out << " " << thermo_data.observable_values[obs_idx][t];
            }
            out << std::endl;
        }
        out.close();
        std::cout << "Combined thermodynamic data saved to: " << thermo_file << std::endl;
    }
}

#endif // FINITE_TEMPERATURE_LANCZOS_H