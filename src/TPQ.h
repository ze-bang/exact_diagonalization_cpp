#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <functional>
#include <thread>
#include <omp.h>

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Structure to hold thermodynamic data
struct TPQThermodynamicData {
    std::vector<double> beta_values;  // Inverse temperature values
    std::vector<double> energy;       // Energy per site
    std::vector<double> specific_heat; // Specific heat per site
    std::vector<double> entropy;      // Entropy per site
    std::vector<double> free_energy;  // Free energy per site
    
    // Save data to file
    void save_to_file(const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        file << "# Beta Energy SpecificHeat Entropy FreeEnergy" << std::endl;
        for (size_t i = 0; i < beta_values.size(); i++) {
            file << std::setprecision(12) << beta_values[i] << " "
                 << energy[i] << " "
                 << specific_heat[i] << " "
                 << entropy[i] << " "
                 << free_energy[i] << std::endl;
        }
        file.close();
    }
};

// Generate a random normalized state
ComplexVector generateRandomQuantumState(int N, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector state(N);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        state[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, state.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, state.data(), 1);
    
    return state;
}

// Compute expectation value <ψ|O|ψ>
Complex expectationValue(const ComplexVector& state, 
                         std::function<void(const Complex*, Complex*, int)> O, 
                         int N) {
    ComplexVector Ostate(N);
    O(state.data(), Ostate.data(), N);
    
    Complex result;
    cblas_zdotc_sub(N, state.data(), 1, Ostate.data(), 1, &result);
    return result;
}

// Compute expectation value <ψ|O²|ψ>
Complex expectationValueSquared(const ComplexVector& state, 
                               std::function<void(const Complex*, Complex*, int)> O, 
                               int N) {
    ComplexVector Ostate(N);
    ComplexVector O2state(N);
    
    O(state.data(), Ostate.data(), N);
    O(Ostate.data(), O2state.data(), N);
    
    Complex result;
    cblas_zdotc_sub(N, state.data(), 1, O2state.data(), 1, &result);
    return result;
}

// Calculate variance <ψ|(O-<O>)²|ψ>
double varianceOperator(const ComplexVector& state, 
                       std::function<void(const Complex*, Complex*, int)> O, 
                       int N) {
    Complex mean = expectationValue(state, O, N);
    Complex mean_squared = mean * mean;
    Complex squared_mean = expectationValueSquared(state, O, N);
    
    return std::abs(squared_mean - mean_squared);
}

// Chebyshev expansion of exp(-β*H/2) applied to a state
// This approach avoids explicit diagonalization
void applyExpBetaH(std::function<void(const Complex*, Complex*, int)> H,
                  double beta, double emin, double emax,
                  ComplexVector& state, int N, int cheby_order = 50) {
    
    // Scale the Hamiltonian to [-1,1] range for Chebyshev
    double a = (emax - emin) / 2.0;
    double b = (emax + emin) / 2.0;
    
    auto scaled_H = [&H, a, b, N](const Complex* in, Complex* out, int size) {
        // Compute (H-b)/a
        ComplexVector temp(N);
        H(in, temp.data(), N);
        
        for (int i = 0; i < N; i++) {
            out[i] = (temp[i] - Complex(b, 0.0)) / Complex(a, 0.0);
        }
    };
    
    // Bessel functions of first kind for coefficients
    std::vector<double> coef(cheby_order);
    double arg = -beta * a / 2.0;
    coef[0] = std::exp(beta * b / 2.0) * std::cyl_bessel_i(0, -arg);
    
    for (int k = 1; k < cheby_order; k++) {
        coef[k] = 2.0 * std::exp(beta * b / 2.0) * std::cyl_bessel_i(k, -arg);
    }
    
    // Store Chebyshev polynomials applied to state
    std::vector<ComplexVector> cheby_states(cheby_order);
    cheby_states[0] = state;  // T_0(x)|ψ⟩ = |ψ⟩
    
    if (cheby_order > 1) {
        cheby_states[1].resize(N);
        scaled_H(state.data(), cheby_states[1].data(), N);  // T_1(x)|ψ⟩ = x|ψ⟩
    }
    
    // Recursively compute T_k(x)|ψ⟩
    for (int k = 2; k < cheby_order; k++) {
        cheby_states[k].resize(N);
        
        scaled_H(cheby_states[k-1].data(), cheby_states[k].data(), N);
        
        Complex two(2.0, 0.0);
        cblas_zscal(N, &two, cheby_states[k].data(), 1);
        
        Complex neg_one(-1.0, 0.0);
        cblas_zaxpy(N, &neg_one, cheby_states[k-2].data(), 1, cheby_states[k].data(), 1);
    }
    
    // Combine all terms with coefficients
    std::fill(state.begin(), state.end(), Complex(0.0, 0.0));
    
    for (int k = 0; k < cheby_order; k++) {
        Complex ck(coef[k], 0.0);
        cblas_zaxpy(N, &ck, cheby_states[k].data(), 1, state.data(), 1);
    }
    
    // Normalize the final state
    double norm = cblas_dznrm2(N, state.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, state.data(), 1);
}

// Time evolution of state |ψ(t)⟩ = e^(-iHt)|ψ(0)⟩ using Chebyshev expansion
void timeEvolveState(std::function<void(const Complex*, Complex*, int)> H,
                    double time, double emin, double emax,
                    ComplexVector& state, int N, int cheby_order = 50) {
    
    // Scale the Hamiltonian to [-1,1] range for Chebyshev
    double a = (emax - emin) / 2.0;
    double b = (emax + emin) / 2.0;
    
    auto scaled_H = [&H, a, b, N](const Complex* in, Complex* out, int size) {
        // Compute (H-b)/a
        ComplexVector temp(N);
        H(in, temp.data(), N);
        
        for (int i = 0; i < N; i++) {
            out[i] = (temp[i] - Complex(b, 0.0)) / Complex(a, 0.0);
        }
    };
    
    // Compute Bessel coefficients
    std::vector<Complex> coef(cheby_order);
    double arg = a * time;
    
    for (int k = 0; k < cheby_order; k++) {
        // c_k = (-i)^k * e^(-i*b*t) * J_k(a*t)
        Complex ik = std::pow(Complex(0.0, -1.0), k);
        coef[k] = ik * std::exp(Complex(0.0, -b*time)) * std::cyl_bessel_j(k, arg);
        
        if (k > 0) {
            coef[k] *= 2.0;  // Multiplier for k > 0
        }
    }
    
    // Store Chebyshev polynomials applied to state
    std::vector<ComplexVector> cheby_states(cheby_order);
    cheby_states[0] = state;  // T_0(x)|ψ⟩ = |ψ⟩
    
    if (cheby_order > 1) {
        cheby_states[1].resize(N);
        scaled_H(state.data(), cheby_states[1].data(), N);  // T_1(x)|ψ⟩ = x|ψ⟩
    }
    
    // Recursively compute T_k(x)|ψ⟩
    for (int k = 2; k < cheby_order; k++) {
        cheby_states[k].resize(N);
        
        scaled_H(cheby_states[k-1].data(), cheby_states[k].data(), N);
        
        Complex two(2.0, 0.0);
        cblas_zscal(N, &two, cheby_states[k].data(), 1);
        
        Complex neg_one(-1.0, 0.0);
        cblas_zaxpy(N, &neg_one, cheby_states[k-2].data(), 1, cheby_states[k].data(), 1);
    }
    
    // Combine all terms with coefficients
    std::fill(state.begin(), state.end(), Complex(0.0, 0.0));
    
    for (int k = 0; k < cheby_order; k++) {
        cblas_zaxpy(N, &coef[k], cheby_states[k].data(), 1, state.data(), 1);
    }
    
    // Normalize the final state
    double norm = cblas_dznrm2(N, state.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, state.data(), 1);
}

// Microcanonical TPQ method
// Computes thermodynamic quantities by time-evolving random states
TPQThermodynamicData microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N,                          // Hilbert space dimension
    int num_sites,                  // Number of physical sites 
    int num_samples = 10,           // Number of random samples
    double time_step = 0.1,         // Time step for evolution
    int num_steps = 100,            // Number of time steps
    std::string dir = "",           // Directory for output
    bool save_states = false        // Whether to save evolved states
) {
    std::cout << "Starting Microcanonical TPQ with " << num_samples 
              << " samples, " << num_steps << " time steps" << std::endl;
    
    // Create output directory
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    
    // Estimate spectral bounds using a few Lanczos steps
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v_rand(N);
    for (int i = 0; i < N; i++) {
        v_rand[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, v_rand.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_rand.data(), 1);
    
    // Power iteration to estimate largest eigenvalue
    const int power_steps = 20;
    ComplexVector Hv(N);
    for (int i = 0; i < power_steps; i++) {
        H(v_rand.data(), Hv.data(), N);
        std::swap(v_rand, Hv);
        
        norm = cblas_dznrm2(N, v_rand.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v_rand.data(), 1);
    }
    
    // Apply H one more time for Rayleigh quotient
    H(v_rand.data(), Hv.data(), N);
    Complex rayleigh_quotient;
    cblas_zdotc_sub(N, v_rand.data(), 1, Hv.data(), 1, &rayleigh_quotient);
    
    // Estimate spectral bounds
    double lambda_max = std::real(rayleigh_quotient);
    double safety_factor = 1.1;  // Add 10% margin
    double emax = lambda_max * safety_factor;
    double emin = -emax;        // Assumes Hamiltonian is centered around zero
    
    std::cout << "Estimated spectral bounds [" << emin << ", " << emax << "]" << std::endl;
    
    // Initialize results
    TPQThermodynamicData results;
    results.beta_values.resize(num_steps);
    results.energy.resize(num_steps, 0.0);
    results.specific_heat.resize(num_steps, 0.0);
    results.entropy.resize(num_steps, 0.0);
    results.free_energy.resize(num_steps, 0.0);
    
    // Data for each sample
    std::vector<std::vector<double>> sample_energy(num_samples, std::vector<double>(num_steps));
    std::vector<std::vector<double>> sample_energy_sq(num_samples, std::vector<double>(num_steps));
    
    // Computing time evolution and averages
    #pragma omp parallel for schedule(dynamic)
    for (int sample = 0; sample < num_samples; sample++) {
        // Generate random initial state
        std::mt19937 local_gen(std::random_device{}() + sample); // Different seed for each sample
        ComplexVector state = generateRandomQuantumState(N, local_gen);
        
        // Time reference for β = 0 (infinite temperature)
        double beta = 0.0;
        
        // Initial energy expectation
        Complex e0 = expectationValue(state, H, N);
        double var0 = varianceOperator(state, H, N);
        
        sample_energy[sample][0] = std::real(e0) / num_sites;
        sample_energy_sq[sample][0] = std::real(e0 * e0) / (num_sites * num_sites);
        
        // Save initial state if requested
        if (save_states && !dir.empty()) {
            std::string state_file = dir + "/mc_tpq_state_sample" + 
                                     std::to_string(sample) + "_step0.bin";
            std::ofstream outfile(state_file, std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<char*>(state.data()), N * sizeof(Complex));
                outfile.close();
            }
        }
        
        // Evolve and measure
        for (int step = 1; step < num_steps; step++) {
            // Evolve the state in time (imaginary time for canonical ensemble)
            timeEvolveState(H, time_step, emin, emax, state, N);
            
            // Update effective beta
            beta += time_step;
            
            // Measure energy
            Complex energy = expectationValue(state, H, N);
            Complex energy_squared = expectationValueSquared(state, H, N);
            
            // Store normalized results
            sample_energy[sample][step] = std::real(energy) / num_sites;
            sample_energy_sq[sample][step] = std::real(energy_squared) / (num_sites * num_sites);
            
            // Save evolved state if requested
            if (save_states && !dir.empty()) {
                std::string state_file = dir + "/mc_tpq_state_sample" + 
                                         std::to_string(sample) + "_step" + 
                                         std::to_string(step) + ".bin";
                std::ofstream outfile(state_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(state.data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
        }
        
        // Progress report
        #pragma omp critical
        {
            std::cout << "Sample " << sample+1 << "/" << num_samples << " completed." << std::endl;
        }
    }
    
    // Compute averages and thermodynamic quantities
    for (int step = 0; step < num_steps; step++) {
        // Beta values
        results.beta_values[step] = step * time_step;
        
        // Compute average energy across samples
        double avg_e = 0.0;
        double avg_e_sq = 0.0;
        
        for (int sample = 0; sample < num_samples; sample++) {
            avg_e += sample_energy[sample][step];
            avg_e_sq += sample_energy_sq[sample][step];
        }
        
        avg_e /= num_samples;
        avg_e_sq /= num_samples;
        
        // Energy per site
        results.energy[step] = avg_e;
        
        // Specific heat per site: C = β² * (<E²> - <E>²)
        double beta = results.beta_values[step];
        if (beta > 1e-10) { // Avoid division by zero
            results.specific_heat[step] = beta * beta * (avg_e_sq - avg_e * avg_e);
        }
        
        // For entropy and free energy, use numerical integration
        if (step > 0) {
            double dbeta = results.beta_values[step] - results.beta_values[step-1];
            
            // Entropy per site: S = S_0 + ∫(0→β) C(β')/β' dβ'
            // Use trapezoid rule for integration
            if (beta > 1e-10) {
                double integrand = 0.5 * (
                    results.specific_heat[step] / beta + 
                    results.specific_heat[step-1] / results.beta_values[step-1]
                );
                results.entropy[step] = results.entropy[step-1] + integrand * dbeta;
            } else {
                // At β=0, entropy is just log(d) where d is local Hilbert space dimension
                results.entropy[step] = std::log(std::pow(2, num_sites)) / num_sites;
            }
            
            // Free energy per site: F = E - TS
            results.free_energy[step] = results.energy[step] - 
                                       (beta > 1e-10 ? results.entropy[step] / beta : 0.0);
        } else {
            // Initial values at infinite temperature
            results.entropy[0] = std::log(std::pow(2, num_sites)) / num_sites;
            results.free_energy[0] = results.energy[0];
        }
    }
    
    // Save results to file
    if (!dir.empty()) {
        std::string results_file = dir + "/microcanonical_tpq_results.dat";
        results.save_to_file(results_file);
        
        std::cout << "Results saved to " << results_file << std::endl;
    }
    
    return results;
}

// Canonical TPQ method
// Computes thermodynamic quantities directly at specified temperatures
TPQThermodynamicData canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H, 
    int N,                          // Hilbert space dimension
    int num_sites,                  // Number of physical sites
    int num_samples = 10,           // Number of random samples
    double beta_min = 0.01,         // Minimum inverse temperature
    double beta_max = 10.0,         // Maximum inverse temperature
    int num_beta = 50,              // Number of temperature points
    std::string dir = "",           // Directory for output
    bool save_states = false        // Whether to save states
) {
    std::cout << "Starting Canonical TPQ with " << num_samples 
              << " samples across " << num_beta << " temperature points" << std::endl;
    
    // Create output directory
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    
    // Estimate spectral bounds
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v_rand(N);
    for (int i = 0; i < N; i++) {
        v_rand[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, v_rand.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_rand.data(), 1);
    
    // Power iteration to estimate largest eigenvalue
    const int power_steps = 20;
    ComplexVector Hv(N);
    for (int i = 0; i < power_steps; i++) {
        H(v_rand.data(), Hv.data(), N);
        std::swap(v_rand, Hv);
        
        norm = cblas_dznrm2(N, v_rand.data(), 1);
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v_rand.data(), 1);
    }
    
    // Apply H one more time for Rayleigh quotient
    H(v_rand.data(), Hv.data(), N);
    Complex rayleigh_quotient;
    cblas_zdotc_sub(N, v_rand.data(), 1, Hv.data(), 1, &rayleigh_quotient);
    
    // Estimate spectral bounds
    double lambda_max = std::real(rayleigh_quotient);
    double safety_factor = 1.1;  // Add 10% margin
    double emax = lambda_max * safety_factor;
    double emin = -emax;         // Assumes Hamiltonian is centered around zero
    
    std::cout << "Estimated spectral bounds [" << emin << ", " << emax << "]" << std::endl;
    
    // Generate logarithmically spaced beta values
    TPQThermodynamicData results;
    results.beta_values.resize(num_beta);
    
    if (beta_min <= 0.0) beta_min = 0.01;  // Avoid zero or negative beta
    
    // Populate beta values (logarithmic spacing)
    double log_beta_min = std::log(beta_min);
    double log_beta_max = std::log(beta_max);
    double log_beta_step = (log_beta_max - log_beta_min) / (num_beta - 1);
    
    for (int i = 0; i < num_beta; i++) {
        results.beta_values[i] = std::exp(log_beta_min + i * log_beta_step);
    }
    
    // Initialize results
    results.energy.resize(num_beta, 0.0);
    results.specific_heat.resize(num_beta, 0.0);
    results.entropy.resize(num_beta, 0.0);
    results.free_energy.resize(num_beta, 0.0);
    
    // Data for each sample
    std::vector<std::vector<double>> sample_energy(num_samples, std::vector<double>(num_beta));
    std::vector<std::vector<double>> sample_energy_sq(num_samples, std::vector<double>(num_beta));
    
    // Computing thermal averages
    #pragma omp parallel for schedule(dynamic)
    for (int sample = 0; sample < num_samples; sample++) {
        // Generate random initial state
        std::mt19937 local_gen(std::random_device{}() + sample); // Different seed for each sample
        
        for (int b = 0; b < num_beta; b++) {
            double beta = results.beta_values[b];
            
            // Create random state
            ComplexVector state = generateRandomQuantumState(N, local_gen);
            
            // Apply exp(-βH/2) to state to create canonical TPQ state
            applyExpBetaH(H, beta, emin, emax, state, N);
            
            // Measure energy
            Complex energy = expectationValue(state, H, N);
            Complex energy_squared = expectationValueSquared(state, H, N);
            
            // Store normalized results
            sample_energy[sample][b] = std::real(energy) / num_sites;
            sample_energy_sq[sample][b] = std::real(energy_squared) / (num_sites * num_sites);
            
            // Save state if requested
            if (save_states && !dir.empty()) {
                std::string state_file = dir + "/canonical_tpq_state_sample" + 
                                        std::to_string(sample) + "_beta" + 
                                        std::to_string(beta) + ".bin";
                std::ofstream outfile(state_file, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<char*>(state.data()), N * sizeof(Complex));
                    outfile.close();
                }
            }
        }
        
        // Progress report
        #pragma omp critical
        {
            std::cout << "Sample " << sample+1 << "/" << num_samples << " completed." << std::endl;
        }
    }
    
    // Compute averages and thermodynamic quantities
    for (int b = 0; b < num_beta; b++) {
        double beta = results.beta_values[b];
        
        // Compute average energy across samples
        double avg_e = 0.0;
        double avg_e_sq = 0.0;
        
        for (int sample = 0; sample < num_samples; sample++) {
            avg_e += sample_energy[sample][b];
            avg_e_sq += sample_energy_sq[sample][b];
        }
        
        avg_e /= num_samples;
        avg_e_sq /= num_samples;
        
        // Energy per site
        results.energy[b] = avg_e;
        
        // Specific heat per site: C = β² * var(E)
        results.specific_heat[b] = beta * beta * (avg_e_sq - avg_e * avg_e);
        
        // For entropy and free energy, use thermodynamic relations
        if (b > 0) {
            // Calculate entropy using numerical integration
            // S = S_0 + ∫(β0→β) β*dE
            double beta_prev = results.beta_values[b-1];
            double dbeta = beta - beta_prev;
            double avg_beta = 0.5 * (beta + beta_prev);
            double dE = results.energy[b] - results.energy[b-1];
            
            results.entropy[b] = results.entropy[b-1] + avg_beta * dE;
            
            // Free energy: F = E - TS = E - S/β
            results.free_energy[b] = results.energy[b] - results.entropy[b] / beta;
        } else {
            // For the first point (highest temperature), entropy is approximately log(dim)
            results.entropy[0] = std::log(std::pow(2, num_sites)) / num_sites;
            results.free_energy[0] = results.energy[0] - results.entropy[0] / beta;
        }
    }
    
    // Fix entropy normalization
    // At infinite temperature, S = log(dim)/N
    double s_inf = std::log(std::pow(2, num_sites)) / num_sites;
    double s_correction = s_inf - results.entropy[0];
    
    for (int b = 0; b < num_beta; b++) {
        results.entropy[b] += s_correction;
        results.free_energy[b] = results.energy[b] - 
                               results.entropy[b] / results.beta_values[b];
    }
    
    // Save results to file
    if (!dir.empty()) {
        std::string results_file = dir + "/canonical_tpq_results.dat";
        results.save_to_file(results_file);
        
        std::cout << "Results saved to " << results_file << std::endl;
    }
    
    return results;
}

// Compute thermodynamic quantities from a canonical TPQ state at a specific beta
void compute_tpq_observables(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int num_sites,
    double beta,
    ComplexVector& state,
    std::vector<std::function<void(const Complex*, Complex*, int)>> observables = {},
    std::string output_file = ""
) {
    // Energy observables
    Complex energy = expectationValue(state, H, N);
    Complex energy_sq = expectationValueSquared(state, H, N);
    
    double e_per_site = std::real(energy) / num_sites;
    double var_e = std::real(energy_sq) - std::real(energy * energy);
    double c_v = beta * beta * var_e / num_sites;
    
    // Compute other observables if provided
    std::vector<Complex> obs_values;
    for (auto& obs : observables) {
        obs_values.push_back(expectationValue(state, obs, N));
    }
    
    // Output results
    std::cout << "TPQ observables at beta = " << beta << ":" << std::endl;
    std::cout << "  Energy per site: " << e_per_site << std::endl;
    std::cout << "  Specific heat per site: " << c_v << std::endl;
    
    for (size_t i = 0; i < obs_values.size(); i++) {
        std::cout << "  Observable " << i << ": " << obs_values[i] << std::endl;
    }
    
    // Save to file if requested
    if (!output_file.empty()) {
        std::ofstream file(output_file);
        if (file) {
            file << "# TPQ observables at beta = " << beta << std::endl;
            file << "# Energy_per_site Specific_heat_per_site" << std::endl;
            file << std::setprecision(12) << e_per_site << " " << c_v << std::endl;
            
            if (!obs_values.empty()) {
                file << "# Additional observables" << std::endl;
                for (const auto& val : obs_values) {
                    file << std::real(val) << " " << std::imag(val) << std::endl;
                }
            }
            file.close();
        }
    }
}

// Generate canonical TPQ state at specified beta
ComplexVector generate_canonical_tpq_state(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double beta,
    double emin,
    double emax,
    unsigned long seed = 0
) {
    // Generate random initial state
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    ComplexVector state = generateRandomQuantumState(N, gen);
    
    // Apply exp(-βH/2) to create canonical TPQ state
    applyExpBetaH(H, beta, emin, emax, state, N);
    
    return state;
}

#endif // TPQ_H