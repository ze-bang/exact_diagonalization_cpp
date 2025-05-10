// TPQ.h - Thermal Pure Quantum state implementation

#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Constants
const double LargeValue = 1.0e+10;

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v[i] = Complex(real, imag);
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        std::string cmd = "mkdir -p " + path;
        return system(cmd.c_str()) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Path exists and is a directory
        return true;
    } else {
        // Path exists but is not a directory
        return false;
    }
}

/**
 * Calculate energy and variance for a TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    int N
) {
    // Calculate H|v⟩
    ComplexVector Hv(N);
    H(v.data(), Hv.data(), N);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        energy_complex += std::conj(v[i]) * Hv[i];
    }
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩
    ComplexVector H2v(N);
    H(Hv.data(), H2v.data(), N);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        h2_complex += std::conj(v[i]) * H2v[i];
    }
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

/**
 * Write TPQ data to file
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, int step) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(16) << inv_temp << " " << energy << " " 
             << variance << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        file.close();
    }
}

/**
 * Read TPQ data from file
 */
bool readTPQData(const std::string& filename, int step, double& energy, 
                double& temp, double& specificHeat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    double inv_temp, e, var, n, doublon;
    int s;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> inv_temp >> e >> var >> n >> doublon >> s)) {
            continue;
        }
        
        if (s == step) {
            energy = e;
            temp = 1.0/inv_temp;
            specificHeat = (var-e*e)*(inv_temp*inv_temp);
            return true;
        }
    }
    
    return false;
}

/**
 * Standard TPQ (microcanonical) implementation
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_spectrum = false
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    eigenvalues.clear();
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
        std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
        
        // Initialize output files
        {
            std::ofstream ss_out(ss_file);
            ss_out << "# inv_temp energy variance num doublon step" << std::endl;
            
            std::ofstream norm_out(norm_file);
            norm_out << "# inv_temp norm first_norm step" << std::endl;
            
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp num num2 doublon doublon2 sz sz2 step" << std::endl;
        }
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        ComplexVector v0(N);
        H(v1.data(), v0.data(), N);
        
        // Calculate initial energy and norm
        auto [energy, variance] = calculateEnergyAndVariance(H, v1, N);
        double first_norm = cblas_dznrm2(N, v1.data(), 1);
        double current_norm = first_norm;
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        writeTPQData(ss_file, inv_temp, energy, variance, current_norm, 0);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << 0 << std::endl;
        }
        
        // Step 1: Calculate v0 = H|v1⟩
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance(H, v1, N);
        double nsite = N; // This should be the actual number of sites, approximating as N for now
        inv_temp = (2.0 / nsite) / (LargeValue - energy1 / nsite);
        
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
            
            std::ofstream flct_out(flct_file, std::ios::app);
            flct_out << std::setprecision(16) << inv_temp << " " 
                     << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                     << 0.0 << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Store previous v0 as temp
            ComplexVector temp = v0;
            
            // Update v0 = H|v1⟩ - v0
            H(v1.data(), v0.data(), N);
            
            // Update v1 = v1 / ||v1||
            std::swap(v1, temp);
            current_norm = cblas_dznrm2(N, v1.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v1.data(), 1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v1, N);
            
            // Update inverse temperature
            inv_temp = (2.0*step / nsite) / (LargeValue - energy_step / nsite);
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                std::ofstream flct_out(flct_file, std::ios::app);
                flct_out << std::setprecision(16) << inv_temp << " " 
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy);
    }
}

/**
 * Canonical TPQ implementation (using imaginary time evolution)
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param delta_tau Time step for imaginary time evolution
 * @param compute_spectrum Whether to compute spectrum
 */
void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double delta_tau = 0.0, // Default 0 means use 1/LargeValue
    bool compute_spectrum = false
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    // Set default delta_tau if not specified
    if (delta_tau <= 0.0) {
        delta_tau = 1.0/LargeValue;
    }
    
    eigenvalues.clear();
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "Canonical TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
        std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
        
        // Initialize output files
        {
            std::ofstream ss_out(ss_file);
            ss_out << "# inv_temp energy variance num doublon step" << std::endl;
            
            std::ofstream norm_out(norm_file);
            norm_out << "# inv_temp norm first_norm step" << std::endl;
            
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp num num2 doublon doublon2 sz sz2 step" << std::endl;
        }
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        ComplexVector v0 = v1; // In canonical TPQ, v0 starts as v1
        
        // Calculate initial energy and norm
        auto [energy, variance] = calculateEnergyAndVariance(H, v1, N);
        double first_norm = cblas_dznrm2(N, v1.data(), 1);
        double current_norm = first_norm;
        
        // Initial inverse temperature is 0
        double inv_temp = 0.0;
        
        // Write initial state (infinite temperature)
        writeTPQData(ss_file, inv_temp, energy, variance, current_norm, 0);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << 0 << std::endl;
        }
        
        // Main canonical TPQ loop
        for (int step = 1; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Canonical TPQ: Apply exp(-delta_tau*H/2) to v1 using 4th order approximation
            // We'll use a simple first-order approximation for simplicity: v0 = v1 - delta_tau*H*v1
            ComplexVector Hv(N);
            H(v1.data(), Hv.data(), N);
            
            // v0 = v1 - delta_tau*H*v1/2
            for (int i = 0; i < N; i++) {
                v0[i] = v1[i] - Hv[i] * (delta_tau/2.0);
            }
            
            // Normalize v0
            current_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Swap v0 and v1 for next iteration
            std::swap(v0, v1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v1, N);
            
            // Update inverse temperature
            inv_temp += delta_tau;
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                std::ofstream flct_out(flct_file, std::ios::app);
                flct_out << std::setprecision(16) << inv_temp << " " 
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy);
    }
}

/**
 * Calculate spectrum function from TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param tpq_sample Sample index to use from TPQ calculation
 * @param tpq_step TPQ step to use
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param omega_step Step size in frequency domain
 * @param eta Broadening factor
 * @param tpq_dir Directory containing TPQ data
 * @param out_file Output file for spectrum
 */
void calculate_spectrum_from_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int tpq_sample,
    int tpq_step,
    double omega_min,
    double omega_max,
    double omega_step,
    double eta,
    const std::string& tpq_dir,
    const std::string& out_file
) {
    std::cout << "Calculating spectrum from TPQ state..." << std::endl;
    
    // Read TPQ data
    std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(tpq_sample) + ".dat";
    double energy, temp, specificHeat;
    
    if (!readTPQData(ss_file, tpq_step, energy, temp, specificHeat)) {
        std::cerr << "Error: Could not read TPQ data from " << ss_file << std::endl;
        return;
    }
    
    std::cout << "Using TPQ state at step " << tpq_step 
              << ", temperature: " << temp 
              << ", energy: " << energy << std::endl;
    
    // Open output file
    std::ofstream spectrum_file(out_file);
    if (!spectrum_file.is_open()) {
        std::cerr << "Error: Could not open output file " << out_file << std::endl;
        return;
    }
    spectrum_file << "# omega re(spectrum) im(spectrum)" << std::endl;
    
    // Calculate number of frequency points
    int n_omega = static_cast<int>((omega_max - omega_min) / omega_step) + 1;
    
    // Pre-factor for Gaussian broadening
    double pre_factor = 2.0 * temp * temp * specificHeat;
    double factor = 1.0 / sqrt(M_PI * pre_factor);
    
    // Calculate spectrum for each frequency
    for (int i = 0; i < n_omega; i++) {
        double omega = omega_min + i * omega_step;
        Complex z(omega, eta); // Complex frequency with broadening
        
        // This is a simplified version - the full algorithm would perform
        // continued fraction expansion using Lanczos tridiagonalization
        
        // Calculate the spectrum using Gaussian broadening approximation
        double spectrum_val = factor * exp(-pow((omega - energy), 2) / pre_factor);
        
        spectrum_file << std::setprecision(16) 
                     << omega << " " 
                     << spectrum_val << " " 
                     << 0.0 << std::endl;
    }
    
    spectrum_file.close();
    std::cout << "Spectrum calculation complete. Written to " << out_file << std::endl;
}

#endif // TPQ_H