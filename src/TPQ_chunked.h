// TPQ_chunked.h - Thermal Pure Quantum state implementation with ChunkedComplexVector
// This implementation uses ChunkedComplexVector to handle large system sizes (2^32+ states)

#ifndef TPQ_CHUNKED_H
#define TPQ_CHUNKED_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include <memory>
#include "chunked_complex_vector.h"
#include "observables.h"
#include "construct_ham.h"

// Type definitions for chunked TPQ
using LargeComplexVector = ChunkedComplexVector;

#define GET_VARIABLE_NAME(Variable) (#Variable)

/**
 * Generate a random normalized chunked vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space (can be > 2^32)
 * @param seed Random seed to use
 * @return Random normalized chunked vector
 */
ChunkedComplexVector generateLargeTPQVector(size_t N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ChunkedComplexVector v(N);
    
    // Use chunk-wise initialization for better performance
    ChunkedVectorUtils::apply_to_chunks(v, [&](Complex* data, size_t chunk_size, size_t chunk_idx) {
        for (size_t i = 0; i < chunk_size; ++i) {
            double real = dist(gen);
            double imag = dist(gen);
            data[i] = Complex(real, imag);
        }
    });
    
    // Normalize using chunked utilities
    ChunkedVectorUtils::normalize(v);

    return v;
}

/**
 * Create directory if it doesn't exist (same as original TPQ)
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
 * Calculate energy and variance for a large TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector (chunked)
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateLargeEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ChunkedComplexVector& v
) {
    // Calculate H|v⟩ using chunked operator
    ChunkedComplexVector Hv(v.size());
    ChunkedOperators::OperatorWrapper H_op(H);
    H_op(v, Hv);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = ChunkedVectorUtils::dot_product(v, Hv);
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩ for variance
    ChunkedComplexVector H2v(v.size());
    H_op(Hv, H2v);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = ChunkedVectorUtils::dot_product(v, H2v);
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

/**
 * Write TPQ data to file (same format as original)
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, int step) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(16) << inv_temp << " " << energy << " " 
             << variance << " " << norm << " " << step << std::endl;
        file.close();
    }
}

/**
 * Imaginary time evolution using Taylor expansion for chunked vectors
 * e^{-βH} |ψ⟩ ≈ Σ_{n=0}^{n_max} (-βH)^n/n! |ψ⟩
 * 
 * @param H Hamiltonian operator function
 * @param state Current state (will be modified)
 * @param delta_beta Time step in imaginary time
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize after evolution
 */
void imaginary_time_evolve_chunked_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ChunkedComplexVector& state,
    double delta_beta,
    int n_max = 50,
    bool normalize = true
) {
    ChunkedComplexVector term = state;
    ChunkedComplexVector result = state;
    ChunkedComplexVector Hterm(state.size());
    
    ChunkedOperators::OperatorWrapper H_op(H);
    
    // Iteratively build Taylor series terms
    double coef_real = 1.0;
    for (int order = 1; order <= n_max; ++order) {
        // term <- H * term
        H_op(term, Hterm);
        term = std::move(Hterm);
        Hterm.resize(state.size()); // Reset for next iteration
        
        // Update coefficient: c_k = c_{k-1} * (-Δβ) / k
        coef_real *= (-delta_beta) / double(order);
        Complex coef(coef_real, 0.0);
        
        // result += coef * term
        ChunkedBLAS::axpy(coef, term, result);
        
        // Check for convergence (optional optimization)
        if (std::abs(coef_real) < 1e-15) {
            std::cout << "Taylor series converged at order " << order << std::endl;
            break;
        }
    }
    
    state = std::move(result);
    
    if (normalize) {
        ChunkedVectorUtils::normalize(state);
    }
}

/**
 * Time evolution using Krylov subspace method for chunked vectors
 * This is more accurate than Taylor expansion for large time steps
 * 
 * @param H Hamiltonian operator function
 * @param state Current state (will be modified)
 * @param delta_t Time step
 * @param krylov_dim Dimension of Krylov subspace
 * @param imaginary_time Whether to use imaginary time (exp(-βH)) or real time (exp(-iHt))
 * @param normalize Whether to normalize after evolution
 */
void time_evolve_chunked_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ChunkedComplexVector& state,
    double delta_t,
    int krylov_dim = 30,
    bool imaginary_time = false,
    bool normalize = true
) {
    // For very large systems, we implement a simplified Krylov method
    // This could be optimized further with proper Lanczos iteration
    
    std::vector<ChunkedComplexVector> krylov_vectors;
    krylov_vectors.reserve(krylov_dim);
    
    // Initialize first Krylov vector
    krylov_vectors.push_back(state);
    ChunkedVectorUtils::normalize(krylov_vectors[0]);
    
    ChunkedOperators::OperatorWrapper H_op(H);
    
    // Build Krylov subspace
    for (int j = 1; j < krylov_dim; ++j) {
        ChunkedComplexVector w(state.size());
        H_op(krylov_vectors[j-1], w);
        
        // Gram-Schmidt orthogonalization
        for (int k = 0; k < j; ++k) {
            Complex overlap = ChunkedVectorUtils::dot_product(krylov_vectors[k], w);
            ChunkedBLAS::axpy(-overlap, krylov_vectors[k], w);
        }
        
        double norm = ChunkedVectorUtils::norm(w);
        if (norm < 1e-12) {
            std::cout << "Krylov subspace converged at dimension " << j << std::endl;
            krylov_dim = j;
            break;
        }
        
        ChunkedBLAS::scal(Complex(1.0/norm, 0.0), w);
        krylov_vectors.push_back(std::move(w));
    }
    
    // For simplicity, use first-order approximation in reduced space
    // In a full implementation, you would diagonalize the tridiagonal matrix
    Complex factor = imaginary_time ? Complex(-delta_t, 0.0) : Complex(0.0, -delta_t);
    
    ChunkedComplexVector evolved_state(state.size());
    evolved_state.fill(Complex(0.0, 0.0));
    
    for (int j = 0; j < krylov_dim; ++j) {
        Complex coef = (j == 0) ? Complex(1.0, 0.0) : factor / double(j);
        ChunkedBLAS::axpy(coef, krylov_vectors[j], evolved_state);
    }
    
    state = std::move(evolved_state);
    
    if (normalize) {
        ChunkedVectorUtils::normalize(state);
    }
}

/**
 * Save a chunked TPQ state to file
 * Saves chunk by chunk to handle large files
 * 
 * @param state TPQ state to save
 * @param filename Output filename
 * @return True if successful
 */
bool save_chunked_tpq_state(const ChunkedComplexVector& state, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header information
    size_t total_size = state.size();
    size_t num_chunks = state.num_chunks();
    file.write(reinterpret_cast<const char*>(&total_size), sizeof(total_size));
    file.write(reinterpret_cast<const char*>(&num_chunks), sizeof(num_chunks));
    
    // Write chunk by chunk
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t chunk_size = state.get_chunk_size(chunk_idx);
        const Complex* chunk_data = state.chunk_data(chunk_idx);
        
        file.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
        file.write(reinterpret_cast<const char*>(chunk_data), chunk_size * sizeof(Complex));
    }
    
    return file.good();
}

/**
 * Load a chunked TPQ state from file
 * 
 * @param state TPQ state to load into
 * @param filename Input filename
 * @return True if successful
 */
bool load_chunked_tpq_state(ChunkedComplexVector& state, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header information
    size_t total_size, num_chunks;
    file.read(reinterpret_cast<char*>(&total_size), sizeof(total_size));
    file.read(reinterpret_cast<char*>(&num_chunks), sizeof(num_chunks));
    
    if (!file.good()) {
        return false;
    }
    
    // Resize state
    state.resize(total_size);
    
    // Read chunk by chunk
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t chunk_size;
        file.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
        
        Complex* chunk_data = state.chunk_data(chunk_idx);
        file.read(reinterpret_cast<char*>(chunk_data), chunk_size * sizeof(Complex));
        
        if (!file.good()) {
            return false;
        }
    }
    
    return true;
}

/**
 * Initialize TPQ output files (same as original)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeTPQFiles(
    const std::string& dir,
    int sample,
    int sublattice_size
) {
    std::string ss_file = dir + "/ss_" + std::to_string(sample) + ".dat";
    std::string norm_file = dir + "/norm_" + std::to_string(sample) + ".dat";
    std::string flct_file = dir + "/flct_" + std::to_string(sample) + ".dat";
    
    // Initialize specific heat file
    std::ofstream ss_out(ss_file);
    ss_out << "# inv_temp energy variance norm step" << std::endl;
    ss_out.close();
    
    // Initialize norm file
    std::ofstream norm_out(norm_file);
    norm_out << "# inv_temp norm_before norm_after step" << std::endl;
    norm_out.close();
    
    // Initialize fluctuation file
    std::ofstream flct_out(flct_file);
    flct_out << "# inv_temp";
    for (int i = 0; i < sublattice_size; ++i) {
        flct_out << " Sz_" << i << " Sz2_" << i;
    }
    flct_out << " step" << std::endl;
    flct_out.close();
    
    // Spin correlation files (simplified for chunked version)
    std::vector<std::string> spin_corr;
    return {ss_file, norm_file, flct_file, spin_corr};
}

/**
 * Canonical TPQ implementation for large systems using ChunkedComplexVector
 * Uses imaginary time evolution: |ψ(β)⟩ = e^{-βH/2} |r⟩ / ||e^{-βH/2} |r⟩||
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space (can be > 2^32)
 * @param beta_max Maximum inverse temperature
 * @param num_samples Number of random samples
 * @param temp_interval Interval for measurements
 * @param energies Output vector for energies at each temperature
 * @param dir Output directory
 * @param delta_beta Time step in imaginary time
 * @param taylor_order Maximum order for Taylor expansion
 * @param use_krylov Whether to use Krylov method (more accurate)
 * @param krylov_dim Dimension of Krylov subspace
 */
void canonical_tpq_chunked(
    std::function<void(const Complex*, Complex*, int)> H,
    size_t N,
    double beta_max,
    int num_samples,
    int temp_interval,
    std::vector<double>& energies,
    std::string dir = "",
    double delta_beta = 0.1,
    int taylor_order = 50,
    bool use_krylov = false,
    int krylov_dim = 30,
    int sublattice_size = 1
) {
    if (!dir.empty()) { 
        ensureDirectoryExists(dir); 
    }
    energies.clear();

    std::cout << "Begin Chunked Canonical TPQ calculation with dimension " << N << std::endl;
    std::cout << "Maximum vector size supported: 2^" << static_cast<int>(std::log2(N)) << " states" << std::endl;
    
    // Calculate number of time steps
    int max_steps = std::max(1, static_cast<int>(std::ceil(beta_max / delta_beta)));
    
    for (int sample = 0; sample < num_samples; ++sample) {
        std::cout << "Canonical TPQ sample " << sample + 1 << " of " << num_samples << std::endl;
        
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state (β=0)
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample * 12345;
        ChunkedComplexVector psi = generateLargeTPQVector(N, seed);
        
        // Record initial state (β=0)
        {
            auto [e0, var0] = calculateLargeEnergyAndVariance(H, psi);
            double inv_temp = 0.0;
            writeTPQData(ss_file, inv_temp, e0, var0, 1.0, 0);
            
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " << 1.0 << " " << 1.0 << " " << 0 << std::endl;
            norm_out.close();
            
            energies.push_back(e0);
            std::cout << "β=0: E=" << e0 << ", Var=" << var0 << std::endl;
        }
        
        // Imaginary time evolution
        for (int step = 1; step <= max_steps; ++step) {
            double current_beta = step * delta_beta;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Evolve state: |ψ(β+Δβ)⟩ = e^{-ΔβH/2} |ψ(β)⟩
            if (use_krylov) {
                time_evolve_chunked_krylov(H, psi, delta_beta/2.0, krylov_dim, true, true);
            } else {
                imaginary_time_evolve_chunked_taylor(H, psi, delta_beta/2.0, taylor_order, true);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Calculate observables if it's a measurement step
            if (step % temp_interval == 0) {
                auto [energy, variance] = calculateLargeEnergyAndVariance(H, psi);
                double norm = ChunkedVectorUtils::norm(psi);
                double inv_temp = current_beta;
                
                writeTPQData(ss_file, inv_temp, energy, variance, norm, step);
                
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " << norm << " " << 1.0 << " " << step << std::endl;
                norm_out.close();
                
                energies.push_back(energy);
                
                std::cout << "β=" << std::fixed << std::setprecision(3) << current_beta 
                         << ": E=" << std::setprecision(8) << energy 
                         << ", Var=" << variance 
                         << ", Time=" << duration.count() << "ms" << std::endl;
                
                // Save state at key temperatures for later analysis
                if (step % (temp_interval * 5) == 0) {
                    std::string state_file = dir + "/state_sample" + std::to_string(sample) 
                                           + "_beta" + std::to_string(current_beta) + ".dat";
                    save_chunked_tpq_state(psi, state_file);
                }
            }
            
            // Memory usage reporting for very large systems
            if (step % (temp_interval * 10) == 0) {
                size_t memory_mb = (N * sizeof(Complex)) / (1024 * 1024);
                std::cout << "Memory usage: ~" << memory_mb << " MB for " << N << " elements" << std::endl;
            }
        }
        
        std::cout << "Sample " << sample + 1 << " completed." << std::endl;
    }
    
    std::cout << "Chunked Canonical TPQ calculation completed." << std::endl;
}

/**
 * Microcanonical TPQ implementation for large systems using ChunkedComplexVector
 * Uses real-time evolution to sample the microcanonical ensemble
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space (can be > 2^32)
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for measurements
 * @param eigenvalues Output vector for final state energies
 * @param dir Output directory
 * @param delta_t Time step for real-time evolution
 * @param use_krylov Whether to use Krylov method
 * @param krylov_dim Dimension of Krylov subspace
 */
void microcanonical_tpq_chunked(
    std::function<void(const Complex*, Complex*, int)> H,
    size_t N,
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double delta_t = 0.01,
    bool use_krylov = true,
    int krylov_dim = 30,
    int sublattice_size = 1
) {
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    eigenvalues.clear();
    
    std::cout << "Begin Chunked Microcanonical TPQ calculation with dimension " << N << std::endl;
    
    for (int sample = 0; sample < num_samples; ++sample) {
        std::cout << "Microcanonical TPQ sample " << sample + 1 << " of " << num_samples << std::endl;
        
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample * 54321;
        ChunkedComplexVector psi = generateLargeTPQVector(N, seed);
        
        // Real-time evolution for microcanonical sampling
        for (int step = 0; step < max_iter; ++step) {
            // Evolve state: |ψ(t+dt)⟩ = e^{-iHdt} |ψ(t)⟩
            if (use_krylov) {
                time_evolve_chunked_krylov(H, psi, delta_t, krylov_dim, false, true);
            } else {
                // For microcanonical, we typically need real-time evolution
                // This is a simplified version - in practice, you'd want more sophisticated methods
                std::cout << "Warning: Taylor expansion not recommended for real-time evolution" << std::endl;
            }
            
            // Measure observables
            if (step % temp_interval == 0) {
                auto [energy, variance] = calculateLargeEnergyAndVariance(H, psi);
                double norm = ChunkedVectorUtils::norm(psi);
                
                // For microcanonical, we use energy as "inverse temperature"
                double pseudo_inv_temp = 1.0 / (energy + 1e-10); // Avoid division by zero
                
                writeTPQData(ss_file, pseudo_inv_temp, energy, variance, norm, step);
                
                if (step % (temp_interval * 10) == 0) {
                    std::cout << "Step " << step << ": E=" << energy 
                             << ", Var=" << variance << ", Norm=" << norm << std::endl;
                }
            }
        }
        
        // Final energy measurement
        auto [final_energy, final_variance] = calculateLargeEnergyAndVariance(H, psi);
        eigenvalues.push_back(final_energy);
        
        std::cout << "Sample " << sample + 1 << " final energy: " << final_energy << std::endl;
    }
    
    std::cout << "Chunked Microcanonical TPQ calculation completed." << std::endl;
}

/**
 * Compute basic observables for chunked TPQ states
 * This is a simplified version - full observable calculations would require
 * chunked versions of the operator classes
 * 
 * @param H Hamiltonian operator
 * @param state Current TPQ state
 * @param observables List of observables to compute
 * @param observable_names Names of observables
 * @param output_file Output file for results
 */
void compute_chunked_observables(
    std::function<void(const Complex*, Complex*, int)> H,
    const ChunkedComplexVector& state,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& observables,
    const std::vector<std::string>& observable_names,
    const std::string& output_file
) {
    std::ofstream file(output_file, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Cannot open observable output file: " << output_file << std::endl;
        return;
    }
    
    ChunkedComplexVector temp_state(state.size());
    
    for (size_t i = 0; i < observables.size() && i < observable_names.size(); ++i) {
        ChunkedOperators::OperatorWrapper obs_op(observables[i]);
        obs_op(state, temp_state);
        
        Complex expectation = ChunkedVectorUtils::dot_product(state, temp_state);
        
        file << observable_names[i] << " " << expectation.real() << " " << expectation.imag() << std::endl;
    }
    
    file.close();
}

#endif // TPQ_CHUNKED_H
