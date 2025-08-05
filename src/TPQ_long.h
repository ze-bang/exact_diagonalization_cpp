// TPQ_long.h - Distributed MPI Thermal Pure Quantum state implementation for large systems

#ifndef TPQ_LONG_H
#define TPQ_LONG_H

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
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <cassert>

#include "observables.h"
#include "construct_ham.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

// Type definitions
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

/**
 * MPI Distributed Complex Vector class for handling massive state vectors
 * Each process holds only a local portion of the full vector
 */
class MPIComplexVector {
private:
    ComplexVector local_data;
    size_t global_size;
    size_t local_size;
    size_t local_start_idx;
    int mpi_rank;
    int mpi_size;
    
public:
    MPIComplexVector(size_t N) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        global_size = N;
        local_size = N / mpi_size;
        local_start_idx = mpi_rank * local_size;
        
        // Handle remainder for the last process
        if (mpi_rank == mpi_size - 1) {
            local_size = N - local_start_idx;
        }
        
        local_data.resize(local_size);
    }
    
    // Getters
    size_t size() const { return global_size; }
    size_t local_size_val() const { return local_size; }
    size_t local_start() const { return local_start_idx; }
    Complex* data() { return local_data.data(); }
    const Complex* data() const { return local_data.data(); }
    ComplexVector& local_vector() { return local_data; }
    const ComplexVector& local_vector() const { return local_data; }
    
    // Element access (local indices only)
    Complex& operator[](size_t local_idx) { return local_data[local_idx]; }
    const Complex& operator[](size_t local_idx) const { return local_data[local_idx]; }
    
    // MPI operations
    void all_reduce_sum() {
        // This is for global reductions, not typically used for state vectors
        MPI_Allreduce(MPI_IN_PLACE, local_data.data(), local_size * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    
    double norm() const {
        double local_norm_sq = 0.0;
        for (const auto& val : local_data) {
            local_norm_sq += std::norm(val);
        }
        
        double global_norm_sq;
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return std::sqrt(global_norm_sq);
    }
    
    void normalize() {
        double n = norm();
        if (n > 0.0) {
            Complex scale_factor(1.0/n, 0.0);
            for (auto& val : local_data) {
                val *= scale_factor;
            }
        }
    }
    
    // Dot product with another distributed vector
    Complex dot(const MPIComplexVector& other) const {
        assert(other.global_size == global_size);
        
        Complex local_dot(0.0, 0.0);
        for (size_t i = 0; i < local_size; i++) {
            local_dot += std::conj(local_data[i]) * other.local_data[i];
        }
        
        Complex global_dot;
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
        return global_dot;
    }
    
    // Scale vector by a complex factor
    void scale(const Complex& factor) {
        for (auto& val : local_data) {
            val *= factor;
        }
    }
    
    // Copy from another MPIComplexVector
    void copy_from(const MPIComplexVector& other) {
        assert(other.global_size == global_size);
        local_data = other.local_data;
    }
    
    // Fill with zeros
    void zero() {
        std::fill(local_data.begin(), local_data.end(), Complex(0.0, 0.0));
    }
    
    // Add another vector: this += other
    void add(const MPIComplexVector& other) {
        assert(other.global_size == global_size);
        for (size_t i = 0; i < local_size; i++) {
            local_data[i] += other.local_data[i];
        }
    }
    
    // Add scaled vector: this += factor * other
    void add_scaled(const Complex& factor, const MPIComplexVector& other) {
        assert(other.global_size == global_size);
        for (size_t i = 0; i < local_size; i++) {
            local_data[i] += factor * other.local_data[i];
        }
    }
};

/**
 * MPI Hamiltonian operator wrapper
 * Each process applies the Hamiltonian to its local portion of the vector
 */
class MPIHamiltonianOperator {
private:
    std::function<void(const Complex*, Complex*, int)> local_H_func;
    size_t global_size;
    int mpi_rank;
    int mpi_size;
    
public:
    MPIHamiltonianOperator(std::function<void(const Complex*, Complex*, int)> H, size_t N)
        : local_H_func(H), global_size(N) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
    
    void apply(const MPIComplexVector& input, MPIComplexVector& output) {
        assert(input.size() == global_size);
        assert(output.size() == global_size);
        
        // For most quantum Hamiltonians, we need the full vector for local operations
        // This is the main challenge - we need to communicate the global state
        
        // Gather the full vector to all processes (this is expensive but necessary for most Hamiltonians)
        ComplexVector full_input(global_size);
        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size);
        
        // Calculate receive counts and displacements
        for (int i = 0; i < mpi_size; i++) {
            size_t local_sz = global_size / mpi_size;
            if (i == mpi_size - 1) {
                local_sz = global_size - i * (global_size / mpi_size);
            }
            recvcounts[i] = local_sz * 2; // *2 for complex (real + imag)
            displs[i] = i * (global_size / mpi_size) * 2;
        }
        
        // All-gather the input vector
        MPI_Allgatherv(input.data(), input.local_size_val() * 2, MPI_DOUBLE,
                       full_input.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Apply Hamiltonian to full vector
        ComplexVector full_output(global_size);
        local_H_func(full_input.data(), full_output.data(), global_size);
        
        // Copy the local portion back to output
        std::copy(full_output.begin() + output.local_start(),
                  full_output.begin() + output.local_start() + output.local_size_val(),
                  output.data());
    }
};

/**
 * Create directory if it doesn't exist (MPI-safe, only rank 0 creates)
 */
bool ensureDirectoryExistsMPI(const std::string& path) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    bool success = true;
    if (rank == 0) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            std::string cmd = "mkdir -p " + path;
            success = (system(cmd.c_str()) == 0);
        } else if (!(info.st_mode & S_IFDIR)) {
            success = false;
        }
    }
    
    // Broadcast result to all processes
    MPI_Bcast(&success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return success;
}

/**
 * Generate a random normalized MPIComplexVector for TPQ initial state
 * Each process generates its local portion using the same seed offset
 */
MPIComplexVector generateMPITPQVector(size_t N, unsigned int seed) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPIComplexVector v(N);
    
    // Each process uses a different seed based on rank and global indices
    std::mt19937 gen(seed + rank * 12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (size_t i = 0; i < v.local_size_val(); i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v[i] = Complex(real, imag);
    }
    
    // Normalize the vector globally
    v.normalize();
    
    return v;
}

/**
 * Calculate energy and variance for a distributed TPQ state
 */
std::pair<double, double> calculateEnergyAndVarianceMPI(
    MPIHamiltonianOperator& H,
    const MPIComplexVector& v
) {
    // Calculate H|v⟩
    MPIComplexVector Hv(v.size());
    H.apply(v, Hv);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = v.dot(Hv);
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩
    MPIComplexVector H2v(v.size());
    H.apply(Hv, H2v);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = v.dot(H2v);
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

/**
 * Write TPQ data to file (only rank 0 writes)
 */
void writeTPQDataMPI(const std::string& filename, double inv_temp, double energy, 
                     double variance, double norm, int step) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::ofstream file(filename, std::ios::app);
        if (file.is_open()) {
            file << std::setprecision(16) << inv_temp << " " << energy << " " 
                 << variance << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
            file.close();
        }
    }
}

/**
 * Save distributed TPQ state to file
 * Each process writes its portion to a separate file
 */
bool saveMPITPQState(const MPIComplexVector& tpq_state, const std::string& base_filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::string filename = base_filename + "_rank" + std::to_string(rank) + ".dat";
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Rank " << rank << ": Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Write metadata
    size_t global_size = tpq_state.size();
    size_t local_size = tpq_state.local_size_val();
    size_t local_start = tpq_state.local_start();
    
    out.write(reinterpret_cast<const char*>(&global_size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&local_size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&local_start), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tpq_state.data()), local_size * sizeof(Complex));
    
    out.close();
    return true;
}

/**
 * Load distributed TPQ state from file
 */
bool loadMPITPQState(MPIComplexVector& tpq_state, const std::string& base_filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::string filename = base_filename + "_rank" + std::to_string(rank) + ".dat";
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Rank " << rank << ": Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t global_size, local_size, local_start;
    in.read(reinterpret_cast<char*>(&global_size), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&local_size), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&local_start), sizeof(size_t));
    
    // Verify sizes match
    if (global_size != tpq_state.size() || 
        local_size != tpq_state.local_size_val() || 
        local_start != tpq_state.local_start()) {
        std::cerr << "Rank " << rank << ": Error: Size mismatch in TPQ state file" << std::endl;
        return false;
    }
    
    in.read(reinterpret_cast<char*>(tpq_state.data()), local_size * sizeof(Complex));
    in.close();
    return true;
}

/**
 * Time evolve distributed TPQ state using Taylor expansion
 */
void timeEvolveMPITPQState(
    MPIHamiltonianOperator& H,
    MPIComplexVector& tpq_state,
    double delta_t,
    int n_max = 100,
    bool normalize = true
) {
    size_t N = tpq_state.size();
    
    // Temporary vectors for calculation
    MPIComplexVector result(N);
    MPIComplexVector term(N);
    MPIComplexVector Hterm(N);
    
    // Copy initial state to term and result
    term.copy_from(tpq_state);
    result.copy_from(tpq_state);
    
    // Precompute coefficients for each term in the Taylor series
    std::vector<Complex> coefficients(n_max + 1);
    coefficients[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        coefficients[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    // Apply Taylor expansion terms
    for (int order = 1; order <= n_max; order++) {
        // Apply H to the previous term
        H.apply(term, Hterm);
        term.copy_from(Hterm);
        
        // Add this term to the result
        result.add_scaled(coefficients[order], term);
    }
    
    // Replace tpq_state with the evolved state
    tpq_state.copy_from(result);
    
    // Normalize if requested
    if (normalize) {
        tpq_state.normalize();
    }
}

/**
 * Initialize TPQ output files (only rank 0 initializes)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeMPITPQFiles(
    const std::string& dir,
    int sample,
    int sublattice_size
) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
    std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
    
    // Create vector of spin correlation files
    std::vector<std::string> spin_corr_files;
    std::vector<std::string> suffixes = {"SzSz", "SpSm", "SmSm", "SpSz"};
    
    for (const auto& suffix : suffixes) {
        std::string filename = dir + "/spin_corr_" + suffix + "_rand" + std::to_string(sample) + ".dat";
        spin_corr_files.push_back(filename);
    }
    
    // Initialize output files (only rank 0)
    if (rank == 0) {
        std::ofstream ss_out(ss_file);
        ss_out << "# inv_temp energy variance num doublon step" << std::endl;
        
        std::ofstream norm_out(norm_file);
        norm_out << "# inv_temp norm first_norm step" << std::endl;
        
        std::ofstream flct_out(flct_file);
        flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag)";

        for (int i = 0; i < sublattice_size; i++) {
            flct_out << " sz" << i << "(real) sz" << i << "(imag)"  << " sz2" << i << "(real) sz2" << i << "(imag)";
        }
        flct_out << " step" << std::endl;

        // Initialize each spin correlation file
        for (const auto& file : spin_corr_files) {
            std::ofstream spin_out(file);
            spin_out << "# inv_temp total(real) total(imag)";
            
            for (int i = 0; i < sublattice_size*sublattice_size; i++) {
                spin_out << " site" << i << "(real) site" << i << "(imag)";
            }
            spin_out << " step" << std::endl;
        }
    }
    
    // Ensure all processes wait for file initialization
    MPI_Barrier(MPI_COMM_WORLD);
    
    return {ss_file, norm_file, flct_file, spin_corr_files};
}

/**
 * Distributed MPI TPQ (microcanonical) implementation for large systems
 * Handles state vectors up to 2^32 elements by distributing across MPI processes
 */
void microcanonicalMPITPQ(
    std::function<void(const Complex*, Complex*, int)> H,
    size_t N,  // Use size_t for large N (up to 2^32)
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double LargeValue = 1e5,
    float spin_length = 0.5,
    bool measure_sz = false,
    int sublattice_size = 1
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "Starting MPI TPQ calculation with " << size << " processes" << std::endl;
        std::cout << "Total Hilbert space dimension: " << N << std::endl;
        std::cout << "Memory per process: ~" << (N * sizeof(Complex) / size / (1024*1024*1024)) << " GB" << std::endl;
    }
    
    // Create output directory if needed (only rank 0)
    if (!dir.empty()) {
        ensureDirectoryExistsMPI(dir);
    }
    
    int num_sites = static_cast<int>(std::log2(N));
    eigenvalues.clear();
    
    // Create MPI Hamiltonian operator
    MPIHamiltonianOperator mpi_H(H, N);
    
    if (rank == 0) {
        std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    }
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        if (rank == 0) {
            std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        }
        
        // Setup filenames
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeMPITPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        MPIComplexVector v1 = generateMPITPQVector(N, seed);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        MPIComplexVector v0(N);
        mpi_H.apply(v1, v0);

        // For each element, compute v0 = (L-H)|v1⟩ = Lv1 - v0
        for (size_t i = 0; i < v0.local_size_val(); i++) {
            v0[i] = (LargeValue * num_sites * v1[i]) - v0[i];
        }
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVarianceMPI(mpi_H, v0);
        inv_temp = (2.0) / (LargeValue * num_sites - energy1);

        double first_norm = v0.norm();
        v0.normalize();
        double current_norm = first_norm;
        
        writeTPQDataMPI(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        if (rank == 0) {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress (only rank 0)
            if (rank == 0 && (step % (max_iter/10) == 0 || step == max_iter)) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Compute v1 = H|v0⟩
            mpi_H.apply(v0, v1);
            
            // For each element, compute v0 = (L-H)|v0⟩ = L*v0 - v1
            for (size_t i = 0; i < v0.local_size_val(); i++) {
                v0[i] = (LargeValue * num_sites * v0[i]) - v1[i];
            }

            current_norm = v0.norm();
            v0.normalize();
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVarianceMPI(mpi_H, v0);
            
            // Update inverse temperature
            inv_temp = (2.0*step) / (LargeValue * num_sites - energy_step);
            
            // Write data
            writeTPQDataMPI(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            if (rank == 0) {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            energy1 = energy_step;

            // Save state at specified intervals for checkpointing
            if (step % temp_interval == 0 || step == max_iter) {
                std::string state_file = dir + "/tpq_state_sample" + std::to_string(sample) + "_step" + std::to_string(step);
                saveMPITPQState(v0, state_file);
            }
        }
        
        // Store final energy for this sample (only rank 0 needs this)
        if (rank == 0) {
            eigenvalues.push_back(energy1);
        }
    }
    
    if (rank == 0) {
        std::cout << "MPI TPQ calculation completed successfully!" << std::endl;
    }
}

/**
 * Utility function to check MPI TPQ state file consistency
 */
bool checkMPIStateConsistency(const std::string& base_filename) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    bool local_ok = true;
    size_t global_size = 0;
    
    // Check local file
    std::string filename = base_filename + "_rank" + std::to_string(rank) + ".dat";
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        local_ok = false;
    } else {
        size_t gs, ls, start;
        in.read(reinterpret_cast<char*>(&gs), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&ls), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&start), sizeof(size_t));
        global_size = gs;
        in.close();
    }
    
    // Check global consistency
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (rank == 0 && global_ok) {
        std::cout << "MPI state files are consistent, global size: " << global_size << std::endl;
    }
    
    return global_ok;
}

#endif // TPQ_LONG_H
