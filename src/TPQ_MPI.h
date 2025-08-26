// TPQ_MPI.h - MPI-distributed Thermal Pure Quantum state implementation

#ifndef TPQ_MPI_H
#define TPQ_MPI_H

#include <mpi.h>
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
#include <memory>
#include <sys/stat.h>
#include "construct_ham.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using Complex = std::complex<double>;

/**
 * MPI-distributed complex vector class
 * Each rank owns a contiguous chunk of the full vector
 */
class DistributedComplexVector {
private:
    std::vector<Complex> local_data;
    int global_size;
    int local_size;
    int local_start;
    int rank;
    int nprocs;
    MPI_Comm comm;

public:
    DistributedComplexVector(int N, MPI_Comm comm_in = MPI_COMM_WORLD) 
        : global_size(N), comm(comm_in) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nprocs);
        
        // Distribute N elements across nprocs ranks
        local_size = N / nprocs;
        int remainder = N % nprocs;
        
        // Give extra elements to first 'remainder' ranks
        if (rank < remainder) {
            local_size++;
            local_start = rank * local_size;
        } else {
            local_start = remainder * (local_size + 1) + (rank - remainder) * local_size;
        }
        
        local_data.resize(local_size);
    }
    
    // Access local elements
    Complex& operator[](int i) { return local_data[i]; }
    const Complex& operator[](int i) const { return local_data[i]; }
    
    // Get local data pointer for BLAS operations
    Complex* data() { return local_data.data(); }
    const Complex* data() const { return local_data.data(); }
    
    int size() const { return local_size; }
    int global_size_val() const { return global_size; }
    int local_start_val() const { return local_start; }
    
    // Get element at global index (only if owned by this rank)
    Complex get_global(int global_idx) const {
        if (global_idx >= local_start && global_idx < local_start + local_size) {
            return local_data[global_idx - local_start];
        }
        return Complex(0.0, 0.0); // Not owned by this rank
    }
    
    // Set element at global index (only if owned by this rank)
    void set_global(int global_idx, const Complex& val) {
        if (global_idx >= local_start && global_idx < local_start + local_size) {
            local_data[global_idx - local_start] = val;
        }
    }
    
    // Check if global index is owned by this rank
    bool owns_global(int global_idx) const {
        return (global_idx >= local_start && global_idx < local_start + local_size);
    }
    
    // Clear vector
    void clear() {
        std::fill(local_data.begin(), local_data.end(), Complex(0.0, 0.0));
    }
    
    // Normalize the distributed vector
    void normalize() {
        double norm = compute_norm();
        if (norm > 0.0) {
            Complex scale_factor = Complex(1.0/norm, 0.0);
            cblas_zscal(local_size, &scale_factor, local_data.data(), 1);
        }
    }
    
    // Compute global norm
    double compute_norm() const {
        double local_norm_squared = cblas_dznrm2(local_size, local_data.data(), 1);
        local_norm_squared *= local_norm_squared;
        
        double global_norm_squared;
        MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, comm);
        
        return std::sqrt(global_norm_squared);
    }
    
    // Compute global dot product with another distributed vector
    Complex dot_product(const DistributedComplexVector& other) const {
        Complex local_dot(0.0, 0.0);
        for (int i = 0; i < local_size; i++) {
            local_dot += std::conj(local_data[i]) * other.local_data[i];
        }
        
        Complex global_dot;
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
        
        return global_dot;
    }
    
    // Scale vector by a complex number
    void scale(const Complex& factor) {
        cblas_zscal(local_size, &factor, local_data.data(), 1);
    }
    
    // Add another vector: this = this + alpha * other
    void axpy(const Complex& alpha, const DistributedComplexVector& other) {
        cblas_zaxpy(local_size, &alpha, other.data(), 1, local_data.data(), 1);
    }
    
    // Copy from another vector
    void copy_from(const DistributedComplexVector& other) {
        std::copy(other.local_data.begin(), other.local_data.end(), local_data.begin());
    }
    
    // Get rank distribution info
    std::vector<int> get_rank_sizes() const {
        std::vector<int> sizes(nprocs);
        MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
        return sizes;
    }
    
    std::vector<int> get_rank_starts() const {
        std::vector<int> starts(nprocs);
        MPI_Allgather(&local_start, 1, MPI_INT, starts.data(), 1, MPI_INT, comm);
        return starts;
    }
};

/**
 * Generate a random normalized distributed vector for TPQ initial state
 */
DistributedComplexVector generateTPQVector_MPI(int N, unsigned int seed, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    DistributedComplexVector v(N, comm);
    
    // Use rank-dependent seed to ensure different random numbers on each rank
    std::mt19937 gen(seed + rank * 12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < v.size(); i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v[i] = Complex(real, imag);
    }
    
    v.normalize();
    return v;
}

/**
 * MPI-aware Hamiltonian application wrapper
 * Takes a local Hamiltonian function and handles halo exchanges for cross-rank terms
 */
class DistributedHamiltonian {
private:
    std::function<void(const Complex*, Complex*, int, int, int)> local_H;
    MPI_Comm comm;
    int rank, nprocs;
    int num_sites;
    
    // Buffers for halo exchange
    std::vector<Complex> send_buffer;
    std::vector<Complex> recv_buffer;
    std::vector<int> send_indices;
    std::vector<int> recv_indices;
    
public:
    DistributedHamiltonian(
        std::function<void(const Complex*, Complex*, int, int, int)> H_local,
        int N_sites,
        MPI_Comm comm_in = MPI_COMM_WORLD
    ) : local_H(H_local), num_sites(N_sites), comm(comm_in) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nprocs);
    }
    
    void apply(const DistributedComplexVector& v_in, DistributedComplexVector& v_out) {
        // Clear output vector
        v_out.clear();
        
        // Apply local Hamiltonian terms
        local_H(v_in.data(), v_out.data(), v_in.size(), v_in.local_start_val(), num_sites);
        
        // TODO: Add halo exchange for cross-rank terms (e.g., nearest neighbor terms)
        // For now, assume Hamiltonian is sufficiently local or implement specific exchange pattern
        // based on the lattice structure and interaction range
    }
};

/**
 * Distributed Krylov time evolution: psi <- exp(-i H dt) psi
 * Uses an Arnoldi process to build an m-dimensional Krylov subspace and
 * computes the small matrix exponential with Eigen.
 */
inline void time_evolve_krylov_distributed(
    DistributedHamiltonian& H,
    DistributedComplexVector& psi,
    int m,
    double dt,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    int rank; MPI_Comm_rank(comm, &rank);
    const int N = psi.global_size_val();
    m = std::max(1, m);
    m = std::min(m, N); // bound m

    // Ensure psi is normalized
    psi.normalize();

    // Allocate Krylov basis V (m+1 vectors), Hessenberg Hm (m+1 x m)
    std::vector<DistributedComplexVector> V;
    V.reserve(m + 1);
    for (int j = 0; j < m + 1; ++j) V.emplace_back(N, comm);
    // V0 = psi
    V[0].copy_from(psi);

    Eigen::MatrixXcd Hm = Eigen::MatrixXcd::Zero(m + 1, m);

    DistributedComplexVector w(N, comm);

    // Arnoldi
    for (int j = 0; j < m; ++j) {
        // w = H * V[j]
        H.apply(V[j], w);

        // Modified Gram-Schmidt
        for (int i = 0; i <= j; ++i) {
            Complex hij = V[i].dot_product(w);
            Hm(i, j) = hij;
            // w = w - hij * V[i]
            Complex minus_hij = Complex(-hij.real(), -hij.imag());
            w.axpy(minus_hij, V[i]);
        }
        // h_{j+1,j} = ||w||
        double norm_w = w.compute_norm();
        Hm(j + 1, j) = norm_w;
        if (norm_w < 1e-14) {
            // happy breakdown
            // shrink matrices accordingly
            int new_m = j + 1;
            Hm.conservativeResize(new_m + 1, new_m);
            // Recreate V to avoid default construction on resize
            std::vector<DistributedComplexVector> V_new;
            V_new.reserve(new_m + 1);
            for (int k = 0; k < new_m + 1; ++k) {
                V_new.emplace_back(N, comm);
                V_new[k].copy_from(V[k]);
            }
            V.swap(V_new);
            m = new_m;
            break;
        }
        // v_{j+1} = w / h_{j+1,j}
        Complex inv = Complex(1.0 / norm_w, 0.0);
        V[j + 1].copy_from(w);
        V[j + 1].scale(inv);
    }

    // Compute y = exp(-i dt Hm(0:m,0:m)) * e1
    int dim = m; // use upper m x m block
    Eigen::MatrixXcd Hm_square = Hm.topLeftCorner(dim, dim);
    std::complex<double> I(0.0, 1.0);
    Eigen::VectorXcd e1 = Eigen::VectorXcd::Zero(dim);
    e1(0) = 1.0;
    Eigen::MatrixXcd expA = (-I * dt * Hm_square).exp();
    Eigen::VectorXcd y = expA * e1;

    // psi = sum_{j=0}^{dim-1} y_j * V[j]
    // Start with psi = y_0 * V[0]
    psi.copy_from(V[0]);
    psi.scale(Complex(y(0).real(), y(0).imag()));
    for (int j = 1; j < dim; ++j) {
        Complex coeff(y(j).real(), y(j).imag());
        psi.axpy(coeff, V[j]);
    }
    // Optional renormalize to counter numerical drift
    psi.normalize();
}

/**
 * Calculate energy and variance for a distributed TPQ state
 */
std::pair<double, double> calculateEnergyAndVariance_MPI(
    DistributedHamiltonian& H,
    const DistributedComplexVector& v,
    DistributedComplexVector& Hv,
    DistributedComplexVector& H2v
) {
    // Calculate H|v⟩
    H.apply(v, Hv);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = v.dot_product(Hv);
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩
    H.apply(Hv, H2v);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = v.dot_product(H2v);
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

/**
 * Create directory if it doesn't exist (only on rank 0)
 */
bool ensureDirectoryExists_MPI(const std::string& path, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    bool success = true;
    if (rank == 0) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            // Directory doesn't exist, create it
            std::string cmd = "mkdir -p " + path;
            success = (system(cmd.c_str()) == 0);
        } else if (!(info.st_mode & S_IFDIR)) {
            // Path exists but is not a directory
            success = false;
        }
    }
    
    // Broadcast result to all ranks
    int success_int = success ? 1 : 0;
    MPI_Bcast(&success_int, 1, MPI_INT, 0, comm);
    
    return (success_int == 1);
}

/**
 * Write TPQ data to file (only on rank 0)
 */
void writeTPQData_MPI(const std::string& filename, double inv_temp, double energy, 
                     double variance, double norm, int step, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (rank == 0) {
        std::ofstream file(filename, std::ios::app);
        if (file.is_open()) {
            file << std::setprecision(16) << inv_temp << " " << energy << " " 
                 << variance << " " << norm << " " << step << std::endl;
            file.close();
        }
    }
}

/**
 * Main MPI-distributed microcanonical TPQ function
 */
void microcanonical_tpq_mpi(
    std::function<void(const Complex*, Complex*, int, int, int)> H_local,
    int N, 
    int num_sites,
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double LargeValue = 1e5,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    // Only rank 0 prints status messages
    auto print_status = [&](const std::string& msg) {
        if (rank == 0) {
            std::cout << msg << std::endl;
        }
    };
    
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists_MPI(dir, comm);
    }
    
    eigenvalues.clear();
    
    // Create distributed Hamiltonian
    DistributedHamiltonian H(H_local, num_sites, comm);
    
    print_status("Setting LargeValue: " + std::to_string(LargeValue));
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        print_status("TPQ sample " + std::to_string(sample+1) + " of " + std::to_string(num_samples));
        
        // Setup filenames
        std::string ss_file = dir + "/tpq_data_" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/tpq_norm_" + std::to_string(sample) + ".dat";
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        DistributedComplexVector v1 = generateTPQVector_MPI(N, seed, comm);
        
        // Create working vectors
        DistributedComplexVector v0(N, comm);
        DistributedComplexVector Hv(N, comm);
        DistributedComplexVector H2v(N, comm);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        H.apply(v1, v0);
        
        // For each element, compute v0 = (L-H)|v1⟩ = Lv1 - v0
        for (int i = 0; i < v0.size(); i++) {
            v0[i] = (LargeValue * num_sites * v1[i]) - v0[i];
        }
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance_MPI(H, v0, Hv, H2v);
        inv_temp = (2.0) / (LargeValue * num_sites - energy1);
        
        double first_norm = v0.compute_norm();
        v0.normalize();
        double current_norm = first_norm;
        
        writeTPQData_MPI(ss_file, inv_temp, energy1, variance1, current_norm, step, comm);
        
        if (rank == 0) {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                print_status("  Step " + std::to_string(step) + " of " + std::to_string(max_iter));
            }
            
            // Compute v1 = H|v0⟩
            H.apply(v0, v1);
            
            // For each element, compute v0 = (L-H)|v0⟩ = L*v0 - v1
            for (int i = 0; i < v0.size(); i++) {
                v0[i] = (LargeValue * num_sites * v0[i]) - v1[i];
            }
            
            current_norm = v0.compute_norm();
            v0.normalize();
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance_MPI(H, v0, Hv, H2v);
            
            // Update inverse temperature
            inv_temp = (2.0*step) / (LargeValue * num_sites - energy_step);
            
            // Write data
            writeTPQData_MPI(ss_file, inv_temp, energy_step, variance_step, current_norm, step, comm);
            
            if (rank == 0) {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            energy1 = energy_step;
        }
        
        // Store final energy
        eigenvalues.push_back(energy1);
    }
    
    print_status("MPI TPQ calculation completed");
}

/**
 * Save distributed TPQ state to file for later analysis
 */
void save_distributed_state(const DistributedComplexVector& state, 
                           const std::string& filename,
                           MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    // Gather vector sizes and starting positions
    auto sizes = state.get_rank_sizes();
    auto starts = state.get_rank_starts();
    
    if (rank == 0) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write header
        int global_size = state.global_size_val();
        file.write(reinterpret_cast<const char*>(&global_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&nprocs), sizeof(int));
        
        // Write rank 0's data
        file.write(reinterpret_cast<const char*>(state.data()), sizes[0] * sizeof(Complex));
        
        // Receive and write data from other ranks
        for (int r = 1; r < nprocs; r++) {
            std::vector<Complex> buffer(sizes[r]);
            MPI_Recv(buffer.data(), sizes[r] * 2, MPI_DOUBLE, r, 0, comm, MPI_STATUS_IGNORE);
            file.write(reinterpret_cast<const char*>(buffer.data()), sizes[r] * sizeof(Complex));
        }
        
        file.close();
    } else {
        // Send data to rank 0
        MPI_Send(state.data(), state.size() * 2, MPI_DOUBLE, 0, 0, comm);
    }
}

/**
 * Load distributed TPQ state from file
 */
DistributedComplexVector load_distributed_state(const std::string& filename,
                                               MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    int global_size, file_nprocs;
    
    if (rank == 0) {
        std::ifstream file(filename, std::ios::binary);
        file.read(reinterpret_cast<char*>(&global_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&file_nprocs), sizeof(int));
        file.close();
    }
    
    // Broadcast header info
    MPI_Bcast(&global_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&file_nprocs, 1, MPI_INT, 0, comm);
    
    // Create distributed vector
    DistributedComplexVector state(global_size, comm);
    
    // TODO: Implement redistribution if file_nprocs != nprocs
    // For now, assume same number of processes
    
    if (rank == 0) {
        std::ifstream file(filename, std::ios::binary);
        file.seekg(2 * sizeof(int)); // Skip header
        
        // Read and distribute data
        for (int r = 0; r < nprocs; r++) {
            auto sizes = state.get_rank_sizes();
            std::vector<Complex> buffer(sizes[r]);
            file.read(reinterpret_cast<char*>(buffer.data()), sizes[r] * sizeof(Complex));
            
            if (r == 0) {
                // Copy to local data
                std::copy(buffer.begin(), buffer.end(), const_cast<Complex*>(state.data()));
            } else {
                // Send to other rank
                MPI_Send(buffer.data(), sizes[r] * 2, MPI_DOUBLE, r, 0, comm);
            }
        }
        
        file.close();
    } else {
        // Receive data from rank 0
        MPI_Recv(const_cast<Complex*>(state.data()), state.size() * 2, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    }
    
    return state;
}

#endif // TPQ_MPI_H
