// gpu_hamiltonian.cu - GPU Hamiltonian operator implementation
#include "gpu_hamiltonian.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>

namespace gpu {

// ============================================================================
// Helper functions
// ============================================================================

__device__ __host__ size_t binomial_coefficient(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k; // Optimization
    
    size_t result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Unrank: convert rank to combination (basis state)
__device__ uint64_t unrank_combination(size_t rank, int n, int k) {
    uint64_t state = 0;
    int bits_set = 0;
    
    for (int i = n - 1; i >= 0; i--) {
        size_t binom = binomial_coefficient(i, k - bits_set);
        if (rank >= binom) {
            state |= (1ULL << i);
            rank -= binom;
            bits_set++;
            if (bits_set == k) break;
        }
    }
    
    return state;
}

// Rank: convert combination (basis state) to rank
__device__ size_t rank_combination(uint64_t state, int n, int k) {
    size_t rank = 0;
    int bits_set = 0;
    
    for (int i = n - 1; i >= 0; i--) {
        if ((state >> i) & 1) {
            rank += binomial_coefficient(i, k - bits_set);
            bits_set++;
            if (bits_set == k) break;
        }
    }
    
    return rank;
}

// Hash function for state lookup
__device__ size_t hash_function_device(uint64_t state, size_t table_size) {
    // Simple multiplicative hash
    const uint64_t HASH_MULT = 11400714819323198485ULL;
    return (state * HASH_MULT) % table_size;
}

// Lookup state in hash table
__device__ int lookup_state_in_hash_table(
    uint64_t state,
    const uint64_t* __restrict__ hash_keys,
    const int* __restrict__ hash_values,
    size_t hash_table_size
) {
    size_t hash = hash_function_device(state, hash_table_size);
    
    // Linear probing
    for (size_t i = 0; i < hash_table_size; i++) {
        size_t idx = (hash + i) % hash_table_size;
        
        if (hash_keys[idx] == state) {
            return hash_values[idx];
        }
        
        // Empty slot means not found
        if (hash_keys[idx] == ULLONG_MAX) {
            return -1;
        }
    }
    
    return -1; // Not found
}

// Apply a single spin operator to a basis state, updating the matrix element accumulator
__device__ inline bool apply_single_operator(
    uint64_t& state,
    int site,
    int op,
    float spin_length,
    cuDoubleComplex& matrix_elem
) {
    switch (op) {
        case 2: {
            double sz = get_sz_eigenvalue(state, site, spin_length);
            matrix_elem = cuCmul(matrix_elem, make_cuDoubleComplex(sz, 0.0));
            return true;
        }
        case 0: {
            device_pair<uint64_t, cuDoubleComplex> result = apply_splus(state, site, spin_length);
            if (result.first == 0) {
                return false;
            }
            matrix_elem = cuCmul(matrix_elem, result.second);
            state = result.first;
            return true;
        }
        case 1: {
            device_pair<uint64_t, cuDoubleComplex> result = apply_sminus(state, site, spin_length);
            if (result.first == 0) {
                return false;
            }
            matrix_elem = cuCmul(matrix_elem, result.second);
            state = result.first;
            return true;
        }
        default:
            return false;
    }
}

// Evaluate an interaction term for a given basis state
__device__ inline bool evaluate_interaction(
    uint64_t state,
    const SpinInteraction& interaction,
    float spin_length,
    uint64_t& target_state,
    cuDoubleComplex& matrix_elem
) {
    target_state = state;
    matrix_elem = interaction.coupling;

    if (fabs(cuCreal(matrix_elem)) < 1e-30 && fabs(cuCimag(matrix_elem)) < 1e-30) {
        return false;
    }

    if (interaction.site2 < 0 || interaction.op2 < 0) {
        return apply_single_operator(target_state, interaction.site1, interaction.op1,
                                      spin_length, matrix_elem);
    }

    if (!apply_single_operator(target_state, interaction.site1, interaction.op1,
                               spin_length, matrix_elem)) {
        return false;
    }

    if (!apply_single_operator(target_state, interaction.site2, interaction.op2,
                               spin_length, matrix_elem)) {
        return false;
    }

    return true;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel: Apply Hamiltonian to full Hilbert space
 * Each thread handles one basis state
 */
__global__ void hamiltonian_apply_kernel(
    const cuDoubleComplex* __restrict__ psi_in,
    cuDoubleComplex* __restrict__ psi_out,
    const SpinInteraction* __restrict__ interactions,
    int num_interactions,
    int num_sites,
    float spin_length,
    size_t hilbert_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hilbert_dim) return;
    
    uint64_t state = idx;
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    // Process each interaction term
    for (int i = 0; i < num_interactions; i++) {
        uint64_t target_state;
        cuDoubleComplex matrix_elem;
        if (evaluate_interaction(state, interactions[i], spin_length, target_state, matrix_elem)) {
            if (target_state < hilbert_dim) {
                cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_state]);
                result = cuCadd(result, contrib);
            }
        }
    }
    
    psi_out[idx] = result;
}

/**
 * Kernel: Apply Hamiltonian to fixed Sz sector
 */
__global__ void hamiltonian_apply_fixed_sz_kernel(
    const cuDoubleComplex* __restrict__ psi_in,
    cuDoubleComplex* __restrict__ psi_out,
    const uint64_t* __restrict__ basis_states,
    const uint64_t* __restrict__ hash_keys,
    const int* __restrict__ hash_values,
    size_t hash_table_size,
    const SpinInteraction* __restrict__ interactions,
    int num_interactions,
    int num_sites,
    float spin_length,
    size_t fixed_sz_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fixed_sz_dim) return;
    
    uint64_t state = basis_states[idx];
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    // Process each interaction term
    for (int i = 0; i < num_interactions; i++) {
        uint64_t target_state;
        cuDoubleComplex matrix_elem;
        if (evaluate_interaction(state, interactions[i], spin_length, target_state, matrix_elem)) {
            int target_idx = (target_state == state)
                ? static_cast<int>(idx)
                : lookup_state_in_hash_table(target_state, hash_keys, hash_values, hash_table_size);

            if (target_idx >= 0) {
                cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_idx]);
                result = cuCadd(result, contrib);
            }
        }
    }
    
    psi_out[idx] = result;
}

/**
 * Kernel: Generate fixed Sz basis
 */
__global__ void generate_fixed_sz_basis_kernel(
    uint64_t* __restrict__ basis_states,
    int num_sites,
    int n_up,
    size_t start_rank,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    size_t rank = start_rank + idx;
    basis_states[idx] = unrank_combination(rank, num_sites, n_up);
}

/**
 * Kernel: Build hash table for state lookup
 */
__global__ void build_hash_table_kernel(
    const uint64_t* __restrict__ basis_states,
    uint64_t* __restrict__ hash_keys,
    int* __restrict__ hash_values,
    size_t basis_dim,
    size_t hash_table_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= basis_dim) return;
    
    uint64_t state = basis_states[idx];
    size_t hash = hash_function_device(state, hash_table_size);
    
    // Linear probing to find empty slot
    for (size_t i = 0; i < hash_table_size; i++) {
        size_t slot = (hash + i) % hash_table_size;
        
        // Try to claim this slot atomically (cast to unsigned long long for atomicCAS)
        unsigned long long old = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
            ULLONG_MAX,
            static_cast<unsigned long long>(state)
        );
        
        if (old == ULLONG_MAX) {
            // Successfully claimed this slot
            hash_values[slot] = idx;
            break;
        }
    }
}

// ============================================================================
// GPUHamiltonianOperator Implementation
// ============================================================================

GPUHamiltonianOperator::GPUHamiltonianOperator(
    int num_sites, 
    float spin_length,
    cublasHandle_t handle
) : num_sites_(num_sites), 
    spin_length_(spin_length),
    hilbert_dim_(1ULL << num_sites),
    d_interactions_(nullptr),
    num_interactions_(0),
    owns_handle_(handle == nullptr)
{
    if (owns_handle_) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    } else {
        cublas_handle_ = handle;
    }
}

GPUHamiltonianOperator::~GPUHamiltonianOperator() {
    if (d_interactions_) {
        cudaFree(d_interactions_);
    }
    if (owns_handle_ && cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

SpinInteraction GPUHamiltonianOperator::parse_interaction_line(const std::string& line) {
    std::istringstream iss(line);
    SpinInteraction inter{};
    
    // Format: site1 op1 site2 op2 coupling
    std::string op1_str, op2_str;
    double coupling_real = 0.0;
    if (!(iss >> inter.site1 >> op1_str >> inter.site2 >> op2_str >> coupling_real)) {
        throw std::runtime_error("Failed to parse interaction line: " + line);
    }
    
    auto map_op = [](const std::string& op) {
        if (op == "S+" || op == "Sp") return 0;
        if (op == "S-" || op == "Sm") return 1;
        if (op == "Sz") return 2;
        if (op == "Sx") return 3;
        if (op == "Sy") return 4;
        return 2;
    };
    
    inter.op1 = map_op(op1_str);
    inter.op2 = map_op(op2_str);
    inter.coupling = make_cuDoubleComplex(coupling_real, 0.0);
    return inter;
}

void GPUHamiltonianOperator::upload_interactions_to_device() {
    num_interactions_ = static_cast<int>(host_interactions_.size());

    if (d_interactions_) {
        cudaFree(d_interactions_);
        d_interactions_ = nullptr;
    }

    if (num_interactions_ == 0) {
        return;
    }

    CUDA_CHECK(cudaMalloc(&d_interactions_,
                          static_cast<size_t>(num_interactions_) * sizeof(SpinInteraction)));
    CUDA_CHECK(cudaMemcpy(d_interactions_, host_interactions_.data(),
                         static_cast<size_t>(num_interactions_) * sizeof(SpinInteraction),
                         cudaMemcpyHostToDevice));
}

void GPUHamiltonianOperator::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<SpinInteraction> new_terms;
    std::string line;

    auto parse_single_site_entry = [&](int op, int idx, const std::complex<double>& coupling) {
        auto append_term = [&](int op_code, const std::complex<double>& coeff) {
            if (std::abs(coeff) < 1e-14) {
                return;
            }
            SpinInteraction inter{};
            inter.site1 = idx;
            inter.site2 = -1;
            inter.op1 = op_code;
            inter.op2 = -1;
            inter.coupling = make_cuDoubleComplex(coeff.real(), coeff.imag());
            new_terms.push_back(inter);
        };

        switch (op) {
            case 0:
            case 1:
            case 2:
                append_term(op, coupling);
                break;
            case 3: { // Sx = (S+ + S-) / 2
                std::complex<double> half = coupling * 0.5;
                append_term(0, half);
                append_term(1, half);
                break;
            }
            case 4: { // Sy = (S+ - S-) / (2i)
                const std::complex<double> coef_plus(0.0, -0.5);
                const std::complex<double> coef_minus(0.0, 0.5);
                append_term(0, coupling * coef_plus);
                append_term(1, coupling * coef_minus);
                break;
            }
            default:
                std::cerr << "Warning: Unsupported single-site operator " << op
                          << " in " << filename << std::endl;
                break;
        }
    };

    // Attempt to parse standard Trans.dat format with header
    file.clear();
    file.seekg(0, std::ios::beg);

    std::string header_line;
    std::string count_line;
    if (std::getline(file, header_line) && std::getline(file, count_line)) {
        std::istringstream iss(count_line);
        std::string label;
        int num_lines = 0;
        if (iss >> label >> num_lines) {
            // Skip next 3 separator lines
            for (int i = 0; i < 3 && std::getline(file, line); ++i) {}

            int line_count = 0;
            while (std::getline(file, line) && line_count < num_lines) {
                if (line.empty() || line[0] == '#') {
                    ++line_count;
                    continue;
                }

                std::istringstream line_stream(line);
                int op = 0;
                int idx = 0;
                double real_part = 0.0;
                double imag_part = 0.0;

                if (line_stream >> op >> idx >> real_part >> imag_part) {
                    std::complex<double> coupling(real_part, imag_part);
                    if (std::abs(coupling) >= 1e-14) {
                        parse_single_site_entry(op, idx, coupling);
                    }
                }
                ++line_count;
            }
        } else {
            // Fallback: treat file as simple whitespace-delimited list without header
            file.clear();
            file.seekg(0, std::ios::beg);
        }
    }

    if (new_terms.empty()) {
        // Parse fallback simple format: op idx real imag
        file.clear();
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream line_stream(line);
            int op = 0;
            int idx = 0;
            double real_part = 0.0;
            double imag_part = 0.0;
            if (!(line_stream >> op >> idx >> real_part >> imag_part)) {
                continue;
            }
            std::complex<double> coupling(real_part, imag_part);
            if (std::abs(coupling) < 1e-14) {
                continue;
            }
            parse_single_site_entry(op, idx, coupling);
        }
    }

    host_interactions_.insert(host_interactions_.end(), new_terms.begin(), new_terms.end());
    upload_interactions_to_device();

    std::cout << "Loaded " << new_terms.size() << " single-site terms from "
              << filename << " (total: " << host_interactions_.size() << ")" << std::endl;
}

void GPUHamiltonianOperator::load_from_interall_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<SpinInteraction> new_terms;
    std::string line;

    auto append_interaction = [&](int op_i, int idx_i, int op_j, int idx_j,
                                  const std::complex<double>& coupling) {
        if (std::abs(coupling) < 1e-14) {
            return;
        }
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            std::cerr << "Warning: Unsupported operator combination (" << op_i << ", "
                      << op_j << ") in " << filename << std::endl;
            return;
        }
        if (idx_i < 0 || idx_i >= num_sites_ || idx_j < 0 || idx_j >= num_sites_) {
            std::cerr << "Warning: Site index out of bounds (" << idx_i << ", "
                      << idx_j << ") in " << filename << std::endl;
            return;
        }
        SpinInteraction inter{};
        inter.site1 = idx_i;
        inter.site2 = idx_j;
        inter.op1 = op_i;
        inter.op2 = op_j;
        inter.coupling = make_cuDoubleComplex(coupling.real(), coupling.imag());
        new_terms.push_back(inter);
    };

    file.clear();
    file.seekg(0, std::ios::beg);

    std::string header_line;
    std::string count_line;
    if (std::getline(file, header_line) && std::getline(file, count_line)) {
        std::istringstream iss(count_line);
        std::string label;
        int num_lines = 0;
        if (iss >> label >> num_lines) {
            for (int i = 0; i < 3 && std::getline(file, line); ++i) {}

            int line_count = 0;
            while (std::getline(file, line) && line_count < num_lines) {
                if (line.empty() || line[0] == '#') {
                    ++line_count;
                    continue;
                }

                std::istringstream line_stream(line);
                int op_i = 0;
                int idx_i = 0;
                int op_j = 0;
                int idx_j = 0;
                double real_part = 0.0;
                double imag_part = 0.0;
                if (line_stream >> op_i >> idx_i >> op_j >> idx_j >> real_part >> imag_part) {
                    append_interaction(op_i, idx_i, op_j, idx_j,
                                       std::complex<double>(real_part, imag_part));
                }
                ++line_count;
            }
        } else {
            file.clear();
            file.seekg(0, std::ios::beg);
        }
    }

    if (new_terms.empty()) {
        file.clear();
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream line_stream(line);
            int op_i = 0;
            int idx_i = 0;
            int op_j = 0;
            int idx_j = 0;
            double real_part = 0.0;
            double imag_part = 0.0;
            if (!(line_stream >> op_i >> idx_i >> op_j >> idx_j >> real_part >> imag_part)) {
                continue;
            }
            append_interaction(op_i, idx_i, op_j, idx_j,
                               std::complex<double>(real_part, imag_part));
        }
    }

    host_interactions_.insert(host_interactions_.end(), new_terms.begin(), new_terms.end());
    upload_interactions_to_device();

    std::cout << "Loaded " << new_terms.size() << " two-site interactions from "
              << filename << " (total: " << host_interactions_.size() << ")" << std::endl;
}

void GPUHamiltonianOperator::apply(const GPUVector& psi_in, GPUVector& psi_out) {
    apply(psi_in.data(), psi_out.data());
}

void GPUHamiltonianOperator::apply(
    const cuDoubleComplex* psi_in, 
    cuDoubleComplex* psi_out
) {
    dim3 block_size = get_optimal_block_size(hilbert_dim_);
    dim3 grid_size = get_optimal_grid_size(hilbert_dim_, block_size);
    
    hamiltonian_apply_kernel<<<grid_size, block_size>>>(
        psi_in, psi_out,
        d_interactions_, num_interactions_,
        num_sites_, spin_length_,
        hilbert_dim_
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// GPUFixedSzOperator Implementation
// ============================================================================

size_t GPUFixedSzOperator::binomial(int n, int k) {
    return binomial_coefficient(n, k);
}

size_t GPUFixedSzOperator::next_prime(size_t n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;
    
    auto is_prime = [](size_t num) {
        if (num <= 1) return false;
        if (num <= 3) return true;
        if (num % 2 == 0 || num % 3 == 0) return false;
        
        for (size_t i = 5; i * i <= num; i += 6) {
            if (num % i == 0 || num % (i + 2) == 0) return false;
        }
        return true;
    };
    
    while (!is_prime(n)) {
        n += 2;
    }
    
    return n;
}

size_t GPUFixedSzOperator::hash_function(uint64_t state, size_t table_size) {
    const uint64_t HASH_MULT = 11400714819323198485ULL;
    return (state * HASH_MULT) % table_size;
}

GPUFixedSzOperator::GPUFixedSzOperator(
    int num_sites,
    float spin_length,
    int n_up,
    cublasHandle_t handle
) : GPUHamiltonianOperator(num_sites, spin_length, handle),
    n_up_(n_up),
    d_basis_states_(nullptr),
    d_hash_keys_(nullptr),
    d_hash_values_(nullptr)
{
    fixed_sz_dim_ = binomial(num_sites, n_up);
    hash_table_size_ = next_prime(2 * fixed_sz_dim_);
    
    std::cout << "Fixed Sz sector: " << num_sites << " sites, " 
              << n_up << " up spins" << std::endl;
    std::cout << "Hilbert space dimension: " << fixed_sz_dim_ << std::endl;
    std::cout << "Memory required: " 
              << fixed_sz_dim_ * sizeof(cuDoubleComplex) / (1024.0 * 1024.0 * 1024.0)
              << " GB per vector" << std::endl;
    
    // Check if we have enough GPU memory
    size_t required_mem = fixed_sz_dim_ * sizeof(cuDoubleComplex) * 3; // 3 vectors
    required_mem += fixed_sz_dim_ * sizeof(uint64_t); // basis states
    required_mem += hash_table_size_ * (sizeof(uint64_t) + sizeof(int)); // hash table
    
    if (!check_gpu_memory(required_mem)) {
        throw std::runtime_error("Insufficient GPU memory for fixed Sz sector");
    }
}

GPUFixedSzOperator::~GPUFixedSzOperator() {
    if (d_basis_states_) cudaFree(d_basis_states_);
    if (d_hash_keys_) cudaFree(d_hash_keys_);
    if (d_hash_values_) cudaFree(d_hash_values_);
}

void GPUFixedSzOperator::generate_basis() {
    std::cout << "Generating fixed Sz basis on GPU..." << std::endl;
    
    // Allocate basis states
    CUDA_CHECK(cudaMalloc(&d_basis_states_, fixed_sz_dim_ * sizeof(uint64_t)));
    
    // Generate basis in parallel
    dim3 block_size = get_optimal_block_size(fixed_sz_dim_);
    dim3 grid_size = get_optimal_grid_size(fixed_sz_dim_, block_size);
    
    generate_fixed_sz_basis_kernel<<<grid_size, block_size>>>(
        d_basis_states_, num_sites_, n_up_, 0, fixed_sz_dim_
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Building hash table for state lookup..." << std::endl;
    
    // Allocate hash table
    CUDA_CHECK(cudaMalloc(&d_hash_keys_, hash_table_size_ * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hash_values_, hash_table_size_ * sizeof(int)));
    
    // Initialize hash table with empty markers
    thrust::device_ptr<uint64_t> keys_ptr(d_hash_keys_);
    thrust::fill(keys_ptr, keys_ptr + hash_table_size_, ULLONG_MAX);
    
    thrust::device_ptr<int> values_ptr(d_hash_values_);
    thrust::fill(values_ptr, values_ptr + hash_table_size_, -1);
    
    // Build hash table
    build_hash_table_kernel<<<grid_size, block_size>>>(
        d_basis_states_, d_hash_keys_, d_hash_values_,
        fixed_sz_dim_, hash_table_size_
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Fixed Sz basis generation complete." << std::endl;
}

void GPUFixedSzOperator::apply(const GPUVector& psi_in, GPUVector& psi_out) {
    apply(psi_in.data(), psi_out.data());
}

void GPUFixedSzOperator::apply(
    const cuDoubleComplex* psi_in,
    cuDoubleComplex* psi_out
) {
    if (!d_basis_states_) {
        throw std::runtime_error("Basis not generated. Call generate_basis() first.");
    }
    
    dim3 block_size = get_optimal_block_size(fixed_sz_dim_);
    dim3 grid_size = get_optimal_grid_size(fixed_sz_dim_, block_size);
    
    hamiltonian_apply_fixed_sz_kernel<<<grid_size, block_size>>>(
        psi_in, psi_out,
        d_basis_states_, d_hash_keys_, d_hash_values_, hash_table_size_,
        d_interactions_, num_interactions_,
        num_sites_, spin_length_,
        fixed_sz_dim_
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpu
