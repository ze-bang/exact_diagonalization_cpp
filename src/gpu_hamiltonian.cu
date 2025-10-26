// gpu_hamiltonian.cu - GPU Hamiltonian operator implementation
#include "gpu_hamiltonian.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cmath>

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
        const SpinInteraction& inter = interactions[i];
        int site1 = inter.site1;
        int site2 = inter.site2;
        int op1 = inter.op1;
        int op2 = inter.op2;
        double coupling = inter.coupling;
        
        // Diagonal terms (Sz-Sz)
        if (op1 == 2 && op2 == 2) {
            double sz1 = get_sz_eigenvalue(state, site1, spin_length);
            double sz2 = get_sz_eigenvalue(state, site2, spin_length);
            cuDoubleComplex contrib = cuCmul(
                make_cuDoubleComplex(coupling * sz1 * sz2, 0.0),
                psi_in[idx]
            );
            result = cuCadd(result, contrib);
        }
        // S+ S- terms
        else if (op1 == 0 && op2 == 1) {
            // Apply S- to site2, then S+ to site1
            device_pair<uint64_t, cuDoubleComplex> result1 = apply_sminus(state, site2, spin_length);
            if (result1.first != 0) {
                device_pair<uint64_t, cuDoubleComplex> result2 = apply_splus(result1.first, site1, spin_length);
                if (result2.first != 0) {
                    cuDoubleComplex matrix_elem = cuCmul(result1.second, result2.second);
                    matrix_elem = cuCmul(matrix_elem, make_cuDoubleComplex(coupling, 0.0));
                    
                    size_t target_idx = result2.first;
                    cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_idx]);
                    result = cuCadd(result, contrib);
                }
            }
        }
        // S- S+ terms
        else if (op1 == 1 && op2 == 0) {
            // Apply S+ to site2, then S- to site1
            device_pair<uint64_t, cuDoubleComplex> result1 = apply_splus(state, site2, spin_length);
            if (result1.first != 0) {
                device_pair<uint64_t, cuDoubleComplex> result2 = apply_sminus(result1.first, site1, spin_length);
                if (result2.first != 0) {
                    cuDoubleComplex matrix_elem = cuCmul(result1.second, result2.second);
                    matrix_elem = cuCmul(matrix_elem, make_cuDoubleComplex(coupling, 0.0));
                    
                    size_t target_idx = result2.first;
                    cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_idx]);
                    result = cuCadd(result, contrib);
                }
            }
        }
        // Other operator combinations can be added here
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
        const SpinInteraction& inter = interactions[i];
        int site1 = inter.site1;
        int site2 = inter.site2;
        int op1 = inter.op1;
        int op2 = inter.op2;
        double coupling = inter.coupling;
        
        // Diagonal terms (Sz-Sz)
        if (op1 == 2 && op2 == 2) {
            double sz1 = get_sz_eigenvalue(state, site1, spin_length);
            double sz2 = get_sz_eigenvalue(state, site2, spin_length);
            cuDoubleComplex contrib = cuCmul(
                make_cuDoubleComplex(coupling * sz1 * sz2, 0.0),
                psi_in[idx]
            );
            result = cuCadd(result, contrib);
        }
        // S+ S- terms
        else if (op1 == 0 && op2 == 1) {
            device_pair<uint64_t, cuDoubleComplex> result1 = apply_sminus(state, site2, spin_length);
            if (result1.first != 0) {
                device_pair<uint64_t, cuDoubleComplex> result2 = apply_splus(result1.first, site1, spin_length);
                if (result2.first != 0) {
                    // Look up new_state2 in hash table
                    int target_idx = lookup_state_in_hash_table(
                        result2.first, hash_keys, hash_values, hash_table_size
                    );
                    
                    if (target_idx >= 0) {
                        cuDoubleComplex matrix_elem = cuCmul(result1.second, result2.second);
                        matrix_elem = cuCmul(matrix_elem, make_cuDoubleComplex(coupling, 0.0));
                        cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_idx]);
                        result = cuCadd(result, contrib);
                    }
                }
            }
        }
        // S- S+ terms
        else if (op1 == 1 && op2 == 0) {
            device_pair<uint64_t, cuDoubleComplex> result1 = apply_splus(state, site2, spin_length);
            if (result1.first != 0) {
                device_pair<uint64_t, cuDoubleComplex> result2 = apply_sminus(result1.first, site1, spin_length);
                if (result2.first != 0) {
                    int target_idx = lookup_state_in_hash_table(
                        result2.first, hash_keys, hash_values, hash_table_size
                    );
                    
                    if (target_idx >= 0) {
                        cuDoubleComplex matrix_elem = cuCmul(result1.second, result2.second);
                        matrix_elem = cuCmul(matrix_elem, make_cuDoubleComplex(coupling, 0.0));
                        cuDoubleComplex contrib = cuCmul(matrix_elem, psi_in[target_idx]);
                        result = cuCadd(result, contrib);
                    }
                }
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
    SpinInteraction inter;
    
    // Format: site1 op1 site2 op2 coupling
    std::string op1_str, op2_str;
    iss >> inter.site1 >> op1_str >> inter.site2 >> op2_str >> inter.coupling;
    
    // Map operator strings to integers
    // 0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy
    auto map_op = [](const std::string& op) {
        if (op == "S+" || op == "Sp") return 0;
        if (op == "S-" || op == "Sm") return 1;
        if (op == "Sz") return 2;
        if (op == "Sx") return 3;
        if (op == "Sy") return 4;
        return 2; // Default to Sz
    };
    
    inter.op1 = map_op(op1_str);
    inter.op2 = map_op(op2_str);
    
    return inter;
}

void GPUHamiltonianOperator::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Try to auto-detect format by reading first line
    std::string first_line;
    std::getline(file, first_line);
    file.seekg(0); // Reset to beginning
    
    // Otherwise, parse as simple format
    std::vector<SpinInteraction> interactions;
    std::string line;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        interactions.push_back(parse_interaction_line(line));
    }
    
    num_interactions_ = interactions.size();
    
    // Allocate and copy to GPU
    if (d_interactions_) {
        cudaFree(d_interactions_);
    }
    
    CUDA_CHECK(cudaMalloc(&d_interactions_, 
                         num_interactions_ * sizeof(SpinInteraction)));
    CUDA_CHECK(cudaMemcpy(d_interactions_, interactions.data(),
                         num_interactions_ * sizeof(SpinInteraction),
                         cudaMemcpyHostToDevice));
    
    std::cout << "Loaded " << num_interactions_ << " interactions from " 
              << filename << std::endl;
}

void GPUHamiltonianOperator::load_from_interall_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<SpinInteraction> interactions;
    std::string line;
    
    // Read header: first line with "==="
    std::getline(file, line);
    
    // Second line: "num XXXX"
    std::getline(file, line);
    std::istringstream iss(line);
    std::string num_str;
    int num_lines;
    iss >> num_str >> num_lines;
    
    // Skip next 3 lines (header separators)
    for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
    }
    
    // Read interaction lines
    int line_count = 0;
    while (std::getline(file, line) && line_count < num_lines) {
        std::istringstream line_stream(line);
        int op_i, indx_i, op_j, indx_j;
        double E, F; // Real and imaginary parts
        
        if (!(line_stream >> op_i >> indx_i >> op_j >> indx_j >> E >> F)) {
            continue;
        }
        
        // Skip zero or negligible interactions
        if (std::abs(E) < 1e-14 && std::abs(F) < 1e-14) {
            line_count++;
            continue;
        }
        
        // For now, only handle real couplings (F should be ~0)
        if (std::abs(F) > 1e-10) {
            std::cerr << "Warning: Skipping complex coupling at line " << line_count 
                      << " (imaginary part not yet supported in GPU code)" << std::endl;
            line_count++;
            continue;
        }
        
        // Create SpinInteraction
        SpinInteraction inter;
        inter.site1 = indx_i;
        inter.site2 = indx_j;
        inter.op1 = op_i;  // 0=S+, 1=S-, 2=Sz
        inter.op2 = op_j;
        inter.coupling = E;
        
        interactions.push_back(inter);
        line_count++;
    }
    
    num_interactions_ = interactions.size();
    
    // Allocate and copy to GPU
    if (d_interactions_) {
        cudaFree(d_interactions_);
    }
    
    CUDA_CHECK(cudaMalloc(&d_interactions_, 
                         num_interactions_ * sizeof(SpinInteraction)));
    CUDA_CHECK(cudaMemcpy(d_interactions_, interactions.data(),
                         num_interactions_ * sizeof(SpinInteraction),
                         cudaMemcpyHostToDevice));
    
    std::cout << "Loaded " << num_interactions_ << " interactions from InterAll file " 
              << filename << std::endl;
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
