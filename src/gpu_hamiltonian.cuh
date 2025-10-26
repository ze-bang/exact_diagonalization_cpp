// gpu_hamiltonian.cuh - GPU-accelerated Hamiltonian operator
#ifndef GPU_HAMILTONIAN_CUH
#define GPU_HAMILTONIAN_CUH

#include "gpu_utils.cuh"
#include "gpu_vector.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace gpu {

/**
 * Structure to store spin-spin interactions on GPU
 */
struct SpinInteraction {
    int site1, site2;      // Site indices
    int op1, op2;          // Operator types: 0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy
    double coupling;       // Coupling strength
};

/**
 * GPU Hamiltonian Operator for full Hilbert space
 * 
 * Implements on-the-fly matrix-vector product H|ψ⟩
 * without storing the full Hamiltonian matrix
 */
class GPUHamiltonianOperator {
protected:
    // System parameters
    int num_sites_;
    float spin_length_;
    size_t hilbert_dim_;
    
    // Interaction data on GPU
    SpinInteraction* d_interactions_;
    int num_interactions_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    bool owns_handle_;
    
public:
    /**
     * Constructor
     */
    GPUHamiltonianOperator(int num_sites, float spin_length, 
                          cublasHandle_t handle = nullptr);
    
    /**
     * Destructor
     */
    virtual ~GPUHamiltonianOperator();
    
    /**
     * Load interactions from file (same format as CPU version)
     */
    void load_from_file(const std::string& filename);
    
    /**
     * Load interactions from InterAll format
     */
    void load_from_interall_file(const std::string& filename);
    
    /**
     * Apply Hamiltonian: psi_out = H * psi_in
     */
    virtual void apply(const GPUVector& psi_in, GPUVector& psi_out);
    
    /**
     * Apply Hamiltonian to raw device pointers
     */
    virtual void apply(const cuDoubleComplex* psi_in, cuDoubleComplex* psi_out);
    
    /**
     * Get Hilbert space dimension
     */
    size_t get_dimension() const { return hilbert_dim_; }
    
    /**
     * Get number of sites
     */
    int get_num_sites() const { return num_sites_; }
    
    /**
     * Get cuBLAS handle
     */
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
    
protected:
    /**
     * Parse interaction term from string
     * Format: "site1 op1 site2 op2 coupling"
     */
    SpinInteraction parse_interaction_line(const std::string& line);
};

/**
 * GPU Hamiltonian Operator for Fixed Sz sector
 * 
 * Restricts to states with fixed total Sz (fixed number of up spins)
 * Reduces dimension from 2^N to C(N, N_up)
 */
class GPUFixedSzOperator : public GPUHamiltonianOperator {
private:
    // Fixed Sz parameters
    int n_up_;                      // Number of up spins
    size_t fixed_sz_dim_;           // C(num_sites, n_up)
    
    // Basis states on GPU
    uint64_t* d_basis_states_;      // List of basis states
    
    // Hash table for state lookup (basis state -> index)
    uint64_t* d_hash_keys_;         // Hash table keys
    int* d_hash_values_;            // Hash table values (indices)
    size_t hash_table_size_;        // Hash table size (must be prime)
    
public:
    /**
     * Constructor
     */
    GPUFixedSzOperator(int num_sites, float spin_length, int n_up,
                      cublasHandle_t handle = nullptr);
    
    /**
     * Destructor
     */
    ~GPUFixedSzOperator() override;
    
    /**
     * Generate basis states for fixed Sz sector
     * Uses combinatorial number system (rank/unrank)
     */
    void generate_basis();
    
    /**
     * Apply Hamiltonian in fixed Sz sector
     */
    void apply(const cuDoubleComplex* psi_in, cuDoubleComplex* psi_out) override;
    
    /**
     * Apply Hamiltonian using GPUVector interface
     */
    void apply(const GPUVector& psi_in, GPUVector& psi_out) override;
    
    /**
     * Get fixed Sz dimension
     */
    size_t get_fixed_sz_dimension() const { return fixed_sz_dim_; }
    
private:
    /**
     * Compute binomial coefficient C(n, k)
     */
    static size_t binomial(int n, int k);
    
    /**
     * Find next prime number >= n
     */
    static size_t next_prime(size_t n);
    
    /**
     * Hash function for state lookup
     */
    static size_t hash_function(uint64_t state, size_t table_size);
};

// ============================================================================
// Kernel declarations (implemented in gpu_hamiltonian.cu)
// ============================================================================

/**
 * Kernel: Apply Hamiltonian to full Hilbert space
 */
__global__ void hamiltonian_apply_kernel(
    const cuDoubleComplex* __restrict__ psi_in,
    cuDoubleComplex* __restrict__ psi_out,
    const SpinInteraction* __restrict__ interactions,
    int num_interactions,
    int num_sites,
    float spin_length,
    size_t hilbert_dim
);

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
);

/**
 * Kernel: Generate fixed Sz basis using combinatorial number system
 */
__global__ void generate_fixed_sz_basis_kernel(
    uint64_t* __restrict__ basis_states,
    int num_sites,
    int n_up,
    size_t start_rank,
    size_t count
);

/**
 * Kernel: Build hash table for state lookup
 */
__global__ void build_hash_table_kernel(
    const uint64_t* __restrict__ basis_states,
    uint64_t* __restrict__ hash_keys,
    int* __restrict__ hash_values,
    size_t basis_dim,
    size_t hash_table_size
);

} // namespace gpu

/**
 * Device function: Lookup state in hash table
 */
__device__ int lookup_state_in_hash_table(
    uint64_t state,
    const uint64_t* __restrict__ hash_keys,
    const int* __restrict__ hash_values,
    size_t hash_table_size
);

/**
 * Device function: Unrank combination (convert rank to basis state)
 */
__device__ uint64_t unrank_combination(size_t rank, int n, int k);

/**
 * Device function: Rank combination (convert basis state to rank)
 */
__device__ size_t rank_combination(uint64_t state, int n, int k);

/**
 * Device function: Binomial coefficient C(n,k)
 */
__device__ __host__ size_t binomial_coefficient(int n, int k);

#endif // GPU_HAMILTONIAN_CUH
