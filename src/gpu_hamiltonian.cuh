// GPU Hamiltonian application prototype for scaling to large systems
// Demonstrates efficient on-the-fly Hamiltonian application without matrix storage
#pragma once

#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <functional>
#include "TPQ_cuda.cuh"

using Complex = std::complex<double>;

// GPU Hamiltonian term types
enum class TermType {
    SINGLE_SITE_SZ,     // S^z_i (diagonal)
    SINGLE_SITE_SP,     // S^+_i (raising operator)
    SINGLE_SITE_SM,     // S^-_i (lowering operator)
    TWO_SITE_SPSP,      // S^+_i S^+_j
    TWO_SITE_SPSM,      // S^+_i S^-_j
    TWO_SITE_SMSM,      // S^-_i S^-_j
    TWO_SITE_SZSP,      // S^z_i S^+_j
    TWO_SITE_SZSZ       // S^z_i S^z_j
};

// Structure to store a single Hamiltonian term
struct HamiltonianTerm {
    TermType type;
    int site1;
    int site2;  // unused for single-site terms
    Complex coefficient;
    float spin_l;  // spin quantum number
    
    HamiltonianTerm(TermType t, int s1, Complex c, float sl) 
        : type(t), site1(s1), site2(-1), coefficient(c), spin_l(sl) {}
    
    HamiltonianTerm(TermType t, int s1, int s2, Complex c, float sl)
        : type(t), site1(s1), site2(s2), coefficient(c), spin_l(sl) {}
};

// GPU-side Hamiltonian term structure (simplified for kernel use)
struct GpuHamTerm {
    int type_id;       // cast from TermType
    int site1;
    int site2;
    double coeff_real;
    double coeff_imag;
    float spin_l;
};

// Hamiltonian builder class for GPU
class GpuHamiltonian {
public:
    GpuHamiltonian(int num_sites, float spin_l);
    ~GpuHamiltonian();
    
    // Add terms to the Hamiltonian
    void addSingleSiteTerm(TermType type, int site, Complex coeff);
    void addTwoSiteTerm(TermType type, int site1, int site2, Complex coeff);
    
    // Load from file (similar to CPU version)
    void loadFromFile(const std::string& filename);
    void loadFromInterAllFile(const std::string& filename);
    
    // Copy terms to GPU memory
    void copyToDevice();
    
    // Apply Hamiltonian on GPU: out = H * in
    void apply(const Complex* d_in, Complex* d_out, size_t N);
    
    // Create a GpuMatvec functor for use with Lanczos/CG
    GpuMatvec getMatvec();
    
    // Get dimension
    size_t getDimension() const { return size_t{1} << num_sites_; }
    
    // Get number of sites
    int getNumSites() const { return num_sites_; }
    
private:
    int num_sites_;
    float spin_l_;
    std::vector<HamiltonianTerm> cpu_terms_;
    
    // GPU storage
    GpuHamTerm* d_terms_ = nullptr;
    int num_terms_ = 0;
    bool device_ready_ = false;
    
    CudaContext ctx_;
};

// Benchmark/validation utilities
struct BenchmarkResult {
    double gflops;
    double bandwidth_gbps;
    double time_ms;
    size_t operations;
    size_t memory_bytes;
};

// Benchmark GPU Hamiltonian application
BenchmarkResult benchmarkGpuHamiltonian(
    GpuHamiltonian& ham,
    int num_iterations = 100
);

// Compare CPU vs GPU results for validation
double validateGpuHamiltonian(
    GpuHamiltonian& gpu_ham,
    const std::function<void(const Complex*, Complex*, int)>& cpu_ham,
    int N
);

// Memory usage estimation
struct MemoryEstimate {
    size_t state_vector_bytes;
    size_t workspace_bytes;
    size_t hamiltonian_bytes;
    size_t total_bytes;
    double total_gb;
    
    void print() const;
};

MemoryEstimate estimateGpuMemoryUsage(
    int num_sites,
    int num_terms,
    int num_workspace_vectors = 5  // typical for Lanczos/CG
);

// Check if the system fits in GPU memory
bool checkGpuMemoryCapacity(int num_sites, int num_terms, int num_workspace_vectors = 5);

// Utility: print GPU properties
void printGpuInfo();

// Multi-GPU support structure (for future scaling)
struct MultiGpuConfig {
    int num_gpus;
    std::vector<int> device_ids;
    size_t basis_per_gpu;
    bool use_nvlink;
};

// Check multi-GPU configuration
MultiGpuConfig getMultiGpuConfig(size_t total_basis_size);
