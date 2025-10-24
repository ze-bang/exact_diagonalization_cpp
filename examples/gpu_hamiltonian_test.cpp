// GPU Hamiltonian Prototype Test
// Demonstrates GPU-accelerated Hamiltonian application and validates scaling
#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include "gpu_hamiltonian.cuh"
#include "construct_ham.h"

using Complex = std::complex<double>;

// Simple test: nearest-neighbor Heisenberg chain
void test_heisenberg_chain_gpu(int num_sites) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST: Heisenberg Chain on GPU (" << num_sites << " sites)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    float spin_l = 0.5;
    double J = 1.0;  // coupling strength
    
    // Create GPU Hamiltonian
    GpuHamiltonian gpu_ham(num_sites, spin_l);
    
    // Add Heisenberg terms: H = J * sum_<i,j> (S+_i S-_j + S-_i S+_j + Sz_i Sz_j)
    for (int i = 0; i < num_sites - 1; ++i) {
        // S+_i S-_{i+1} term
        gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SPSM, i, i+1, Complex(0.5*J, 0.0));
        // S-_i S+_{i+1} term  
        gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SPSM, i+1, i, Complex(0.5*J, 0.0));
        // Sz_i Sz_{i+1} term
        gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SZSZ, i, i+1, Complex(J, 0.0));
    }
    
    // Periodic boundary conditions
    gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SPSM, num_sites-1, 0, Complex(0.5*J, 0.0));
    gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SPSM, 0, num_sites-1, Complex(0.5*J, 0.0));
    gpu_ham.addTwoSiteTerm(TermType::TWO_SITE_SZSZ, num_sites-1, 0, Complex(J, 0.0));
    
    // Check memory requirements
    checkGpuMemoryCapacity(num_sites, (num_sites-1)*3 + 3, 5);
    
    // Benchmark
    benchmarkGpuHamiltonian(gpu_ham, 100);
    
    // Validate against CPU if small enough
    if (num_sites <= 16) {
        std::cout << "\n=== CPU Validation ===" << std::endl;
        
        // Create CPU Hamiltonian
        Operator cpu_ham(num_sites, spin_l);
        
        for (int i = 0; i < num_sites - 1; ++i) {
            // S+_i S-_{i+1}
            cpu_ham.addTransform([=](int basis) -> std::pair<int, Complex> {
                if ((basis >> i) & 1 && !((basis >> (i+1)) & 1)) {
                    int new_basis = (basis & ~(1 << i)) | (1 << (i+1));
                    return {new_basis, Complex(0.5*J, 0.0)};
                }
                return {-1, Complex(0.0, 0.0)};
            });
            
            // S-_i S+_{i+1}
            cpu_ham.addTransform([=](int basis) -> std::pair<int, Complex> {
                if (!((basis >> i) & 1) && ((basis >> (i+1)) & 1)) {
                    int new_basis = (basis | (1 << i)) & ~(1 << (i+1));
                    return {new_basis, Complex(0.5*J, 0.0)};
                }
                return {-1, Complex(0.0, 0.0)};
            });
            
            // Sz_i Sz_{i+1}
            cpu_ham.addTransform([=](int basis) -> std::pair<int, Complex> {
                double sz_i = spin_l * (((basis >> i) & 1) ? 1.0 : -1.0);
                double sz_j = spin_l * (((basis >> (i+1)) & 1) ? 1.0 : -1.0);
                return {basis, Complex(J * sz_i * sz_j, 0.0)};
            });
        }
        
        // CPU apply function
        auto cpu_apply = [&](const Complex* in, Complex* out, int N) {
            std::vector<Complex> in_vec(in, in + N);
            std::vector<Complex> out_vec = cpu_ham.apply(in_vec);
            std::copy(out_vec.begin(), out_vec.end(), out);
        };
        
        int N = 1 << num_sites;
        double error = validateGpuHamiltonian(gpu_ham, cpu_apply, N);
        
        if (error < 1e-10) {
            std::cout << "✓ Validation PASSED" << std::endl;
        } else {
            std::cout << "✗ Validation FAILED" << std::endl;
        }
    }
}

// Test memory scaling
void test_memory_scaling() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Memory Scaling Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    printGpuInfo();
    
    std::cout << "\nMemory requirements for different system sizes:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int n = 10; n <= 32; n += 2) {
        std::cout << "\nN = " << n << " sites:" << std::endl;
        
        size_t dim = size_t{1} << n;
        std::cout << "  Hilbert space dimension: " << dim;
        if (dim >= 1e9) {
            std::cout << " (" << dim / 1e9 << "B)";
        } else if (dim >= 1e6) {
            std::cout << " (" << dim / 1e6 << "M)";
        }
        std::cout << std::endl;
        
        MemoryEstimate est = estimateGpuMemoryUsage(n, 100, 5);
        std::cout << "  Single state vector: " << est.state_vector_bytes / (1024.0*1024.0) << " MB" << std::endl;
        std::cout << "  Working memory (5 vectors): " << est.workspace_bytes / (1024.0*1024.0) << " MB" << std::endl;
        std::cout << "  Total required: " << est.total_gb << " GB" << std::endl;
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        bool fits = est.total_bytes < (free_mem * 0.9);
        std::cout << "  Fits in GPU: " << (fits ? "YES ✓" : "NO ✗") << std::endl;
        
        if (n == 32) {
            std::cout << "\n  For 32 sites:" << std::endl;
            std::cout << "  - Full space is " << est.total_gb << " GB (likely too large)" << std::endl;
            std::cout << "  - Symmetry reduction strategies:" << std::endl;
            std::cout << "    • Fixed Sz sector: reduces by ~√N ≈ factor of 180" << std::endl;
            std::cout << "    • Momentum conservation: reduces by factor of N = 32" << std::endl;
            std::cout << "    • Point group symmetry: factor of 4-48 depending on lattice" << std::endl;
            std::cout << "    • Combined: could reduce to <1GB per block" << std::endl;
        }
    }
}

// Bandwidth test
void test_bandwidth() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "GPU Bandwidth Test" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    printGpuInfo();
    
    std::vector<int> test_sizes = {20, 22, 24, 26};
    
    for (int n : test_sizes) {
        size_t dim = size_t{1} << n;
        
        if (dim * sizeof(Complex) * 6 > 40e9) {  // Skip if >40GB required
            std::cout << "\nN = " << n << " sites: SKIPPED (too large)" << std::endl;
            continue;
        }
        
        std::cout << "\nN = " << n << " sites (dim = " << dim << "):" << std::endl;
        
        GpuHamiltonian ham(n, 0.5);
        
        // Add a simple Hamiltonian (just Sz terms for speed)
        for (int i = 0; i < n; ++i) {
            ham.addSingleSiteTerm(TermType::SINGLE_SITE_SZ, i, Complex(1.0, 0.0));
        }
        
        benchmarkGpuHamiltonian(ham, 50);
    }
}

// Test point group symmetry potential
void test_symmetry_reduction() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Symmetry Reduction Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nFor 32-site systems with point group symmetry:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    size_t full_dim = size_t{1} << 32;
    double full_mem_gb = (full_dim * sizeof(Complex) * 6) / 1e9;
    
    std::cout << "\nFull Hilbert space:" << std::endl;
    std::cout << "  Dimension: " << full_dim << " ≈ 4.3 billion" << std::endl;
    std::cout << "  Memory: " << full_mem_gb << " GB" << std::endl;
    
    struct Reduction {
        std::string name;
        double factor;
    };
    
    std::vector<Reduction> reductions = {
        {"No symmetry", 1.0},
        {"Sz conservation (Sz=0 sector)", 184.8},
        {"Translation (momentum)", 32.0},
        {"Point group (typical)", 8.0},
        {"Sz + Translation", 184.8 * 32.0},
        {"Sz + Point group", 184.8 * 8.0},
        {"All symmetries", 184.8 * 32.0 * 8.0}
    };
    
    std::cout << "\nWith symmetry reductions:" << std::endl;
    for (const auto& red : reductions) {
        size_t reduced_dim = full_dim / red.factor;
        double reduced_mem = full_mem_gb / red.factor;
        
        std::cout << "\n  " << red.name << ":" << std::endl;
        std::cout << "    Reduction factor: " << red.factor << std::endl;
        std::cout << "    Reduced dimension: " << reduced_dim;
        if (reduced_dim >= 1e6) {
            std::cout << " ≈ " << reduced_dim / 1e6 << "M";
        }
        std::cout << std::endl;
        std::cout << "    Memory required: " << reduced_mem << " GB" << std::endl;
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        double free_gb = free_mem / 1e9;
        
        std::cout << "    Fits in GPU: " << (reduced_mem < free_gb * 0.9 ? "YES ✓" : "NO ✗") << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Recommendation for 32 sites:" << std::endl;
    std::cout << "  Use Sz + Translation symmetry to get ~20-40 MB blocks" << std::endl;
    std::cout << "  This makes GPU diagonalization feasible!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "GPU Hamiltonian Prototype for 32-Site Scaling" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    try {
        // Test 1: Memory scaling analysis
        test_memory_scaling();
        
        // Test 2: Small system validation
        test_heisenberg_chain_gpu(12);
        
        // Test 3: Bandwidth test on moderate systems
        test_bandwidth();
        
        // Test 4: Symmetry reduction analysis
        test_symmetry_reduction();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Summary and Recommendations" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << "\n1. Full 32-site ED requires ~200-400 GB with current approach" << std::endl;
        std::cout << "   → Not feasible on single GPU without symmetries" << std::endl;
        
        std::cout << "\n2. With point-group + Sz symmetry:" << std::endl;
        std::cout << "   → Can reduce to ~1-10 GB per symmetry block" << std::endl;
        std::cout << "   → Single GPU (80GB) can handle individual blocks" << std::endl;
        
        std::cout << "\n3. GPU implementation advantages:" << std::endl;
        std::cout << "   • No sparse matrix storage (applies H on-the-fly)" << std::endl;
        std::cout << "   • High memory bandwidth (~1-2 TB/s)" << std::endl;
        std::cout << "   • Efficient for Lanczos/CG iterations" << std::endl;
        
        std::cout << "\n4. Next steps for 32-site ED:" << std::endl;
        std::cout << "   ✓ Implement symmetry projection in construct_ham.h" << std::endl;
        std::cout << "   ✓ Port symmetry-reduced basis to GPU kernels" << std::endl;
        std::cout << "   ✓ Test on 24-28 site systems with partial symmetries" << std::endl;
        std::cout << "   ✓ Scale to 32 sites with full symmetry reduction" << std::endl;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
