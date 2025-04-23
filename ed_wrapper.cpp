
#include "TPQ.h"
#include "CG.h"
#include "lanczos"

// Enum for selecting diagonalization method
enum class DiagMethod {
    FULL_DIAGONALIZATION,       // Direct full diagonalization for small matrices
    LANCZOS,                    // Standard Lanczos (with reorthogonalization)
    LANCZOS_NO_ORTHO,           // Lanczos without reorthogonalization
    LANCZOS_SELECTIVE,          // Lanczos with selective reorthogonalization
    SHIFT_INVERT,               // Shift-invert Lanczos for interior eigenvalues
    ROBUST_SHIFT_INVERT,        // More robust implementation of shift-invert
    BLOCK_LANCZOS,              // Block Lanczos for degenerate eigenvalues
    CHEBYSHEV,                  // Chebyshev filtered Lanczos
    KRYLOV_SCHUR,               // Krylov-Schur algorithm
    IMPLICITLY_RESTARTED,       // Implicitly restarted Lanczos
    SPECTRUM_SLICING,           // Spectrum slicing solver
    OPTIMAL_SOLVER              // Optimal solver for full spectrum
};

// Enum for specifying which part of spectrum to compute
enum class SpectrumPart {
    LOWEST,                     // Lowest eigenvalues
    HIGHEST,                    // Highest eigenvalues
    INTERIOR,                   // Interior eigenvalues around sigma
    FULL                        // Full spectrum
};



// Struct containing diagonalization parameters
struct DiagParams {
    DiagMethod method = DiagMethod::LANCZOS;      // Diagonalization method
    SpectrumPart target = SpectrumPart::LOWEST;   // Which part of spectrum to compute
    int num_eigenvalues = 10;                     // Number of eigenvalues to compute
    int max_iterations = 1000;                    // Maximum number of iterations
    double tolerance = 1e-10;                     // Convergence tolerance
    double sigma = 0.0;                           // Target energy for shift-invert methods
    int block_size = 3;                           // Block size for block methods
    std::string output_dir = "";                  // Directory for output files
    bool compute_eigenvectors = true;             // Whether to compute eigenvectors
    bool verbose = true;                          // Whether to print detailed progress
};

// Wrapper function for exact diagonalization with selected method
bool exact_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                               // Hilbert space dimension
    std::vector<double>& eigenvalues,                    // Output eigenvalues
    std::vector<ComplexVector>* eigenvectors = nullptr,  // Optional output eigenvectors
    const DiagParams& params = DiagParams()             // Diagonalization parameters
) {
    // Create a local copy of parameters for potential modification
    DiagParams local_params = params;
    
    // Create output directory if needed
    if (!local_params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + local_params.output_dir;
        system(cmd.c_str());
    }
    
    // Print initial information if verbose
    if (local_params.verbose) {
        std::cout << "Starting exact diagonalization with dimension N = " << N << std::endl;
        std::cout << "Method: ";
        switch (local_params.method) {
            case DiagMethod::FULL_DIAGONALIZATION:  std::cout << "Full diagonalization"; break;
            case DiagMethod::LANCZOS:               std::cout << "Lanczos"; break;
            case DiagMethod::LANCZOS_NO_ORTHO:      std::cout << "Lanczos without reorthogonalization"; break;
            case DiagMethod::LANCZOS_SELECTIVE:     std::cout << "Lanczos with selective reorthogonalization"; break;
            case DiagMethod::SHIFT_INVERT:          std::cout << "Shift-invert Lanczos"; break;
            case DiagMethod::ROBUST_SHIFT_INVERT:   std::cout << "Robust shift-invert Lanczos"; break;
            case DiagMethod::BLOCK_LANCZOS:         std::cout << "Block Lanczos"; break;
            case DiagMethod::CHEBYSHEV:             std::cout << "Chebyshev filtered Lanczos"; break;
            case DiagMethod::KRYLOV_SCHUR:          std::cout << "Krylov-Schur"; break;
            case DiagMethod::IMPLICITLY_RESTARTED:  std::cout << "Implicitly restarted Lanczos"; break;
            case DiagMethod::SPECTRUM_SLICING:      std::cout << "Spectrum slicing"; break;
            case DiagMethod::OPTIMAL_SOLVER:        std::cout << "Optimal solver"; break;
        }
        std::cout << std::endl;
        
        std::cout << "Target: ";
        switch (local_params.target) {
            case SpectrumPart::LOWEST:   std::cout << "Lowest " << local_params.num_eigenvalues << " eigenvalues"; break;
            case SpectrumPart::HIGHEST:  std::cout << "Highest " << local_params.num_eigenvalues << " eigenvalues"; break;
            case SpectrumPart::INTERIOR: std::cout << local_params.num_eigenvalues << " eigenvalues around σ = " 
                                                  << local_params.sigma; break;
            case SpectrumPart::FULL:     std::cout << "Full spectrum"; break;
        }
        std::cout << std::endl;
    }
    
    // For highest eigenvalues, transform the problem
    std::function<void(const Complex*, Complex*, int)> effective_H = H;
    if (local_params.target == SpectrumPart::HIGHEST) {
        // Estimate spectral bounds if using highest eigenvalues
        if (local_params.verbose) {
            std::cout << "Estimating spectral range for highest eigenvalues transformation..." << std::endl;
        }
        
        // Run a quick Lanczos to estimate bounds
        std::vector<double> bounds_evals;
        lanczos_no_ortho(H, N, 30, 30, local_params.tolerance, bounds_evals);
        
        double lambda_min = bounds_evals.front();
        double lambda_max = bounds_evals.back();
        double spectral_bound = 2.0 * std::max(std::abs(lambda_min), std::abs(lambda_max));
        
        if (local_params.verbose) {
            std::cout << "Estimated spectral bound: " << spectral_bound << std::endl;
        }
        
        // Create transformed operator -H to find highest eigenvalues as lowest
        effective_H = [H, spectral_bound, N](const Complex* v, Complex* result, int size) {
            // Apply H to v
            H(v, result, size);
            
            // Negate the result (using -H)
            for (int i = 0; i < size; i++) {
                result[i] = -result[i];
            }
        };
    }
    
    // For interior eigenvalues, shift-invert is preferred
    if (local_params.target == SpectrumPart::INTERIOR &&
        local_params.method != DiagMethod::SHIFT_INVERT &&
        local_params.method != DiagMethod::ROBUST_SHIFT_INVERT) {
        
        if (local_params.verbose) {
            std::cout << "Warning: For interior eigenvalues, shift-invert is recommended. "
                     << "Switching to robust shift-invert method." << std::endl;
        }
        local_params.method = DiagMethod::ROBUST_SHIFT_INVERT;
    }
    
    // Choose appropriate defaults based on matrix size
    if (N < 1000) {
        // For very small matrices, full diagonalization is fastest
        if (local_params.method != DiagMethod::FULL_DIAGONALIZATION && local_params.verbose) {
            std::cout << "Small matrix detected (N < 1000). Full diagonalization is recommended for better performance." << std::endl;
        }
    } else if (N > 10000 && local_params.target == SpectrumPart::FULL) {
        // For large matrices and full spectrum, spectrum slicing is better
        if (local_params.method != DiagMethod::SPECTRUM_SLICING && 
            local_params.method != DiagMethod::OPTIMAL_SOLVER && local_params.verbose) {
                
            std::cout << "Large matrix detected (N > 10000) with full spectrum request. "
                     << "Spectrum slicing is recommended for better performance." << std::endl;
        }
    }
    
    // Timer for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Call the appropriate diagonalization method
    bool success = true;
    
    try {
        switch (local_params.method) {
            case DiagMethod::FULL_DIAGONALIZATION:
                full_diagonalization(effective_H, N, eigenvalues, local_params.output_dir, 
                                    local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::LANCZOS:
                lanczos(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                      local_params.tolerance, eigenvalues, local_params.output_dir, 
                      local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::LANCZOS_NO_ORTHO:
                lanczos_no_ortho(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                             local_params.tolerance, eigenvalues, local_params.output_dir, 
                             local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::LANCZOS_SELECTIVE:
                lanczos_selective_reorth(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                     local_params.tolerance, eigenvalues, local_params.output_dir, 
                                     local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::SHIFT_INVERT:
                shift_invert_lanczos(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                  local_params.sigma, local_params.tolerance, eigenvalues, 
                                  local_params.output_dir, local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::ROBUST_SHIFT_INVERT:
                shift_invert_lanczos_robust(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                         local_params.sigma, local_params.tolerance, eigenvalues, 
                                         local_params.output_dir, local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::BLOCK_LANCZOS:
                block_lanczos_biorthogonal(effective_H, N, local_params.max_iterations, local_params.block_size,
                                        local_params.num_eigenvalues, local_params.tolerance, eigenvalues, 
                                        local_params.output_dir, local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::CHEBYSHEV:
                if (eigenvectors) {
                    chebyshev_filtered_lanczos(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                          local_params.tolerance, eigenvalues, eigenvectors);
                } else {
                    std::vector<ComplexVector> temp_evecs;
                    chebyshev_filtered_lanczos(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                          local_params.tolerance, eigenvalues, 
                                          local_params.compute_eigenvectors ? &temp_evecs : nullptr);
                }
                break;
                
            case DiagMethod::KRYLOV_SCHUR:
                krylov_schur(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                          local_params.tolerance, eigenvalues, local_params.output_dir, 
                          local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::IMPLICITLY_RESTARTED:
                implicitly_restarted_lanczos(effective_H, N, local_params.max_iterations, local_params.num_eigenvalues,
                                         local_params.tolerance, eigenvalues, local_params.output_dir, 
                                         local_params.compute_eigenvectors);
                break;
                
            case DiagMethod::SPECTRUM_SLICING:
                spectrum_slicing_solver(effective_H, N, eigenvalues, local_params.output_dir, 
                                      local_params.compute_eigenvectors, local_params.tolerance);
                break;
                
            case DiagMethod::OPTIMAL_SOLVER:
                optimal_spectrum_solver(effective_H, N, eigenvalues, local_params.output_dir, 
                                      local_params.compute_eigenvectors, local_params.tolerance);
                break;
        }
        
        // If we computed highest eigenvalues by negating H, we need to negate the eigenvalues back
        if (local_params.target == SpectrumPart::HIGHEST) {
            for (auto& val : eigenvalues) {
                val = -val;
            }
        }
        
        // Copy eigenvectors if requested and not already handled by the algorithm
        if (eigenvectors && local_params.compute_eigenvectors && 
            local_params.method != DiagMethod::CHEBYSHEV) {
            
            if (!local_params.output_dir.empty()) {
                eigenvectors->clear();
                eigenvectors->resize(eigenvalues.size(), ComplexVector(N));
                
                std::string evec_subdir;
                switch (local_params.method) {
                    case DiagMethod::LANCZOS:              evec_subdir = "/lanczos_eigenvectors"; break;
                    case DiagMethod::LANCZOS_NO_ORTHO:     evec_subdir = "/lanczos_eigenvectors"; break;
                    case DiagMethod::LANCZOS_SELECTIVE:    evec_subdir = "/lanczos_eigenvectors"; break;
                    case DiagMethod::SHIFT_INVERT:         evec_subdir = "/shift_invert_lanczos_results"; break;
                    case DiagMethod::ROBUST_SHIFT_INVERT:  evec_subdir = "/robust_shift_invert_lanczos_results"; break;
                    case DiagMethod::BLOCK_LANCZOS:        evec_subdir = "/block_lanczos_eigenvectors"; break;
                    case DiagMethod::KRYLOV_SCHUR:         evec_subdir = "/krylov_schur_eigenvectors"; break;
                    case DiagMethod::IMPLICITLY_RESTARTED: evec_subdir = "/irl_eigenvectors"; break;
                    case DiagMethod::SPECTRUM_SLICING:     evec_subdir = "/spectrum_slicing/results"; break;
                    case DiagMethod::OPTIMAL_SOLVER:       evec_subdir = "/optimal_solver/eigenvectors"; break;
                    default:                               evec_subdir = "";
                }
                
                for (size_t i = 0; i < eigenvalues.size(); i++) {
                    std::string evec_file = local_params.output_dir + evec_subdir + "/eigenvector_" + std::to_string(i) + ".bin";
                    std::ifstream infile(evec_file, std::ios::binary);
                    if (infile) {
                        infile.read(reinterpret_cast<char*>((*eigenvectors)[i].data()), N * sizeof(Complex));
                        infile.close();
                    } else {
                        std::cerr << "Warning: Could not read eigenvector file " << evec_file << std::endl;
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during diagonalization: " << e.what() << std::endl;
        success = false;
    }
    
    // Calculate timing information
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    if (local_params.verbose) {
        std::cout << "Diagonalization " << (success ? "completed" : "failed") 
                 << " in " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Found " << eigenvalues.size() << " eigenvalues" << std::endl;
        
        if (!eigenvalues.empty()) {
            std::cout << "Eigenvalue range: [" << eigenvalues.front() << ", " << eigenvalues.back() << "]" << std::endl;
        }
    }
    
    return success;
}

// Higher-level convenience function to solve common diagonalization cases
bool solve_eigenvalue_problem(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    int N,                                               // Hilbert space dimension
    const std::string& problem_type,                     // "ground", "excited", "full", "interior", "thermal"
    std::vector<double>& eigenvalues,                    // Output eigenvalues
    std::vector<ComplexVector>* eigenvectors = nullptr,  // Optional output eigenvectors
    const std::string& output_dir = "",                 // Directory for output files
    double target_energy = 0.0,                         // Target energy for interior eigenvalues
    int num_states = 10,                                // Number of states to compute
    double temperature = 0.0                           // Temperature for thermal calculations
) {
    DiagParams params;
    params.output_dir = output_dir;
    params.compute_eigenvectors = (eigenvectors != nullptr);
    params.tolerance = 1e-10;
    
    // Set up parameters based on problem type
    if (problem_type == "ground" || problem_type == "low") {
        // Ground state or low-lying states
        params.method = (N < 1000) ? DiagMethod::FULL_DIAGONALIZATION : DiagMethod::LANCZOS;
        params.target = SpectrumPart::LOWEST;
        params.num_eigenvalues = num_states;
        
    } else if (problem_type == "excited" || problem_type == "high") {
        // Highest excited states
        params.method = (N < 1000) ? DiagMethod::FULL_DIAGONALIZATION : DiagMethod::LANCZOS;
        params.target = SpectrumPart::HIGHEST;
        params.num_eigenvalues = num_states;
        
    } else if (problem_type == "full") {
        // Full spectrum
        if (N < 1000) {
            params.method = DiagMethod::FULL_DIAGONALIZATION;
        } else if (N < 10000) {
            params.method = DiagMethod::LANCZOS;
            params.num_eigenvalues = N;
        } else {
            params.method = DiagMethod::SPECTRUM_SLICING;
        }
        params.target = SpectrumPart::FULL;
        
    } else if (problem_type == "interior") {
        // Interior eigenvalues around target energy
        params.method = DiagMethod::ROBUST_SHIFT_INVERT;
        params.target = SpectrumPart::INTERIOR;
        params.sigma = target_energy;
        params.num_eigenvalues = num_states;
        
    } else if (problem_type == "thermal") {
        // Thermal properties calculation
        if (N < 1000) {
            // For small matrices, compute full spectrum and calculate thermal properties directly
            params.method = DiagMethod::FULL_DIAGONALIZATION;
            params.target = SpectrumPart::FULL;
        } else {
            // For larger matrices, use Finite Temperature Lanczos Method
            params.method = DiagMethod::LANCZOS;
            params.target = SpectrumPart::LOWEST;
            params.num_eigenvalues = std::min(1000, N/2);  // Use more states for better accuracy
        }
        
    } else if (problem_type == "degenerate") {
        // For handling degenerate eigenvalues
        params.method = DiagMethod::BLOCK_LANCZOS;
        params.target = SpectrumPart::LOWEST;
        params.num_eigenvalues = num_states;
        params.block_size = 3;  // Default block size, adjust if needed
        
    } else {
        std::cerr << "Unknown problem type: " << problem_type << std::endl;
        return false;
    }
    
    // Perform diagonalization
    bool success = exact_diagonalization(H, N, eigenvalues, eigenvectors, params);
    
    // Additional processing for thermal calculations
    if (success && problem_type == "thermal" && temperature > 0.0) {
        // Calculate thermodynamic quantities
        ThermodynamicData thermo = calculate_thermodynamics_from_spectrum(
            eigenvalues,
            temperature / 10.0,  // T_min
            temperature * 10.0,  // T_max
            100                  // num_points
        );
        
        // Save thermodynamic data if output directory provided
        if (!output_dir.empty()) {
            std::string thermo_dir = output_dir + "/thermodynamics";
            system(("mkdir -p " + thermo_dir).c_str());
            
            std::string thermo_file = thermo_dir + "/thermo_data.dat";
            std::ofstream out(thermo_file);
            if (out.is_open()) {
                out << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                for (size_t i = 0; i < thermo.temperatures.size(); i++) {
                    out << thermo.temperatures[i] << " "
                       << thermo.energy[i] << " "
                       << thermo.specific_heat[i] << " "
                       << thermo.entropy[i] << " "
                       << thermo.free_energy[i] << std::endl;
                }
                out.close();
                std::cout << "Thermodynamic data saved to " << thermo_file << std::endl;
            }
        }
    }
    
    return success;
}