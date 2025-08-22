#ifndef ED_WRAPPER_H
#define ED_WRAPPER_H

#include "TPQ.h"
#include "CG.h"
#include "arpack.h"
#include "lanczos.h"
#include "construct_ham.h"
#include "observables.h"
#include "finite_temperature_lanczos.h"
#include <sys/stat.h>

// #include "TPQ_cuda.cuh"
// #include "lanczos_cuda.cuh"

std::vector<Complex> operator+ (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}
std::vector<Complex> operator- (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

std::vector<Complex> operator+= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
    return a;
}

std::vector<Complex> operator-= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] -= b[i];
    }
    return a;
}

std::vector<Complex> operator* (const std::vector<Complex>& a, const Complex& b) {
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b;
    }
    return result;
}

// Enum for available diagonalization methods
enum class DiagonalizationMethod {
    LANCZOS,               // Standard Lanczos algorithm
    LANCZOS_SELECTIVE,     // Lanczos with selective reorthogonalization
    LANCZOS_NO_ORTHO,      // Lanczos without reorthogonalization
    BLOCK_LANCZOS,         // Block Lanczos
    SHIFT_INVERT,          // Shift-invert Lanczos
    SHIFT_INVERT_ROBUST,   // Robust shift-invert Lanczos
    CG,                    // Conjugate gradient
    BLOCK_CG,              // Block conjugate gradient
    DAVIDSON,              // Davidson method
    BICG,                  // Biconjugate gradient
    LOBPCG,                // Locally optimal block preconditioned conjugate gradient
    KRYLOV_SCHUR,          // Krylov-Schur algorithm
    FULL,                  // Full diagonalization
    mTPQ,                  // Thermal Pure Quantum states
    cTPQ,
    OSS,                   
    mTPQ_CUDA,             // CUDA microcanonical Thermal Pure Quantum states
    FTLM,                  // Finite Temperature Lanczos Method
    LTLM,                   // Low Temperature Lanczos Method
    ARPACK_SM,                // ARPACK for eigenvalue problems
    ARPACK_LM,
    ARPACK_SHIFT_INVERT,   // ARPACK in shift-invert mode
    ARPACK_ADVANCED        // ARPACK advanced multi-attempt strategy
};

// Structure to hold exact diagonalization results
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed;
    std::string eigenvectors_path;
    
    // For thermal calculations
    ThermodynamicData thermo_data;
};

// Structure for ED parameters
struct EDParameters {
    int max_iterations = 100000;
    int num_eigenvalues = 1;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    std::string output_dir = "";
    
    // Method-specific parameters
    double shift = 0.0;  // For shift-invert methods
    int block_size = 10; // For block methods
    int max_subspace = 100; // For Davidson
    
    // TPQ-specific parameters
    int num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20;
    int num_temp_bins = 100;
    int num_order = 100; // order in which canonical ensemble is calculated
    int num_measure_freq = 100; // frequency of measurements
    int delta_tau = 1e-4; // time step for imaginary time evolution for cTPQ
    double large_value = 1e5; // Large value for TPQ

    mutable std::vector<Operator> observables = {}; // Observables to calculate for TPQ
    mutable std::vector<std::string> observable_names = {}; // Names of observables to calculate for TPQ
    double omega_min = -10.0; // Minimum frequency for spectral function
    double omega_max = 10.0; // Maximum frequency for spectral function
    int num_points = 1000; // Number of points for spectral function
    double t_end = 50.0; // End time for time evolution
    double dt = 0.01; // Time step for time evolution

    // Required lattice parameters
    int num_sites = 0; // Number of sites in the system
    float spin_length = 0.5; // Spin length
    int sublattice_size = 1; // Size of the sublattice

    bool calc_observables = false; // Calculate custom observables
    bool measure_spin = false; // Measure spins

    // ARPACK advanced options (used when method == ARPACK_ADVANCED)
    // These mirror (a subset of) detail_arpack::ArpackAdvancedOptions
    bool arpack_advanced_verbose = false;    // --arpack-verbose
    std::string arpack_which = "SM";         // --arpack-which=SM|LM|SR|LR|... (Hermitian typical: SM/LM)
    int arpack_ncv = -1;                     // --arpack-ncv=<int>
    int arpack_max_restarts = 2;             // --arpack-max-restarts=<int>
    double arpack_ncv_growth = 1.5;          // --arpack-ncv-growth=<double>
    bool arpack_auto_enlarge_ncv = true;     // --arpack-no-auto-enlarge-ncv to disable
    bool arpack_two_phase_refine = true;     // --arpack-no-two-phase to disable
    double arpack_relaxed_tol = 1e-6;        // --arpack-relaxed-tol=<double>
    bool arpack_shift_invert = false;        // --arpack-shift-invert
    double arpack_sigma = 0.0;               // --arpack-sigma=<double>
    bool arpack_auto_switch_shift_invert = true; // --arpack-no-auto-switch-si
    double arpack_switch_sigma = 0.0;        // --arpack-switch-sigma=<double>
    bool arpack_adaptive_inner_tol = true;   // --arpack-no-adaptive-inner-tol
    double arpack_inner_tol_factor = 1e-2;   // --arpack-inner-tol-factor=<double>
    double arpack_inner_tol_min = 1e-14;     // --arpack-inner-tol-min=<double>
    int arpack_inner_max_iter = 300;         // --arpack-inner-max-iter=<int>
};

// Main wrapper function for exact diagonalization
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    int hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters()
) {
    EDResults results;
    
    // Initialize output directory if needed
    if (!params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + params.output_dir;
        system(cmd.c_str());
    }
    
    // Set eigenvectors flag in results
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Call the appropriate diagonalization method
    switch (method) {
        case DiagonalizationMethod::FULL:
            std::cout << "Using full diagonalization" << std::endl;
            full_diagonalization(H, hilbert_space_dim, params.num_eigenvalues, results.eigenvalues, 
                                 params.output_dir, params.compute_eigenvectors);
            break;


        case DiagonalizationMethod::LANCZOS:
            std::cout << "Using standard Lanczos method" << std::endl;
            lanczos(H, hilbert_space_dim, params.max_iterations, params.num_eigenvalues, 
                    params.tolerance, results.eigenvalues, params.output_dir, 
                    params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::LANCZOS_SELECTIVE:
            std::cout << "Using Lanczos with selective reorthogonalization" << std::endl;
            lanczos_selective_reorth(H, hilbert_space_dim, params.max_iterations, 
                                    params.num_eigenvalues, params.tolerance, 
                                    results.eigenvalues, params.output_dir, 
                                    params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::LANCZOS_NO_ORTHO:
            std::cout << "Using Lanczos without reorthogonalization" << std::endl;
            lanczos_no_ortho(H, hilbert_space_dim, params.max_iterations, 
                           params.num_eigenvalues, params.tolerance, 
                           results.eigenvalues, params.output_dir, 
                           params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::SHIFT_INVERT:
            std::cout << "Using shift-invert Lanczos method with shift = " << params.shift << std::endl;
            shift_invert_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.shift, 
                                params.tolerance, results.eigenvalues, 
                                params.output_dir, params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::CG:
            std::cout << "Using conjugate gradient method" << std::endl;
            cg_diagonalization(H, hilbert_space_dim, params.max_iterations, 
                             params.num_eigenvalues, params.tolerance, 
                             results.eigenvalues, params.output_dir, 
                             params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::BLOCK_CG:
            {
                std::cout << "Using block conjugate gradient method with block size = " 
                         << params.block_size << std::endl;
                std::vector<ComplexVector> eigenvectors;
                block_cg(H, hilbert_space_dim, params.max_iterations, 
                      params.block_size, params.tolerance, 
                      results.eigenvalues, eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::DAVIDSON:
            {
                std::cout << "Using Davidson method" << std::endl;
                std::vector<ComplexVector> eigenvectors;
                davidson_method(H, hilbert_space_dim, params.max_iterations, 
                             params.max_subspace, params.num_eigenvalues, 
                             params.tolerance, results.eigenvalues, 
                             eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::BICG:
            {
                std::cout << "Using biconjugate gradient method" << std::endl;
                std::vector<ComplexVector> eigenvectors;
                bicg_eigenvalues(H, hilbert_space_dim, params.max_iterations, 
                              params.tolerance, results.eigenvalues, 
                              eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::LOBPCG:
            std::cout << "Using LOBPCG method" << std::endl;
            lobpcg_diagonalization(H, hilbert_space_dim, params.max_iterations, 
                            params.num_eigenvalues, params.tolerance, 
                            results.eigenvalues, params.output_dir, 
                            params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::KRYLOV_SCHUR:
            std::cout << "Using Krylov-Schur method" << std::endl;
            krylov_schur(H, hilbert_space_dim, params.max_iterations, 
                       params.num_eigenvalues, params.tolerance, 
                       results.eigenvalues, params.output_dir, 
                       params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::OSS:
            std::cout << "Spectrum slicing for full diagonalization" << std::endl;
            optimal_spectrum_solver(
                H, hilbert_space_dim, params.max_iterations,
                results.eigenvalues, params.output_dir, 
                params.compute_eigenvectors
            );
            break;
            
        case DiagonalizationMethod::mTPQ:
            std::cout << "Using microcanonical TPQ method" << std::endl;
            
            // Search for observable files and load them as operators
            if (params.calc_observables) {
                std::string base_dir;
                if (!params.output_dir.empty()) {
                    size_t pos = params.output_dir.find_last_of("/\\");
                    base_dir = (pos != std::string::npos) ? params.output_dir.substr(0, pos) : ".";
                } else {
                    base_dir = ".";
                }
                
                std::cout << "Searching for observable files in: " << base_dir << std::endl;
                
                // Create a temporary file to store the list of observable files
                std::string temp_list_file = params.output_dir + "/observable_files.txt";
                std::string find_command = "find \"" + base_dir + "\" -name \"observables*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
                system(find_command.c_str());
                
                // Read the list of observable files
                std::ifstream file_list(temp_list_file);
                if (file_list.is_open()) {
                    std::string observable_file;
                    int loaded_count = 0;
                    
                    while (std::getline(file_list, observable_file)) {
                        if (observable_file.empty()) continue;
                        
                        std::cout << "Loading observable from file: " << observable_file << std::endl;
                        
                        try {
                            Operator obs_op(params.num_sites, params.spin_length);
                            
                            // Extract observable name from the filename
                            std::string obs_name = "observable";
                            size_t name_pos = observable_file.find("observables_");
                            if (name_pos != std::string::npos) {
                                size_t start = name_pos + 12; // Length of "observables_"
                                size_t end = observable_file.find(".dat", start);
                                if (end != std::string::npos) {
                                    obs_name = observable_file.substr(start, end - start);
                                }
                            } else {
                                // Get the filename without path
                                size_t last_slash = observable_file.find_last_of("/\\");
                                if (last_slash != std::string::npos) {
                                    obs_name = observable_file.substr(last_slash + 11);
                                    // Remove .dat extension if present
                                    size_t dot_pos = obs_name.find(".dat");
                                    if (dot_pos != std::string::npos) {
                                        obs_name = obs_name.substr(0, dot_pos);
                                    }
                                }
                            }
                            
                            // Determine file type and load accordingly
                            if (observable_file.find("InterAll") != std::string::npos) {
                                obs_op.loadFromInterAllFile(observable_file);
                            } else {
                                obs_op.loadFromFile(observable_file);
                            }
                            
                            params.observables.push_back(obs_op);
                            params.observable_names.push_back(obs_name);
                            loaded_count++;
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Error loading observable from " << observable_file << ": " << e.what() << std::endl;
                        }
                    }
                    
                    file_list.close();
                    std::remove(temp_list_file.c_str());
                    
                    std::cout << "Loaded " << loaded_count << " observables for TPQ calculations" << std::endl;
                }
            }


            microcanonical_tpq(H, hilbert_space_dim,
                             params.max_iterations, params.num_samples,
                             params.num_measure_freq,
                             results.eigenvalues,
                             params.output_dir,
                             params.compute_eigenvectors,
                             params.large_value,
                             params.calc_observables,params.observables, params.observable_names,
                            params.omega_min, params.omega_max,
                            params.num_points, params.t_end, params.dt, params.spin_length, params.measure_spin, params.sublattice_size); 
            break;

        case DiagonalizationMethod::mTPQ_CUDA:
            std::cout << "Using microcanonical TPQ method with CUDA acceleration" << std::endl;

            // Search for observable files and load them as operators
            if (params.calc_observables) {
                std::string base_dir;
                if (!params.output_dir.empty()) {
                    size_t pos = params.output_dir.find_last_of("/\\");
                    base_dir = (pos != std::string::npos) ? params.output_dir.substr(0, pos) : ".";
                } else {
                    base_dir = ".";
                }

                std::cout << "Searching for observable files in: " << base_dir << std::endl;

                // Create a temporary file to store the list of observable files
                std::string temp_list_file = params.output_dir + "/observable_files.txt";
                std::string find_command = "find \"" + base_dir + "\" -name \"observables*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
                system(find_command.c_str());

                // Read the list of observable files
                std::ifstream file_list(temp_list_file);
                if (file_list.is_open()) {
                    std::string observable_file;
                    int loaded_count = 0;

                    while (std::getline(file_list, observable_file)) {
                        if (observable_file.empty()) continue;

                        std::cout << "Loading observable from file: " << observable_file << std::endl;

                        try {
                            Operator obs_op(params.num_sites, params.spin_length);

                            // Extract observable name from the filename
                            std::string obs_name = "observable";
                            size_t name_pos = observable_file.find("observables_");
                            if (name_pos != std::string::npos) {
                                size_t start = name_pos + 12; // Length of "observables_"
                                size_t end = observable_file.find(".dat", start);
                                if (end != std::string::npos) {
                                    obs_name = observable_file.substr(start, end - start);
                                }
                            } else {
                                // Get the filename without path
                                size_t last_slash = observable_file.find_last_of("/\\");
                                if (last_slash != std::string::npos) {
                                    obs_name = observable_file.substr(last_slash + 11);
                                    // Remove .dat extension if present
                                    size_t dot_pos = obs_name.find(".dat");
                                    if (dot_pos != std::string::npos) {
                                        obs_name = obs_name.substr(0, dot_pos);
                                    }
                                }
                            }

                            // Determine file type and load accordingly
                            if (observable_file.find("InterAll") != std::string::npos) {
                                obs_op.loadFromInterAllFile(observable_file);
                            } else {
                                obs_op.loadFromFile(observable_file);
                            }

                            params.observables.push_back(obs_op);
                            params.observable_names.push_back(obs_name);
                            loaded_count++;
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Error loading observable from " << observable_file << ": " << e.what() << std::endl;
                        }
                    }

                    file_list.close();
                    std::remove(temp_list_file.c_str());

                    std::cout << "Loaded " << loaded_count << " observables for TPQ calculations" << std::endl;
                }
            }

            microcanonical_tpq_unified(H, hilbert_space_dim,
                             params.max_iterations, params.num_samples,
                             params.num_measure_freq,
                             results.eigenvalues,
                             params.output_dir,
                             params.compute_eigenvectors,
                             params.large_value,
                             params.calc_observables, params.observables, params.observable_names,
                            params.omega_min, params.omega_max,
                            params.num_points, params.t_end, params.dt, params.spin_length, params.measure_spin, params.sublattice_size,
                            /*use_cuda=*/true);
            break;

        case DiagonalizationMethod::BLOCK_LANCZOS:
            std::cout << "Using block Lanczos method" << std::endl;
            block_lanczos(H, hilbert_space_dim, 
                        params.max_iterations, params.num_eigenvalues, params.block_size, 
                        params.tolerance, results.eigenvalues, 
                        params.output_dir, params.compute_eigenvectors);
            break;
            
        
        case DiagonalizationMethod::ARPACK_SM:
            std::cout << "Using ARPACK standard eigenvalue solver" << std::endl;
            arpack_ground_state(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::ARPACK_LM:
            std::cout << "Using ARPACK standard eigenvalue solver" << std::endl;
            arpack_largest(H, hilbert_space_dim,
                            params.max_iterations, params.num_eigenvalues, params.tolerance,
                            results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_SHIFT_INVERT:
            std::cout << "Using ARPACK shift-invert method with shift = " << params.shift << std::endl;
            arpack_shift_invert(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                params.shift,
                                results.eigenvalues, params.output_dir,
                                params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_ADVANCED: {
            std::cout << "Using ARPACK advanced multi-attempt solver" << std::endl;
            detail_arpack::ArpackAdvancedOptions opts;
            opts.nev = params.num_eigenvalues;
            opts.which = params.arpack_which;
            opts.tol = params.tolerance;
            opts.max_iter = params.max_iterations;
            opts.ncv = params.arpack_ncv;
            opts.auto_enlarge_ncv = params.arpack_auto_enlarge_ncv;
            opts.max_restarts = params.arpack_max_restarts;
            opts.ncv_growth = params.arpack_ncv_growth;
            opts.two_phase_refine = params.arpack_two_phase_refine;
            opts.relaxed_tol = params.arpack_relaxed_tol;
            opts.shift_invert = params.arpack_shift_invert;
            opts.sigma = params.arpack_sigma;
            opts.auto_switch_to_shift_invert = params.arpack_auto_switch_shift_invert;
            opts.switch_sigma = params.arpack_switch_sigma;
            opts.adaptive_inner_tol = params.arpack_adaptive_inner_tol;
            opts.inner_tol_factor = params.arpack_inner_tol_factor;
            opts.inner_tol_min = params.arpack_inner_tol_min;
            opts.inner_max_iter = params.arpack_inner_max_iter;
            opts.verbose = params.arpack_advanced_verbose;
            std::vector<Complex> evecs; // optionally capture
            int info = arpack_eigs_advanced(H, hilbert_space_dim, opts,
                                            results.eigenvalues,
                                            params.output_dir,
                                            params.compute_eigenvectors,
                                            params.compute_eigenvectors ? &evecs : nullptr);
            if (info != 0) {
                std::cerr << "ARPACK advanced solver returned info=" << info << std::endl;
            }
            break; }

        default:
            std::cerr << "Unknown diagonalization method selected" << std::endl;
            break;
    }
    
    if (params.compute_eigenvectors) {
        std::cout << "Eigenvectors computed and saved to " << params.output_dir << std::endl;
    }

    if (params.calc_observables) {
        // Call the function to calculate observables
        std::cout << "Calculating custom observables..." << std::endl;
        // This would call the appropriate function to calculate observables
        // Calculate thermal expectation values for correlation operators
        std::cout << "Calculating thermal expectation values for correlation operators..." << std::endl;

        // Create a directory for thermal correlation results
        std::string output_correlations_dir = params.output_dir + "/thermal_correlations";
        std::string cmd_mkdir = "mkdir -p " + output_correlations_dir;
        system(cmd_mkdir.c_str());

        // Get the base directory where correlation files might be located
        std::string base_dir;
        if (!params.output_dir.empty()) {
            size_t pos = params.output_dir.find_last_of("/\\");
            base_dir = (pos != std::string::npos) ? params.output_dir.substr(0, pos) : ".";
        } else {
            base_dir = ".";
        }

        std::cout << "Looking for correlation files in: " << base_dir << std::endl;

        // Define correlation file patterns to search for
        std::vector<std::pair<std::string, std::string>> patterns = {
            {"one_body_correlations", "one_body_correlations*.dat"},
            {"two_body_correlations", "two_body_correlations*.dat"}
        };

        // Process each type of correlation file
        for (const auto& [prefix, pattern] : patterns) {
            // Find matching files
            std::string temp_list_file = output_correlations_dir + "/" + prefix + "_files.txt";
            std::string find_command = "find \"" + base_dir + "\" -name \"" + pattern + "\" 2>/dev/null > \"" + temp_list_file + "\"";
            system(find_command.c_str());
            
            // Read the list of files
            std::ifstream file_list(temp_list_file);
            if (!file_list.is_open()) continue;
            
            std::string correlation_file;
            int file_count = 0;

            // Compute thermal expectations at different temperatures
            std::string results_file_path = output_correlations_dir + "/thermal_expectation_" + 
            prefix + ".dat";
            std::ofstream results_file(results_file_path);
            
            if (!results_file.is_open()) {
                std::cerr << "Error: Could not open output file: " << results_file_path << std::endl;
                continue;
            }
                        
            while (std::getline(file_list, correlation_file)) {
                if (correlation_file.empty()) continue;
                file_count++;
                
                // Extract operator type from filename
                size_t prefix_pos = correlation_file.find(prefix);
                if (prefix_pos == std::string::npos) continue;
                
                
                std::cout << "Processing " << prefix << " file: " << correlation_file << std::endl;
                
                try {
                    // Load the operator
                    

                    std::ifstream file(correlation_file);
                    if (!file.is_open()) {
                        throw std::runtime_error("Could not open file: " + correlation_file);
                    }
                    std::cout << "Reading file: " << correlation_file << std::endl;
                    std::string line;
                    
                    // Skip the first line (header)
                    std::getline(file, line);
                    
                    // Read the number of lines
            
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int numLines;
                    std::string m;
                    iss >> m >> numLines;
                    // std::cout << "Number of lines: " << numLines << std::endl;
                    
                    // Skip the next 3 lines (separators/headers)
                    for (int i = 0; i < 3; ++i) {
                        std::getline(file, line);
                    }
                                            
                    if (prefix == "one_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    } else if (prefix == "two_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Op2" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Index2" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    }

                    // Process transform data
                    int lineCount = 0;
                    while (std::getline(file, line) && lineCount < numLines) {
                        Operator correlation_op(params.num_sites, params.spin_length);
                        std::istringstream lineStream(line);
                        int Op1, indx1, Op2, indx2;
                        double E, F;
                        if (prefix == "one_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> E >> F)) {
                                continue; // Skip invalid lines
                            }

                            correlation_op.loadonebodycorrelation(Op1, indx1);
                        } else if (prefix == "two_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                                continue; // Skip invalid lines
                            }
                            correlation_op.loadtwobodycorrelation(Op1, indx1, Op2, indx2);
                        }

                        // Create a lambda to apply the operator
                        auto apply_correlation_op = [&correlation_op](const Complex* in, Complex* out, int n) {
                            std::vector<Complex> in_vec(in, in + n);
                            std::vector<Complex> out_vec = correlation_op.apply(in_vec);
                            std::copy(out_vec.begin(), out_vec.end(), out);
                        };
                        

                        // Calculate thermal expectations at temperature points
                        int num_temps = std::min(params.num_temp_bins, 20);
                        double log_temp_min = std::log(params.temp_min);
                        double log_temp_max = std::log(params.temp_max);
                        double log_temp_step = (log_temp_max - log_temp_min) / std::max(1, num_temps - 1);

                        for (int i = 0; i < num_temps; i++) {
                            double T = std::exp(log_temp_min + i * log_temp_step);
                            double beta = 1.0 / T;
                                // Compute thermal expectation
                            Complex expectation = calculate_thermal_expectation(
                                apply_correlation_op, hilbert_space_dim, beta, params.output_dir + "/eigenvectors/");
                            
                            std::cout << "T: " << T << ", beta: " << beta << ", expectation: " 
                                        << expectation.real() << " + " << expectation.imag() << "i" << std::endl;

                            // Write to file
                            if (prefix == "one_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            } else if (prefix == "two_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << Op2 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx2 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            }
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing " << correlation_file << ": " << e.what() << std::endl;
                }
            }
            results_file.close();
            std::cout << "Thermal expectations saved to: " << results_file_path << std::endl;
            file_list.close();
            std::cout << "Processed " << file_count << " " << prefix << " files" << std::endl;
            std::remove(temp_list_file.c_str());
        }

        std::cout << "Thermal expectation calculations complete!" << std::endl;
    }

    return results;
}

// Enum for Hamiltonian file formats
enum class HamiltonianFileFormat {
    STANDARD,       // InterAll.dat and Trans.dat format
    SPARSE_MATRIX,  // Sparse matrix format
    CUSTOM          // Custom format requiring a parser function
};

// Wrapper function to perform exact diagonalization from Hamiltonian files
EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    // 1. Determine the number of sites and create the Hamiltonian
    int num_sites = params.num_sites;
    Operator hamiltonian(num_sites, params.spin_length);  // Initialize with dummy size, will update later
    std::cerr << "[DEBUG] exact_diagonalization_from_files: format=STANDARD, num_sites=" << num_sites
              << ", method=" << static_cast<int>(method) << std::endl;
    
    switch (format) {
        case HamiltonianFileFormat::STANDARD: {            
            // Create the Hamiltonian with the correct number of sites
            hamiltonian = Operator(num_sites, params.spin_length);
            
            // Load the terms from files
            if (!single_site_file.empty()) {
                hamiltonian.loadFromFile(single_site_file);
            }
            
            if (!interaction_file.empty()) {
                hamiltonian.loadFromInterAllFile(interaction_file);
            }

            if (method != DiagonalizationMethod::FULL){
                hamiltonian.buildSparseMatrix();
            }
            break;
        }
        
        case HamiltonianFileFormat::SPARSE_MATRIX:
            // Implement loading from sparse matrix format
            throw std::runtime_error("Sparse matrix format not yet implemented");
            break;
            
        case HamiltonianFileFormat::CUSTOM:
            // This would require a custom parser function
            throw std::runtime_error("Custom format requires a parser function");
            break;
            
        default:
            throw std::runtime_error("Unknown Hamiltonian file format");
    }
    
    // 2. Calculate the Hilbert space dimension
    int hilbert_space_dim = static_cast<int>(1ULL << num_sites);  // 2^num_sites using 64-bit safe shift
    std::cerr << "[DEBUG] exact_diagonalization_from_files: hilbert_space_dim=" << hilbert_space_dim << std::endl;
    // 3. Create a lambda function to apply the Hamiltonian
    auto apply_hamiltonian = [&hamiltonian](const Complex* in, Complex* out, int n) {
        // Convert raw pointers to vectors for the Operator::apply method
        std::vector<Complex> in_vec(in, in + n);
        std::vector<Complex> out_vec = hamiltonian.apply(in_vec);
        
        // Copy result back to the output array
        std::copy(out_vec.begin(), out_vec.end(), out);
    };
    
    // 4. Call the exact diagonalization core with our Hamiltonian function
    return exact_diagonalization_core(apply_hamiltonian, hilbert_space_dim, method, params);
}

// Wrapper function to perform exact diagonalization from a directory containing Hamiltonian files
EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    // Call the file-based wrapper
    EDResults results = exact_diagonalization_from_files(
        interaction_file, single_site_file, method, params, format
    );
    return results;
}


// Wrapper function to perform exact diagonalization using symmetrized basis
EDResults exact_diagonalization_from_directory_symmetrized(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cerr << "[DEBUG] exact_diagonalization_from_directory_symmetrized: num_sites=" << params.num_sites
              << ", method=" << static_cast<int>(method) << std::endl;

    // Check if automorphism results already exist
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    struct stat automorphism_buffer;
    bool automorphisms_exist = (stat((automorphism_file).c_str(), &automorphism_buffer) == 0);

    if (!automorphisms_exist) {
        // Run the automorphism finder Python script to generate symmetry data
        std::string automorphism_finder_path = std::string(__FILE__);
        automorphism_finder_path = automorphism_finder_path.substr(0, automorphism_finder_path.find_last_of("/\\"));
        automorphism_finder_path += "/automorphism_finder.py";

        std::string automorphism_cmd = "python " + automorphism_finder_path + " --data_dir=\"" + directory + "\"";
        std::cout << "Running automorphism finder: " << automorphism_cmd << std::endl;
        int result = system(automorphism_cmd.c_str());
        if (result != 0) {
            std::cerr << "Warning: Automorphism finder script returned non-zero code: " << result << std::endl;
            std::cerr << "Continuing without symmetry analysis." << std::endl;
        }
    } else {
        std::cout << "Using existing automorphism results from: " << automorphism_file << std::endl;
    }

    // Check for automorphism results
    if (automorphisms_exist) {
        std::cout << "Found automorphisms for symmetry analysis: " << automorphism_file << std::endl;
    }
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    // Initialize results
    EDResults results;
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    int num_sites = params.num_sites;

    
    // Create the Hamiltonian with the correct number of sites
    Operator hamiltonian(num_sites, params.spin_length);
    
    // Load the terms from files
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    std::string sym_basis_dir = directory + "/sym_basis";
    std::string sym_blocks_dir = directory + "/sym_blocks";

    // Check if directories exist
    struct stat buffer;
    bool sym_basis_exists = (stat((sym_basis_dir).c_str(), &buffer) == 0);
    bool sym_blocks_exists = (stat((sym_blocks_dir).c_str(), &buffer) == 0);

    // Generate symmetrized basis if needed
    if (!sym_basis_exists) {
        std::cout << "Symmetrized basis not found. Generating..." << std::endl;
    hamiltonian.generateSymmetrizedBasis(directory);
    } else {
        std::cout << "Using existing symmetrized basis from " << sym_basis_dir << std::endl;
    }
    std::vector<int> block_sizes;

    // Build and save symmetrized blocks if needed
    if (!sym_blocks_exists) {
        hamiltonian.buildAndSaveSymmetrizedBlocks(directory);
    }else{
        hamiltonian.loadAllSymmetrizedBlocks(directory);
    }
    // hamiltonian.printEntireSymmetrizedMatrix(directory);
    block_sizes = hamiltonian.symmetrized_block_ham_sizes;
    std::cout << "Found " << block_sizes.size() << " symmetrized blocks." << std::endl;
    std::cout << "With sizes: ";
    for (const auto& size : block_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // Get the sizes of the symmetrized blocks    
    if (block_sizes.empty()) {
        throw std::runtime_error("No symmetrized blocks found. Failed to generate symmetrized basis.");
    }
    
    // Create output directory if needed
    if (!params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + params.output_dir;
        system(cmd.c_str());
    }
    
    // Structure to keep track of eigenvalues and their source block/index
    struct EigenInfo {
        double value;
        int block_idx;
        int eigen_idx;
        
        bool operator<(const EigenInfo& other) const {
            return value < other.value;
        }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    int block_start_dim = 0;
    // 4. Diagonalize each block separately    
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        int block_dim = block_sizes[block_idx];
        std::cerr << "[DEBUG] Diagonalizing block " << block_idx << " dim=" << block_dim << std::endl;
        
        std::cout << "Diagonalizing block " << block_idx + 1 << "/" << block_sizes.size() 
                  << " (dimension: " << block_dim << ")" << std::endl;
        
        // Load the block Hamiltonian from file
        std::string block_file = directory + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
        std::ifstream file(block_file);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open symmetrized block file: " + block_file);
        }
        
        // Read the sparse matrix format (row, col, real, imag)
        std::vector<std::tuple<int, int, Complex>> block_entries;
        int row, col;
        double real, imag;
        
        // Skip the first line which might contain header/metadata
        std::string header_line;
        std::getline(file, header_line);
        while (file >> row >> col >> real >> imag) {
            block_entries.emplace_back(row, col, Complex(real, imag));
        }
        file.close();
        
        // Create a lambda to apply the block Hamiltonian
        auto apply_block_hamiltonian = [&block_entries, block_dim](const Complex* in, Complex* out, int n) {
            // Initialize output to zero
            std::fill(out, out + n, Complex(0.0, 0.0));
            
            // Apply the sparse block matrix
            for (const auto& [row, col, val] : block_entries) {
                out[row] += val * in[col];
            }
        };
        
        // Modify diagonalization parameters for the block
        EDParameters block_params = params;
        block_params.num_eigenvalues = std::min(params.num_eigenvalues - block_start_dim, block_dim);

        if (params.compute_eigenvectors) {
            block_params.output_dir = params.output_dir + "/block_" + std::to_string(block_idx);
            std::string cmd = "mkdir -p " + block_params.output_dir;
            system(cmd.c_str());
        }
        
        // Perform exact diagonalization on this block
        EDResults block_results = exact_diagonalization_core(
            apply_block_hamiltonian, block_dim, method, block_params
        );
        
        // Store the eigenvalues with their block and index information
        for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                block_results.eigenvalues[i],
                static_cast<int>(block_idx),
                static_cast<int>(i)
            });
        }
        
        // Transform eigenvectors or TPQ states if requested
        if (params.compute_eigenvectors || method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::mTPQ_CUDA) {
            
            // Make directory for eigenvectors/states
            std::string eigenvector_dir = params.output_dir + "/eigenvectors";
            std::string cmd = "mkdir -p " + eigenvector_dir;
            system(cmd.c_str());

            // Handle TPQ state transformation
            if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::mTPQ_CUDA) {
                std::cout << "Transforming TPQ states for block " << block_idx << std::endl;
                
                // Search for all TPQ state files in the block directory (not in tpq_data subdirectory)
                std::string temp_list_file = params.output_dir + "/tpq_state_files_" + std::to_string(block_idx) + ".txt";
                std::string find_command = "find \"" + block_params.output_dir + "\" -name \"tpq_state_*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
                system(find_command.c_str());
                
                std::ifstream file_list(temp_list_file);
                if (file_list.is_open()) {
                    std::string tpq_state_file;
                    while (std::getline(file_list, tpq_state_file)) {
                        if (tpq_state_file.empty()) continue;
                        
                        // Extract sample number and beta value from filename
                        size_t sample_pos = tpq_state_file.find("tpq_state_");
                        size_t beta_pos = tpq_state_file.find("_beta=");
                        size_t dat_pos = tpq_state_file.find(".dat");
                        
                        if (sample_pos != std::string::npos && beta_pos != std::string::npos && dat_pos != std::string::npos) {
                            std::string sample_str = tpq_state_file.substr(sample_pos + 10, beta_pos - sample_pos - 10);
                            std::string beta_str = tpq_state_file.substr(beta_pos + 6, dat_pos - beta_pos - 6);
                            
                            std::cout << "Processing TPQ state: sample=" << sample_str << ", beta=" << beta_str << std::endl;
                            
                            // Read the block TPQ state
                            ComplexVector block_tpq_state(block_dim);
                            std::ifstream tpq_file(tpq_state_file, std::ios::binary);
                            if (!tpq_file.is_open()) {
                                std::cerr << "Warning: Could not open TPQ state file: " << tpq_state_file << std::endl;
                                continue;
                            }
                            
                            // Read the TPQ state (assuming binary format as in save_tpq_state)
                            tpq_file.read(reinterpret_cast<char*>(block_tpq_state.data()), block_dim * sizeof(Complex));
                            tpq_file.close();
                            
                            // Transform to full Hilbert space
                            ComplexVector full_tpq_state(1ULL << params.num_sites, Complex(0.0, 0.0));
                            
                            for (int i = 0; i < block_dim; ++i) {
                                std::vector<Complex> basis_vector = hamiltonian.read_sym_basis(i + block_start_dim, directory);
                                for (size_t j = 0; j < full_tpq_state.size(); ++j) {
                                    full_tpq_state[j] += basis_vector[j] * block_tpq_state[i];
                                }
                            }
                            
                            // Save the transformed TPQ state
                            std::string transformed_file = params.output_dir + "/tpq_state_" + sample_str + 
                                                          "_beta=" + beta_str + "_block" + std::to_string(block_idx) + ".dat";
                            
                            std::ofstream out_file(transformed_file, std::ios::binary);
                            if (out_file.is_open()) {
                                out_file.write(reinterpret_cast<const char*>(full_tpq_state.data()), 
                                             full_tpq_state.size() * sizeof(Complex));
                                out_file.close();
                                std::cout << "Saved transformed TPQ state to: " << transformed_file << std::endl;
                            } else {
                                std::cerr << "Warning: Could not save transformed TPQ state to: " << transformed_file << std::endl;
                            }
                        }
                    }
                    file_list.close();
                    std::remove(temp_list_file.c_str());
                }
            }
            
            // Handle regular eigenvector transformation
            if (params.compute_eigenvectors && method != DiagonalizationMethod::mTPQ && method != DiagonalizationMethod::mTPQ_CUDA) {
                std::cout << "Processing eigenvectors for block " << block_idx << std::endl;
                
                for (size_t eigen_idx = 0; eigen_idx < block_results.eigenvalues.size(); ++eigen_idx) {
                    // Path to the block eigenvector
                    std::string block_eigenvector_file = block_params.output_dir + 
                                                       "/eigenvector_" + std::to_string(eigen_idx) + ".dat";

                    std::cout << "Reading eigenvector from: " << block_eigenvector_file << std::endl;

                    // Read the block eigenvector
                    std::vector<Complex> block_eigenvector(block_dim);
                    std::ifstream eigen_file(block_eigenvector_file);
                    if (!eigen_file.is_open()) {
                        std::cerr << "Warning: Could not open eigenvector file: " << block_eigenvector_file << std::endl;
                        continue;
                    }
                    std::string line;
                    int num_entries = 0;
                    while(std::getline(eigen_file, line) && num_entries < block_dim) {
                        if (line.empty()) continue;  // Skip empty lines
                        std::istringstream iss(line);
                        double real_part, imag_part;
                        if (!(iss >> real_part >> imag_part)) {
                            std::cerr << "Warning: Invalid line in eigenvector file: " << line << std::endl;
                            continue;
                        }
                        block_eigenvector[num_entries] = Complex(real_part, imag_part);
                        num_entries++;

                    }

                    // Verify the eigenvector is normalized and contains valid values
                    double norm = 0.0;
                    bool has_invalid_values = false;
                    for (const auto& val : block_eigenvector) {
                        if (!std::isfinite(val.real()) || !std::isfinite(val.imag())) {
                            has_invalid_values = true;
                            break;
                        }
                        norm += std::norm(val);
                    }
                    
                    if (has_invalid_values) {
                        std::cerr << "Warning: Eigenvector contains invalid (NaN/Inf) values in file: " << block_eigenvector_file << std::endl;
                        continue;
                    }
                    
                    norm = std::sqrt(norm);
                    if (norm < 1e-10) {
                        std::cerr << "Warning: Eigenvector has zero norm in file: " << block_eigenvector_file << std::endl;
                        continue;
                    }
                    
                    // Create a file for the transformed eigenvector
                    std::string transformed_file = params.output_dir + "/eigenvector_block" + 
                                                std::to_string(block_idx) + "_" + 
                                                std::to_string(eigen_idx) + ".dat";
                    std::ofstream out_file(transformed_file);
                    if (!out_file.is_open()) {
                        std::cerr << "Warning: Could not open file for transformed eigenvector: " << transformed_file << std::endl;
                        continue;
                    }

                    std::vector<Complex> transformed_eigenvector((1ULL<<params.num_sites), Complex(0.0, 0.0));
                    std::cerr << "[DEBUG] Transforming eigenvector to full dim=" << (1ULL<<params.num_sites) << std::endl;
                    
                    for (size_t i = 0; i < block_dim; ++i) {
                        std::vector<Complex> temp_eigenvector = hamiltonian.read_sym_basis(i+block_start_dim, directory);
                        transformed_eigenvector += temp_eigenvector * block_eigenvector[i];
                    }
                    
                    std::cout << " norm of transformed eigenvector: " << norm << std::endl;
                    if (norm > 1e-10) {  // Avoid division by zero
                        for (auto& val : transformed_eigenvector) {
                            val /= norm;
                        }
                    } else {
                        std::cerr << "Warning: Transformed eigenvector has near-zero norm!" << std::endl;
                    }
                    
                    out_file << transformed_eigenvector.size() << std::endl;
                    // Write the transformed eigenvector to file
                    for (size_t i = 0; i < transformed_eigenvector.size(); ++i) {
                        // Write only non-zero entries
                        if (std::abs(transformed_eigenvector[i]) < 1e-10) continue;
                        out_file << i << " " << transformed_eigenvector[i].real() << " " 
                                << transformed_eigenvector[i].imag() << std::endl;
                    }
                    out_file << std::endl;

                    out_file.close();
                }
            }
        
        }
        block_start_dim += block_dim;  // Update the starting dimension for the next block
    }
    
    // 5. Sort eigenvalues
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    // 6. Keep only the requested number of eigenvalues
    if (all_eigen_info.size() > static_cast<size_t>(params.num_eigenvalues)) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    // 7. Extract the final eigenvalues
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // 8. Create a mapping file for eigenvectors if they were computed
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        std::string mapping_file = params.output_dir + "/eigenvector_mapping.txt";
        std::ofstream map_file(mapping_file);
        if (map_file.is_open()) {
            map_file << "# Global Index, Eigenvalue, Block Index, Block Eigenvalue Index, Filename" << std::endl;
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                std::string filename = "eigenvector_block" + std::to_string(info.block_idx) + 
                                     "_" + std::to_string(info.eigen_idx) + ".dat";
                map_file << i << " " << info.value << " " << info.block_idx << " " 
                        << info.eigen_idx << " " << filename << std::endl;
            }
            map_file.close();
        }
    }
    
    return results;
}



#endif