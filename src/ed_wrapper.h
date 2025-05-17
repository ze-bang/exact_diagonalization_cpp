
#ifndef ED_WRAPPER_H
#define ED_WRAPPER_H

#include "TPQ.h"
#include "CG.h"
#include "lanczos.h"
#include "construct_ham.h"
#include "observables.h"
#include <sys/stat.h>

#ifdef ENABLE_CUDA
#include "lanczos_cuda.h"
#endif

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
    ARPACK,                   // ARPACK
    LANCZOS_CUDA,
    LANCZOS_CUDA_SELECTIVE,
    LANCZOS_CUDA_NO_ORTHO,
    FULL_CUDA
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
    double t_end = 100.0; // End time for time evolution
    double dt = 0.1; // Time step for time evolution

    // Required lattice parameters
    int num_sites = 0; // Number of sites in the system
    float spin_length = 0.5; // Spin length
    int sublattice_size = 1; // Size of the sublattice

    bool calc_observables = false; // Calculate custom observables
    bool measure_spin = false; // Measure spins
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
            
        case DiagonalizationMethod::SHIFT_INVERT_ROBUST:
            std::cout << "Using robust shift-invert Lanczos method with shift = " << params.shift << std::endl;
            shift_invert_lanczos_robust(H, hilbert_space_dim, params.max_iterations, 
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
            lobpcg_eigenvalues(H, hilbert_space_dim, params.max_iterations, 
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
            
        case DiagonalizationMethod::FULL:
            std::cout << "Using full diagonalization" << std::endl;
            full_diagonalization(H, hilbert_space_dim, results.eigenvalues, 
                               params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::OSS:
            std::cout << "Spectrum slicing for full diagonalization" << std::endl;
            optimal_spectrum_solver(
                H, hilbert_space_dim, 
                results.eigenvalues, params.output_dir, 
                params.compute_eigenvectors, params.tolerance
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

        case DiagonalizationMethod::cTPQ:
            std::cout << "Using canonical TPQ method" << std::endl;

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
                            
                            // Determine file type and load accordingly
                            if (observable_file.find("InterAll") != std::string::npos) {
                                obs_op.loadFromInterAllFile(observable_file);
                            } else {
                                obs_op.loadFromFile(observable_file);
                            }
                            
                            params.observables.push_back(obs_op);
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

            canonical_tpq(H, hilbert_space_dim,
                        params.max_iterations, params.num_samples,
                        params.num_measure_freq,
                        results.eigenvalues,
                        params.output_dir,
                        params.delta_tau, 
                        params.compute_eigenvectors,
                        params.num_order,
                        params.calc_observables,params.observables, params.observable_names,
                        params.omega_min, params.omega_max,
                        params.num_points, params.t_end, params.dt, params.spin_length, params.measure_spin, params.sublattice_size); // n_max order for Taylor expansion
            break;
        
        case DiagonalizationMethod::BLOCK_LANCZOS:
            std::cout << "Using block Lanczos method" << std::endl;
            block_lanczos(H, hilbert_space_dim, 
                        params.max_iterations, params.num_eigenvalues, 
                        params.tolerance, results.eigenvalues, 
                        params.output_dir, params.compute_eigenvectors, params.block_size);
            break;
        
        // case DiagonalizationMethod::LANCZOS_CUDA:
        //     std::cout << "Using CUDA Lanczos method" << std::endl;
        //     lanczos_cuda(H, hilbert_space_dim, 
        //                 params.max_iterations, params.num_eigenvalues, 
        //                 params.tolerance, results.eigenvalues, 
        //                 params.output_dir, params.compute_eigenvectors);
        //     break;
        // case DiagonalizationMethod::LANCZOS_CUDA_SELECTIVE:
        //     std::cout << "Using CUDA Lanczos with selective reorthogonalization" << std::endl;
        //     lanczos_selective_reorth_cuda(H, hilbert_space_dim, 
        //                         params.max_iterations, params.num_eigenvalues, 
        //                         params.tolerance, results.eigenvalues, 
        //                         params.output_dir, params.compute_eigenvectors);
        //     break;
        // case DiagonalizationMethod::LANCZOS_CUDA_NO_ORTHO:
        //     std::cout << "Using CUDA Lanczos without reorthogonalization" << std::endl;
        //     lanczos_no_ortho_cuda(H, hilbert_space_dim, 
        //                         params.max_iterations, params.num_eigenvalues, 
        //                         params.tolerance, results.eigenvalues, 
        //                         params.output_dir, params.compute_eigenvectors);
        //     break;

        // case DiagonalizationMethod::FULL_CUDA:
        //     std::cout << "Using CUDA full diagonalization" << std::endl;
        //     full_diagonalization_cuda(H, hilbert_space_dim, params.num_eigenvalues, 
        //                             results.eigenvalues, params.output_dir, 
        //                             params.compute_eigenvectors);
        //     break;

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
    int hilbert_space_dim = pow(2, num_sites);  // 2^num_sites
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

    // Check if we need to convert binary eigenvectors to sparse text format
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        std::string eigenvectors_dir = params.output_dir + "/eigenvectors";
        std::string sparse_dir = params.output_dir + "/eigenvectors";
        
        // Create directory for sparse format eigenvectors
        std::string cmd = "mkdir -p " + sparse_dir;
        system(cmd.c_str());
        
        std::cout << "Converting binary eigenvectors to sparse text format..." << std::endl;
        
        // Find all eigenvector binary files
        for (size_t i = 0; i < results.eigenvalues.size(); i++) {
            std::string binary_file = eigenvectors_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            std::string sparse_file = sparse_dir + "/eigenvector_" + std::to_string(i) + ".txt";
            
            // Check if binary file exists
            std::ifstream bin_file(binary_file, std::ios::binary);
            if (!bin_file.is_open()) {
                std::cerr << "Warning: Could not open binary eigenvector file: " << binary_file << std::endl;
                continue;
            }
            
            // Get file size to determine vector dimension
            bin_file.seekg(0, std::ios::end);
            std::streamsize file_size = bin_file.tellg();
            int vector_dim = file_size / sizeof(Complex);
            bin_file.seekg(0, std::ios::beg);
            
            // Read binary data
            std::vector<Complex> eigenvector(vector_dim);
            bin_file.read(reinterpret_cast<char*>(&eigenvector[0]), file_size);
            bin_file.close();
            
            // Write sparse text format
            std::ofstream sparse_out(sparse_file);
            if (!sparse_out.is_open()) {
                std::cerr << "Error: Could not create sparse eigenvector file: " << sparse_file << std::endl;
                continue;
            }
            
            sparse_out << vector_dim << std::endl;
            
            // Write only non-zero entries
            for (int j = 0; j < vector_dim; j++) {
                if (std::abs(eigenvector[j]) > 1e-10) {
                    sparse_out << j << " " << std::scientific << std::setprecision(15) 
                              << eigenvector[j].real() << " " << eigenvector[j].imag() << std::endl;
                }
            }
            sparse_out.close();
            std::cout << "Converted eigenvector " << i << " to sparse format" << std::endl;
            // Delete binary file after successful conversion to save space
            if (std::remove(binary_file.c_str()) == 0) {
                std::cout << "Deleted binary file: " << binary_file << std::endl;
            } else {
                std::cerr << "Warning: Failed to delete binary file: " << binary_file << std::endl;
            }
        }
        std::cout << "Conversion complete. Sparse format eigenvectors saved in: " << sparse_dir << std::endl;
    }


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
    // if (!sym_blocks_exists) {
    hamiltonian.buildAndSaveSymmetrizedBlocks(directory);
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
        
        // Transform and save eigenvectors if requested
        if (params.compute_eigenvectors) {

            // Make directory for eigenvectors
            std::string eigenvector_dir = params.output_dir + "/eigenvectors";
            std::string cmd = "mkdir -p " + eigenvector_dir;
            system(cmd.c_str());


            std::cout << "Processing eigenvectors for block " << block_idx << std::endl;
            
            for (size_t eigen_idx = 0; eigen_idx < block_results.eigenvalues.size(); ++eigen_idx) {
                // Path to the block eigenvector
                std::string block_eigenvector_file = block_params.output_dir + 
                                                   "/eigenvectors/eigenvector_" + std::to_string(eigen_idx) + ".dat";

                std::cout << "Reading eigenvector from: " << block_eigenvector_file << std::endl;

                // Read the block eigenvector
                std::vector<Complex> block_eigenvector(block_dim);
                std::ifstream eigen_file(block_eigenvector_file, std::ios::binary);
                if (!eigen_file.is_open()) {
                    std::cerr << "Warning: Could not open eigenvector file: " << block_eigenvector_file << std::endl;
                    continue;
                }

                eigen_file.read(reinterpret_cast<char*>(&block_eigenvector[0]), block_dim * sizeof(Complex));
                eigen_file.close();
                
                // Create a file for the transformed eigenvector
                std::string transformed_file = params.output_dir + "/eigenvectors/eigenvector_block" + 
                                            std::to_string(block_idx) + "_" + 
                                            std::to_string(eigen_idx) + ".dat";
                std::ofstream out_file(transformed_file);
                if (!out_file.is_open()) {
                    std::cerr << "Warning: Could not open file for transformed eigenvector: " << transformed_file << std::endl;
                    continue;
                }

                std::vector<Complex> transformed_eigenvector((1ULL<<params.num_sites), Complex(0.0, 0.0));
                
                for (size_t i = 0; i < block_dim; ++i) {
                    std::vector<Complex> temp_eigenvector = hamiltonian.read_sym_basis(i+block_start_dim, directory);
                    transformed_eigenvector += temp_eigenvector * block_eigenvector[i];
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