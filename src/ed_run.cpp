#include <iostream>
#include "construct_ham.h"
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include "ed_wrapper.h"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <directory> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --method=<method>    : Diagonalization method (LANCZOS, FULL, ARPACK, etc.)" << std::endl;
        std::cout << "  --eigenvalues=<n>    : Number of eigenvalues to compute" << std::endl;
        std::cout << "  --eigenvectors       : Compute eigenvectors" << std::endl;
        std::cout << "  --output=<dir>       : Output directory" << std::endl;
        std::cout << "  --tolerance=<tol>    : Convergence tolerance" << std::endl;
        std::cout << "  --iterations=<iter>  : Maximum iterations" << std::endl;
        std::cout << "  --block-size=<size>  : Block size for block methods" << std::endl;
        std::cout << "  --shift=<value>      : Shift value for shift-invert methods" << std::endl;
        std::cout << "  --format=<format>    : Hamiltonian file format (STANDARD, SPARSE_MATRIX)" << std::endl;
        std::cout << "  --standard           : standard diagonalization" << std::endl;
        std::cout << "  --symmetrized        : symmetrized diagonalization" << std::endl;
        std::cout << "  --thermo             : Compute thermodynamic data" << std::endl;
        std::cout << "  --temp-min=<value>   : Minimum inverse temperature (for thermo)" << std::endl; 
        std::cout << "  --temp-max=<value>   : Maximum inverse temperature (for thermo)" << std::endl;
        std::cout << "  --temp-bins=<n>      : Number of temperature points (for thermo)" << std::endl;
        std::cout << "  --measure-spin       : Compute spin expectation values" << std::endl;
        std::cout << "  --samples=<n>        : Number of samples for TPQ method" << std::endl;
        std::cout << "  --num_sites=<n>      : Number of sites in the system" << std::endl;
        std::cout << "  --spin_length=<value> : Spin length" << std::endl;
        std::cout << "  --calc_observables   : Calculate all custom operators" << std::endl;
        std::cout << "  --skip_ED            : Skip ED calculation" << std::endl;
        return 1;
    }

    // Directory with Hamiltonian files
    std::string directory = argv[1];

    // Default parameters
    EDParameters params;
    params.num_eigenvalues = (1<<10);
    params.compute_eigenvectors = false;
    params.output_dir = directory + "/output";
    params.tolerance = 1e-10;
    params.max_iterations = (1<<10);
    params.block_size = 10;
    params.shift = 0.0;
    params.temp_min = 0.001;
    params.temp_max = 20.0;
    params.num_temp_bins = 100;
    params.num_samples = 20;

    // Required parameters - must be specified by user
    params.num_sites = 0;
    params.spin_length = 0.0;
    
    // Logging configuration
    bool enable_logging = true;
    std::string log_file_path = directory + "/ed_log.txt";
    std::ofstream log_file;
    
    if (enable_logging) {
        log_file.open(log_file_path, std::ios::out | std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Warning: Could not open log file at " << log_file_path << std::endl;
            enable_logging = false;
        } else {
            // Log start of execution with timestamp
            auto now = std::chrono::system_clock::now();
            std::time_t time_now = std::chrono::system_clock::to_time_t(now);
            log_file << "\n\n=== Execution started at " << std::ctime(&time_now);
            log_file << "Command line: ";
            for (int i = 0; i < argc; i++) {
                log_file << argv[i] << " ";
            }
            log_file << std::endl;
        }
    }
    
    // Helper function to log command output
    auto log_command = [&](const std::string& command, const std::string& output) {
        if (enable_logging && log_file.is_open()) {
            log_file << "\n--- Command: " << command << " ---\n";
            log_file << output << std::endl;
            log_file << "--- End output ---\n";
            log_file.flush();
        }
    };
    bool num_sites_specified = false;
    bool spin_length_specified = false;
    bool full_spectrum = false;
    
    // Default method
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS;
    
    // Default file format
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD;
    
    // Control flags
    bool run_standard = false;
    bool run_symmetrized = false;
    bool compute_thermo = false;
    bool measure_spin = false;
    bool skip_ED = false;

    // Parse command line options
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--method=") == 0) {
            std::string method_str = arg.substr(9);
            if (method_str == "LANCZOS") method = DiagonalizationMethod::LANCZOS;
            else if (method_str == "FULL") {
                method = DiagonalizationMethod::FULL; 
                full_spectrum = true; 
            }
            else if (method_str == "mTPQ") method = DiagonalizationMethod::mTPQ;
            else if (method_str == "cTPQ") method = DiagonalizationMethod::cTPQ;
            else if (method_str == "KRYLOV_SCHUR") method = DiagonalizationMethod::KRYLOV_SCHUR;
            else if (method_str == "DAVIDSON") method = DiagonalizationMethod::DAVIDSON;
            else if (method_str == "BICG") method = DiagonalizationMethod::BICG;
            else if (method_str == "LOBPCG") method = DiagonalizationMethod::LOBPCG;
            else if (method_str == "CG") method = DiagonalizationMethod::CG;
            else if (method_str == "BLOCK_CG") method = DiagonalizationMethod::BLOCK_CG;
            else if (method_str == "LANCZOS_SELECTIVE") method = DiagonalizationMethod::LANCZOS_SELECTIVE;
            else if (method_str == "LANCZOS_NO_ORTHO") method = DiagonalizationMethod::LANCZOS_NO_ORTHO;
            else if (method_str == "SHIFT_INVERT") method = DiagonalizationMethod::SHIFT_INVERT;
            else if (method_str == "SHIFT_INVERT_ROBUST") method = DiagonalizationMethod::SHIFT_INVERT_ROBUST;
            else if (method_str == "ARPACK") method = DiagonalizationMethod::ARPACK;
            else if (method_str == "BLOCK_LANCZOS") method = DiagonalizationMethod::BLOCK_LANCZOS;
            else if (method_str == "OSS") method = DiagonalizationMethod::OSS;
            else if (method_str == "LANCZOS_CUDA") method = DiagonalizationMethod::LANCZOS_CUDA;
            else if (method_str == "LANCZOS_CUDA_SELECTIVE") method = DiagonalizationMethod::LANCZOS_CUDA_SELECTIVE;
            else if (method_str == "LANCZOS_CUDA_NO_ORTHO") method = DiagonalizationMethod::LANCZOS_CUDA_NO_ORTHO;
            else if (method_str == "FULL_CUDA") method = DiagonalizationMethod::FULL_CUDA;
            else std::cerr << "Unknown method: " << method_str << std::endl;
        }
        else if (arg.find("--eigenvalues=") == 0) {
            if (arg.substr(14) == "FULL") {
                full_spectrum = true;
            }
            else{
                params.num_eigenvalues = std::stoi(arg.substr(14));
            }
        }
        else if (arg == "--skip_ED") {
            skip_ED = true;
        }
        else if (arg == "--eigenvectors") {
            params.compute_eigenvectors = true;
        }
        else if (arg.find("--output=") == 0) {
            params.output_dir = arg.substr(9);
        }
        else if (arg.find("--tolerance=") == 0) {
            params.tolerance = std::stod(arg.substr(12));
        }
        else if (arg.find("--iterations=") == 0) {
            params.max_iterations = std::stoi(arg.substr(13));
        }
        else if (arg.find("--block-size=") == 0) {
            params.block_size = std::stoi(arg.substr(13));
        }
        else if (arg.find("--shift=") == 0) {
            params.shift = std::stod(arg.substr(8));
        }
        else if (arg.find("--format=") == 0) {
            std::string format_str = arg.substr(9);
            if (format_str == "STANDARD") format = HamiltonianFileFormat::STANDARD;
            else if (format_str == "SPARSE_MATRIX") format = HamiltonianFileFormat::SPARSE_MATRIX;
            else std::cerr << "Unknown format: " << format_str << std::endl;
        }
        else if (arg == "--standard") {
            run_standard = true;
        }
        else if (arg == "--symmetrized") {
            run_symmetrized = true;
        }
        else if (arg == "--thermo") {
            compute_thermo = true;
        }
        else if (arg.find("--temp-min=") == 0) {
            params.temp_min = std::stod(arg.substr(11));
        }
        else if (arg.find("--temp-max=") == 0) {
            params.temp_max = std::stod(arg.substr(11));
        }
        else if (arg.find("--temp-bins=") == 0) {
            params.num_temp_bins = std::stoi(arg.substr(12));
        }
        else if (arg == "--measure-spin") {
            measure_spin = true;
            // Spin measurements require eigenvectors
            params.compute_eigenvectors = true;
        }
        else if (arg.find("--samples=") == 0) {
            params.num_samples = std::stoi(arg.substr(10));
        }
        else if (arg.find("--num_sites=") == 0) {
            params.num_sites = std::stoi(arg.substr(12));
            num_sites_specified = true;
        }
        else if (arg.find("--spin_length=") == 0) {
            params.spin_length = std::stod(arg.substr(14));
            spin_length_specified = true;
        }
        else if (arg == "--calc_observables") {
            params.compute_eigenvectors = true;
            params.calc_observables = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
        }
    }

    if (!run_standard && !run_symmetrized) {
        run_standard = true; // Default to running standard diagonalization
    }
    
    if (full_spectrum) {
        params.num_eigenvalues = (1ULL << params.num_sites);
    }

    params.max_iterations = std::max(params.max_iterations, params.num_eigenvalues);

    // Check if required parameters were provided
    if (!num_sites_specified) {
        std::cerr << "Error: --num_sites parameter is required" << std::endl;
        std::cout << "Usage: " << argv[0] << " <directory> [options]" << std::endl;
        std::cout << "Required options:" << std::endl;
        std::cout << "  --num_sites=<n>      : Number of sites in the system" << std::endl;
        return 1;
    }

    if (!spin_length_specified) {
        std::cerr << "Error: --spin_length parameter is required" << std::endl;
        std::cout << "Usage: " << argv[0] << " <directory> [options]" << std::endl;
        std::cout << "Required options:" << std::endl;
        std::cout << "  --spin_length=<value> : Spin length" << std::endl;
        return 1;
    }

    
    // Create output directories
    std::string standard_output = params.output_dir;
    std::string symmetrized_output = params.output_dir;
    std::string thermo_output = params.output_dir.substr(0, params.output_dir.rfind("/")) + "/thermo";
    
    std::string cmd = "mkdir -p " + standard_output;
    if (run_symmetrized) cmd += " " + symmetrized_output;
    if (compute_thermo) cmd += " " + thermo_output;
    system(cmd.c_str());
    
    // Store results
    EDResults standard_results;
    EDResults sym_results;
    
    // Run standard diagonalization
    if (run_standard) {
        std::cout << "==========================================" << std::endl;
        std::cout << "Starting Standard Exact Diagonalization" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Method: ";
        switch (method) {
            case DiagonalizationMethod::LANCZOS: std::cout << "Lanczos"; break;
            case DiagonalizationMethod::FULL: std::cout << "Full Diagonalization"; break;
            case DiagonalizationMethod::mTPQ: std::cout << "microcanonical Thermal Pure Quantum (mTPQ)"; break;
            case DiagonalizationMethod::cTPQ: std::cout << "canonical Thermal Pure Quantum (cTPQ)"; break;
            case DiagonalizationMethod::OSS: std::cout << "Optimal spectrum solver"; break;
            default: std::cout << "Other"; break;
        }
        std::cout << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            if (!skip_ED) {
                standard_results = exact_diagonalization_from_directory(
                    directory, method, params, format
                );
            
                // Display eigenvalues
                std::cout << "Eigenvalues (standard):" << std::endl;
                for (size_t i = 0; i < standard_results.eigenvalues.size() && i < 10; i++) {
                    std::cout << i << ": " << standard_results.eigenvalues[i] << std::endl;
                }
                if (standard_results.eigenvalues.size() > 10) {
                    std::cout << "... (" << standard_results.eigenvalues.size() - 10 << " more eigenvalues)" << std::endl;
                }
                
                // Save eigenvalues to file
                std::ofstream standard_file(standard_output + "/eigenvalues.txt");
                if (standard_file.is_open()) {
                    for (const auto& val : standard_results.eigenvalues) {
                        standard_file << val << std::endl;
                    }
                    standard_file.close();
                    std::cout << "Saved " << standard_results.eigenvalues.size() << " eigenvalues to " 
                            << standard_output + "/eigenvalues.txt" << std::endl;
                }
            }

            // If thermodynamic data computed, save it
            if (compute_thermo) {
                if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                    std::ofstream thermo_file(thermo_output + "/thermo_data.txt");
                    if (thermo_file.is_open()) {
                        thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                        for (size_t i = 0; i < standard_results.thermo_data.temperatures.size(); i++) {
                            thermo_file << standard_results.thermo_data.temperatures[i] << " "
                                       << standard_results.thermo_data.energy[i] << " "
                                       << standard_results.thermo_data.specific_heat[i] << " "
                                       << standard_results.thermo_data.entropy[i] << " "
                                       << standard_results.thermo_data.free_energy[i] << std::endl;
                        }
                        thermo_file.close();
                        std::cout << "Saved thermodynamic data to " << thermo_output + "/thermo_data.txt" << std::endl;
                    }
                }
                // Check if full spectrum is calculated
                else if (standard_results.eigenvalues.size() == (1ULL << params.num_sites)) {
                    std::cout << "Full spectrum calculated. Computing thermodynamic properties..." << std::endl;
                    
                    // Call the function to calculate thermodynamics from spectrum
                    ThermodynamicData thermo_data = calculate_thermodynamics_from_spectrum(
                        standard_results.eigenvalues,
                        params.temp_min,  // T_min
                        params.temp_max,  // T_max
                        params.num_temp_bins  // num_points
                    );
                    
                    // Save the calculated thermodynamic data
                    std::ofstream thermo_file(thermo_output + "/thermo_data.txt");
                    if (thermo_file.is_open()) {
                        thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                        for (size_t i = 0; i < thermo_data.temperatures.size(); i++) {
                            thermo_file << thermo_data.temperatures[i] << " "
                                       << thermo_data.energy[i] << " "
                                       << thermo_data.specific_heat[i] << " "
                                       << thermo_data.entropy[i] << " "
                                       << thermo_data.free_energy[i] << std::endl;
                        }
                        thermo_file.close();
                        std::cout << "Saved thermodynamic data to " << thermo_output + "/thermo_data.txt" << std::endl;
                    }
                }
            }
            
            // Measure spin if requested
            if (measure_spin && params.compute_eigenvectors) {
                // This would call the appropriate function to measure spin
                // For now, just print a message
                std::cout << "Spin measurements not implemented yet" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error in standard ED: " << e.what() << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Standard ED completed in " << duration / 1000.0 << " seconds" << std::endl;
    }
    
    // Run symmetrized diagonalization
    if (run_symmetrized) {
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Starting Symmetrized Exact Diagonalization" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Set up parameters for symmetrized diagonalization
        EDParameters sym_params = params;
        sym_params.output_dir = symmetrized_output;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            if (!skip_ED) {
                sym_results = exact_diagonalization_from_directory_symmetrized(
                    directory, method, sym_params, format
                );
                
                // Display eigenvalues
                std::cout << "Eigenvalues (symmetrized):" << std::endl;
                for (size_t i = 0; i < sym_results.eigenvalues.size() && i < 10; i++) {
                    std::cout << i << ": " << sym_results.eigenvalues[i] << std::endl;
                }
                if (sym_results.eigenvalues.size() > 10) {
                    std::cout << "... (" << sym_results.eigenvalues.size() - 10 << " more eigenvalues)" << std::endl;
                }
                
                // Save eigenvalues to file
                std::ofstream sym_file(symmetrized_output + "/eigenvalues.txt");
                if (sym_file.is_open()) {
                    for (const auto& val : sym_results.eigenvalues) {
                        sym_file << val << std::endl;
                    }
                    sym_file.close();
                    std::cout << "Saved " << sym_results.eigenvalues.size() << " eigenvalues to " 
                            << symmetrized_output + "/eigenvalues.txt" << std::endl;
                }
            }

            
            // If thermodynamic data computed, save it
            if (compute_thermo) {
                if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                    std::ofstream thermo_file(thermo_output + "/thermo_data.txt");
                    if (thermo_file.is_open()) {
                        thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                        for (size_t i = 0; i < standard_results.thermo_data.temperatures.size(); i++) {
                            thermo_file << standard_results.thermo_data.temperatures[i] << " "
                                       << standard_results.thermo_data.energy[i] << " "
                                       << standard_results.thermo_data.specific_heat[i] << " "
                                       << standard_results.thermo_data.entropy[i] << " "
                                       << standard_results.thermo_data.free_energy[i] << std::endl;
                        }
                        thermo_file.close();
                        std::cout << "Saved thermodynamic data to " << thermo_output + "/thermo_data.txt" << std::endl;
                    }
                }
                // Check if full spectrum is calculated
                else if (standard_results.eigenvalues.size() == (1ULL << params.num_sites)) {
                    std::cout << "Full spectrum calculated. Computing thermodynamic properties..." << std::endl;
                    
                    // Call the function to calculate thermodynamics from spectrum
                    ThermodynamicData thermo_data = calculate_thermodynamics_from_spectrum(
                        standard_results.eigenvalues,
                        params.temp_min,  // T_min
                        params.temp_max,  // T_max
                        params.num_temp_bins  // num_points
                    );
                    
                    // Save the calculated thermodynamic data
                    std::ofstream thermo_file(thermo_output + "/thermo_data.txt");
                    if (thermo_file.is_open()) {
                        thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                        for (size_t i = 0; i < thermo_data.temperatures.size(); i++) {
                            thermo_file << thermo_data.temperatures[i] << " "
                                       << thermo_data.energy[i] << " "
                                       << thermo_data.specific_heat[i] << " "
                                       << thermo_data.entropy[i] << " "
                                       << thermo_data.free_energy[i] << std::endl;
                        }
                        thermo_file.close();
                        std::cout << "Saved thermodynamic data to " << thermo_output + "/thermo_data.txt" << std::endl;
                    }
                }
            }
            
        }
        catch (const std::exception& e) {
            std::cerr << "Error in symmetrized ED: " << e.what() << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Symmetrized ED completed in " << duration / 1000.0 << " seconds" << std::endl;
    }
    
    // Compare results if both calculations were run
    if (run_standard && run_symmetrized) {
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Comparing Results" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        int compare_count = std::min(standard_results.eigenvalues.size(), sym_results.eigenvalues.size());
        if (compare_count > 0) {
            std::cout << "    Standard      Symmetrized    Difference" << std::endl;
            double max_diff = 0.0;
            double avg_diff = 0.0;
            for (int i = 0; i < compare_count && i < 20; i++) {
                double diff = std::abs(standard_results.eigenvalues[i] - sym_results.eigenvalues[i]);
                max_diff = std::max(max_diff, diff);
                avg_diff += diff;
                std::cout << i << ": " << standard_results.eigenvalues[i] << "  " 
                          << sym_results.eigenvalues[i] << "  " << diff << std::endl;
            }
            if (compare_count > 20) {
                std::cout << "... (" << compare_count - 20 << " more comparisons)" << std::endl;
            }
            avg_diff /= compare_count;
            std::cout << "Maximum difference: " << max_diff << std::endl;
            std::cout << "Average difference: " << avg_diff << std::endl;
        } else {
            std::cout << "No eigenvalues to compare." << std::endl;
        }
    }
    
    return 0;
}

