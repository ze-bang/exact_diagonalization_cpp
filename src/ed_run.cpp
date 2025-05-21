#include <iostream>
#include "construct_ham.h"
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include "ed_wrapper.h"
#include <map>


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
        std::cout << "  --temp_min=<value>   : Minimum inverse temperature (for thermo)" << std::endl; 
        std::cout << "  --temp_max=<value>   : Maximum inverse temperature (for thermo)" << std::endl;
        std::cout << "  --temp_bins=<n>      : Number of temperature points (for thermo)" << std::endl;
        std::cout << "  --measure_spin       : Compute spin expectation values" << std::endl;
        std::cout << "  --samples=<n>        : Number of samples for TPQ method" << std::endl;
        std::cout << "  --num_sites=<n>      : Number of sites in the system" << std::endl;
        std::cout << "  --spin_length=<value> : Spin length" << std::endl;
        std::cout << "  --calc_observables   : Calculate all custom operators" << std::endl;
        std::cout << "  --skip_ED            : Skip ED calculation" << std::endl;
        std::cout << "  --large_value=<value> : Large value for TPQ" << std::endl;
        std::cout << "  --max_subspace=<value> : Maximum subspace for Davidson" << std::endl;
        std::cout << "  --num_order=<value>    : Order for Taylor expansion" << std::endl;
        std::cout << "  --num_measure_freq=<value> : Frequency of measurements" << std::endl;
        std::cout << "  --delta_tau=<value>    : Time step for imaginary time evolution" << std::endl;
        std::cout << "  --omega_min=<value>    : Minimum frequency for spectral function" << std::endl;
        std::cout << "  --omega_max=<value>    : Maximum frequency for spectral function" << std::endl;
        std::cout << "  --num_points=<value>   : Number of points for spectral function" << std::endl;
        std::cout << "  --t_end=<value>        : End time for time evolution" << std::endl;
        std::cout << "  --dt=<value>           : Time step for time evolution" << std::endl;
        std::cout << "  --help                : Show this help message" << std::endl;
        std::cout << "  --sublattice_size=<value> : Size of the sublattice" << std::endl;
        return 0;
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
    params.num_samples = 1;

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
    bool num_eigenvalues_override = false;

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
            num_eigenvalues_override = true;
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
        else if (arg.find("--block_size=") == 0) {
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
        else if (arg.find("--temp_min=") == 0) {
            params.temp_min = std::stod(arg.substr(11));
        }
        else if (arg.find("--temp_max=") == 0) {
            params.temp_max = std::stod(arg.substr(11));
        }
        else if (arg.find("--temp_bins=") == 0) {
            params.num_temp_bins = std::stoi(arg.substr(12));
        }
        else if (arg == "--measure_spin") {
            measure_spin = true;
            // Spin measurements require eigenvectors
            params.measure_spin = true;
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
        else if (arg.find("--max_subspace=") == 0) {
            params.max_subspace = std::stoi(arg.substr(14));
        }
        else if (arg.find("--num_order=") == 0) {
            params.num_order = std::stoi(arg.substr(12));
        }
        else if (arg.find("--measure_freq=") == 0) {
            params.num_measure_freq = std::stoi(arg.substr(15));
        }
        else if (arg.find("--delta_tau=") == 0) {
            params.delta_tau = std::stod(arg.substr(12));
        }
        else if (arg.find("--large_value=") == 0) {
            params.large_value = std::stod(arg.substr(14));
        }else if (arg.find("--omega_min=") == 0) {
            params.omega_min = std::stod(arg.substr(12));
        }else if (arg.find("--omega_max=") == 0) {
            params.omega_max = std::stod(arg.substr(12));
        }else if (arg.find("--num_points=") == 0) {
            params.num_points = std::stoi(arg.substr(13));
        }else if (arg.find("--t_end=") == 0) {
            params.t_end = std::stod(arg.substr(8));
        }else if (arg.find("--dt=") == 0) {
            params.dt = std::stod(arg.substr(5));
        }else if (arg.find("--sublattice_size=") == 0) {
            params.sublattice_size = std::stoi(arg.substr(18));
        }
        else if (arg == "--help") {
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
            std::cout << "  --measure_spin       : Compute spin expectation values" << std::endl;
            std::cout << "  --samples=<n>        : Number of samples for TPQ method" << std::endl;
            std::cout << "  --num_sites=<n>      : Number of sites in the system" << std::endl;
            std::cout << "  --spin_length=<value> : Spin length" << std::endl;
            std::cout << "  --calc_observables   : Calculate all custom operators" << std::endl;
            std::cout << "  --skip_ED            : Skip ED calculation" << std::endl;
            std::cout << "  --large_value=<value> : Large value for TPQ" << std::endl;
            std::cout << "  --max_subspace=<value> : Maximum subspace for Davidson" << std::endl;
            std::cout << "  --num_order=<value>    : Order for Taylor expansion" << std::endl;
            std::cout << "  --num_measure_freq=<value> : Frequency of measurements" << std::endl;
            std::cout << "  --delta_tau=<value>    : Time step for imaginary time evolution" << std::endl;
            return 0;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
        }
    }

    if (!run_standard && !run_symmetrized) {
        run_standard = true; // Default to running standard diagonalization
    }
    
    if (full_spectrum && !num_eigenvalues_override) {
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
                if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                    std::cout << "Thermal Pure Quantum (TPQ) method completed." << std::endl;
                    return 0;
                }
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
            
            // Generate logspace temperatures to compute thermodynamic data
            std::vector<double> temperatures(params.num_temp_bins);
            for (int i = 0; i < params.num_temp_bins; i++) {
                temperatures[i] = std::exp(std::log(params.temp_min) + i * (std::log(params.temp_max) - std::log(params.temp_min)) / (params.num_temp_bins - 1));
            }

            std::cout << "Compute thermo " << compute_thermo << " full spectrum: " << full_spectrum << std::endl;
            std::cout << "Compute observables " << params.calc_observables << std::endl;
            std::cout << "Compute spin " << measure_spin << std::endl;

            // If thermodynamic data computed, save it
            if (compute_thermo) {
                if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                }
                // Check if full spectrum is calculated
                else {
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
            //Compute observables if requested
            if (params.calc_observables) {
                std::cout << "Calculating observables..." << std::endl;
                // Find all observable files
                std::vector<std::string> observable_files;
                std::vector<std::string> observable_names;
                std::string cmd = "find " + directory + " -name 'observables_*.dat' > " + directory + "/observable_files.txt";
                system(cmd.c_str());

                std::ifstream file_list(directory + "/observable_files.txt");
                if (file_list) {
                    std::string filename;
                    while (std::getline(file_list, filename)) {
                        if (!filename.empty()) {
                            observable_files.push_back(filename);
                            // Extract observable name from filename (e.g., "Sz_X" from "observables_Sz_X.dat")
                            std::string base_name = filename.substr(filename.find_last_of("/\\") + 1);
                            std::string obs_name = base_name.substr(12, base_name.length() - 4); // Remove "observables_" and ".dat"
                            observable_names.push_back(obs_name);
                        }
                    }
                }
                std::cout << "Found " << observable_files.size() << " observable files." << std::endl;

                // Create directory for observable results
                std::string obs_output_dir = params.output_dir + "/observables";
                system(("mkdir -p " + obs_output_dir).c_str());

                

                // Parse observable files and create operators
                for (const auto& filename : observable_files) {
                    // Extract observable name from filename (e.g., "Sz_X" from "observables_Sz_X.dat")
                    std::string base_name = filename.substr(filename.find_last_of("/\\") + 1);
                    std::string obs_name = base_name.substr(11, base_name.length() - 15); // Remove "observables_" and ".dat"
                    
                    std::cout << "Processing observable: " << obs_name << " from " << filename << std::endl;
                    
                    // Open and parse the file
                    std::ifstream infile(filename);
                    if (!infile) {
                        std::cerr << "Error: Cannot open observable file " << filename << std::endl;
                        continue;
                    }
                    
                    // Read the file header and data
                    std::string line;
                    int loc = 0;
                    
                    // Find and extract the 'loc' parameter
                    while (std::getline(infile, line)) {
                        if (line.find("loc") != std::string::npos) {
                            std::istringstream iss(line);
                            std::string dummy;
                            iss >> dummy >> loc;
                            break;
                        }
                    }
                    
                    // Skip separator lines
                    while (std::getline(infile, line) && line.find("=") != std::string::npos) {}
                    
                    // Read operator data
                    std::vector<std::tuple<int, int, double, double>> operator_data;
                    
                    // Parse the data section
                    while (std::getline(infile, line)) {
                        if (line.empty()) continue;
                        
                        std::istringstream iss(line);
                        int type, site;
                        double real_coef, imag_coef;
                        
                        if (iss >> type >> site >> real_coef >> imag_coef) {
                            operator_data.emplace_back(type, site, real_coef, imag_coef);
                        }
                    }
                    
                    // Create operator function
                    auto observable_op = [operator_data, params](const Complex* in, Complex* out, int n) {
                        // Initialize output to zero
                        std::fill(out, out + n, Complex(0.0, 0.0));
                        
                        // Apply each term in the operator
                        for (const auto& term : operator_data) {
                            int type = std::get<0>(term);
                            int site = std::get<1>(term);
                            double real_coef = std::get<2>(term);
                            double imag_coef = std::get<3>(term);
                            Complex coef(real_coef, imag_coef);
                            
                            // Create single-site operator
                            SingleSiteOperator op(params.num_sites, params.spin_length, type, site);
                            
                            // Apply operator to input
                            std::vector<Complex> temp_in(in, in + n);
                            std::vector<Complex> temp_out = op.apply(temp_in);
                            
                            // Add contribution to output
                            for (int i = 0; i < n; i++) {
                                out[i] += coef * temp_out[i];
                            }
                        }
                    };
                    
                    // Calculate spectral function and dynamical susceptibility for all temperatures
                    std::cout << "Calculating spectral function and dynamical susceptibility for " << obs_name << " at all temperatures" << std::endl;
                    
                    for (double T : temperatures) {
                        std::cout << "Processing temperature T=" << T << std::endl;
                        
                        // Calculate spectral function
                        SpectralFunctionData spectral_data = calculate_spectral_function(
                            observable_op,
                            1 << params.num_sites,
                            standard_output,
                            params.omega_min,  // omega_min
                            params.omega_max,   // omega_max
                            params.num_points,   // num_points
                            0.1,    // eta (broadening)
                            T,      // temperature
                            false   // use Gaussian broadening
                        );
                        
                        // Save spectral function data with temperature in filename
                        std::string spectral_file = obs_output_dir + "/spectral_" + obs_name + "_T" + std::to_string(T) + ".dat";
                        std::ofstream spectral_out(spectral_file);
                        if (spectral_out) {
                            spectral_out << "# Frequency Spectral_Function T=" << T << std::endl;
                            for (size_t i = 0; i < spectral_data.frequencies.size(); i++) {
                                spectral_out << spectral_data.frequencies[i] << " " 
                                             << spectral_data.spectral_function[i] << std::endl;
                            }
                            spectral_out.close();
                            std::cout << "Saved spectral function to " << spectral_file << std::endl;
                        }
                        
                        // Calculate dynamical susceptibility
                        DynamicalSusceptibilityData chi_data = calculate_dynamical_susceptibility(
                            observable_op,
                            1 << params.num_sites,
                            standard_output,
                            params.omega_min,  // omega_min
                            params.omega_max,   // omega_max
                            params.num_points,   // num_points
                            0.1,    // eta (broadening)
                            T       // temperature
                        );
                        
                        // Save dynamical susceptibility data with temperature in filename
                        std::string chi_file = obs_output_dir + "/chi_" + obs_name + "_T" + std::to_string(T) + ".dat";
                        std::ofstream chi_out(chi_file);
                        if (chi_out) {
                            chi_out << "# Frequency Chi_Real Chi_Imag T=" << T << std::endl;
                            for (size_t i = 0; i < chi_data.frequencies.size(); i++) {
                                chi_out << chi_data.frequencies[i] << " " 
                                       << chi_data.chi[i].real() << " "
                                       << chi_data.chi[i].imag() << std::endl;
                            }
                            chi_out.close();
                            std::cout << "Saved dynamical susceptibility to " << chi_file << std::endl;
                        }
                    }
                    
                    // Calculate quantum Fisher information for different temperatures
                    std::string qfi_file = obs_output_dir + "/qfi_" + obs_name + ".dat";
                    std::ofstream qfi_out(qfi_file);
                    if (qfi_out) {
                        qfi_out << "# Temperature QFI" << std::endl;
                        for (double T : temperatures) {
                            std::cout << "Calculating QFI for " << obs_name << " at T=" << T << std::endl;
                            double qfi = calculate_quantum_fisher_information(
                                observable_op,
                                1 << params.num_sites,
                                T,
                                standard_output
                            );
                            qfi_out << T << " " << qfi << std::endl;
                        }
                        qfi_out.close();
                        std::cout << "Saved quantum Fisher information to " << qfi_file << std::endl;
                    }

                }
            }

                // Calculate spin expectations if requested
            if (measure_spin) {
                std::cout << "Calculating spin expectations..." << std::endl;
                std::string obs_output_dir = params.output_dir + "/spin_expectations";
                system(("mkdir -p " + obs_output_dir).c_str());
                
                for (double T : temperatures) {
                    std::cout << "Computing spin expectations at T=" << T << std::endl;
                    compute_spin_expectations(
                        standard_output,
                        obs_output_dir,
                        params.num_sites,
                        params.spin_length,
                        T,
                        true  // print output
                    );
                }
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
        
        if (!skip_ED) {
            sym_results = exact_diagonalization_from_directory_symmetrized(
                directory, method, sym_params, format
            );
            
            if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                std::cout << "Thermal Pure Quantum (TPQ) method completed" << std::endl;
                return 0;
            }
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

            if (params.compute_eigenvectors) {
                std::cout << "Computing eigenvectors..." << std::endl;
                // Create directory for renamed eigenvectors
                std::string eigenvectors_dir = symmetrized_output + "/eigenvectors";
                system(("mkdir -p " + eigenvectors_dir).c_str());

                // Read the eigenvector mapping file
                std::string mapping_path = standard_output + "/eigenvector_mapping.txt";
                std::ifstream mapping_file(mapping_path);
                if (!mapping_file) {
                    std::cerr << "Error: Cannot open eigenvector mapping file: " << mapping_path << std::endl;
                }

                // Skip header line
                std::string header_line;
                std::getline(mapping_file, header_line);

                // Create a map from original filename to global index
                std::map<std::string, int> filename_to_global_index;
                int global_idx, block_idx, block_eigen_idx;
                double eigenvalue;
                std::string filename;

                while (mapping_file >> global_idx >> eigenvalue >> block_idx >> block_eigen_idx >> filename) {
                    filename_to_global_index[filename] = global_idx;
                }
                mapping_file.close();

                std::cout << "Loaded mapping for " << filename_to_global_index.size() << " eigenvector files" << std::endl;

                // Process each eigenvector file
                int processed_count = 0;
                for (const auto& entry : filename_to_global_index) {
                    const std::string& src_filename = entry.first;
                    int global_index = entry.second;
                    
                    std::string src_path = standard_output + "/eigenvectors/" + src_filename;
                    std::string dst_path = eigenvectors_dir + "/eigenvector_" + std::to_string(global_index) + ".dat";
                    
                    // Read the source file
                    std::ifstream src_file(src_path);
                    if (!src_file) {
                        std::cerr << "Warning: Cannot open eigenvector file " << src_path << std::endl;
                        continue;
                    }
                    
                    // Write to the destination file
                    std::ofstream dst_file(dst_path);
                    if (!dst_file) {
                        std::cerr << "Error: Cannot create eigenvector file " << dst_path << std::endl;
                        continue;
                    }
                    
                    // Copy content
                    dst_file << src_file.rdbuf();
                    
                    // Close files
                    src_file.close();
                    dst_file.close();
                    
                    // Delete the original file
                    if (std::remove(src_path.c_str()) != 0) {
                        std::cerr << "Warning: Could not delete file " << src_path << std::endl;
                    }
                    
                    processed_count++;
                    if (processed_count % 100 == 0) {
                        std::cout << "Processed " << processed_count << " eigenvector files..." << std::endl;
                    }
                }

                std::cout << "Renamed " << processed_count << " eigenvector files according to global indices" << std::endl;
            }
        }
        std::cout << "Compute thermo " << compute_thermo << " full spectrum: " << full_spectrum << std::endl;
        std::cout << "Compute observables " << params.calc_observables << std::endl;
        std::cout << "Compute spin " << measure_spin << std::endl;
        // Generate logspace temperatures to compute thermodynamic data
        std::vector<double> temperatures(params.num_temp_bins);
        for (int i = 0; i < params.num_temp_bins; i++) {
            temperatures[i] = std::exp(std::log(params.temp_min) + i * (std::log(params.temp_max) - std::log(params.temp_min)) / (params.num_temp_bins - 1));
        }
        // If thermodynamic data computed, save it
        if (compute_thermo) {
            if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
                std::ofstream thermo_file(thermo_output + "/thermo_data.txt");
                if (thermo_file.is_open()) {
                    thermo_file << "# Temperature Energy SpecificHeat Entropy FreeEnergy" << std::endl;
                    for (size_t i = 0; i < sym_results.thermo_data.temperatures.size(); i++) {
                        thermo_file << sym_results.thermo_data.temperatures[i] << " "
                                    << sym_results.thermo_data.energy[i] << " "
                                    << sym_results.thermo_data.specific_heat[i] << " "
                                    << sym_results.thermo_data.entropy[i] << " "
                                    << sym_results.thermo_data.free_energy[i] << std::endl;
                    }
                    thermo_file.close();
                    std::cout << "Saved thermodynamic data to " << thermo_output + "/thermo_data.txt" << std::endl;
                }
            }
            // Check if full spectrum is calculated
            else {
                std::cout << "Full spectrum calculated. Computing thermodynamic properties..." << std::endl;
                
                if (sym_results.eigenvalues.empty()) {
                    std::cerr << "No eigenvalues found in symmetrized results. Trying to load " << std::endl;
                }
                // Load the eigenvalues from the symmetrized results
                std::string eigenvalue_file = symmetrized_output + "/eigenvalues.txt";
                std::ifstream eigenvalue_stream(eigenvalue_file);
                if (!eigenvalue_stream) {
                    std::cerr << "Error: Cannot open eigenvalue file " << eigenvalue_file << std::endl;
                    return 1;
                }
                std::vector<double> eigenvalues;
                double eigenvalue;
                while (eigenvalue_stream >> eigenvalue) {
                    eigenvalues.push_back(eigenvalue);
                }
                eigenvalue_stream.close();
                std::cout << "Loaded " << eigenvalues.size() << " eigenvalues from " << eigenvalue_file << std::endl;

                sym_results.eigenvalues = eigenvalues;

                // Call the function to calculate thermodynamics from spectrum
                ThermodynamicData thermo_data = calculate_thermodynamics_from_spectrum(
                    sym_results.eigenvalues,
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
        //Compute observables if requested
        if (params.calc_observables) {
            std::cout << "Calculating observables..." << std::endl;
            // Find all observable files
            std::vector<std::string> observable_files;
            std::string cmd = "find " + directory + " -name 'observables_*.dat' > " + directory + "/observable_files.txt";
            system(cmd.c_str());

            std::ifstream file_list(directory + "/observable_files.txt");
            if (file_list) {
                std::string filename;
                while (std::getline(file_list, filename)) {
                    if (!filename.empty()) {
                        observable_files.push_back(filename);
                    }
                }
            }
            std::cout << "Found " << observable_files.size() << " observable files." << std::endl;

            // Create directory for observable results
            std::string obs_output_dir = params.output_dir + "/observables";
            system(("mkdir -p " + obs_output_dir).c_str());

            

            // Parse observable files and create operators
            for (const auto& filename : observable_files) {
                // Extract observable name from filename (e.g., "Sz_X" from "observables_Sz_X.dat")
                std::string base_name = filename.substr(filename.find_last_of("/\\") + 1);
                std::string obs_name = base_name.substr(11, base_name.length() - 15); // Remove "observables_" and ".dat"
                
                std::cout << "Processing observable: " << obs_name << " from " << filename << std::endl;
                
                // Open and parse the file
                std::ifstream infile(filename);
                if (!infile) {
                    std::cerr << "Error: Cannot open observable file " << filename << std::endl;
                    continue;
                }
                
                // Read the file header and data
                std::string line;
                int loc = 0;
                
                // Find and extract the 'loc' parameter
                while (std::getline(infile, line)) {
                    if (line.find("loc") != std::string::npos) {
                        std::istringstream iss(line);
                        std::string dummy;
                        iss >> dummy >> loc;
                        break;
                    }
                }
                
                // Skip separator lines
                while (std::getline(infile, line) && line.find("=") != std::string::npos) {}
                
                // Read operator data
                std::vector<std::tuple<int, int, double, double>> operator_data;
                
                // Parse the data section
                while (std::getline(infile, line)) {
                    if (line.empty()) continue;
                    
                    std::istringstream iss(line);
                    int type, site;
                    double real_coef, imag_coef;
                    
                    if (iss >> type >> site >> real_coef >> imag_coef) {
                        operator_data.emplace_back(type, site, real_coef, imag_coef);
                    }
                }
                
                // Create operator function
                auto observable_op = [operator_data, params](const Complex* in, Complex* out, int n) {
                    // Initialize output to zero
                    std::fill(out, out + n, Complex(0.0, 0.0));
                    
                    // Apply each term in the operator
                    for (const auto& term : operator_data) {
                        int type = std::get<0>(term);
                        int site = std::get<1>(term);
                        double real_coef = std::get<2>(term);
                        double imag_coef = std::get<3>(term);
                        Complex coef(real_coef, imag_coef);
                        
                        // Create single-site operator
                        SingleSiteOperator op(params.num_sites, params.spin_length, type, site);
                        
                        // Apply operator to input
                        std::vector<Complex> temp_in(in, in + n);
                        std::vector<Complex> temp_out = op.apply(temp_in);
                        
                        // Add contribution to output
                        for (int i = 0; i < n; i++) {
                            out[i] += coef * temp_out[i];
                        }
                    }
                };
                
                // // Calculate spectral function and dynamical susceptibility for all temperatures
                // std::cout << "Calculating spectral function and dynamical susceptibility for " << obs_name << " at all temperatures" << std::endl;
                
                // for (double T : temperatures) {
                //     std::cout << "Processing temperature T=" << T << std::endl;
                    
                //     // Calculate spectral function
                //     SpectralFunctionData spectral_data = calculate_spectral_function(
                //         observable_op,
                //         1 << params.num_sites,
                //         standard_output,
                //         params.omega_min,  // omega_min
                //         params.omega_max,   // omega_max
                //         params.num_points,   // num_points
                //         0.1,    // eta (broadening)
                //         T,      // temperature
                //         false   // use Gaussian broadening
                //     );
                    
                //     // Save spectral function data with temperature in filename
                //     std::string spectral_file = obs_output_dir + "/spectral_" + obs_name + "_T" + std::to_string(T) + ".dat";
                //     std::ofstream spectral_out(spectral_file);
                //     if (spectral_out) {
                //         spectral_out << "# Frequency Spectral_Function T=" << T << std::endl;
                //         for (size_t i = 0; i < spectral_data.frequencies.size(); i++) {
                //             spectral_out << spectral_data.frequencies[i] << " " 
                //                             << spectral_data.spectral_function[i] << std::endl;
                //         }
                //         spectral_out.close();
                //         std::cout << "Saved spectral function to " << spectral_file << std::endl;
                //     }
                    
                //     // Calculate dynamical susceptibility
                //     DynamicalSusceptibilityData chi_data = calculate_dynamical_susceptibility(
                //         observable_op,
                //         1 << params.num_sites,
                //         standard_output,
                //         params.omega_min,  // omega_min
                //         params.omega_max,   // omega_max
                //         params.num_points,   // num_points
                //         0.1,    // eta (broadening)
                //         T       // temperature
                //     );
                    
                //     // Save dynamical susceptibility data with temperature in filename
                //     std::string chi_file = obs_output_dir + "/chi_" + obs_name + "_T" + std::to_string(T) + ".dat";
                //     std::ofstream chi_out(chi_file);
                //     if (chi_out) {
                //         chi_out << "# Frequency Chi_Real Chi_Imag T=" << T << std::endl;
                //         for (size_t i = 0; i < chi_data.frequencies.size(); i++) {
                //             chi_out << chi_data.frequencies[i] << " " 
                //                     << chi_data.chi[i].real() << " "
                //                     << chi_data.chi[i].imag() << std::endl;
                //         }
                //         chi_out.close();
                //         std::cout << "Saved dynamical susceptibility to " << chi_file << std::endl;
                //     }
                // }
                
                // Calculate quantum Fisher information for different temperatures
                std::string qfi_file = obs_output_dir + "/qfi_" + obs_name + ".dat";
                std::ofstream qfi_out(qfi_file);
                std::vector<double> temperatures = {{0.1, 0.01, 0.001}};

                if (qfi_out) {
                    qfi_out << "# Temperature QFI" << std::endl;
                    for (double T : temperatures) {
                        std::cout << "Calculating QFI for " << obs_name << " at T=" << T << std::endl;
                        double qfi = calculate_quantum_fisher_information(
                            observable_op,
                            1 << params.num_sites,
                            T,
                            standard_output
                        );
                        qfi_out << T << " " << qfi << std::endl;
                    }
                    qfi_out.close();
                    std::cout << "Saved quantum Fisher information to " << qfi_file << std::endl;
                }

            }
        }

            // Calculate spin expectations if requested
        if (measure_spin) {
            std::cout << "Calculating spin expectations..." << std::endl;
            std::string obs_output_dir = params.output_dir + "/spin_expectations";
            system(("mkdir -p " + obs_output_dir).c_str());
            
            std::cout << "Computing spin expectations at T=0" << std::endl;
            compute_spin_expectations(
                directory + "/output",
                obs_output_dir,
                params.num_sites,
                params.spin_length,
                0.0,
                true  // print output
            );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Standard ED completed in " << duration / 1000.0 << " seconds" << std::endl;            
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

