#include "observables.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <functional>
#include <unistd.h> // For getopt

// Helper function to print usage information
void print_usage() {
    std::cout << "Usage: compute_observables [command] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Available commands:" << std::endl;
    std::cout << "  thermo        - Calculate thermodynamic quantities from energy spectrum" << std::endl;
    std::cout << "  expect        - Calculate expectation value of an operator in an eigenstate" << std::endl;
    std::cout << "  thermal       - Calculate thermal expectation value of an operator" << std::endl;
    std::cout << "  matrix        - Calculate matrix element between two eigenstates" << std::endl;
    std::cout << "  spectral      - Calculate spectral function" << std::endl;
    std::cout << "  suscept       - Calculate dynamical susceptibility" << std::endl;
    std::cout << "  qfi           - Calculate quantum Fisher information" << std::endl;
    std::cout << "  spinexp       - Calculate thermal spin expectation values" << std::endl;
    std::cout << "  spinstate     - Calculate spin expectation values for a specific eigenstate" << std::endl;
    std::cout << std::endl;
    std::cout << "Run 'compute_observables [command] --help' for command-specific options." << std::endl;
}

// Helper function to print command-specific usage
void print_command_usage(const std::string& command) {
    if (command == "thermo") {
        std::cout << "Usage: compute_observables thermo [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -f, --file FILE     Input file with eigenvalues" << std::endl;
        std::cout << "  -o, --output FILE   Output file for results" << std::endl;
        std::cout << "  -l, --tmin TMIN     Minimum temperature (default: 0.01)" << std::endl;
        std::cout << "  -h, --tmax TMAX     Maximum temperature (default: 10.0)" << std::endl;
        std::cout << "  -n, --num NUM       Number of temperature points (default: 100)" << std::endl;
    }
    else if (command == "expect") {
        std::cout << "Usage: compute_observables expect [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -d, --dim DIM       Hilbert space dimension" << std::endl;
        std::cout << "  -o, --operator OP   Operator type (sz, sp, sm, etc.)" << std::endl;
        std::cout << "  -s, --site SITE     Site number for site operators" << std::endl;
        std::cout << "  -a, --state IDX     Eigenstate index (default: 0 for ground state)" << std::endl;
        std::cout << "  -f, --file DIR      Directory with eigenvalues/eigenvectors" << std::endl;
    }
    // Add similar help for other commands
    else if (command == "thermal") {
        std::cout << "Usage: compute_observables thermal [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -d, --dim DIM       Hilbert space dimension" << std::endl;
        std::cout << "  -o, --operator OP   Operator type (sz, sp, sm, etc.)" << std::endl;
        std::cout << "  -s, --site SITE     Site number for site operators" << std::endl;
        std::cout << "  -t, --temp TEMP     Temperature (default: 1.0)" << std::endl;
        std::cout << "  -f, --file DIR      Directory with eigenvalues/eigenvectors" << std::endl;
    }
    // Add other command help here...
    else {
        print_usage();
    }
}

// Define operator function based on type
std::function<void(const Complex*, Complex*, int)> create_operator_function(
    const std::string& op_type, int num_sites, int site, float spin_l) {
    
    if (op_type == "sz") {
        SingleSiteOperator sz(num_sites, spin_l, 2, site);
        return [sz](const Complex* in, Complex* out, int n) {
            std::vector<Complex> in_vec(in, in + n);
            std::vector<Complex> out_vec = sz.apply(in_vec);
            std::copy(out_vec.begin(), out_vec.end(), out);
        };
    }
    else if (op_type == "sp") {
        SingleSiteOperator sp(num_sites, spin_l, 0, site);
        return [sp](const Complex* in, Complex* out, int n) {
            std::vector<Complex> in_vec(in, in + n);
            std::vector<Complex> out_vec = sp.apply(in_vec);
            std::copy(out_vec.begin(), out_vec.end(), out);
        };
    }
    else if (op_type == "sm") {
        SingleSiteOperator sm(num_sites, spin_l, 1, site);
        return [sm](const Complex* in, Complex* out, int n) {
            std::vector<Complex> in_vec(in, in + n);
            std::vector<Complex> out_vec = sm.apply(in_vec);
            std::copy(out_vec.begin(), out_vec.end(), out);
        };
    }
    // Add more operator types as needed
    
    // Default: identity operator
    return [](const Complex* in, Complex* out, int n) {
        std::copy(in, in + n, out);
    };
}

// Function to calculate thermodynamic quantities
void calculate_thermodynamics(int argc, char** argv) {
    std::string input_file;
    std::string output_file = "thermodynamics.dat";
    double t_min = 0.01;
    double t_max = 10.0;
    int num_points = 100;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:o:l:h:n:")) != -1) {
        switch (opt) {
            case 'f': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'l': t_min = std::stod(optarg); break;
            case 'h': t_max = std::stod(optarg); break;
            case 'n': num_points = std::stoi(optarg); break;
            default:
                print_command_usage("thermo");
                return;
        }
    }
    
    if (input_file.empty()) {
        std::cerr << "Error: Input file not specified" << std::endl;
        print_command_usage("thermo");
        return;
    }
    
    // Read eigenvalues from file
    std::vector<double> eigenvalues;
    std::ifstream infile(input_file);
    if (!infile) {
        std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
        return;
    }
    
    double value;
    while (infile >> value) {
        eigenvalues.push_back(value);
    }
    infile.close();
    
    std::cout << "Calculating thermodynamic quantities from " << eigenvalues.size() << " eigenvalues..." << std::endl;
    
    // Calculate thermodynamic quantities
    ThermodynamicData results = calculate_thermodynamics_from_spectrum(eigenvalues, t_min, t_max, num_points);
    
    // Write results to file
    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "Error: Cannot open output file: " << output_file << std::endl;
        return;
    }
    
    outfile << "# T Energy Specific_Heat Entropy Free_Energy" << std::endl;
    for (size_t i = 0; i < results.temperatures.size(); i++) {
        outfile << results.temperatures[i] << " "
                << results.energy[i] << " "
                << results.specific_heat[i] << " "
                << results.entropy[i] << " "
                << results.free_energy[i] << std::endl;
    }
    outfile.close();
    
    std::cout << "Results written to " << output_file << std::endl;
}

// Function to calculate expectation values
void calculate_expectation_values(int argc, char** argv) {
    int dimension = 0;
    std::string op_type = "sz";
    int site = 0;
    int state_idx = 0;
    std::string eig_dir;
    int num_sites = 0;
    float spin_l = 0.5; // Default: spin-1/2
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "d:o:s:a:f:n:l:h")) != -1) {
        switch (opt) {
            case 'd': dimension = std::stoi(optarg); break;
            case 'o': op_type = optarg; break;
            case 's': site = std::stoi(optarg); break;
            case 'a': state_idx = std::stoi(optarg); break;
            case 'f': eig_dir = optarg; break;
            case 'n': num_sites = std::stoi(optarg); break;
            case 'l': spin_l = std::stof(optarg); break;
            case 'h':
                print_command_usage("expect");
                return;
            default:
                print_command_usage("expect");
                return;
        }
    }
    
    if (dimension == 0 || eig_dir.empty() || num_sites == 0) {
        std::cerr << "Error: Required parameters missing" << std::endl;
        print_command_usage("expect");
        return;
    }
    
    std::cout << "Calculating expectation value of " << op_type << " at site " << site << " for state " << state_idx << std::endl;
    
    // Create the operator function
    auto op_func = create_operator_function(op_type, num_sites, site, spin_l);
    
    // Create a dummy Hamiltonian function (we don't need it for this calculation)
    auto H_func = [](const Complex* in, Complex* out, int n) {
        std::copy(in, in + n, out);
    };
    
    // Load eigenstate directly from file
    std::string evec_file = eig_dir + "/eigenvector_" + std::to_string(state_idx) + ".dat";
    ComplexVector eigenstate = load_eigenstate_from_file(evec_file, dimension);
    
    if (eigenstate.empty()) {
        std::cerr << "Error: Failed to load eigenstate" << std::endl;
        return;
    }
    
    // Apply operator directly
    ComplexVector op_psi(dimension);
    std::vector<Complex> in_vec(eigenstate.begin(), eigenstate.end());
    std::vector<Complex> out_vec(dimension);
    
    op_func(in_vec.data(), out_vec.data(), dimension);
    
    // Calculate expectation value
    Complex expectation(0.0, 0.0);
    for (int i = 0; i < dimension; i++) {
        expectation += std::conj(in_vec[i]) * out_vec[i];
    }
    
    std::cout << "Expectation value: " << expectation.real() << " + " << expectation.imag() << "i" << std::endl;
}

// Function to calculate thermal expectation values
void calculate_thermal_expectation_values(int argc, char** argv) {
    int dimension = 0;
    std::string op_type = "sz";
    int site = 0;
    double temperature = 1.0;
    std::string eig_dir;
    int num_sites = 0;
    float spin_l = 0.5; // Default: spin-1/2
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "d:o:s:t:f:n:l:h")) != -1) {
        switch (opt) {
            case 'd': dimension = std::stoi(optarg); break;
            case 'o': op_type = optarg; break;
            case 's': site = std::stoi(optarg); break;
            case 't': temperature = std::stod(optarg); break;
            case 'f': eig_dir = optarg; break;
            case 'n': num_sites = std::stoi(optarg); break;
            case 'l': spin_l = std::stof(optarg); break;
            case 'h':
                print_command_usage("thermal");
                return;
            default:
                print_command_usage("thermal");
                return;
        }
    }
    
    if (dimension == 0 || eig_dir.empty() || num_sites == 0) {
        std::cerr << "Error: Required parameters missing" << std::endl;
        print_command_usage("thermal");
        return;
    }
    
    std::cout << "Calculating thermal expectation value of " << op_type << " at site " << site 
              << " at temperature " << temperature << std::endl;
    
    // Create the operator function
    auto op_func = create_operator_function(op_type, num_sites, site, spin_l);
    
    // Calculate thermal expectation value
    double beta = 1.0 / temperature;
    Complex result = calculate_thermal_expectation(op_func, dimension, beta, eig_dir);
    
    std::cout << "Thermal expectation value: " << result.real() << " + " << result.imag() << "i" << std::endl;
}

// Function to calculate spin expectation values
void compute_spin_expectations_thermal(int argc, char** argv) {
    std::string eig_dir;
    std::string output_dir = ".";
    int num_sites = 0;
    float spin_l = 0.5; // Default: spin-1/2
    double temperature = 1.0;
    bool print_output = true;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:o:n:l:t:ph")) != -1) {
        switch (opt) {
            case 'f': eig_dir = optarg; break;
            case 'o': output_dir = optarg; break;
            case 'n': num_sites = std::stoi(optarg); break;
            case 'l': spin_l = std::stof(optarg); break;
            case 't': temperature = std::stod(optarg); break;
            case 'p': print_output = false; break;
            case 'h':
                print_command_usage("spinexp");
                return;
            default:
                print_command_usage("spinexp");
                return;
        }
    }
    
    if (eig_dir.empty() || num_sites == 0) {
        std::cerr << "Error: Required parameters missing" << std::endl;
        print_command_usage("spinexp");
        return;
    }
    
    std::cout << "Computing spin expectation values for " << num_sites << " sites at temperature " << temperature << std::endl;
    
    compute_spin_expectations(eig_dir, output_dir, num_sites, spin_l, temperature, print_output);
}

void compute_spin_expectations_eigenstate(int argc, char** argv) {
    std::string eigenstate_file;
    std::string output_file = "";
    int num_sites = 0;
    float spin_l = 0.5; // Default: spin-1/2
    bool print_output = true;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:o:n:l:ph")) != -1) {
        switch (opt) {
            case 'f': eigenstate_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'n': num_sites = std::stoi(optarg); break;
            case 'l': spin_l = std::stof(optarg); break;
            case 'p': print_output = false; break;
            case 'h':
                print_command_usage("spinstate");
                return;
            default:
                print_command_usage("spinstate");
                return;
        }
    }
    
    if (eigenstate_file.empty() || num_sites == 0) {
        std::cerr << "Error: Required parameters missing" << std::endl;
        print_command_usage("spinstate");
        return;
    }
    
    std::cout << "Computing spin expectation values for eigenstate in " << eigenstate_file << std::endl;
    
    compute_eigenstate_spin_expectations_from_file(eigenstate_file, num_sites, spin_l, output_file, print_output);
}

// Functions for other calculations can be implemented similarly

// Main function to parse command and dispatch to appropriate function
int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "thermo") {
        calculate_thermodynamics(argc-1, argv+1);
    }
    else if (command == "expect") {
        calculate_expectation_values(argc-1, argv+1);
    }
    else if (command == "thermal") {
        calculate_thermal_expectation_values(argc-1, argv+1);
    }
    else if (command == "spinexp") {
        compute_spin_expectations_thermal(argc-1, argv+1);
    }
    else if (command == "spinstate") {
        compute_spin_expectations_eigenstate(argc-1, argv+1);
    }
    // Add other commands here as they're implemented
    else if (command == "--help" || command == "-h") {
        print_usage();
    }
    else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage();
        return 1;
    }
    
    return 0;
}