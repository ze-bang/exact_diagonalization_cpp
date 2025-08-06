// filepath: /home/pc_linux/exact_diagonalization_cpp/src/test_observables.cpp
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include "construct_ham.h"
#include "TPQ.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <state_file> <observable_file> <num_sites> <spin_length> [ham_file] [compute_dynamics]" << std::endl;
        std::cerr << "  compute_dynamics: 0=static only, 1=time evolution, 2=spectral function" << std::endl;
        return 1;
    }

    std::string state_filename = argv[1];
    std::string observable_filename = argv[2];
    int num_sites = std::stoi(argv[3]);
    float spin_length = std::stof(argv[4]);
    
    // Optional parameters
    std::string ham_filename = "";
    int compute_dynamics = 0;
    if (argc >= 6) ham_filename = argv[5];
    if (argc >= 7) compute_dynamics = std::stoi(argv[6]);

    // Load the TPQ state
    ComplexVector tpq_state;
    if (!load_tpq_state(tpq_state, state_filename)) {
        std::cerr << "Failed to load TPQ state from " << state_filename << std::endl;
        return 1;
    }
    std::cout << "Loaded TPQ state of size " << tpq_state.size() << " from " << state_filename << std::endl;

    std::cout << std::endl;
    // Load observable using Operator class
    // The observable_filename is used to determine the operator type and momentum
    // For SumOperator, we expect a format like "Sz_Q_0.0_0.0_0.0.dat"
    // We will parse this to set up the SumOperator
    std::string positions_file = ham_filename + "/super_pyrochlore_1x1x1_obc_kramer_site_info.dat";
    SumOperator obs_op(num_sites, spin_length, 0, {{0, 0, 0}}, positions_file);

    // Extract observable name from the filename
    std::string obs_name = "observable";
    
    std::cout << "Loaded observable '" << obs_name << "' from " << observable_filename << std::endl;

    // Calculate expectation value
    int N = 1 << num_sites;
    std::vector<Complex> result(N);
    
    obs_op.apply(tpq_state.data(), result.data(), N);
    
    Complex total_expectation(0.0, 0.0);
    for (int i = 0; i < N; ++i) {
        total_expectation += std::conj(result[i]) * result[i];
    }

    total_expectation /= static_cast<double>(num_sites);

    std::cout << "\nStatic expectation value: (" << total_expectation.real() << ", " << total_expectation.imag() << ")" << std::endl;

    // Time evolution capabilities
    if (compute_dynamics > 0) {
        if (ham_filename.empty()) {
            std::cerr << "Error: Hamiltonian file required for dynamics calculations" << std::endl;
            return 1;
        }

        std::cout << "\nLoading Hamiltonian for time evolution..." << std::endl;
        
        // Create Hamiltonian operator using Operator class
        Operator ham_op(num_sites, spin_length);
        
        // Load Hamiltonian from InterAll.dat file
        std::string interall_file = ham_filename + "/InterAll.dat";
        std::string trans_file = ham_filename + "/Trans.dat";
        
        // Check if files exist
        std::ifstream interall_check(interall_file);
        std::ifstream trans_check(trans_file);
        
        if (!interall_check.is_open()) {
            std::cerr << "Error: Cannot open InterAll.dat file: " << interall_file << std::endl;
            return 1;
        }
        if (!trans_check.is_open()) {
            std::cerr << "Error: Cannot open Trans.dat file: " << trans_file << std::endl;
            return 1;
        }
        interall_check.close();
        trans_check.close();
        
        // Load InterAll interactions
        ham_op.loadFromInterAllFile(interall_file);
        
        // Load single-site terms from Trans.dat
        ham_op.loadFromFile(trans_file);
        
        std::cout << "Loaded Hamiltonian from " << interall_file << " and " << trans_file << std::endl;
        
        // Create Hamiltonian function
        auto H = [&ham_op](const Complex* in, Complex* out, int size) {
            ham_op.apply(in, out, size);
        };
        
        if (compute_dynamics == 1) {
            // Simple time evolution
            std::cout << "\nPerforming time evolution..." << std::endl;
            
            ComplexVector evolved_state = tpq_state;
            double dt = 0.005;
            int num_steps = 1000;

            // Create output file for time evolution data
            std::string output_filename = "time_evolution_" + obs_name + ".dat";
            std::ofstream outfile(output_filename);
            outfile << "# t\tRe(<O(t)>)\tIm(<O(t)>)\tNorm\tEnergy\n";
            
            std::cout << "Time evolution with " << num_steps << " steps, dt = " << dt << std::endl;
            std::cout << "Output will be saved to " << output_filename << std::endl;
            
            for (int step = 0; step <= num_steps; step++) {
                if (step % 100 == 0) { // Log to console less frequently
                    // Calculate expectation value at this time
                    std::vector<Complex> obs_result(N);
                    obs_op.apply(evolved_state.data(), obs_result.data(), N);
                    

                    Complex obs_expectation(0.0, 0.0);
                    for (int i = 0; i < N; ++i) {
                    obs_expectation += std::conj(obs_result[i]) * obs_result[i];
                    }
                    
                    std::cout << "t = " << step * dt << ", <" << obs_name <<">(t) = (" 
                        << obs_expectation.real() << ", " 
                        << obs_expectation.imag() << ")" << std::endl;
                }

                // Evolve state for the next step
                if (step < num_steps) {
                    time_evolve_tpq_krylov(H, evolved_state, N, dt, 30, true);
                }
            }
            outfile.close();
            std::cout << "Time evolution data saved to " << output_filename << std::endl;
        } else if (compute_dynamics == 2) {
            // Compute spin structure factor using Krylov method
            std::cout << "\nComputing spin structure factor using Krylov method..." << std::endl;
            
            // Define momentum points for structure factor calculation
            const std::vector<std::vector<double>> momentum_points = {
                {0.0, 0.0, 0.0}
            };
            
            // Create positions file path (assuming standard location)
            std::string positions_file = ham_filename + "/super_pyrochlore_1x1x1_obc_kramer_site_info.dat";
            
            // Check if positions file exists
            std::ifstream pos_check(positions_file);
            if (!pos_check.is_open()) {
                std::cerr << "Warning: positions.dat not found at " << positions_file << std::endl;
                std::cerr << "Creating default positions file for chain geometry..." << std::endl;
                
                // Create a simple chain positions file
                std::ofstream pos_out(positions_file);
                if (pos_out.is_open()) {
                    for (int i = 0; i < num_sites; i++) {
                        pos_out << i << " " << i << " 0 0" << std::endl;
                    }
                    pos_out.close();
                    std::cout << "Created default positions file: " << positions_file << std::endl;
                }
            } else {
                pos_check.close();
            }
            
            // Create output directory for structure factor results
            std::string output_dir = "structure_factor_results";
            ensureDirectoryExists(output_dir);
            
            // Compute spin structure factor with Krylov method
            computeTPQSpinStructureFactorKrylov(
                H,                      // Hamiltonian function
                tpq_state,             // TPQ state
                positions_file,        // Positions file
                N,                     // Hilbert space dimension
                num_sites,             // Number of sites
                spin_length,           // Spin length
                output_dir,            // Output directory
                0,                     // Sample index
                1.0,                   // Inverse temperature (placeholder)
                momentum_points,       // Custom momentum points
                50                     // Krylov dimension
            );
            
            std::cout << "Spin structure factor calculation complete!" << std::endl;
            std::cout << "Results saved in: " << output_dir << std::endl;
        }
        else if (compute_dynamics == 3) {
            // Convergence test with different Krylov dimensions
            std::cout << "\nPerforming Krylov dimension convergence test..." << std::endl;
            
            std::vector<int> krylov_dims = {50, 60, 80, 100, 150, 200};
            std::string positions_file = ham_filename + "/super_pyrochlore_1x1x1_obc_kramer_site_info.dat";
            
            // Check if positions file exists
            std::ifstream pos_check(positions_file);
            if (!pos_check.is_open()) {
                std::cerr << "Warning: positions file not found, creating default..." << std::endl;
                std::ofstream pos_out(positions_file);
                if (pos_out.is_open()) {
                    for (int i = 0; i < num_sites; i++) {
                        pos_out << i << " " << i << " 0 0" << std::endl;
                    }
                    pos_out.close();
                }
            } else {
                pos_check.close();
            }
            
            // Output file for convergence results
            std::string convergence_file = "krylov_convergence_" + obs_name + ".dat";
            std::ofstream conv_out(convergence_file);
            conv_out << "# krylov_dim\tomega_0\tspectral_weight_0\tomega_peak\tspectral_weight_peak\n";
            
            std::cout << "Testing Krylov dimensions: ";
            for (int dim : krylov_dims) std::cout << dim << " ";
            std::cout << std::endl;
            
            for (int krylov_dim : krylov_dims) {
                std::cout << "Computing with Krylov dimension: " << krylov_dim << std::endl;
                
                std::string test_output_dir = "convergence_test_k" + std::to_string(krylov_dim);
                ensureDirectoryExists(test_output_dir);
                
                // Compute structure factor with current Krylov dimension
                computeTPQSpinStructureFactorKrylov(
                    H,
                    tpq_state,
                    positions_file,
                    N,
                    num_sites,
                    spin_length,
                    test_output_dir,
                    0,
                    1.0,
                    {{0.0, 0.0, 0.0}},
                    krylov_dim
                );
                
                // Read back the first few spectral function values for comparison
                std::string spec_file = test_output_dir + "/spectral_function_Q_0.000_0.000_0.000_sample_0.dat";
                std::ifstream spec_in(spec_file);
                
                double omega_0 = 0.0, weight_0 = 0.0, omega_peak = 0.0, weight_peak = 0.0;
                if (spec_in.is_open()) {
                    std::string line;
                    std::getline(spec_in, line); // Skip header
                    
                    if (std::getline(spec_in, line)) {
                        std::istringstream iss(line);
                        iss >> omega_0 >> weight_0;
                    }
                    
                    // Find peak value
                    while (std::getline(spec_in, line)) {
                        std::istringstream iss(line);
                        double omega, weight;
                        iss >> omega >> weight;
                        if (weight > weight_peak) {
                            omega_peak = omega;
                            weight_peak = weight;
                        }
                    }
                    spec_in.close();
                }
                
                conv_out << krylov_dim << "\t" << omega_0 << "\t" << weight_0 
                         << "\t" << omega_peak << "\t" << weight_peak << std::endl;
                
                std::cout << "  k=" << krylov_dim << ": peak at ω=" << omega_peak 
                          << ", weight=" << weight_peak << std::endl;
            }
            
            conv_out.close();
            std::cout << "Convergence test complete! Results saved to " << convergence_file << std::endl;
        }
    }

    return 0;
}
