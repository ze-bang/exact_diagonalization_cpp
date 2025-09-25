#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <limits>
#include <cstring>
#include <iomanip> // added for std::setprecision and std::fixed
#include <cstdlib> // for getenv, setenv
#include <omp.h>   // for OpenMP thread control
#include "construct_ham.h"
#include "TPQ.h"
#include "observables.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
namespace fs = std::filesystem;
#include <mpi.h>

void printSpinCorrelation(ComplexVector &state, int num_sites, float spin_length, const std::string &dir) {
    // Compute and print <S_i . S_j> for all pairs (i,j)
    std::vector<std::vector<std::vector<Complex>>> result(2, std::vector<std::vector<Complex>>(num_sites, std::vector<Complex>(num_sites)));

    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            SingleSiteOperator S_plus_i(num_sites, spin_length, 0, i);
            SingleSiteOperator S_plus_j(num_sites, spin_length, 0, j);
            SingleSiteOperator S_z_i(num_sites, spin_length, 2, i);
            SingleSiteOperator S_z_j(num_sites, spin_length, 2, j);

            ComplexVector temp_plus_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_plus_j(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_j(state.size(), Complex(0.0, 0.0));

            S_plus_i.apply(state.data(), temp_plus_i.data(), state.size());
            S_z_i.apply(state.data(), temp_z_i.data(), state.size());
            S_plus_j.apply(state.data(), temp_plus_j.data(), state.size());
            S_z_j.apply(state.data(), temp_z_j.data(), state.size());

            Complex expectation_plus = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_plus += std::conj(temp_plus_i[k]) * temp_plus_j[k];
            }
            result[0][i][j] = expectation_plus;
            Complex expectation_z = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_z += std::conj(temp_z_i[k]) * temp_z_j[k];
            }
            result[1][i][j] = expectation_z;
        }
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_correlation.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "i j <S+_i S-_j> <Sz_i Sz_j>\n";
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            outfile << i << " " << j << " " << result[0][i][j] << " " << result[1][i][j] << "\n";
        }
    }
    outfile.close();

    // Print sublattice correlations
    std::ofstream subfile(dir + "/sublattice_correlation.txt");
    subfile << std::fixed << std::setprecision(6);
    subfile << "sub_i sub_j <S+_i S-_j>_sum <Sz_i Sz_j>_sum count\n";
    
    // Compute sublattice sums
    std::vector<std::vector<Complex>> sublattice_sums_plus(4, std::vector<Complex>(4, 0.0));
    std::vector<std::vector<Complex>> sublattice_sums_z(4, std::vector<Complex>(4, 0.0));
    std::vector<std::vector<int>> sublattice_counts(4, std::vector<int>(4, 0));
    
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            int sub_i = i % 4;
            int sub_j = j % 4;
            sublattice_sums_plus[sub_i][sub_j] += result[0][i][j];
            sublattice_sums_z[sub_i][sub_j] += result[1][i][j];
            sublattice_counts[sub_i][sub_j]++;
        }
    }
    
    // Write sublattice results
    for (int sub_i = 0; sub_i < 4; sub_i++) {
        for (int sub_j = 0; sub_j < 4; sub_j++) {
            subfile << sub_i << " " << sub_j << " " 
                   << sublattice_sums_plus[sub_i][sub_j] << " "
                   << sublattice_sums_z[sub_i][sub_j] << " "
                   << sublattice_counts[sub_i][sub_j] << "\n";
        }
    }
    subfile.close();

    // Print total sums for verification
    std::ofstream sumfile(dir + "/total_sums.txt");
    sumfile << std::fixed << std::setprecision(6);

    Complex total_plus_sum = 0.0;
    Complex total_z_sum = 0.0;
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            total_plus_sum += result[0][i][j];
            total_z_sum += result[1][i][j];
        }
    }

    sumfile << "Total <S+_i S-_j> sum: " << total_plus_sum << "\n";
    sumfile << "Total <Sz_i Sz_j> sum: " << total_z_sum << "\n";
    sumfile.close();

    std::cout << "Total correlation sums:" << std::endl;
    std::cout << "  <S+_i S-_j> sum: " << total_plus_sum << std::endl;
    std::cout << "  <Sz_i Sz_j> sum: " << total_z_sum << std::endl;

    std::cout << "Spin correlation data saved to spin_correlation.txt" << std::endl;

    // Print sublattice correlations
    std::ofstream subfile_sans_diag(dir + "/sublattice_correlation_sans_diag.txt");
    subfile_sans_diag << std::fixed << std::setprecision(6);
    subfile_sans_diag << "sub_i sub_j <S+_i S-_j>_sum <Sz_i Sz_j>_sum count\n";

    // Compute sublattice sums
    std::vector<std::vector<Complex>> sublattice_sums_plus_sans_diag(4, std::vector<Complex>(4, 0.0));
    std::vector<std::vector<Complex>> sublattice_sums_z_sans_diag(4, std::vector<Complex>(4, 0.0));
    std::vector<std::vector<int>> sublattice_counts_sans_diag(4, std::vector<int>(4, 0));
    
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            if (i == j) continue; // Skip diagonal elements
            int sub_i = i % 4;
            int sub_j = j % 4;
            sublattice_sums_plus_sans_diag[sub_i][sub_j] += result[0][i][j];
            sublattice_sums_z_sans_diag[sub_i][sub_j] += result[1][i][j];
            sublattice_counts_sans_diag[sub_i][sub_j]++;
        }
    }
    
    // Write sublattice results
    for (int sub_i = 0; sub_i < 4; sub_i++) {
        for (int sub_j = 0; sub_j < 4; sub_j++) {
            subfile_sans_diag << sub_i << " " << sub_j << " " 
                   << sublattice_sums_plus_sans_diag[sub_i][sub_j] << " "
                   << sublattice_sums_z_sans_diag[sub_i][sub_j] << " "
                   << sublattice_counts_sans_diag[sub_i][sub_j] << "\n";
        }
    }
    subfile_sans_diag.close();

    // Print total sums for verification
    std::ofstream sumfile_sans_diag(dir + "/total_sums.txt");
    sumfile_sans_diag << std::fixed << std::setprecision(6);

    Complex total_plus_sum_sans_diag = 0.0;
    Complex total_z_sum_sans_diag = 0.0;
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            if (i == j) continue; // Skip diagonal elements
            total_plus_sum_sans_diag += result[0][i][j];
            total_z_sum_sans_diag += result[1][i][j];
        }
    }

    sumfile_sans_diag << "Total <S+_i S-_j> sum: " << total_plus_sum_sans_diag << "\n";
    sumfile_sans_diag << "Total <Sz_i Sz_j> sum: " << total_z_sum_sans_diag << "\n";
    sumfile_sans_diag.close();

    std::cout << "Total correlation sums:" << std::endl;
    std::cout << "  <S+_i S-_j> sum: " << total_plus_sum_sans_diag << std::endl;
    std::cout << "  <Sz_i Sz_j> sum: " << total_z_sum_sans_diag << std::endl;

    std::cout << "Spin correlation data saved to spin_correlation.txt" << std::endl;

    // Print diagonal-only correlations
    std::ofstream diagfile(dir + "/diagonal_correlation.txt");
    diagfile << std::fixed << std::setprecision(6);
    diagfile << "i <S+_i S-_i> <Sz_i Sz_i>\n";
    
    for (int i = 0; i < num_sites; i++) {
        diagfile << i << " " << result[0][i][i] << " " << result[1][i][i] << "\n";
    }
    diagfile.close();
    
    // Print sublattice diagonal correlations
    std::ofstream sub_diagfile(dir + "/sublattice_diagonal_correlation.txt");
    sub_diagfile << std::fixed << std::setprecision(6);
    sub_diagfile << "sub_i <S+_i S-_i>_sum <Sz_i Sz_i>_sum count\n";
    
    std::vector<Complex> sublattice_diag_plus(4, 0.0);
    std::vector<Complex> sublattice_diag_z(4, 0.0);
    std::vector<int> sublattice_diag_counts(4, 0);
    
    for (int i = 0; i < num_sites; i++) {
        int sub_i = i % 4;
        sublattice_diag_plus[sub_i] += result[0][i][i];
        sublattice_diag_z[sub_i] += result[1][i][i];
        sublattice_diag_counts[sub_i]++;
    }
    
    for (int sub_i = 0; sub_i < 4; sub_i++) {
        sub_diagfile << sub_i << " " 
                     << sublattice_diag_plus[sub_i] << " "
                     << sublattice_diag_z[sub_i] << " "
                     << sublattice_diag_counts[sub_i] << "\n";
    }
    sub_diagfile.close();
    
    std::cout << "Diagonal correlation data saved to diagonal_correlation.txt" << std::endl;

}

// Helper function to distribute operators among ranks
struct OperatorInfo {
    size_t q_idx;
    size_t combo_idx;
    size_t sublattice_1;
    size_t sublattice_2;
    std::string name;
};

std::vector<OperatorInfo> distributeOperators(int rank, int size, 
    const std::vector<std::vector<double>>& momentum_points,
    const std::vector<std::pair<int, int>>& spin_combinations,
    const std::vector<const char*>& spin_combination_names,
    const std::string& method) {
    
    std::vector<OperatorInfo> all_operators;
    
    // Build complete operator list based on method
    if (method == "taylor") {
        for (size_t q_idx = 0; q_idx < momentum_points.size(); ++q_idx) {
            const auto &Q = momentum_points[q_idx];
            for (size_t combo_idx = 0; combo_idx < spin_combinations.size(); ++combo_idx) {
                OperatorInfo op_info;
                op_info.q_idx = q_idx;
                op_info.combo_idx = combo_idx;
                op_info.sublattice_1 = 0; // Not used for taylor
                op_info.sublattice_2 = 0; // Not used for taylor
                
                std::stringstream name_ss;
                name_ss << spin_combination_names[combo_idx] << "_q" << "_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                op_info.name = name_ss.str();
                all_operators.push_back(op_info);
            }
        }
    } else if (method == "pedantic") {
        for (size_t q_idx = 0; q_idx < momentum_points.size(); ++q_idx) {
            const auto &Q = momentum_points[q_idx];
            for (size_t combo_idx = 0; combo_idx < spin_combinations.size(); ++combo_idx) {
                for (size_t sublattice = 0; sublattice < 4; ++sublattice) {
                    for (size_t sublattice2 = 0; sublattice2 < 4; ++sublattice2) {
                        OperatorInfo op_info;
                        op_info.q_idx = q_idx;
                        op_info.combo_idx = combo_idx;
                        op_info.sublattice_1 = sublattice;
                        op_info.sublattice_2 = sublattice2;
                        
                        std::stringstream name_ss;
                        name_ss << spin_combination_names[combo_idx] << "_sub" << sublattice 
                               << "_sub" << sublattice2 << "_q" << "_Qx" << Q[0] 
                               << "_Qy" << Q[1] << "_Qz" << Q[2];
                        op_info.name = name_ss.str();
                        all_operators.push_back(op_info);
                    }
                }
            }
        }
    }
    
    // Distribute operators to this rank
    std::vector<OperatorInfo> local_operators;
    for (size_t i = rank; i < all_operators.size(); i += size) {
        local_operators.push_back(all_operators[i]);
    }
    
    return local_operators;
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Configure OpenMP to avoid resource exhaustion
    // Set reasonable defaults if not already set
    char* omp_num_threads = std::getenv("OMP_NUM_THREADS");
    if (omp_num_threads == nullptr) {
        // Default: use 1 thread per MPI rank to avoid oversubscription
        omp_set_num_threads(1);
        if (rank == 0) {
            std::cout << "OMP_NUM_THREADS not set. Using 1 OpenMP thread per MPI rank." << std::endl;
        }
    } else {
        int num_omp_threads = std::atoi(omp_num_threads);
        if (num_omp_threads <= 0) {
            omp_set_num_threads(1);
            if (rank == 0) {
                std::cout << "Invalid OMP_NUM_THREADS value. Using 1 OpenMP thread per MPI rank." << std::endl;
            }
        } else {
            omp_set_num_threads(num_omp_threads);
            if (rank == 0) {
                std::cout << "Using " << num_omp_threads << " OpenMP threads per MPI rank." << std::endl;
            }
        }
    }
    
    // Report thread configuration
    if (rank == 0) {
        std::cout << "MPI Configuration: " << size << " ranks" << std::endl;
        std::cout << "OpenMP Configuration: " << omp_get_max_threads() << " threads per rank" << std::endl;
        std::cout << "Total threads: " << size * omp_get_max_threads() << std::endl;
        
        // Warn about potential oversubscription
        int total_threads = size * omp_get_max_threads();
        if (total_threads > 64) {  // Reasonable threshold for most systems
            std::cout << "WARNING: Using " << total_threads << " total threads." << std::endl;
            std::cout << "         This may cause resource exhaustion on some systems." << std::endl;
            std::cout << "         Consider reducing OMP_NUM_THREADS if you encounter errors." << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (argc < 6 || argc > 9) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory> <num_sites> <spin_length> <krylov_dim_or_nmax> <spin_combinations> [method] [dt,t_end] [steps]" << std::endl;
            std::cerr << "  method (optional): krylov (default) | taylor | pedantic" << std::endl;
            std::cerr << "  dt,t_end (optional, only for taylor/pedantic): e.g. 0.01,50.0" << std::endl;
            std::cerr << "  spin_combinations format: \"op1,op2;op3,op4;...\" where op is 0(Sp), 1(Sm), or 2(Sz)" << std::endl;
            std::cerr << "  Example: \"0,1;2,2\" for SpSm, SzSz combinations" << std::endl;
            std::cerr << "  Note: Operator-level parallelization - each rank evolves a subset of operators" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Threading recommendations:" << std::endl;
            std::cerr << "  For large systems, set: export OMP_NUM_THREADS=1" << std::endl;
            std::cerr << "  Or use: export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK))" << std::endl;
            std::cerr << "  Total threads should not exceed available CPU cores" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string directory = argv[1];
    int num_sites = std::stoi(argv[2]);
    float spin_length = std::stof(argv[3]);
    int krylov_dim_or_nmax = std::stoi(argv[4]);
    std::string spin_combinations_str = argv[5];
    std::string method = (argc >= 7) ? std::string(argv[6]) : std::string("krylov");
    double dt_opt = 0.01;
    double t_end_opt = 50.0;
    int steps_opt = -1; // optional
    if (argc >= 8) {
        // Parse dt,t_end combined argument
        std::string dt_tend = argv[7];
        auto comma_pos = dt_tend.find(',');
        if (comma_pos != std::string::npos) {
            try {
                dt_opt = std::stod(dt_tend.substr(0, comma_pos));
                t_end_opt = std::stod(dt_tend.substr(comma_pos + 1));
            } catch (...) {
                if (rank == 0) {
                    std::cerr << "Warning: failed to parse dt,t_end argument. Using defaults 0.01,50.0" << std::endl;
                }
            }
        }
    }
    if (argc == 9) {
        try { steps_opt = std::stoi(argv[8]); } catch (...) { steps_opt = -1; }
    }

    // Parse spin combinations
    std::vector<std::pair<int, int>> spin_combinations;
    std::stringstream ss(spin_combinations_str);
    std::string pair_str;
    
    while (std::getline(ss, pair_str, ';')) {
        std::stringstream pair_ss(pair_str);
        std::string op1_str, op2_str;
        
        if (std::getline(pair_ss, op1_str, ',') && std::getline(pair_ss, op2_str)) {
            try {
                int op1 = std::stoi(op1_str);
                int op2 = std::stoi(op2_str);
                
                if (op1 >= 0 && op1 <= 2 && op2 >= 0 && op2 <= 2) {
                    spin_combinations.push_back({op1, op2});
                } else {
                    if (rank == 0) {
                        std::cerr << "Warning: Invalid spin operator " << op1 << "," << op2 
                                  << ". Operators must be 0, 1, or 2." << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "Warning: Failed to parse spin combination: " << pair_str << std::endl;
                }
            }
        }
    }
    
    if (spin_combinations.empty()) {
        if (rank == 0) {
            std::cerr << "Error: No valid spin combinations provided. Using default SzSz." << std::endl;
        }
        spin_combinations = {{2, 2}};
    }

    auto spin_combination_name = [](int op) {
        switch (op) {
            case 2:
                return "Sz";
            case 0:
                return "Sp";
            case 1:
                return "Sm";
            default:
                return "Unknown";
        }
    };

    std::vector<const char*> spin_combination_names;
    for (const auto& pair : spin_combinations) {
        int first = pair.first;
        first = first == 2 ? 2 : 1 - first; // Convert 0->1(Sp), 1->0(Sm)
        std::string combined_name = std::string(spin_combination_name(first)) + std::string(spin_combination_name(pair.second));
        char* name = new char[combined_name.size() + 1];
        std::strcpy(name, combined_name.c_str());
        spin_combination_names.push_back(name);
    }

    // Regex to match tpq_state_i_beta=*.dat files where i is the sample index
    std::regex state_pattern("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");

    // Load Hamiltonian (all processes need this)
    if (rank == 0) {
        std::cout << "Loading Hamiltonian..." << std::endl;
    }
    
    // Ensure consistent threading during Hamiltonian operations
    int saved_num_threads = omp_get_max_threads();
    
    Operator ham_op(num_sites, spin_length);
    
    std::string interall_file = directory + "/InterAll.dat";
    std::string trans_file = directory + "/Trans.dat";
    std::string positions_file = directory + "/positions.dat";
    
    // Check if files exist
    if (!fs::exists(interall_file) || !fs::exists(trans_file)) {
        if (rank == 0) {
            std::cerr << "Error: Hamiltonian files not found in " << directory << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Load Hamiltonian with controlled threading
    try {
        ham_op.loadFromInterAllFile(interall_file);
        ham_op.loadFromFile(trans_file);
        
        if (rank == 0) {
            std::cout << "Hamiltonian loaded successfully." << std::endl;
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error loading Hamiltonian: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    auto H = [&ham_op](const Complex* in, Complex* out, int size) {
        ham_op.apply(in, out, size);
    };
    
    // Use 64-bit to compute Hilbert space dimension and guard against int overflow
    size_t N64 = 1ULL << num_sites;
    if (N64 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: 2^num_sites exceeds 32-bit int range (num_sites=" << num_sites
                      << "). Refactor APIs to use size_t for N or reduce num_sites." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    int N = static_cast<int>(N64);
    
    // Define momentum points
    const std::vector<std::vector<double>> momentum_points = {
        {0.0, 0.0, 0.0},
        {0, 0, 2*M_PI},
        {0, 0, 4*M_PI},
        {4*M_PI, 4*M_PI, 0}
    };
    
    // Create output directory (only rank 0)
    std::string output_base_dir = directory + "/structure_factor_results";
    if (rank == 0) {
        ensureDirectoryExists(output_base_dir);
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Collect all tpq_state files from the output subdirectory (all ranks need this info)
    std::vector<std::string> tpq_files;
    std::vector<int> sample_indices;
    std::vector<double> beta_values;
    std::vector<std::string> beta_strings;
    
    if (rank == 0) {
        std::string tpq_directory = directory + "/output";
        for (const auto& entry : fs::directory_iterator(tpq_directory)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::smatch match;
            
            if (std::regex_match(filename, match, state_pattern)) {
                tpq_files.push_back(entry.path().string());
                sample_indices.push_back(std::stoi(match[1]));
                beta_strings.push_back(match[2]);
                beta_values.push_back(std::stod(match[2]));
            }
        }

        // Optionally include zero-temperature ground-state eigenvector
        const std::string gs_file = tpq_directory + "/eigenvectors/eigenvector_0.dat";
        if (fs::exists(gs_file)) {
            tpq_files.push_back(gs_file);
            sample_indices.push_back(0); // use 0 as a conventional index for ground state
            beta_strings.push_back("inf");
            beta_values.push_back(std::numeric_limits<double>::infinity());
        }
        
        std::cout << "Found " << tpq_files.size() << " state file(s) to process (including ground state if present)" << std::endl;
        std::cout << "Using " << size << " MPI processes for operator-level parallelization" << std::endl;
    }
    
    // Broadcast the number of files to all processes
    int num_files = tpq_files.size();
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (num_files == 0) {
        if (rank == 0) {
            std::cout << "No TPQ state files found." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    
    // Resize vectors on non-root processes
    if (rank != 0) {
        tpq_files.resize(num_files);
        sample_indices.resize(num_files);
        beta_values.resize(num_files);
        beta_strings.resize(num_files);
    }
    
    // Broadcast file information to all processes
    for (int i = 0; i < num_files; i++) {
        int filename_size;
        if (rank == 0) {
            filename_size = tpq_files[i].size();
        }
        MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            tpq_files[i].resize(filename_size);
        }
        MPI_Bcast(&tpq_files[i][0], filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        
        // Broadcast beta string
        int beta_str_size;
        if (rank == 0) {
            beta_str_size = beta_strings[i].size();
        }
        MPI_Bcast(&beta_str_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            beta_strings[i].resize(beta_str_size);
        }
        MPI_Bcast(&beta_strings[i][0], beta_str_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    MPI_Bcast(sample_indices.data(), num_files, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_values.data(), num_files, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Distribute operators among ranks for operator-level parallelization
    auto local_operators = distributeOperators(rank, size, momentum_points, 
                                             spin_combinations, spin_combination_names, method);
    
    if (rank == 0) {
        std::cout << "Total operators: " << (momentum_points.size() * spin_combinations.size() * 
                     (method == "pedantic" ? 16 : 1)) << std::endl;
        std::cout << "Rank 0 handling " << local_operators.size() << " operators" << std::endl;
    }
    
    // Process all TPQ files for the operators assigned to this rank
    int local_processed_count = 0;
    
    if (method == "spin_correlation") {
        // Special case: spin_correlation doesn't use operator parallelization
        for (int file_idx = 0; file_idx < num_files; file_idx++) {
            if (file_idx % size != rank) continue; // Simple file-level distribution for this method
            
            int sample_index = sample_indices[file_idx];
            double beta = beta_values[file_idx];
            std::string beta_str = beta_strings[file_idx];
            std::string filename = fs::path(tpq_files[file_idx]).filename().string();
            
            std::cout << "Rank " << rank << " processing " << filename << " (sample " << sample_index << ", beta = " << beta << ")" << std::endl;
            
            // Load state
            ComplexVector tpq_state;
            bool loaded_ok = false;
            if (filename.find("eigenvector") != std::string::npos) {
                loaded_ok = load_raw_data(tpq_state, tpq_files[file_idx], N64);
            } else {
                loaded_ok = load_tpq_state(tpq_state, tpq_files[file_idx]);
            }
            if (!loaded_ok || (int)tpq_state.size() != N) {
                std::cerr << "Rank " << rank << " failed to load or validate state from " << filename << std::endl;
                continue;
            }
            
            std::string output_dir = output_base_dir + "/beta_" + beta_str;
            ensureDirectoryExists(output_dir);
            printSpinCorrelation(tpq_state, num_sites, spin_length, output_dir);
            local_processed_count++;
        }
    } else if (method == "krylov") {
        // Use existing krylov method (file-level parallelization for now)
        for (int file_idx = 0; file_idx < num_files; file_idx++) {
            if (file_idx % size != rank) continue;
            
            int sample_index = sample_indices[file_idx];
            double beta = beta_values[file_idx];
            std::string beta_str = beta_strings[file_idx];
            std::string filename = fs::path(tpq_files[file_idx]).filename().string();
            
            std::cout << "Rank " << rank << " processing " << filename << " (sample " << sample_index << ", beta = " << beta << ")" << std::endl;
            
            ComplexVector tpq_state;
            bool loaded_ok = false;
            if (filename.find("eigenvector") != std::string::npos) {
                loaded_ok = load_raw_data(tpq_state, tpq_files[file_idx], N64);
            } else {
                loaded_ok = load_tpq_state(tpq_state, tpq_files[file_idx]);
            }
            if (!loaded_ok || (int)tpq_state.size() != N) {
                std::cerr << "Rank " << rank << " failed to load or validate state from " << filename << std::endl;
                continue;
            }
            
            std::string output_dir = output_base_dir + "/beta_" + beta_str;
            ensureDirectoryExists(output_dir);
            
            int krylov_dim = krylov_dim_or_nmax;
            computeTPQSpinStructureFactorKrylov(
                H, tpq_state, positions_file, N, num_sites, spin_length,
                output_dir, sample_index, beta, momentum_points, krylov_dim,
                spin_combinations, spin_combination_names
            );
            local_processed_count++;
        }
    } else if (method == "taylor" || method == "pedantic") {
        // Operator-level parallelization for taylor and pedantic methods
        if (rank == 0) {
            std::cout << "Using " << method << " evolution with operator-level parallelization (n_max=" 
                      << krylov_dim_or_nmax << ", dt=" << dt_opt << ", t_end=" << t_end_opt << ")" << std::endl;
        }
        
        // Build time evolution operator (all ranks need this)
        if (rank == 0) {
            std::cout << "Creating time evolution operator with " << omp_get_max_threads() 
                      << " OpenMP threads..." << std::endl;
        }
        
        // Ensure we don't exceed thread limits during operator construction
        omp_set_dynamic(0); // Disable dynamic thread adjustment
        auto U_t = create_time_evolution_operator(H, dt_opt, krylov_dim_or_nmax, true);
        
        if (rank == 0) {
            std::cout << "Time evolution operator created successfully." << std::endl;
        }
        
        // Process each TPQ state file for local operators
        for (int file_idx = 0; file_idx < num_files; file_idx++) {
            int sample_index = sample_indices[file_idx];
            double beta = beta_values[file_idx];
            std::string beta_str = beta_strings[file_idx];
            std::string filename = fs::path(tpq_files[file_idx]).filename().string();
            
            std::cout << "Rank " << rank << " processing " << filename << " with " 
                      << local_operators.size() << " operators (sample " << sample_index 
                      << ", beta = " << beta << ")" << std::endl;
            
            // All ranks load all states (needed for operator evolution)
            ComplexVector tpq_state;
            bool loaded_ok = false;
            if (filename.find("eigenvector") != std::string::npos) {
                loaded_ok = load_raw_data(tpq_state, tpq_files[file_idx], N64);
            } else {
                loaded_ok = load_tpq_state(tpq_state, tpq_files[file_idx]);
            }
            if (!loaded_ok || (int)tpq_state.size() != N) {
                std::cerr << "Rank " << rank << " failed to load or validate state from " << filename << std::endl;
                continue;
            }
            
            std::string output_dir = output_base_dir + "/beta_" + beta_str;
            std::string taylor_dir = output_dir + "/taylor";
            if (rank == 0) {
                ensureDirectoryExists(output_dir);
                ensureDirectoryExists(taylor_dir);
            }
            MPI_Barrier(MPI_COMM_WORLD); // Ensure directories are created
            
            // Build local operators and compute their evolution
            std::vector<Operator> observables_1;
            std::vector<Operator> observables_2;
            std::vector<std::string> observable_names;
            
            for (const auto& op_info : local_operators) {
                const auto &Q = momentum_points[op_info.q_idx];
                int op_type_1 = spin_combinations[op_info.combo_idx].first;
                int op_type_2 = spin_combinations[op_info.combo_idx].second;
                
                try {
                    if (method == "taylor") {
                        SumOperator sum_op(num_sites, spin_length, op_type_1, Q, positions_file);
                        SumOperator sum_op_2(num_sites, spin_length, op_type_2, Q, positions_file);
                        observables_1.push_back(std::move(sum_op));
                        observables_2.push_back(std::move(sum_op_2));
                    } else { // pedantic
                        SublatticeOperator sum_op(op_info.sublattice_1, 4, num_sites, spin_length, op_type_1, Q, positions_file);
                        SublatticeOperator sum_op_2(op_info.sublattice_2, 4, num_sites, spin_length, op_type_2, Q, positions_file);
                        observables_1.push_back(std::move(sum_op));
                        observables_2.push_back(std::move(sum_op_2));
                    }
                    observable_names.push_back(op_info.name);
                } catch (const std::exception &e) {
                    std::cerr << "Rank " << rank << " failed to build operator " << op_info.name 
                              << ": " << e.what() << std::endl;
                }
            }
            
            if (!observables_1.empty() && !observables_2.empty()) {
                computeObservableDynamics_U_t(
                    U_t, tpq_state, observables_1, observables_2, observable_names,
                    N, taylor_dir, sample_index, beta, t_end_opt, dt_opt
                );
            }
        }
        local_processed_count = num_files; // Each rank processes all files with their operators
    } else {
        if (rank == 0) {
            std::cerr << "Unknown method '" << method << "'. Supported: krylov, taylor, pedantic, spin_correlation" << std::endl;
        }
    }
    
    // Synchronize all processes before final output
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Gather total processed count and operator statistics
    int total_processed_count;
    MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    int local_op_count = local_operators.size();
    int total_op_count;
    MPI_Reduce(&local_op_count, &total_op_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (method == "taylor" || method == "pedantic") {
            std::cout << "\nOperator-level parallelization completed:" << std::endl;
            std::cout << "  Total operators distributed: " << total_op_count << std::endl;
            std::cout << "  Total TPQ states processed: " << num_files << std::endl;
            std::cout << "  Each operator evolved for all " << num_files << " states" << std::endl;
        } else {
            std::cout << "\nProcessed " << total_processed_count << " TPQ state files with file-level parallelization." << std::endl;
        }
        std::cout << "Results saved in: " << output_base_dir << std::endl;
    }
    
    // Clean up dynamically allocated memory
    for (const char* name : spin_combination_names) {
        delete[] name;
    }
    
    MPI_Finalize();
    return 0;
}

