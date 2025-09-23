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


void printSpinExpectation(ComplexVector &state, int num_sites, float spin_length, const std::string &dir) {
    // Compute and print <S_i . S_j> for all pairs (i,j)
    std::vector<std::vector<Complex>> result(2, std::vector<Complex>(num_sites));

    for (int i = 0; i < num_sites; i++) {
        SingleSiteOperator S_plus_i(num_sites, spin_length, 0, i);
        SingleSiteOperator S_z_i(num_sites, spin_length, 2, i);

        ComplexVector temp_plus_i(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_z_i(state.size(), Complex(0.0, 0.0));

        S_plus_i.apply(state.data(), temp_plus_i.data(), state.size());
        S_z_i.apply(state.data(), temp_z_i.data(), state.size());

        Complex expectation_plus = 0.0;
        for (size_t k = 0; k < state.size(); k++) {
            expectation_plus += std::conj(state[k]) * temp_plus_i[k];
        }
        result[0][i] = expectation_plus;
        Complex expectation_z = 0.0;
        for (size_t k = 0; k < state.size(); k++) {
            expectation_z += std::conj(state[k]) * temp_z_i[k];
        }
        result[1][i] = expectation_z;
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_expectation.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "i <S+_i> <Sz_i>\n";
    for (int i = 0; i < num_sites; i++) {
        outfile << i << " " << result[0][i] << " " << result[1][i] << "\n";
    }
    outfile.close();

    std::cout << "Spin expectation data saved to spin_expectation.txt" << std::endl;
}


int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 6 || argc > 9) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory> <num_sites> <spin_length> <krylov_dim_or_nmax> <spin_combinations> [method] [dt,t_end] [steps]" << std::endl;
            std::cerr << "  method (optional): krylov (default) | taylor" << std::endl;
            std::cerr << "  dt,t_end (optional, only for taylor): e.g. 0.01,50.0" << std::endl;
            std::cerr << "  spin_combinations format: \"op1,op2;op3,op4;...\" where op is 0(Sp), 1(Sm), or 2(Sz)" << std::endl;
            std::cerr << "  Example: \"0,1;2,2\" for SpSm, SzSz combinations" << std::endl;
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
    
    // Load Hamiltonian
    ham_op.loadFromInterAllFile(interall_file);
    ham_op.loadFromFile(trans_file);
    
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
        {0, 0, 4*M_PI}
    };
    
    // Create output directory (only rank 0)
    std::string output_base_dir = directory + "/structure_factor_results";
    if (rank == 0) {
        ensureDirectoryExists(output_base_dir);
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Collect all tpq_state files from the output subdirectory (only rank 0)
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
        std::cout << "Using " << size << " MPI processes" << std::endl;
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
    
    // Process files assigned to this rank
    int local_processed_count = 0;
    for (int i = rank; i < num_files; i += size) {
        int sample_index = sample_indices[i];
        double beta = beta_values[i];
        std::string beta_str = beta_strings[i];
        std::string filename = fs::path(tpq_files[i]).filename().string();
        
        std::cout << "Rank " << rank << " processing " << filename << " (sample " << sample_index << ", beta = " << beta << ")" << std::endl;
        
        // Load state: TPQ binary or text eigenvector
        ComplexVector tpq_state;
        bool loaded_ok = false;
        if (filename.find("eigenvector") != std::string::npos) {
            // Text eigenvector format: first line dimension, then index real imag
            loaded_ok = load_raw_data(tpq_state, tpq_files[i], N64);
        } else {
            loaded_ok = load_tpq_state(tpq_state, tpq_files[i]);
        }
        if (!loaded_ok) {
            std::cerr << "Rank " << rank << " failed to load state from " << filename << std::endl;
            continue;
        }
        if ((int)tpq_state.size() != N) {
            std::cerr << "Rank " << rank << ": state dimension mismatch for " << filename
                      << ". got " << tpq_state.size() << ", expected " << N << std::endl;
            continue;
        }
        
        // Create output directory for this beta (each process creates its own)
        std::string output_dir = output_base_dir + "/beta_" + beta_str;
        ensureDirectoryExists(output_dir);
        
        if (method == "krylov") {
            int krylov_dim = krylov_dim_or_nmax;
            computeTPQSpinStructureFactorKrylov(
                H,
                tpq_state,
                positions_file,
                N,
                num_sites,
                spin_length,
                output_dir,
                sample_index,
                beta,
                momentum_points,
                krylov_dim,
                spin_combinations,
                spin_combination_names
            );
        } else if (method == "taylor") {
            if (rank == 0) {
                std::cout << "Using Taylor (create_time_evolution_operator) evolution (n_max=" << krylov_dim_or_nmax
                          << ", dt=" << dt_opt << ", t_end=" << t_end_opt << ")" << std::endl;
            }
            // Build U(dt) and U(-dt)
            auto U_t = create_time_evolution_operator(H, dt_opt, krylov_dim_or_nmax, true);

            // Build observables: momentum-dependent sum operators for the FIRST operator in each pair.
            std::vector<Operator> observables_1;
            std::vector<Operator> observables_2;
            std::vector<std::string> observable_names;
            for (size_t q_idx = 0; q_idx < momentum_points.size(); ++q_idx) {
                const auto &Q = momentum_points[q_idx];
                for (size_t combo_idx = 0; combo_idx < spin_combinations.size(); ++combo_idx) {
                    int op_type_1 = spin_combinations[combo_idx].first; // 0 Sp,1 Sm,2 Sz
                    int op_type_2 = spin_combinations[combo_idx].second; // 0 Sp,1 Sm,2 Sz
                    std::stringstream name_ss;
                    name_ss << spin_combination_names[combo_idx] << "_q" << "_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                    try {
                        SumOperator sum_op(num_sites, spin_length, op_type_1, momentum_points[q_idx], positions_file);
                        SumOperator sum_op_2(num_sites, spin_length, op_type_2, momentum_points[q_idx], positions_file);
                        observable_names.push_back(name_ss.str());
                        observables_1.push_back(std::move(sum_op));
                        observables_2.push_back(std::move(sum_op_2));
                    } catch (const std::exception &e) {
                        if (rank == 0) {
                            std::cerr << "Failed to build SumOperator for q_idx=" << q_idx << ": " << e.what() << std::endl;
                        }
                    }
                }
            }
            if (observables_1.empty() && observables_2.empty()) {
                if (rank == 0) {
                    std::cerr << "No observables constructed. Skipping Taylor evolution for this state." << std::endl;
                }
            } else {
                std::string taylor_dir = output_dir + "/taylor";
                ensureDirectoryExists(taylor_dir);
                computeObservableDynamics_U_t(
                    U_t,
                    tpq_state,
                    observables_1,
                    observables_2,
                    observable_names,
                    N,
                    taylor_dir,
                    sample_index,
                    beta,
                    t_end_opt,
                    dt_opt
                );
            }
        } else if (method == "pedantic"){
            if (rank == 0) {
                std::cout << "Using Taylor (create_time_evolution_operator) evolution for sublattice operator evolution (n_max=" << krylov_dim_or_nmax
                          << ", dt=" << dt_opt << ", t_end=" << t_end_opt << ")" << std::endl;
            }
            // Build U(dt) and U(-dt)
            auto U_t = create_time_evolution_operator(H, dt_opt, krylov_dim_or_nmax, true);

            // Build observables: momentum-dependent sum operators for the FIRST operator in each pair.
            std::vector<Operator> observables_1;
            std::vector<Operator> observables_2;
            std::vector<std::string> observable_names;
            for (size_t q_idx = 0; q_idx < momentum_points.size(); ++q_idx) {
                const auto &Q = momentum_points[q_idx];
                for (size_t combo_idx = 0; combo_idx < spin_combinations.size(); ++combo_idx) {
                    for(size_t sublattice = 0; sublattice < 4; ++sublattice){
                        for(size_t sublattice2 = 0; sublattice2 < 4; ++sublattice2){
                            int op_type_1 = spin_combinations[combo_idx].first; // 0 Sp,1 Sm,2 Sz
                            int op_type_2 = spin_combinations[combo_idx].second; // 0 Sp,1 Sm,2 Sz
                            std::stringstream name_ss;
                            name_ss << spin_combination_names[combo_idx] << "_sub" << sublattice << "_sub" << sublattice2 << "_q" << "_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                            try {
                                SublatticeOperator sum_op(sublattice, 4, num_sites, spin_length, op_type_1, momentum_points[q_idx], positions_file);
                                SublatticeOperator sum_op_2(sublattice2, 4, num_sites, spin_length, op_type_2, momentum_points[q_idx], positions_file);
                                observable_names.push_back(name_ss.str());
                                observables_1.push_back(std::move(sum_op));
                                observables_2.push_back(std::move(sum_op_2));
                            } catch (const std::exception &e) {
                                if (rank == 0) {
                                    std::cerr << "Failed to build SumOperator for q_idx=" << q_idx << ": " << e.what() << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            if (observables_1.empty() && observables_2.empty()) {
                if (rank == 0) {
                    std::cerr << "No observables constructed. Skipping Taylor evolution for this state." << std::endl;
                }
            } else {
                std::string taylor_dir = output_dir + "/taylor";
                ensureDirectoryExists(taylor_dir);
                computeObservableDynamics_U_t(
                    U_t,
                    tpq_state,
                    observables_1,
                    observables_2,
                    observable_names,
                    N,
                    taylor_dir,
                    sample_index,
                    beta,
                    t_end_opt,
                    dt_opt
                );
            }
        }
        else if (method == "spin_correlation") {
            printSpinCorrelation(tpq_state, num_sites, spin_length, output_dir);
            printSpinExpectation(tpq_state, num_sites, spin_length, output_dir);
        } else {
            if (rank == 0) {
                std::cerr << "Unknown method '" << method << "'. Supported: krylov, taylor" << std::endl;
            }
        }
        
        local_processed_count++;
        std::cout << "Rank " << rank << " completed structure factor calculation for sample " << sample_index << ", beta = " << beta << std::endl;
    }
    
    // Gather total processed count
    int total_processed_count;
    MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\nProcessed " << total_processed_count << " TPQ state files." << std::endl;
        std::cout << "Results saved in: " << output_base_dir << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

