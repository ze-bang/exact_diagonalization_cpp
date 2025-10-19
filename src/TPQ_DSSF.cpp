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
#include <algorithm> // for std::sort, std::max_element, std::min_element
#include <numeric> // for std::accumulate
#include "construct_ham.h"
#include "TPQ.h"
#include "observables.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
namespace fs = std::filesystem;
#include <mpi.h>


void printSpinConfiguration(ComplexVector &state, int num_sites, float spin_length, const std::string &dir) {
    // Compute and print <S_i> for all sites
    std::vector<std::vector<Complex>> result(num_sites, std::vector<Complex>(3));

    for (int i = 0; i < num_sites; i++) {
        SingleSiteOperator S_plus(num_sites, spin_length, 0, i);
        SingleSiteOperator S_minus(num_sites, spin_length, 1, i);
        SingleSiteOperator S_z(num_sites, spin_length, 2, i);

        ComplexVector temp_plus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_minus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_z(state.size(), Complex(0.0, 0.0));

        S_plus.apply(state.data(), temp_plus.data(), state.size());
        S_minus.apply(state.data(), temp_minus.data(), state.size());
        S_z.apply(state.data(), temp_z.data(), state.size());

        Complex expectation_plus = 0.0;
        Complex expectation_minus = 0.0;
        Complex expectation_z = 0.0;
        for (size_t k = 0; k < state.size(); k++) {
            expectation_plus += std::conj(state[k]) * temp_plus[k];
            expectation_minus += std::conj(state[k]) * temp_minus[k];
            expectation_z += std::conj(state[k]) * temp_z[k];
        }
        result[i][0] = expectation_plus;
        result[i][1] = expectation_minus;
        result[i][2] = expectation_z;
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_configuration.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "Site S+ S- Sz\n";
    for (int i = 0; i < num_sites; i++) {
        outfile << i << " " << result[i][0] << " " << result[i][1] << " " << result[i][2] << "\n";
    }
    outfile.close();
}

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
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 6 || argc > 10) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory> <num_sites> <spin_length> <krylov_dim_or_nmax> <spin_combinations> [method] [dt,t_end] [steps] [unit_cell_size]" << std::endl;
            std::cerr << "  method (optional): krylov (default) | taylor | pedantic" << std::endl;
            std::cerr << "  dt,t_end (optional, only for taylor/pedantic): e.g. 0.01,50.0" << std::endl;
            std::cerr << "  spin_combinations format: \"op1,op2;op3,op4;...\" where op is 0(Sp), 1(Sm), or 2(Sz)" << std::endl;
            std::cerr << "  Example: \"0,1;2,2\" for SpSm, SzSz combinations" << std::endl;
            std::cerr << "  unit_cell_size (optional, for pedantic mode): number of sublattices (default: 4)" << std::endl;
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
    if (argc >= 9) {
        try { steps_opt = std::stoi(argv[8]); } catch (...) { steps_opt = -1; }
    }
    
    int unit_cell_size = 4; // Default for pyrochlore
    if (argc == 10) {
        try { unit_cell_size = std::stoi(argv[9]); } catch (...) { unit_cell_size = 4; }
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
    // const std::vector<std::vector<double>> momentum_points = {
    //     {0.0, 0.0, 0.0},
    //     {0, 0, 2*M_PI},
    //     {4*M_PI, 4*M_PI, 0}
    // };
    // Define momentum points
    const std::vector<std::vector<double>> momentum_points = {
        {0.0, 0.0, 0.0},
        {0, 0, 2*M_PI}
    };
    // const std::vector<std::vector<double>> momentum_points = {
    //     // {0.0, 0.0, 0.0},
    //     // {0, 0, 2*M_PI},
    //     {0, 0, 4*M_PI}
    // };
    
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
    
    // Get file sizes for workload estimation (only rank 0)
    std::vector<size_t> file_sizes(num_files, 0);
    if (rank == 0) {
        for (int i = 0; i < num_files; i++) {
            try {
                file_sizes[i] = fs::file_size(tpq_files[i]);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not get size of " << tpq_files[i] << ": " << e.what() << std::endl;
                file_sizes[i] = 0;
            }
        }
    }
    MPI_Bcast(file_sizes.data(), num_files, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    // Build fine-grained task list for operator-level parallelization
    // Each task is (state_idx, momentum_idx, combo_idx, sublattice_i, sublattice_j)
    struct Task {
        int state_idx;
        int momentum_idx;
        int combo_idx;
        int sublattice_i;
        int sublattice_j;
        size_t weight;  // file_size as proxy for cost
    };
    
    std::vector<Task> all_tasks;
    int num_momentum = momentum_points.size();
    int num_combos = spin_combinations.size();
    
    if (rank == 0) {
        if (method == "krylov" || method == "spin_correlation" || method == "spin_configuration") {
            // These methods process entire states atomically
            for (int s = 0; s < num_files; s++) {
                all_tasks.push_back({s, -1, -1, -1, -1, file_sizes[s]});
            }
            std::cout << "Parallelization: per-state (" << num_files << " tasks)" << std::endl;
        } else if (method == "pedantic") {
            // pedantic: parallelize across (state, momentum, combo, sublattice pairs)
            // Only compute upper triangle: sublattice_i <= sublattice_j (symmetry)
            int num_sublattice_pairs = unit_cell_size * (unit_cell_size + 1) / 2;
            for (int s = 0; s < num_files; s++) {
                for (int q = 0; q < num_momentum; q++) {
                    for (int c = 0; c < num_combos; c++) {
                        for (int sub_i = 0; sub_i < unit_cell_size; sub_i++) {
                            for (int sub_j = sub_i; sub_j < unit_cell_size; sub_j++) {
                                size_t task_weight = file_sizes[s] / (num_momentum * num_combos * num_sublattice_pairs);
                                all_tasks.push_back({s, q, c, sub_i, sub_j, task_weight});
                            }
                        }
                    }
                }
            }
            std::cout << "Parallelization: per-sublattice-pair (upper triangle, " << all_tasks.size() << " tasks = "
                      << num_files << " states × " << num_momentum << " momenta × "
                      << num_combos << " combos × " << num_sublattice_pairs << " unique sublattice pairs)" << std::endl;
        } else {
            // taylor/global: can parallelize across (state, momentum, combo)
            for (int s = 0; s < num_files; s++) {
                for (int q = 0; q < num_momentum; q++) {
                    for (int c = 0; c < num_combos; c++) {
                        size_t task_weight = file_sizes[s] / (num_momentum * num_combos);
                        all_tasks.push_back({s, q, c, -1, -1, task_weight});
                    }
                }
            }
            std::cout << "Parallelization: per-operator (" << all_tasks.size() << " tasks = "
                      << num_files << " states × " << num_momentum << " momenta × "
                      << num_combos << " combos)" << std::endl;
        }
        
        // Sort by weight (descending) for better load balance
        std::sort(all_tasks.begin(), all_tasks.end(), 
                  [](const Task& a, const Task& b) { return a.weight > b.weight; });
    }
    
    // Broadcast task list
    int num_tasks = all_tasks.size();
    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        all_tasks.resize(num_tasks);
    }
    
    // Broadcast all tasks (struct is POD-like)
    for (int i = 0; i < num_tasks; i++) {
        int buf[5] = {all_tasks[i].state_idx, all_tasks[i].momentum_idx, all_tasks[i].combo_idx, 
                      all_tasks[i].sublattice_i, all_tasks[i].sublattice_j};
        size_t w = all_tasks[i].weight;
        MPI_Bcast(buf, 5, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            all_tasks[i] = {buf[0], buf[1], buf[2], buf[3], buf[4], w};
        }
    }
    
    // Resize vectors on non-root processes
    if (rank != 0) {
        tpq_files.resize(num_files);
        sample_indices.resize(num_files);
        beta_values.resize(num_files);
        beta_strings.resize(num_files);
    }
    
    // Optimized broadcast: use single buffer for all strings
    if (rank == 0) {
        // Pack all string data into a single buffer
        std::vector<char> string_buffer;
        std::vector<int> offsets;
        std::vector<int> lengths;
        
        for (int i = 0; i < num_files; i++) {
            offsets.push_back(string_buffer.size());
            lengths.push_back(tpq_files[i].size());
            string_buffer.insert(string_buffer.end(), tpq_files[i].begin(), tpq_files[i].end());
            
            offsets.push_back(string_buffer.size());
            lengths.push_back(beta_strings[i].size());
            string_buffer.insert(string_buffer.end(), beta_strings[i].begin(), beta_strings[i].end());
        }
        
        int buffer_size = string_buffer.size();
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int buffer_size;
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<char> string_buffer(buffer_size);
        std::vector<int> offsets(num_files * 2);
        std::vector<int> lengths(num_files * 2);
        
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
        
        // Unpack strings
        for (int i = 0; i < num_files; i++) {
            int file_offset = offsets[i * 2];
            int file_length = lengths[i * 2];
            tpq_files[i].assign(string_buffer.begin() + file_offset, 
                               string_buffer.begin() + file_offset + file_length);
            
            int beta_offset = offsets[i * 2 + 1];
            int beta_length = lengths[i * 2 + 1];
            beta_strings[i].assign(string_buffer.begin() + beta_offset, 
                                  string_buffer.begin() + beta_offset + beta_length);
        }
    }
    
    MPI_Bcast(sample_indices.data(), num_files, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_values.data(), num_files, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Pre-compute time evolution operator and transverse bases if needed
    std::function<void(const Complex*, Complex*, int)> U_t;
    std::vector<std::array<double,3>> transverse_basis_1, transverse_basis_2;
    bool precomputed_U = false;
    
    if (method == "taylor" || method == "global" || method == "pedantic") {
        if (rank == 0) {
            std::cout << "Pre-computing time evolution operator (n_max=" << krylov_dim_or_nmax
                      << ", dt=" << dt_opt << ", t_end=" << t_end_opt << ")" << std::endl;
        }
        U_t = create_time_evolution_operator(H, dt_opt, krylov_dim_or_nmax, true);
        precomputed_U = true;
    }
    
    if (method == "global") {
        transverse_basis_1.resize(num_momentum);
        transverse_basis_2.resize(num_momentum);
        transverse_basis_1[0] = {1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};
        transverse_basis_2[0] = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};
        transverse_basis_1[1] = {1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};
        transverse_basis_2[1] = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};
        transverse_basis_1[2] = {0.0, 0.0, 1.0};
        transverse_basis_2[2] = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};
        
        if (rank == 0) {
            std::cout << "Transverse bases for momentum points:" << std::endl;
            for (int qi = 0; qi < num_momentum; ++qi) {
                const auto &Q = momentum_points[qi];
                const auto &b1 = transverse_basis_1[qi];
                const auto &b2 = transverse_basis_2[qi];
                std::cout << "  Q[" << qi << "] = (" << Q[0] << "," << Q[1] << "," << Q[2] 
                          << "), e1=(" << b1[0] << "," << b1[1] << "," << b1[2] 
                          << "), e2=(" << b2[0] << "," << b2[1] << "," << b2[2] << ")" << std::endl;
            }
        }
    }
    
    // Lambda to process a single task
    auto process_task = [&](const Task& task) -> bool {
        int state_idx = task.state_idx;
        int momentum_idx = task.momentum_idx;
        int combo_idx = task.combo_idx;
        int sublattice_i = task.sublattice_i;
        int sublattice_j = task.sublattice_j;
        
        int sample_index = sample_indices[state_idx];
        double beta = beta_values[state_idx];
        std::string beta_str = beta_strings[state_idx];
        std::string filename = fs::path(tpq_files[state_idx]).filename().string();
        std::string output_dir = output_base_dir + "/beta_" + beta_str;
        
        // Load state (or reuse from cache - TODO: implement caching for efficiency)
        ComplexVector tpq_state;
        bool loaded_ok = false;
        if (filename.find("eigenvector") != std::string::npos) {
            loaded_ok = load_raw_data(tpq_state, tpq_files[state_idx], N64);
        } else {
            loaded_ok = load_tpq_state(tpq_state, tpq_files[state_idx]);
        }
        
        if (!loaded_ok || (int)tpq_state.size() != N) {
            std::cerr << "Rank " << rank << " failed to load/validate state from " << filename << std::endl;
            return false;
        }
        
        ensureDirectoryExists(output_dir);
        
        // Process based on method and task granularity
        if (method == "krylov") {
            // Full-state processing
            int krylov_dim = krylov_dim_or_nmax;
            computeTPQSpinStructureFactorKrylov(
                H, tpq_state, positions_file, N, num_sites, spin_length,
                output_dir, sample_index, beta, momentum_points, krylov_dim,
                spin_combinations, spin_combination_names
            );
        } else if (method == "spin_correlation") {
            printSpinCorrelation(tpq_state, num_sites, spin_length, output_dir);
        } else if (method == "spin_configuration") {
            printSpinConfiguration(tpq_state, num_sites, spin_length, output_dir);
        } else if (method == "taylor") {
            // Process single (momentum, combo) pair
            const auto &Q = momentum_points[momentum_idx];
            int op_type_1 = spin_combinations[combo_idx].first;
            int op_type_2 = spin_combinations[combo_idx].second;
            
            std::stringstream name_ss;
            name_ss << spin_combination_names[combo_idx] << "_q_Qx" << Q[0] 
                    << "_Qy" << Q[1] << "_Qz" << Q[2];
            std::string obs_name = name_ss.str();
            
            try {
                SumOperator sum_op_1(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], positions_file);
                SumOperator sum_op_2(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], positions_file);
                
                std::vector<Operator> obs_1 = {std::move(sum_op_1)};
                std::vector<Operator> obs_2 = {std::move(sum_op_2)};
                std::vector<std::string> obs_names = {obs_name};
                
                std::string taylor_dir = output_dir + "/taylor";
                ensureDirectoryExists(taylor_dir);
                
                computeObservableDynamics_U_t(
                    U_t, tpq_state, obs_1, obs_2, obs_names, N,
                    taylor_dir, sample_index, beta, t_end_opt, dt_opt
                );
            } catch (const std::exception &e) {
                std::cerr << "Rank " << rank << " failed operator construction for " 
                          << obs_name << ": " << e.what() << std::endl;
                return false;
            }
        } else if (method == "global") {
            // Process single (momentum, combo) pair with both transverse components
            const auto &Q = momentum_points[momentum_idx];
            const auto &b1 = transverse_basis_1[momentum_idx];
            const auto &b2 = transverse_basis_2[momentum_idx];
            std::vector<double> e1_vec = {b1[0], b1[1], b1[2]};
            std::vector<double> e2_vec = {b2[0], b2[1], b2[2]};
            
            int op_type_1 = spin_combinations[combo_idx].first;
            int op_type_2 = spin_combinations[combo_idx].second;
            std::string base_name = std::string(spin_combination_names[combo_idx]);
            
            std::vector<Operator> obs_1, obs_2;
            std::vector<std::string> obs_names;
            
            try {
                // SF component
                std::stringstream name_sf;
                name_sf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_SF";
                TransverseOperator op1_sf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                TransverseOperator op2_sf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                obs_1.push_back(std::move(op1_sf));
                obs_2.push_back(std::move(op2_sf));
                obs_names.push_back(name_sf.str());
                
                // NSF component
                std::stringstream name_nsf;
                name_nsf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_NSF";
                TransverseOperator op1_nsf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                TransverseOperator op2_nsf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                obs_1.push_back(std::move(op1_nsf));
                obs_2.push_back(std::move(op2_nsf));
                obs_names.push_back(name_nsf.str());
                
                std::string global_dir = output_dir + "/global";
                ensureDirectoryExists(global_dir);
                
                computeObservableDynamics_U_t(
                    U_t, tpq_state, obs_1, obs_2, obs_names, N,
                    global_dir, sample_index, beta, t_end_opt, dt_opt
                );
            } catch (const std::exception &e) {
                std::cerr << "Rank " << rank << " failed transverse operator construction: " 
                          << e.what() << std::endl;
                return false;
            }
        } else if (method == "pedantic") {
            // Process single sublattice pair correlation at given (momentum, combo)
            const auto &Q = momentum_points[momentum_idx];
            int op_type_1 = spin_combinations[combo_idx].first;
            int op_type_2 = spin_combinations[combo_idx].second;
            
            std::stringstream name_ss;
            name_ss << spin_combination_names[combo_idx] 
                    << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2]
                    << "_sub" << sublattice_i << "_sub" << sublattice_j;
            std::string obs_name = name_ss.str();
            
            try {
                // Create sublattice operators for sites i and j
                std::vector<double> Q_vec = {Q[0], Q[1], Q[2]};
                SublatticeOperator sub_op_1(sublattice_i, unit_cell_size, num_sites, spin_length, op_type_1, Q_vec, positions_file);
                SublatticeOperator sub_op_2(sublattice_j, unit_cell_size, num_sites, spin_length, op_type_2, Q_vec, positions_file);
                
                std::vector<Operator> obs_1 = {std::move(sub_op_1)};
                std::vector<Operator> obs_2 = {std::move(sub_op_2)};
                std::vector<std::string> obs_names = {obs_name};
                
                std::string pedantic_dir = output_dir + "/pedantic";
                ensureDirectoryExists(pedantic_dir);
                
                computeObservableDynamics_U_t(
                    U_t, tpq_state, obs_1, obs_2, obs_names, N,
                    pedantic_dir, sample_index, beta, t_end_opt, dt_opt
                );
            } catch (const std::exception &e) {
                std::cerr << "Rank " << rank << " failed sublattice operator construction for " 
                          << obs_name << ": " << e.what() << std::endl;
                return false;
            }
        }
        
        return true;
    };
    
    // Dynamic work distribution using master-worker pattern
    int local_processed_count = 0;
    double start_time = MPI_Wtime();
    
    if (size == 1) {
        // Serial execution
        for (const auto& task : all_tasks) {
            if (process_task(task)) {
                local_processed_count++;
            }
        }
    } else {
        // Master-worker dynamic scheduling
        const int TASK_TAG = 1;
        const int STOP_TAG = 2;
        const int DONE_TAG = 3;
        
        if (rank == 0) {
            // Master: dispatch tasks
            int next_task = 0;
            int active_workers = std::min(size - 1, num_tasks);
            
            // Send initial batch
            for (int r = 1; r <= active_workers; ++r) {
                MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                next_task++;
            }
            
            // Idle remaining workers
            for (int r = active_workers + 1; r < size; ++r) {
                int dummy = -1;
                MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
            }
            
            // Dispatch remaining tasks as workers finish
            int completed = 0;
            while (completed < num_tasks) {
                int done_task;
                MPI_Status status;
                MPI_Recv(&done_task, 1, MPI_INT, MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &status);
                completed++;
                
                if (next_task < num_tasks) {
                    MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
                } else {
                    int dummy = -1;
                    MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                }
            }
        } else {
            // Worker: request and process tasks
            while (true) {
                int task_id;
                MPI_Status status;
                MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == STOP_TAG) {
                    break;
                }
                
                std::cout << "Rank " << rank << " processing task " << task_id << "/" << num_tasks;
                if (all_tasks[task_id].momentum_idx >= 0) {
                    std::cout << " (state=" << all_tasks[task_id].state_idx 
                              << ", Q=" << all_tasks[task_id].momentum_idx
                              << ", combo=" << all_tasks[task_id].combo_idx;
                    if (all_tasks[task_id].sublattice_i >= 0) {
                        std::cout << ", sub_i=" << all_tasks[task_id].sublattice_i 
                                  << ", sub_j=" << all_tasks[task_id].sublattice_j;
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
                
                if (process_task(all_tasks[task_id])) {
                    local_processed_count++;
                }
                
                MPI_Send(&task_id, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Gather timing and count statistics
    std::vector<double> all_times;
    if (rank == 0) {
        all_times.resize(size);
    }
    MPI_Gather(&elapsed_time, 1, MPI_DOUBLE, rank == 0 ? all_times.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int total_processed_count;
    MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing complete!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Processed " << total_processed_count << "/" << num_tasks << " tasks successfully." << std::endl;
        std::cout << "Results saved in: " << output_base_dir << std::endl;
        
        // Print timing statistics
        std::cout << "\nTiming statistics:" << std::endl;
        auto max_it = std::max_element(all_times.begin(), all_times.end());
        auto min_it = std::min_element(all_times.begin(), all_times.end());
        double max_time = *max_it;
        double min_time = *min_it;
        double avg_time = std::accumulate(all_times.begin(), all_times.end(), 0.0) / size;
        double load_imbalance = 0.0;
        if (max_time > 0.0) {
            load_imbalance = (max_time - min_time) / max_time * 100.0;
        }
        
        std::cout << "  Max time: " << max_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), max_it) << ")" << std::endl;
        std::cout << "  Min time: " << min_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), min_it) << ")" << std::endl;
        std::cout << "  Avg time: " << avg_time << " seconds" << std::endl;
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(2) << load_imbalance << "%" << std::endl;
        
        std::cout << "\nPer-rank timing:" << std::endl;
        for (int r = 0; r < size; r++) {
            std::cout << "  Rank " << r << ": " << all_times[r] << " seconds" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
