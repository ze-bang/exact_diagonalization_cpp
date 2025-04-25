
#ifndef ED_WRAPPER_H
#define ED_WRAPPER_H

#include "TPQ.h"
#include "CG.h"
#include "lanczos.h"
#include "construct_ham.h"
#include <sys/stat.h>
// Enum for available diagonalization methods
enum class DiagonalizationMethod {
    LANCZOS,               // Standard Lanczos algorithm
    LANCZOS_SELECTIVE,     // Lanczos with selective reorthogonalization
    LANCZOS_NO_ORTHO,      // Lanczos without reorthogonalization
    SHIFT_INVERT,          // Shift-invert Lanczos
    SHIFT_INVERT_ROBUST,   // Robust shift-invert Lanczos
    CG,                    // Conjugate gradient
    BLOCK_CG,              // Block conjugate gradient
    DAVIDSON,              // Davidson method
    BICG,                  // Biconjugate gradient
    LOBPCG,                // Locally optimal block preconditioned conjugate gradient
    KRYLOV_SCHUR,          // Krylov-Schur algorithm
    FULL,                  // Full diagonalization
    TPQ,                    // Thermal Pure Quantum states
    ARPACK                  // ARPACK
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
    int max_iterations = 1000;
    int num_eigenvalues = 1;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    std::string output_dir = "";
    
    // Method-specific parameters
    double shift = 0.0;  // For shift-invert methods
    int block_size = 10; // For block methods
    int max_subspace = 100; // For Davidson
    
    // TPQ-specific parameters
    int num_samples = 20;
    double beta_min = 0.01;
    double beta_max = 100.0;
    int num_beta_bins = 100;
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
        
        case DiagonalizationMethod::ARPACK:
            std::cout << "Using ARPACK method" << std::endl;
            arpack_diagonalization(H, hilbert_space_dim, params.max_iterations, 
                             params.num_eigenvalues, params.tolerance, 
                             results.eigenvalues, params.output_dir, 
                             params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::TPQ:
            {
                std::cout << "Using TPQ method for thermal calculations" << std::endl;
                TPQResults tpq_results = perform_tpq_calculation(
                    H, hilbert_space_dim, params.num_samples, 50, 
                    -10.0, 10.0, 0.0, params.beta_min, params.beta_max, 
                    params.num_beta_bins);
                
                // Convert TPQ results to thermodynamic data
                results.thermo_data.temperatures = tpq_results.temperatures;
                results.thermo_data.energy = tpq_results.energies;
                results.thermo_data.specific_heat = tpq_results.specific_heats;
                results.thermo_data.entropy = tpq_results.entropies;
                results.thermo_data.free_energy = tpq_results.free_energies;
                
                // Save TPQ results if output directory is provided
                if (!params.output_dir.empty()) {
                    save_tpq_results(tpq_results, params.output_dir + "/tpq_results.txt", 0);
                }
            }
            break;
            
        default:
            std::cerr << "Unknown diagonalization method selected" << std::endl;
            break;
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
    int num_sites = 0;
    Operator hamiltonian(1);  // Initialize with dummy size, will update later
    
    switch (format) {
        case HamiltonianFileFormat::STANDARD: {
            // Read the number of sites from Trans.dat
            std::ifstream trans_file(single_site_file);
            if (!trans_file.is_open()) {
                throw std::runtime_error("Error: Cannot open file " + single_site_file);
            }

            // Skip the first line
            std::string dummy_line;
            std::getline(trans_file, dummy_line);

            // Read the second line to get num_sites
            std::string dum;
            trans_file >> dum >> num_sites;
            trans_file.close();
            
            // Create the Hamiltonian with the correct number of sites
            hamiltonian = Operator(num_sites);
            
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
    int hilbert_space_dim = 1 << num_sites;  // 2^num_sites
    
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
    return exact_diagonalization_from_files(
        interaction_file, single_site_file, method, params, format
    );
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
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    // Initialize results
    EDResults results;
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // 1. Determine the number of sites and create the Hamiltonian
    int num_sites = 0;
    
    if (format != HamiltonianFileFormat::STANDARD) {
        throw std::runtime_error("Only STANDARD format is supported for symmetrized diagonalization");
    }
    
    // Read the number of sites from the file
    std::ifstream trans_file(single_site_file);
    if (!trans_file.is_open()) {
        throw std::runtime_error("Error: Cannot open file " + single_site_file);
    }

    // Skip the first line
    std::string dummy_line;
    std::getline(trans_file, dummy_line);

    // Read the second line to get num_sites
    std::string dum;
    trans_file >> dum >> num_sites;
    trans_file.close();
    
    // Create the Hamiltonian with the correct number of sites
    Operator hamiltonian(num_sites);
    
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
        std::cout << "Symmetrized blocks not found. Building..." << std::endl;
        hamiltonian.buildAndSaveSymmetrizedBlocks(directory);
        block_sizes = hamiltonian.symmetrized_block_ham_sizes;
    } else {
        std::cout << "Using existing symmetrized blocks from " << sym_blocks_dir << std::endl;
        
        // We need to scan the sym_blocks directory to get block sizes
        for (int block_idx = 0; ; block_idx++) {
            std::string block_file = sym_blocks_dir + "/block_" + std::to_string(block_idx) + ".dat";
            if (stat((block_file).c_str(), &buffer) != 0) {
                break; // File doesn't exist, no more blocks
            }
            
            // Open the block file to determine its dimension
            std::ifstream file(block_file);
            if (!file.is_open()) {
                std::cerr << "Warning: Could not open block file: " << block_file << std::endl;
                continue;
            }
            
            // Read the first line which contains the block size
            int block_size;
            file >> block_size;
            block_sizes.push_back(block_size);
            
            file.close();
        }
    }
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
    
    // 4. Diagonalize each block separately
    std::cout << "Found " << block_sizes.size() << " symmetrized blocks." << std::endl;
    
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

        // Print the block Hamiltonian matrix for debugging
        std::cout << "Matrix representation of block " << block_idx << " (dim: " << block_dim << "):" << std::endl;

        // // For large matrices, only print summary or write to file
        // if (block_dim > 20) {
        //     std::cout << "Matrix too large to display directly. Writing to file..." << std::endl;
        //     std::string matrix_file = params.output_dir + "/block_" + std::to_string(block_idx) + "_matrix.txt";
        //     std::ofstream mat_out(matrix_file);
        //     if (mat_out.is_open()) {
        //         mat_out << "# Block Hamiltonian Matrix " << block_idx << " (dim: " << block_dim << ")" << std::endl;
        //         mat_out << "# Format: row col real imag" << std::endl;
        //         for (const auto& [row, col, val] : block_entries) {
        //             mat_out << row << " " << col << " " << val.real() << " " << val.imag() << std::endl;
        //         }
        //         mat_out.close();
        //         std::cout << "Matrix saved to " << matrix_file << std::endl;
        //     } else {
        //         std::cout << "Failed to open file for matrix output." << std::endl;
        //     }
        // } else {
        //     // Create a dense matrix representation for visualization
        //     std::vector<std::vector<Complex>> dense_matrix(block_dim, std::vector<Complex>(block_dim, Complex(0.0, 0.0)));
            
        //     // Fill in the entries
        //     for (const auto& [row, col, val] : block_entries) {
        //         dense_matrix[row][col] = val;
        //     }
            
        //     // Print the matrix
        //     for (int i = 0; i < block_dim; ++i) {
        //         for (int j = 0; j < block_dim; ++j) {
        //             std::cout << "(" << dense_matrix[i][j].real() << "," << dense_matrix[i][j].imag() << ") ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << std::endl;
        // }
        
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
            std::cout << "Processing eigenvectors for block " << block_idx << std::endl;
            
            for (size_t eigen_idx = 0; eigen_idx < block_results.eigenvalues.size(); ++eigen_idx) {
                // Path to the block eigenvector
                std::string block_eigenvector_file = block_params.output_dir + 
                                                   "/eigenvector_" + std::to_string(eigen_idx) + ".dat";
                
                // Read the block eigenvector
                std::vector<Complex> block_eigenvector;
                std::ifstream eigen_file(block_eigenvector_file);
                if (!eigen_file.is_open()) {
                    std::cerr << "Warning: Could not open eigenvector file: " << block_eigenvector_file << std::endl;
                    continue;
                }
                
                block_eigenvector.resize(block_dim);
                for (int i = 0; i < block_dim; ++i) {
                    double real, imag;
                    eigen_file >> real >> imag;
                    block_eigenvector[i] = Complex(real, imag);
                }
                eigen_file.close();
                
                // Create a file for the transformed eigenvector
                std::string transformed_file = params.output_dir + "/eigenvector_block" + 
                                            std::to_string(block_idx) + "_" + 
                                            std::to_string(eigen_idx) + ".dat";
                std::ofstream out_file(transformed_file);
                if (!out_file.is_open()) {
                    std::cerr << "Warning: Could not open file for transformed eigenvector: " << transformed_file << std::endl;
                    continue;
                }
                
                // Read the symmetry basis for this block
                std::vector<Complex> sym_basis = hamiltonian.read_sym_basis(block_idx, directory);
                
                // Transform the eigenvector using the symmetry basis
                int hilbert_space_dim = 1 << num_sites;
                for (int orig_idx = 0; orig_idx < hilbert_space_dim; ++orig_idx) {
                    Complex coeff(0.0, 0.0);
                    
                    // Compute the coefficient in the original basis
                    for (int sym_idx = 0; sym_idx < block_dim; ++sym_idx) {
                        // Get the coefficient of the original basis state in the symmetry basis
                        // and multiply by the eigenvector coefficient
                        coeff += sym_basis[orig_idx + sym_idx * hilbert_space_dim] * block_eigenvector[sym_idx];
                    }
                    
                    // Only write non-zero coefficients to save space
                    if (std::abs(coeff) > 1e-10) {
                        out_file << orig_idx << " " << coeff.real() << " " << coeff.imag() << std::endl;
                    }
                }
                
                out_file.close();
            }
        }
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