#include <iostream>
#include "construct_ham.h"
#include <fstream>
#include <string>
#include "ed_wrapper.h"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <directory> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --method=<method>    : Diagonalization method (LANCZOS, FULL, etc.)" << std::endl;
        std::cout << "  --eigenvalues=<n>    : Number of eigenvalues to compute" << std::endl;
        std::cout << "  --eigenvectors       : Compute eigenvectors" << std::endl;
        std::cout << "  --output=<dir>       : Output directory" << std::endl;
        std::cout << "  --tolerance=<tol>    : Convergence tolerance" << std::endl;
        std::cout << "  --iterations=<iter>  : Maximum iterations" << std::endl;
        return 1;
    }

    // Directory with Hamiltonian files
    std::string directory = argv[1];
    
    // Default parameters
    EDParameters params;
    params.num_eigenvalues = (1<<10);
    params.compute_eigenvectors = false;
    params.output_dir = "ed_results";
    params.tolerance = 1e-10;
    params.max_iterations = (1<<10);
    
    // Default method
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS;
    
    // Parse command line options
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--method=") == 0) {
            std::string method_str = arg.substr(9);
            if (method_str == "LANCZOS") method = DiagonalizationMethod::LANCZOS;
            else if (method_str == "FULL") method = DiagonalizationMethod::FULL;
            else if (method_str == "TPQ") method = DiagonalizationMethod::TPQ;
            // Add other methods as needed
        }
        else if (arg.find("--eigenvalues=") == 0) {
            params.num_eigenvalues = std::stoi(arg.substr(14));
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
    }
    
    // Create output directories
    params.output_dir = params.output_dir + "/standard";
    std::string symmetrized_output = params.output_dir.substr(0, params.output_dir.rfind("/")) + "/symmetrized";
    std::string cmd = "mkdir -p " + params.output_dir + " " + symmetrized_output;
    system(cmd.c_str());
    
    std::cout << "==========================================" << std::endl;
    std::cout << "Starting Standard Exact Diagonalization" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Run standard diagonalization
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    try {
        results = exact_diagonalization_from_directory(
            directory, method, params, HamiltonianFileFormat::STANDARD
        );
        
        // Display eigenvalues
        std::cout << "Eigenvalues (standard):" << std::endl;
        for (size_t i = 0; i < results.eigenvalues.size(); i++) {
            std::cout << i << ": " << results.eigenvalues[i] << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in standard ED: " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Standard ED completed in " << duration / 1000.0 << " seconds" << std::endl;
    
    // Save eigenvalues to file
    std::ofstream standard_file(params.output_dir + "/eigenvalues.txt");
    if (standard_file.is_open()) {
        for (const auto& val : results.eigenvalues) {
            standard_file << val << std::endl;
        }
        standard_file.close();
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Starting Symmetrized Exact Diagonalization" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Set up parameters for symmetrized diagonalization
    EDParameters sym_params = params;
    sym_params.output_dir = symmetrized_output;
    
    // Run symmetrized diagonalization
    start_time = std::chrono::high_resolution_clock::now();
    
    EDResults sym_results;
    try {
        sym_results = exact_diagonalization_from_directory_symmetrized(
            directory, method, sym_params, HamiltonianFileFormat::STANDARD
        );
        
        // Display eigenvalues
        std::cout << "Eigenvalues (symmetrized):" << std::endl;
        for (size_t i = 0; i < sym_results.eigenvalues.size(); i++) {
            std::cout << i << ": " << sym_results.eigenvalues[i] << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in symmetrized ED: " << e.what() << std::endl;
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Symmetrized ED completed in " << duration / 1000.0 << " seconds" << std::endl;
    
    // Save eigenvalues to file
    std::ofstream sym_file(symmetrized_output + "/eigenvalues.txt");
    if (sym_file.is_open()) {
        for (const auto& val : sym_results.eigenvalues) {
            sym_file << val << std::endl;
        }
        sym_file.close();
    }
    
    // Compare results
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Comparing Results" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    int compare_count = std::min(results.eigenvalues.size(), sym_results.eigenvalues.size());
    if (compare_count > 0) {
        std::cout << "    Standard      Symmetrized    Difference" << std::endl;
        for (int i = 0; i < compare_count; i++) {
            double diff = std::abs(results.eigenvalues[i] - sym_results.eigenvalues[i]);
            std::cout << i << ": " << results.eigenvalues[i] << "  " 
                      << sym_results.eigenvalues[i] << "  " << diff << std::endl;
        }
    } else {
        std::cout << "No eigenvalues to compare." << std::endl;
    }
    
    return 0;
}