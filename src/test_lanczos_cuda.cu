// test_lanczos_cuda.cu - Test program for CUDA-accelerated Lanczos implementations
#include "lanczos_cuda.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>

// Timer utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() {
        reset();
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

// Generate a random Hermitian matrix
void generate_random_hermitian_matrix(ComplexVector& matrix, int N) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Allocate matrix in row-major order
    matrix.resize(N * N);
    
    // Generate random Hermitian matrix
    for (int i = 0; i < N; i++) {
        // Diagonal elements (real)
        matrix[i * N + i] = Complex(dist(gen), 0.0);
        
        // Off-diagonal elements
        for (int j = i + 1; j < N; j++) {
            double real = dist(gen);
            double imag = dist(gen);
            matrix[i * N + j] = Complex(real, imag);
            matrix[j * N + i] = Complex(real, -imag); // Hermitian condition
        }
    }
}

// Matrix-vector product function
void matrix_vector_product(const Complex* v_in, Complex* v_out, int N, const ComplexVector& matrix) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Complex sum(0.0, 0.0);
        for (int j = 0; j < N; j++) {
            sum += matrix[i * N + j] * v_in[j];
        }
        v_out[i] = sum;
    }
}

// Full diagonalization using LAPACKE
std::vector<double> full_diagonalization(const ComplexVector& matrix, int N) {
    // Copy matrix for LAPACK (column-major)
    std::vector<double> eigenvalues(N);
    ComplexVector matrix_copy(N * N);
    
    // Convert from row-major to column-major
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix_copy[j * N + i] = matrix[i * N + j];
        }
    }
    
    // Workspace
    int info;
    
    // Call LAPACK
    info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', N, reinterpret_cast<lapack_complex_double*>(matrix_copy.data()),
                         N, eigenvalues.data());
    
    if (info != 0) {
        std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
        return std::vector<double>();
    }
    
    return eigenvalues;
}

// Compare eigenvalue arrays
double compare_eigenvalues(const std::vector<double>& exact, const std::vector<double>& approx, int count) {
    double max_error = 0.0;
    int compare_count = std::min(count, static_cast<int>(std::min(exact.size(), approx.size())));
    
    std::cout << "Comparing " << compare_count << " eigenvalues:" << std::endl;
    std::cout << std::setw(5) << "Index" << std::setw(15) << "Exact" << std::setw(15) << "Lanczos"
              << std::setw(15) << "Error" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (int i = 0; i < compare_count; i++) {
        double error = std::abs(exact[i] - approx[i]);
        max_error = std::max(max_error, error);
        
        std::cout << std::setw(5) << i 
                  << std::setw(15) << std::fixed << std::setprecision(8) << exact[i]
                  << std::setw(15) << std::fixed << std::setprecision(8) << approx[i]
                  << std::setw(15) << std::scientific << std::setprecision(2) << error
                  << std::endl;
    }
    
    return max_error;
}

int main(int argc, char** argv) {
    // Configuration
    int N = 500;           // Matrix dimension
    int max_iter = 100;    // Maximum Lanczos iterations
    int num_eigenvalues = 5; // Number of eigenvalues to compute
    double tolerance = 1e-10; // Convergence tolerance
    std::string output_dir = "./lanczos_test_output"; // Output directory
    bool compute_eigenvectors = false;  // Whether to compute eigenvectors
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "-i" && i + 1 < argc) max_iter = std::stoi(argv[++i]);
        else if (arg == "-e" && i + 1 < argc) num_eigenvalues = std::stoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) tolerance = std::stod(argv[++i]);
        else if (arg == "-v") compute_eigenvectors = true;
        else if (arg == "-o" && i + 1 < argc) output_dir = argv[++i];
    }
    
    std::cout << "Running CUDA Lanczos test with:" << std::endl;
    std::cout << "  Matrix dimension: " << N << std::endl;
    std::cout << "  Max iterations: " << max_iter << std::endl;
    std::cout << "  Eigenvalues to compute: " << num_eigenvalues << std::endl;
    std::cout << "  Tolerance: " << tolerance << std::endl;
    std::cout << "  Computing eigenvectors: " << (compute_eigenvectors ? "Yes" : "No") << std::endl;
    std::cout << "  Output directory: " << output_dir << std::endl << std::endl;
    
    // Create output directory
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());
    
    // Generate random Hermitian matrix
    Timer timer;
    std::cout << "Generating random Hermitian matrix..." << std::endl;
    ComplexVector matrix;
    generate_random_hermitian_matrix(matrix, N);
    std::cout << "Matrix generation time: " << timer.elapsed() << " seconds" << std::endl << std::endl;
    
    // Create matrix-vector product function
    auto matrix_vector_function = [&matrix](const Complex* v_in, Complex* v_out, int dim) {
        matrix_vector_product(v_in, v_out, dim, matrix);
    };
    
    // Full diagonalization for comparison
    std::cout << "Running full diagonalization..." << std::endl;
    timer.reset();
    std::vector<double> exact_eigenvalues = full_diagonalization(matrix, N);
    double full_diag_time = timer.elapsed();
    std::cout << "Full diagonalization time: " << full_diag_time << " seconds" << std::endl << std::endl;
    
    // Sort eigenvalues (they should be sorted already from LAPACK, but to be safe)
    std::sort(exact_eigenvalues.begin(), exact_eigenvalues.end());
    
    // Run Lanczos without reorthogonalization
    std::vector<double> lanczos_no_ortho_evals;
    std::cout << "Running Lanczos without reorthogonalization..." << std::endl;
    timer.reset();
    lanczos_no_ortho_cuda(matrix_vector_function, N, max_iter, num_eigenvalues, 
                         tolerance, lanczos_no_ortho_evals, output_dir + "/no_ortho", 
                         compute_eigenvectors);
    double no_ortho_time = timer.elapsed();
    std::cout << "Lanczos without reorthogonalization time: " << no_ortho_time << " seconds" << std::endl;
    std::cout << "Speedup vs full diagonalization: " << full_diag_time / no_ortho_time << "x" << std::endl;
    double no_ortho_error = compare_eigenvalues(exact_eigenvalues, lanczos_no_ortho_evals, num_eigenvalues);
    std::cout << "Maximum error: " << std::scientific << no_ortho_error << std::endl << std::endl;
    
    // Run Lanczos with selective reorthogonalization
    std::vector<double> lanczos_selective_evals;
    std::cout << "Running Lanczos with selective reorthogonalization..." << std::endl;
    timer.reset();
    lanczos_selective_reorth_cuda(matrix_vector_function, N, max_iter, num_eigenvalues, 
                                 tolerance, lanczos_selective_evals, output_dir + "/selective", 
                                 compute_eigenvectors);
    double selective_time = timer.elapsed();
    std::cout << "Lanczos with selective reorthogonalization time: " << selective_time << " seconds" << std::endl;
    std::cout << "Speedup vs full diagonalization: " << full_diag_time / selective_time << "x" << std::endl;
    double selective_error = compare_eigenvalues(exact_eigenvalues, lanczos_selective_evals, num_eigenvalues);
    std::cout << "Maximum error: " << std::scientific << selective_error << std::endl << std::endl;
    
    // Run Lanczos with full reorthogonalization
    std::vector<double> lanczos_full_evals;
    std::cout << "Running Lanczos with full reorthogonalization..." << std::endl;
    timer.reset();
    lanczos_cuda(matrix_vector_function, N, max_iter, num_eigenvalues, 
                tolerance, lanczos_full_evals, output_dir + "/full", 
                compute_eigenvectors);
    double full_ortho_time = timer.elapsed();
    std::cout << "Lanczos with full reorthogonalization time: " << full_ortho_time << " seconds" << std::endl;
    std::cout << "Speedup vs full diagonalization: " << full_diag_time / full_ortho_time << "x" << std::endl;
    double full_ortho_error = compare_eigenvalues(exact_eigenvalues, lanczos_full_evals, num_eigenvalues);
    std::cout << "Maximum error: " << std::scientific << full_ortho_error << std::endl << std::endl;
    
    // Final summary
    std::cout << "=== SUMMARY ===" << std::endl;
    std::cout << "Method                            | Time (s) | Speedup | Max Error" << std::endl;
    std::cout << "-----------------------------------|----------|---------|----------" << std::endl;
    std::cout << "Full Diagonalization              | " << std::setw(8) << full_diag_time 
              << " | " << std::setw(7) << "1.00" << " | " << std::setw(9) << "0.00" << std::endl;
    std::cout << "Lanczos (No Reorth)               | " << std::setw(8) << no_ortho_time 
              << " | " << std::setw(7) << std::fixed << std::setprecision(2) << full_diag_time/no_ortho_time 
              << " | " << std::setw(9) << std::scientific << std::setprecision(2) << no_ortho_error << std::endl;
    std::cout << "Lanczos (Selective Reorth)        | " << std::setw(8) << selective_time 
              << " | " << std::setw(7) << std::fixed << std::setprecision(2) << full_diag_time/selective_time 
              << " | " << std::setw(9) << std::scientific << std::setprecision(2) << selective_error << std::endl;
    std::cout << "Lanczos (Full Reorth)             | " << std::setw(8) << full_ortho_time 
              << " | " << std::setw(7) << std::fixed << std::setprecision(2) << full_diag_time/full_ortho_time 
              << " | " << std::setw(9) << std::scientific << std::setprecision(2) << full_ortho_error << std::endl;
              
    return 0;
}