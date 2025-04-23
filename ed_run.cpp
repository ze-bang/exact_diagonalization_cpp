#include <iostream>
#include "construct_ham.h"
#include <fstream>
#include <string>
#include "ed_wrapper.h"


int main(int argc, char* argv[]) {
    // Test parameters
    const std::string dir = argv[1];
    // Read num_site from the second line of Trans.dat

    // Read num_site from the second line of Trans.dat
    std::ifstream trans_file(dir + "/Trans.dat");
    if (!trans_file.is_open()) {
        std::cerr << "Error: Cannot open file " << dir + "/Trans.dat" << std::endl;
        return 1;
    }

    // Skip the first line
    std::string dummy_line;
    std::getline(trans_file, dummy_line);

    // Read the second line to get num_site
    std::string dum;
    int num_site;
    trans_file >> dum >> num_site;
    trans_file.close();

    std::cout << "Number of sites: " << num_site << std::endl;

    // Create operator with correct number of sites
    Operator op(num_site);
    op.generateSymmetrizedBasis(dir);
    // Load operator definitions
    op.loadFromFile(dir + "/Trans.dat");
    op.loadFromInterAllFile(dir + "/InterAll.dat");

    // Print the regular matrix representation
    std::cout << "\nRegular matrix representation:" << std::endl;
    Matrix regular_matrix = op.returnMatrix();
    int dim = 1 << num_site;
    
    // Determine max width for formatting
    int max_dim = std::min(20, dim); // Limit display size for large matrices
    
    if (dim <= max_dim) {
        // Print full matrix
        for (int i = 0; i < dim; i++) {
            std::cout << "[";
            for (int j = 0; j < dim; j++) {
                Complex val = regular_matrix[i][j];
                if (std::abs(val) < 1e-10) {
                    std::cout << " 0 ";
                } else if (std::abs(val.imag()) < 1e-10) {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() << " ";
                } else {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() 
                              << (val.imag() >= 0 ? "+" : "") << std::setprecision(2) << val.imag() << "i ";
                }
                if (j < dim - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "Matrix too large to display fully (" << dim << "x" << dim << "). Showing top-left " 
                 << max_dim << "x" << max_dim << " corner:" << std::endl;
        for (int i = 0; i < max_dim; i++) {
            std::cout << "[";
            for (int j = 0; j < max_dim; j++) {
                Complex val = regular_matrix[i][j];
                if (std::abs(val) < 1e-10) {
                    std::cout << " 0 ";
                } else if (std::abs(val.imag()) < 1e-10) {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() << " ";
                } else {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() 
                              << (val.imag() >= 0 ? "+" : "") << std::setprecision(2) << val.imag() << "i ";
                }
                if (j < max_dim - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "..." << std::endl;
    }

    // Print the symmetrized matrix representation
    std::cout << "\nSymmetrized matrix representation:" << std::endl;
    Matrix sym_matrix = op.returnSymmetrizedMatrix(dir);
    int sym_dim = sym_matrix.size();
    max_dim = std::min(20, sym_dim); // Limit display size for large matrices
    
    if (sym_dim <= max_dim) {
        // Print full matrix
        for (int i = 0; i < sym_dim; i++) {
            std::cout << "[";
            for (int j = 0; j < sym_matrix[i].size(); j++) {
                Complex val = sym_matrix[i][j];
                if (std::abs(val) < 1e-10) {
                    std::cout << " 0.00 ";
                } else if (std::abs(val.imag()) < 1e-10) {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() << " ";
                } else {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() 
                              << (val.imag() >= 0 ? "+" : "") << std::setprecision(2) << val.imag() << "i ";
                }
                if (j < sym_matrix[i].size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "Matrix too large to display fully (" << sym_dim << "x" << sym_dim << "). Showing top-left " 
                 << max_dim << "x" << max_dim << " corner:" << std::endl;
        for (int i = 0; i < max_dim; i++) {
            std::cout << "[";
            int cols = std::min(max_dim, (int)sym_matrix[i].size());
            for (int j = 0; j < cols; j++) {
                Complex val = sym_matrix[i][j];
                if (std::abs(val) < 1e-10) {
                    std::cout << " 0.00 ";
                } else if (std::abs(val.imag()) < 1e-10) {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() << " ";
                } else {
                    std::cout << " " << std::fixed << std::setprecision(2) << val.real() 
                              << (val.imag() >= 0 ? "+" : "") << std::setprecision(2) << val.imag() << "i ";
                }
                if (j < cols - 1) std::cout << ",";
            }
            if (sym_matrix[i].size() > max_dim) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}