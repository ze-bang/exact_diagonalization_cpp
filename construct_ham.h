#include <vector>
#include <complex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

// Define complex number type and matrix type for convenience
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

/**
 * Operator class that can represent arbitrary quantum operators
 * through bit flip operations and scalar multiplications
 */
class Operator {
public:
    // Function type for transforming basis states
    using TransformFunction = std::function<std::pair<int, Complex>(int)>;

private:
    std::vector<TransformFunction> transforms_;
    int n_bits_; // Number of bits in the basis representation

public:
    // Constructor
    Operator(int n_bits) : n_bits_(n_bits) {}

    // Add a transform operation to the operator
    void addTransform(TransformFunction transform) {
        transforms_.push_back(transform);
    }

    // Apply the operator to a complex vector
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        int dim = n_bits_;  // Dimension of the vector space (2^n_bits)
        
        // Check if the input vector has the correct dimension
        if (vec.size() != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input vector dimension does not match operator dimension");
        }
        
        // Initialize result vector with zeros
        std::vector<Complex> result(dim, 0.0);
        
        // For each input basis state
        for (int i = 0; i < dim; ++i) {
            // Apply each transform to the current basis state
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {  // Ensure the output state is valid
                    // Add the contribution to the result
                    result[j] += scalar * vec[i];
                }
            }
        }
        
        return result;
    }

    // Print the operator as a matrix
    Matrix returnMatrix(){
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {
                    matrix[j][i] += scalar;
                }
            }
        }
        return matrix;
    }
    

    // Load operator definition from a file
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        // Skip the first line (header)
        std::getline(file, line);
        
        // Read the number of lines
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        std::cout << "Number of lines: " << numLines << std::endl;
        // Skip the next 3 lines (separators/headers)
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        // Process transform data
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> Op >> indx >> E >> F)) {
                continue; // Skip invalid lines
            }
            
            // // Update n_bits if needed
            // n_bits_ = std::max({n_bits_, A + 1, C + 1});
            
            // Add a transform function for this interaction
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                if (Op == 2){
                    return {basis, Complex(E*0.5*pow(-1,(basis >> indx) & 1),F)};
                }
                else{
                    bool A_matches = ((basis >> indx) & 1) == Op;
                    if (A_matches) {
                        return {-1, Complex(0.0, 0.0)}; // No contribution
                    }
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << indx);
                    return {flipped_basis, Complex(E, F)};
                }
                // }
                // else{
                //     return {basis, Complex(0.0, 0.0)};
                // }
            });
            
            
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;
    }
    void loadFromInterAllFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        // Skip the first line (header)
        std::getline(file, line);
        
        // Read the number of lines

        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        std::cout << "Number of lines: " << numLines << std::endl;
        
        // Skip the next 3 lines (separators/headers)
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        // Process transform data
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op1, indx1, Op2, indx2;
            double E, F;
            std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                continue; // Skip invalid lines
            }
            
            // Add a transform function for this interaction
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                if (Op2 == 2){
                    return {basis, Complex(0.5*pow(-1,(basis >> indx2) & 1), 0)};
                }
                else{
                    bool A_matches = ((basis >> indx2) & 1) == Op2;
                    if (A_matches) {
                        return {-1, Complex(0.0, 0.0)}; // No contribution
                    }
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << indx2);
                    return {flipped_basis, Complex(1.0, 0)};
                }
            });

            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                if (Op1 == 2){
                    return {basis, Complex(E*0.5*pow(-1,(basis >> indx1) & 1), F)};
                }
                else{
                    bool A_matches = ((basis >> indx1) & 1) == Op1;
                    if (A_matches) {
                        return {-1, Complex(0.0, 0.0)}; // No contribution
                    }
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << indx1);
                    return {flipped_basis, Complex(E, F)};
                }
            });
            
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;
    }
    
};
    // Load operator definition from an InterAll.def file
    
// Example of how to use the Operator class
