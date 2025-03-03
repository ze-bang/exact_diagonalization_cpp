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

    // Generate matrix representation of the operator
    Matrix generateMatrix() const {
        int dim = 1 << n_bits_;  // Dimension of the vector space (2^n_bits)
        std::cout << "Matrix dimension: " << dim << "x" << dim << std::endl;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        // For each input basis state
        for (int i = 0; i < dim; ++i) {
            // Apply each transform and accumulate the results
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {  // Ensure the output state is valid
                    matrix[j][i] += scalar;
                }
            }
        }
        std::cout << "Matrix generated." << std::endl;
        return matrix;
    }

    // Apply the operator to a complex vector
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        int dim = 1 << n_bits_;  // Dimension of the vector space (2^n_bits)
        
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
            int A, B, C, D;
            double E, F;
            std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> A >> B >> C >> D >> E >> F)) {
                continue; // Skip invalid lines
            }
            
            // // Update n_bits if needed
            // n_bits_ = std::max({n_bits_, A + 1, C + 1});
            
            // Add a transform function for this interaction
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                bool A_matches = ((basis >> A) & 1) == B;
                bool B_matches = ((basis >> C) & 1) == D;

                
                if (!(A_matches && B_matches)) {
                    return {-1, Complex(0.0, 0.0)}; // No contribution
                }
                
                // Flip all four bits
                int flipped_basis = basis ^ ((1 << A) | (1 << C));
                return {flipped_basis, Complex(E, F)};
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
            int A1, A2, B1, B2, C1, C2, D1, D2;
            double E, F;
            std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> A1 >> A2 >> B1 >> B2 >> C1 >> C2 >> D1 >> D2 >> E >> F)) {
                continue; // Skip invalid lines
            }
            
            // Add a transform function for this interaction
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                bool A_matches = ((basis >> A1) & 1) == A2;
                bool B_matches = ((basis >> B1) & 1) == B2;
                bool C_matches = ((basis >> C1) & 1) == C2;
                bool D_matches = ((basis >> D1) & 1) == D2;
                
                if (!(A_matches && B_matches && C_matches && D_matches)) {
                    return {-1, Complex(0.0, 0.0)}; // No contribution
                }
                
                // Flip all four bits
                int flipped_basis = basis ^ ((1 << A1) | (1 << B1) | (1 << C1) | (1 << D1));
                return {flipped_basis, Complex(E, F)};
            });
            
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;
    }
    
};
    // Load operator definition from an InterAll.def file
    

    // Helper function to print a binary representation of a number
std::string toBinary(int n, int bits) {
    std::string result;
    for (int i = bits - 1; i >= 0; --i) {
        result += ((n >> i) & 1) ? '1' : '0';
    }
    return result;
}

// Helper function to print a matrix
void printMatrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            if (std::abs(elem.imag()) < 1e-10) {
                std::cout << std::setw(8) << elem.real() << " ";
            } else {
                std::cout << std::setw(8) << elem.real()
                          << (elem.imag() >= 0 ? "+" : "") 
                          << elem.imag() << "i ";
            }
        }
        std::cout << std::endl;
    }
}

// Example of how to use the Operator class
int main() {
    // Create an operator for a 3-bit system
    int n_bits = 16;
    Operator op(n_bits);

    // Example: Add a transform that flips the first bit (position 0) and multiplies by 2
    op.loadFromFile("Trans.def");
    op.loadFromInterAllFile("InterAll.def");
    // op.generateMatrix();
    return 0;
}