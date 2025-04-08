#include <vector>
#include <complex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <Eigen/Dense>


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

    // Constructor
    
    Operator(int n_bits) : n_bits_(n_bits) {}

    // Mark matrix as needing rebuild when new transform is added
    void addTransform(TransformFunction transform) {
        transforms_.push_back(transform);
        matrixBuilt_ = false;  // Matrix needs to be rebuilt
    }

    // Apply the operator to a complex vector using Eigen sparse matrix operations
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        int dim = 1 << n_bits_;
        
        if (vec.size() != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input vector dimension does not match operator dimension");
        }
        
        // Build the sparse matrix if not already built
        buildSparseMatrix();
        
        // Convert input vector to Eigen vector
        Eigen::VectorXcd eigenVec(dim);
        for (int i = 0; i < dim; ++i) {
            eigenVec(i) = vec[i];
        }
        
        // Perform sparse matrix-vector multiplication
        Eigen::VectorXcd result = sparseMatrix_ * eigenVec;
        
        // Convert back to std::vector
        std::vector<Complex> resultVec(dim);
        for (int i = 0; i < dim; ++i) {
            resultVec[i] = result(i);
        }
        
        return resultVec;
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
        // std::cout << "Number of lines: " << numLines << std::endl;
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
            // std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> Op >> indx >> E >> F)) {
                continue; // Skip invalid lines
            }
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                if (Op == 2){
                    return {basis, Complex(E,F)*0.5*pow(-1,(basis >> indx) & 1)};
                }
                else{
                    if (((basis >> indx) & 1) != Op) {
                        // Flip the A bit
                        int flipped_basis = basis ^ (1 << indx);
                        return {flipped_basis, Complex(E, F)};                    
                    }
                }
                // Default case: no transformation applies
                return {-1, Complex(0.0, 0.0)};
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
        // std::cout << "Number of lines: " << numLines << std::endl;
        
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
            // std::cout << "Reading line: " << line << std::endl;
            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                continue; // Skip invalid lines
            }
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check what type of operators we're dealing with
                if (Op1 == 2 && Op2 == 2) {
                    // Both are identity operators with phase factors
                    int bit1 = (basis >> indx1) & 1;
                    int bit2 = (basis >> indx2) & 1;
                    return {basis, Complex(E, F)* 0.25 * pow(-1, bit1) * pow(-1, bit2)};
                } 
                else if (Op1 == 2) {
                    // Op1 is identity with phase, Op2 is bit flip
                    int bit1 = (basis >> indx1) & 1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit2_matches) {
                        int flipped_basis = basis ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F) * 0.5 * pow(-1, bit1)};
                    }
                } 
                else if (Op2 == 2) {
                    // Op2 is identity with phase, Op1 is bit flip
                    int bit2 = (basis >> indx2) & 1;
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    
                    if (bit1_matches) {
                        // Flip the first bit
                        int flipped_basis = basis ^ (1 << indx1);
                        return {flipped_basis, Complex(E, F)* 0.5 * pow(-1, bit2)};
                    }
                } 
                else {
                    // Both are bit flip operators
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit1_matches && bit2_matches) {
                        // Flip both bits
                        int flipped_basis = basis ^ (1 << indx1) ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F)};
                    }
                }
                // Default case: no transformation applies
                return {-1, Complex(0.0, 0.0)};
            });
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;    
    }
private:
    std::vector<TransformFunction> transforms_;
    int n_bits_; // Number of bits in the basis representation
    mutable Eigen::SparseMatrix<Complex> sparseMatrix_;
    mutable bool matrixBuilt_ = false;

    // Build the sparse matrix from transforms if needed
    void buildSparseMatrix() const {
        if (matrixBuilt_) return;
        
        int dim = 1 << n_bits_;
        sparseMatrix_.resize(dim, dim);
        
        // Use triplets to efficiently build the sparse matrix
        std::vector<Eigen::Triplet<Complex>> triplets;
        
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {
                    triplets.emplace_back(j, i, scalar);
                }
            }
        }
        
        sparseMatrix_.setFromTriplets(triplets.begin(), triplets.end());
        matrixBuilt_ = true;
    }
};


/**
 * Creates a graphical representation of interactions between quantum sites
 * based on the interall.def file
 */
class InteractionGraph {
    // Add accessor method to InteractionGraph to get the adjacency matrix
public:
    // Constructor
    InteractionGraph(int n_sites) : n_sites_(n_sites) {
        adjacency_matrix_.resize(n_sites, std::vector<bool>(n_sites, false));
    }

    // Parse the interall.def file to build the interaction graph
    void buildFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "Building interaction graph from: " << filename << std::endl;
        std::string line;
        
        // Skip header
        std::getline(file, line);
        
        // Read number of lines
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Skip format lines
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        // Process each interaction
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op1, indx1, Op2, indx2;
            double E, F;
            
            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                continue;
            }
            
            // Add edge between the sites
            if (indx1 >= 0 && indx1 < n_sites_ && indx2 >= 0 && indx2 < n_sites_) {
                adjacency_matrix_[indx1][indx2] = true;
                adjacency_matrix_[indx2][indx1] = true;  // Undirected graph
            }
            
            lineCount++;
        }
        std::cout << "Interaction graph built successfully." << std::endl;
    }

    // Export to DOT format for visualization with Graphviz
    void exportToDOT(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        file << "graph InteractionNetwork {\n";
        file << "  node [shape=circle style=filled fillcolor=lightblue];\n";
        
        // Add edges
        for (int i = 0; i < n_sites_; ++i) {
            for (int j = i+1; j < n_sites_; ++j) {
                if (adjacency_matrix_[i][j]) {
                    file << "  " << i << " -- " << j << ";\n";
                }
            }
        }
        
        file << "}\n";
        file.close();
        
        std::cout << "Graph exported to " << filename << std::endl;
        std::cout << "Visualize with: dot -Tpng " << filename << " -o graph.png" << std::endl;
    }

    // Print the adjacency matrix
    void printAdjacencyMatrix() const {
        std::cout << "Interaction Adjacency Matrix:" << std::endl;
        for (int i = 0; i < n_sites_; ++i) {
            for (int j = 0; j < n_sites_; ++j) {
                std::cout << (adjacency_matrix_[i][j] ? "1 " : "0 ");
            }
            std::cout << std::endl;
        }
    }

    const std::vector<std::vector<bool>>& getAdjacencyMatrix() const {
        return adjacency_matrix_;
    }

private:
    int n_sites_;
    std::vector<std::vector<bool>> adjacency_matrix_;
};

/**
 * Finds all automorphisms of the interaction graph
 * An automorphism is a permutation of vertices that preserves adjacency relationships
 */
class GraphAutomorphismFinder {
public:
    GraphAutomorphismFinder(const InteractionGraph& graph, int n_sites) 
        : adjacency_matrix_(graph.getAdjacencyMatrix()), n_sites_(n_sites) {}
    
    // Get all automorphisms (permutations that preserve graph structure)
    std::vector<std::vector<int>> findAllAutomorphisms() {
        std::vector<std::vector<int>> automorphisms;
        std::vector<int> perm(n_sites_);
        
        // Initialize the first permutation as identity
        for (int i = 0; i < n_sites_; i++) {
            perm[i] = i;
        }
        
        // Generate all permutations and check if they're automorphisms
        do {
            if (isAutomorphism(perm)) {
                automorphisms.push_back(perm);
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        
        return automorphisms;
    }
    
    // Print the automorphisms in a readable format
    void printAutomorphisms(const std::vector<std::vector<int>>& automorphisms) const {
        std::cout << "Found " << automorphisms.size() << " automorphisms:" << std::endl;
        for (size_t i = 0; i < automorphisms.size(); ++i) {
            std::cout << "Automorphism " << (i+1) << ": ";
            for (int v : automorphisms[i]) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
    }
    
private:
    const std::vector<std::vector<bool>>& adjacency_matrix_;
    int n_sites_;
    
    // Check if a permutation is an automorphism
    bool isAutomorphism(const std::vector<int>& perm) const {
        // For each pair of vertices (i,j), check if adjacency is preserved
        for (int i = 0; i < n_sites_; ++i) {
            for (int j = 0; j < n_sites_; ++j) {
                // original adjacency between i and j
                bool orig_adj = adjacency_matrix_[i][j];
                
                // adjacency between mapped vertices perm[i] and perm[j]
                bool mapped_adj = adjacency_matrix_[perm[i]][perm[j]];
                
                // If adjacency relation isn't preserved, this isn't an automorphism
                if (orig_adj != mapped_adj) {
                    return false;
                }
            }
        }
        return true;
    }
};

