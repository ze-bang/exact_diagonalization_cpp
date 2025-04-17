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
#include <set>

// Define complex number type and matrix type for convenience
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;


std::array<double, 2> operator* (const std::array<double, 4>& a, const std::array<double, 4>& b) {
    return {a[0] * b[0] + a[1] * b[1], a[2] * b[0] + a[3] * b[1]};
}
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
                return {basis, Complex(0.0, 0.0)};
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
                return {basis, Complex(0.0, 0.0)};
            });
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;    
    }
private:
    std::vector<TransformFunction> transforms_;
    int n_bits_; // Number of bits in the basis representation
    const std::array<std::array<double, 4>, 3> operators = {
        {{0, 1, 0, 0}, {0, 0, 1, 0},{1, 0, 0, -1}}
    };

    const std::array<std::array<double, 2>, 2> basis = {
        {{1, 0}, {0, 1}}
    };

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

/**
 * Creates a graph showing commutation relations between automorphisms
 * Vertices are automorphisms, edges indicate that the two automorphisms commute
 */
class AutomorphismCommutationGraph {
public:
    // Constructor
    AutomorphismCommutationGraph(const std::vector<std::vector<int>>& automorphisms) 
        : automorphisms_(automorphisms) {
        buildGraph();
    }
    
    // Build the graph of commuting automorphisms
    void buildGraph() {
        int n = automorphisms_.size();
        adjacency_matrix_.resize(n, std::vector<bool>(n, false));
        
        // Check each pair of automorphisms
        for (int i = 0; i < n; ++i) {
            // An automorphism always commutes with itself
            adjacency_matrix_[i][i] = true;
            
            for (int j = i+1; j < n; ++j) {
                if (commute(automorphisms_[i], automorphisms_[j])) {
                    adjacency_matrix_[i][j] = true;
                    adjacency_matrix_[j][i] = true;
                }
            }
        }
    }
    
    // Export to DOT format for visualization with Graphviz
    void exportToDOT(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        file << "graph AutomorphismCommutationGraph {\n";
        file << "  node [shape=circle style=filled fillcolor=lightgreen];\n";
        
        // Add edges
        for (size_t i = 0; i < automorphisms_.size(); ++i) {
            for (size_t j = i+1; j < automorphisms_.size(); ++j) {
                if (adjacency_matrix_[i][j]) {
                    file << "  \"";
                    printAutomorphism(file, automorphisms_[i]);
                    file << "\" -- \"";
                    printAutomorphism(file, automorphisms_[j]);
                    file << "\";\n";
                }
            }
        }
        
        // Add isolated vertices if any
        for (size_t i = 0; i < automorphisms_.size(); ++i) {
            bool hasEdge = false;
            for (size_t j = 0; j < automorphisms_.size(); ++j) {
                if (i != j && adjacency_matrix_[i][j]) {
                    hasEdge = true;
                    break;
                }
            }
            
            if (!hasEdge) {
                file << "  \"";
                printAutomorphism(file, automorphisms_[i]);
                file << "\";\n";
            }
        }
        
        file << "}\n";
        file.close();
        
        std::cout << "Automorphism commutation graph exported to " << filename << std::endl;
        std::cout << "Visualize with: dot -Tpng " << filename << " -o automorphism_graph.png" << std::endl;
    }
    
    // Print the adjacency matrix
    void printAdjacencyMatrix() const {
        std::cout << "Automorphism Commutation Matrix:" << std::endl;
        for (size_t i = 0; i < adjacency_matrix_.size(); ++i) {
            for (size_t j = 0; j < adjacency_matrix_.size(); ++j) {
                std::cout << (adjacency_matrix_[i][j] ? "1 " : "0 ");
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<std::vector<int>> automorphisms_;
    std::vector<std::vector<bool>> adjacency_matrix_;
    
    // Check if two automorphisms commute
    bool commute(const std::vector<int>& a, const std::vector<int>& b) const {
        int n = a.size();
        
        // Compute a○b
        std::vector<int> ab(n);
        for (int i = 0; i < n; ++i) {
            ab[i] = a[b[i]];
        }
        
        // Compute b○a
        std::vector<int> ba(n);
        for (int i = 0; i < n; ++i) {
            ba[i] = b[a[i]];
        }
        
        // Check if a○b = b○a
        return ab == ba;
    }
    
    // Helper function to print an automorphism to a stream
    void printAutomorphism(std::ofstream& file, const std::vector<int>& auto_perm) const {
        for (size_t k = 0; k < auto_perm.size(); ++k) {
            file << auto_perm[k];
            if (k < auto_perm.size() - 1) {
                file << ",";
            }
        }
    }
};

/**
 * Finds the maximum clique in a graph using a recursive backtracking algorithm
 * A clique is a subset of vertices where every two vertices are connected by an edge
 */
class MaximumCliqueFinder {
public:
    MaximumCliqueFinder(const std::vector<std::vector<bool>>& adjacency_matrix) 
        : adjacency_matrix_(adjacency_matrix) {
        max_clique_size_ = 0;
    }
    
    // Find the maximum clique
    std::vector<int> findMaximumClique() {
        int n = adjacency_matrix_.size();
        std::vector<int> current;
        std::vector<int> remaining;
        max_clique_.clear();
        max_clique_size_ = 0;
        
        // Start with all vertices as candidates
        for (int i = 0; i < n; ++i) {
            remaining.push_back(i);
        }
        
        findClique(current, remaining);
        return max_clique_;
    }
    
    // Print the maximum clique for an automorphism graph
    void printMaximumCliqueForAutomorphisms(const std::vector<std::vector<int>>& automorphisms) {
        std::vector<int> clique_indices = findMaximumClique();
        
        std::cout << "Maximum clique size: " << clique_indices.size() << std::endl;
        std::cout << "Maximum clique (indices): ";
        for (int idx : clique_indices) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Corresponding automorphisms: " << std::endl;
        for (int idx : clique_indices) {
            std::cout << "  ";
            for (int v : automorphisms[idx]) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
    }
    
private:
    const std::vector<std::vector<bool>>& adjacency_matrix_;
    std::vector<int> max_clique_;
    int max_clique_size_;
    
    // Recursive function to find maximal cliques
    void findClique(std::vector<int>& current, std::vector<int>& remaining) {
        // If no more candidates, we have a maximal clique
        if (remaining.empty()) {
            if (current.size() > max_clique_size_) {
                max_clique_ = current;
                max_clique_size_ = current.size();
            }
            return;
        }
        
        // If even adding all remaining vertices won't beat the current max, stop
        if (current.size() + remaining.size() <= max_clique_size_) {
            return;
        }
        
        // Copy remaining for iteration safety
        std::vector<int> remaining_copy = remaining;
        
        for (int v : remaining_copy) {
            // Create new candidate set of vertices adjacent to v
            std::vector<int> new_remaining;
            for (int u : remaining) {
                if (u != v && adjacency_matrix_[v][u]) {
                    new_remaining.push_back(u);
                }
            }
            
            // Add v to current clique and recurse
            current.push_back(v);
            findClique(current, new_remaining);
            current.pop_back();
            
            // Remove v from remaining
            remaining.erase(std::remove(remaining.begin(), remaining.end(), v), remaining.end());
        }
    }
};

/**
 * Finds the minimal set of automorphism generators that can produce all automorphisms in a clique
 * through composition
 */
class AutomorphismGeneratorFinder {
public:
    AutomorphismGeneratorFinder(const std::vector<std::vector<int>>& automorphisms)
        : automorphisms_(automorphisms) {}
    
    // Find the minimal set of generators for the given clique indices
    std::vector<int> findMinimalGenerators(const std::vector<int>& clique_indices) {
        std::vector<int> generators;
        std::vector<std::vector<int>> clique_automorphisms;
        
        // Extract the automorphisms in the clique
        for (int idx : clique_indices) {
            clique_automorphisms.push_back(automorphisms_[idx]);
        }
        
        std::vector<bool> included(clique_indices.size(), false);
        std::vector<int> current_generators;
        
        findMinimalGeneratorsRecursive(clique_automorphisms, included, current_generators, 0, generators);
        
        // Convert back to original indices
        std::vector<int> result;
        for (int gen : generators) {
            result.push_back(clique_indices[gen]);
        }
        
        return result;
    }
    
    // Print the minimal generators for a clique
    void printMinimalGenerators(const std::vector<int>& clique_indices) {
        std::vector<int> generator_indices = findMinimalGenerators(clique_indices);
        
        std::cout << "Minimal generator set size: " << generator_indices.size() << std::endl;
        std::cout << "Generator automorphisms: " << std::endl;
        
        for (int idx : generator_indices) {
            std::cout << "  ";
            for (int v : automorphisms_[idx]) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
    }
private:
    const std::vector<std::vector<int>>& automorphisms_;
    
    // Recursive function to find minimal generators
    void findMinimalGeneratorsRecursive(
        const std::vector<std::vector<int>>& clique_autos, 
        std::vector<bool>& included,
        std::vector<int>& current_gens,
        int start_idx,
        std::vector<int>& best_gens) {
        
        // Check if current generator set generates all automorphisms
        if (canGenerateAll(clique_autos, current_gens)) {
            // Update best solution if current is better
            if (best_gens.empty() || current_gens.size() < best_gens.size()) {
                best_gens = current_gens;
            }
            return;
        }
        
        // If we've tried all automorphisms, return
        if (start_idx >= clique_autos.size()) {
            return;
        }
        
        // Skip this automorphism
        findMinimalGeneratorsRecursive(clique_autos, included, current_gens, start_idx + 1, best_gens);
        
        // Include this automorphism
        if (!included[start_idx]) {
            included[start_idx] = true;
            current_gens.push_back(start_idx);
            
            findMinimalGeneratorsRecursive(clique_autos, included, current_gens, start_idx + 1, best_gens);
            
            // Backtrack
            current_gens.pop_back();
            included[start_idx] = false;
        }
    }
    
    // Check if generator set can produce all automorphisms in the clique
    bool canGenerateAll(const std::vector<std::vector<int>>& clique_autos, const std::vector<int>& gen_indices) {
        if (gen_indices.empty()) return clique_autos.size() <= 1;
        
        // Get the generator automorphisms
        std::vector<std::vector<int>> generators;
        for (int idx : gen_indices) {
            generators.push_back(clique_autos[idx]);
        }
        
        // Generate all possible automorphisms from the generators
        std::set<std::vector<int>> generated = generateClosure(generators);
        
        // Check if all clique automorphisms are in the generated set
        for (const auto& auto_perm : clique_autos) {
            if (generated.find(auto_perm) == generated.end()) {
                return false;
            }
        }
        
        return true;
    }
    
    // Generate all automorphisms that can be created from the given generators
    std::set<std::vector<int>> generateClosure(const std::vector<std::vector<int>>& generators) {
        std::set<std::vector<int>> result;
        
        // Add identity permutation
        std::vector<int> identity(generators[0].size());
        for (size_t i = 0; i < identity.size(); ++i) {
            identity[i] = i;
        }
        result.insert(identity);
        
        // Add all generators
        for (const auto& gen : generators) {
            result.insert(gen);
        }
        
        bool changed = true;
        while (changed) {
            changed = false;
            int start_size = result.size();
            
            // Try composing each generator with each element in result
            std::vector<std::vector<int>> current_elements(result.begin(), result.end());
            
            for (const auto& gen : generators) {
                for (const auto& elem : current_elements) {
                    // Compose gen ○ elem
                    std::vector<int> composed = compose(gen, elem);
                    
                    // Add the new element if not already present
                    if (result.find(composed) == result.end()) {
                        result.insert(composed);
                        changed = true;
                    }
                }
            }
        }
        
        return result;
    }
    
    // Compose two automorphisms: a(b(x))
    std::vector<int> compose(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[b[i]];
        }
        return result;
    }
};

/**
 * Decomposes automorphisms in a clique as powers of their minimal generators
 */
class AutomorphismDecomposer {
public:
    AutomorphismDecomposer(const std::vector<std::vector<int>>& automorphisms)
        : automorphisms_(automorphisms) {}
    
    // Decompose each automorphism in the clique into generator powers
    std::vector<std::vector<int>> decomposeClique(
        const std::vector<int>& clique_indices,
        const std::vector<int>& generator_indices) {
        
        std::vector<std::vector<int>> result;
        std::vector<std::vector<int>> clique_autos;
        std::vector<std::vector<int>> generators;
        
        // Extract the automorphisms in the clique
        for (int idx : clique_indices) {
            clique_autos.push_back(automorphisms_[idx]);
        }
        
        // Extract the generator automorphisms
        for (int idx : generator_indices) {
            generators.push_back(automorphisms_[clique_indices[idx]]);
        }
        
        // Find the order of each generator
        std::vector<int> generator_orders = findGeneratorOrders(generators);
        
        // For each automorphism in the clique
        for (const auto& auto_perm : clique_autos) {
            // Find powers of generators that produce this automorphism
            std::vector<int> powers = findGeneratorPowers(auto_perm, generators, generator_orders);
            result.push_back(powers);
        }
        
        return result;
    }
    
    // Print the decomposition
    void printDecomposition(
        const std::vector<int>& clique_indices,
        const std::vector<int>& generator_indices) {
        
        std::vector<std::vector<int>> decomposition = 
            decomposeClique(clique_indices, generator_indices);
        
        std::cout << "Automorphism decomposition in terms of generator powers:" << std::endl;
        for (size_t i = 0; i < clique_indices.size(); ++i) {
            std::cout << "Automorphism " << clique_indices[i] << ": [";
            for (size_t j = 0; j < decomposition[i].size(); ++j) {
                std::cout << decomposition[i][j];
                if (j < decomposition[i].size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }
    
private:
    const std::vector<std::vector<int>>& automorphisms_;
    
    // Find the order of each generator
    std::vector<int> findGeneratorOrders(const std::vector<std::vector<int>>& generators) {
        std::vector<int> orders;
        
        for (const auto& gen : generators) {
            std::vector<int> current = gen;
            int order = 1;
            
            while (!isIdentity(current)) {
                current = compose(gen, current);
                order++;
            }
            
            orders.push_back(order);
        }
        
        return orders;
    }
    
    bool isIdentity(const std::vector<int>& perm) {
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] != i) return false;
        }
        return true;
    }
    
    std::vector<int> findGeneratorPowers(
        const std::vector<int>& target,
        const std::vector<std::vector<int>>& generators,
        const std::vector<int>& generator_orders) {
        
        std::vector<int> current_powers(generators.size(), 0);
        std::vector<int> best_powers;
        
        if (tryAllPowerCombinations(target, generators, generator_orders, current_powers, 0, best_powers)) {
            return best_powers;
        }
        
        return std::vector<int>(generators.size(), 0);
    }
    
    bool tryAllPowerCombinations(
        const std::vector<int>& target,
        const std::vector<std::vector<int>>& generators,
        const std::vector<int>& generator_orders,
        std::vector<int>& current_powers,
        int gen_idx,
        std::vector<int>& best_powers) {
        
        if (gen_idx >= generators.size()) {
            std::vector<int> result = applyGeneratorPowers(generators, current_powers);
            if (result == target) {
                best_powers = current_powers;
                return true;
            }
            return false;
        }
        
        for (int power = 0; power < generator_orders[gen_idx]; ++power) {
            current_powers[gen_idx] = power;
            if (tryAllPowerCombinations(target, generators, generator_orders, current_powers, gen_idx + 1, best_powers)) {
                return true;
            }
        }
        
        return false;
    }
    
    std::vector<int> applyGeneratorPowers(
        const std::vector<std::vector<int>>& generators,
        const std::vector<int>& powers) {
        
        std::vector<int> result(generators[0].size());
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = i;
        }
        
        for (size_t g = 0; g < generators.size(); ++g) {
            std::vector<int> gen_power = powerOf(generators[g], powers[g]);
            result = compose(gen_power, result);
        }
        
        return result;
    }
    
    std::vector<int> powerOf(const std::vector<int>& auto_perm, int power) {
        if (power == 0) {
            std::vector<int> identity(auto_perm.size());
            for (size_t i = 0; i < identity.size(); ++i) {
                identity[i] = i;
            }
            return identity;
        }
        
        std::vector<int> result = auto_perm;
        for (int i = 1; i < power; ++i) {
            result = compose(auto_perm, result);
        }
        return result;
    }
    
    std::vector<int> compose(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[b[i]];
        }
        return result;
    }
};

