#ifndef CONSTRUCT_HAM_H
#define CONSTRUCT_HAM_H

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
#include <queue>
#include <random>

// Define complex number type and matrix type for convenience
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;


int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

std::array<double, 2> operator* (const std::array<double, 4>& a, const std::array<double, 4>& b) {
    return {a[0] * b[0] + a[1] * b[1], a[2] * b[0] + a[3] * b[1]};
}



/**
 * HamiltonianVisualizer class for creating graphical representations of Hamiltonians
 */
class HamiltonianVisualizer {
public:
    HamiltonianVisualizer(int n_sites) : n_sites_(n_sites) {}

    void loadEdgesFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        std::string num;
        int numInteractions;
        iss >> num >> numInteractions;
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numInteractions) {
            std::istringstream lineStream(line);
            int op1, site1, op2, site2;
            double real, imag;
            
            if (!(lineStream >> op1 >> site1 >> op2 >> site2 >> real >> imag)) {
                continue;
            }
            
            Edge edge;
            edge.site1 = site1;
            edge.site2 = site2;
            edge.op1 = op1;
            edge.op2 = op2;
            edge.weight = std::complex<double>(real, imag);
            edges.push_back(edge);
            
            lineCount++;
        }
    }

    void loadVerticesFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        std::string num;
        int numVertices;
        iss >> num >> numVertices;
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numVertices) {
            std::istringstream lineStream(line);
            int op, site;
            double real, imag;
            
            if (!(lineStream >> op >> site >> real >> imag)) {
                continue;
            }
            
            Vertex vertex;
            vertex.site = site;
            vertex.op = op;
            vertex.weight = std::complex<double>(real, imag);
            vertices.push_back(vertex);
            
            lineCount++;
        }
    }

    void generateDotFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        file << "graph Hamiltonian {\n";
        file << "  node [shape=circle];\n";
        
        for (int i = 0; i < n_sites_; ++i) {
            file << "  " << i << " [label=\"Site " << i << "\"";
            
            for (const auto& vertex : vertices) {
                if (vertex.site == i) {
                    file << ", tooltip=\"Op: " << getOperatorName(vertex.op) 
                         << ", Weight: " << vertex.weight.real();
                    if (vertex.weight.imag() != 0) {
                        file << " + " << vertex.weight.imag() << "i";
                    }
                    file << "\"";
                    
                    double magnitude = std::abs(vertex.weight);
                    if (magnitude > 0) {
                        int intensity = std::min(255, static_cast<int>(255 * magnitude / 0.5));
                        file << ", fillcolor=\"#" << std::hex << intensity << "0000\"";
                        file << ", style=filled";
                    }
                    break;
                }
            }
            
            file << "];\n";
        }
        
        for (const auto& edge : edges) {
            if (edge.site1 != edge.site2) {
                file << "  " << edge.site1 << " -- " << edge.site2;
                
                file << " [label=\"" << getOperatorName(edge.op1) << "-" << getOperatorName(edge.op2) << "\"";
                
                file << ", tooltip=\"Weight: " << edge.weight.real();
                if (edge.weight.imag() != 0) {
                    file << " + " << edge.weight.imag() << "i";
                }
                file << "\"";
                
                double magnitude = std::abs(edge.weight);
                double penwidth = 1.0 + 5.0 * magnitude / 0.5;
                file << ", penwidth=" << penwidth;
                
                if (magnitude > 0) {
                    file << ", color=\"#0000FF\"";
                }
                
                file << "];\n";
            }
        }
        
        file << "}\n";
        
        std::cout << "Generated DOT file: " << filename << std::endl;
        std::cout << "Visualize with: dot -Tpng " << filename << " -o graph.png" << std::endl;
    }

    void saveGraphImage(const std::string& outputFile, const std::string& dotFile = "temp_hamiltonian.dot") {
        generateDotFile(dotFile);
        std::string command = "dot -Tpng " + dotFile + " -o " + outputFile;
        int result = system(command.c_str());
        
        if (result != 0) {
            std::cerr << "Failed to generate graph image. Make sure GraphViz is installed." << std::endl;
            return;
        }
        
        std::cout << "Generated graph image: " << outputFile << std::endl;
    }

private:
    int n_sites_;

    struct Edge {
        int site1;
        int site2;
        int op1;
        int op2;
        std::complex<double> weight;
    };

    struct Vertex {
        int site;
        int op;
        std::complex<double> weight;
    };

    std::vector<Edge> edges;
    std::vector<Vertex> vertices;

    std::string getOperatorName(int op) {
        switch (op) {
            case 0: return "X";
            case 1: return "Y";
            case 2: return "Z";
            default: return "?";
        }
    }
};

/**
 * HamiltonianAutomorphismFinder class to find automorphisms of a Hamiltonian
 */
class HamiltonianAutomorphismFinder {
public:
    struct Edge {
        int site1;
        int site2;
        int op1;
        int op2;
        std::complex<double> weight;
    };

    struct Vertex {
        int site;
        int op;
        std::complex<double> weight;
    };
    
    HamiltonianAutomorphismFinder(int n_sites) : n_sites_(n_sites) {}
    
    void loadEdgesFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        std::string num;
        int numInteractions;
        iss >> num >> numInteractions;
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numInteractions) {
            std::istringstream lineStream(line);
            int op1, site1, op2, site2;
            double real, imag;
            
            if (!(lineStream >> op1 >> site1 >> op2 >> site2 >> real >> imag)) {
                continue;
            }
            
            Edge edge;
            edge.site1 = site1;
            edge.site2 = site2;
            edge.op1 = op1;
            edge.op2 = op2;
            edge.weight = std::complex<double>(real, imag);
            edges.push_back(edge);
            
            lineCount++;
        }
    }
    
    void loadVerticesFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        std::string num;
        int numVertices;
        iss >> num >> numVertices;
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numVertices) {
            std::istringstream lineStream(line);
            int op, site;
            double real, imag;
            
            if (!(lineStream >> op >> site >> real >> imag)) {
                continue;
            }
            
            Vertex vertex;
            vertex.site = site;
            vertex.op = op;
            vertex.weight = std::complex<double>(real, imag);
            vertices.push_back(vertex);
            
            lineCount++;
        }
    }
    
    bool isAutomorphism(const std::vector<int>& permutation) const {
        if (permutation.size() != n_sites_) {
            throw std::invalid_argument("Permutation size must match number of sites");
        }
        
        // Check edges
        for (const auto& edge : edges) {
            int permuted_site1 = permutation[edge.site1];
            int permuted_site2 = permutation[edge.site2];
            
            bool found = false;
            for (const auto& other_edge : edges) {
                // Check both orientations (undirected graph)
                bool matches1 = (other_edge.site1 == permuted_site1 && 
                                other_edge.site2 == permuted_site2 &&
                                other_edge.op1 == edge.op1 && 
                                other_edge.op2 == edge.op2 &&
                                std::abs(other_edge.weight - edge.weight) < 1e-10);
                
                bool matches2 = (other_edge.site1 == permuted_site2 && 
                                other_edge.site2 == permuted_site1 &&
                                other_edge.op1 == edge.op2 && 
                                other_edge.op2 == edge.op1 &&
                                std::abs(other_edge.weight - edge.weight) < 1e-10);
                
                if (matches1 || matches2) {
                    found = true;
                    break;
                }
            }
            
            if (!found) return false;
        }
        
        // Check vertices
        for (const auto& vertex : vertices) {
            int permuted_site = permutation[vertex.site];
            
            bool found = false;
            for (const auto& other_vertex : vertices) {
                if (other_vertex.site == permuted_site && 
                    other_vertex.op == vertex.op &&
                    std::abs(other_vertex.weight - vertex.weight) < 1e-10) {
                    found = true;
                    break;
                }
            }
            
            if (!found) return false;
        }
        
        return true;
    }
    
    std::vector<std::vector<int>> findAllAutomorphisms() const {
        std::vector<std::vector<int>> automorphisms;
        
        // Start with identity permutation
        std::vector<int> permutation(n_sites_);
        for (int i = 0; i < n_sites_; ++i) {
            permutation[i] = i;
        }
        int count = 0;
        // Generate all permutations and check if they're automorphisms
        do {
            count++;
            // Update the loading bar periodically
            if (count % 1000 == 0 || count == 1) {
                double percentage = (double)count / factorial(n_sites_) * 100.0;
                int barWidth = 40;
                int pos = barWidth * percentage / 100.0;
                
                std::cout << "\r[";
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%" << std::flush;
            }
            if (isAutomorphism(permutation)) {
                automorphisms.push_back(permutation);
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        
        return automorphisms;
    }
    
    std::string permutationToCycleNotation(const std::vector<int>& permutation) const {
        std::vector<bool> visited(permutation.size(), false);
        std::string result;
        
        for (size_t i = 0; i < permutation.size(); ++i) {
            if (visited[i] || permutation[i] == i) continue;
            
            result += "(";
            size_t j = i;
            do {
                result += std::to_string(j);
                visited[j] = true;
                j = permutation[j];
                if (visited[j] && j != i) break;
                if (j != i) result += " ";
            } while (j != i);
            result += ")";
        }
        
        // Add fixed points
        for (size_t i = 0; i < permutation.size(); ++i) {
            if (permutation[i] == i) {
                result += "(" + std::to_string(i) + ")";
            }
        }
        
        if (result.empty()) result = "()"; // Identity permutation
        
        return result;
    }
    
private:
    int n_sites_;
    std::vector<Edge> edges;
    std::vector<Vertex> vertices;
};

/**
 * Class for sampling automorphisms in large systems
 */
class AutomorphismSampler {
public:
    AutomorphismSampler(int n_sites) : n_sites_(n_sites) {}
    
    // Find a subset of automorphisms for large systems
    std::vector<std::vector<int>> sampleAutomorphisms(
        const HamiltonianAutomorphismFinder& finder, 
        int max_samples = 10000, 
        int max_automorphisms = 100) {
        
        std::vector<std::vector<int>> result;
        
        // Start with identity permutation
        std::vector<int> identity(n_sites_);
        for (int i = 0; i < n_sites_; ++i) {
            identity[i] = i;
        }
        result.push_back(identity);
        
        // Try common symmetries first
        tryCommonSymmetries(finder, result);
        
        // If we already found enough symmetries, return early
        if (result.size() >= max_automorphisms) {
            return result;
        }
        
        // Setup random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Random sampling approach
        std::cout << "Sampling random permutations for automorphisms..." << std::endl;
        int attempts = 0;
        
        while (attempts < max_samples && result.size() < max_automorphisms) {
            // Generate a random permutation
            std::vector<int> perm = identity;
            std::shuffle(perm.begin(), perm.end(), gen);
            
            if (finder.isAutomorphism(perm)) {
                bool is_new = std::find(result.begin(), result.end(), perm) == result.end();
                if (is_new) {
                    result.push_back(perm);
                    std::cout << "Found automorphism #" << result.size() << std::endl;
                    
                    // Also try compositions with existing automorphisms
                    tryCompositions(finder, result, perm, max_automorphisms);
                }
            }
            
            // Show progress
            if (++attempts % 100 == 0) {
                std::cout << "Sampled " << attempts << "/" << max_samples << " permutations, found " 
                         << result.size() << " automorphisms" << std::endl;
            }
        }
        
        return result;
    }
    
private:
    int n_sites_;
    
    // Try common symmetries like reflections, translations, etc.
    void tryCommonSymmetries(const HamiltonianAutomorphismFinder& finder, std::vector<std::vector<int>>& result) {
        std::cout << "Checking for common symmetries..." << std::endl;
        
        // 1. Reflection symmetry (for chain-like systems)
        std::vector<int> reflection(n_sites_);
        for (int i = 0; i < n_sites_; ++i) {
            reflection[i] = n_sites_ - 1 - i;
        }
        if (finder.isAutomorphism(reflection)) {
            result.push_back(reflection);
            std::cout << "Found reflection symmetry" << std::endl;
        }
        
        // 2. Translations (for periodic systems)
        for (int shift = 1; shift < n_sites_; ++shift) {
            std::vector<int> translation(n_sites_);
            for (int i = 0; i < n_sites_; ++i) {
                translation[i] = (i + shift) % n_sites_;
            }
            if (finder.isAutomorphism(translation)) {
                result.push_back(translation);
                std::cout << "Found translation symmetry with shift " << shift << std::endl;
            }
        }
        
        // 3. Site swap symmetries
        for (int i = 0; i < n_sites_; ++i) {
            for (int j = i + 1; j < n_sites_; ++j) {
                std::vector<int> swap_perm(n_sites_);
                for (int k = 0; k < n_sites_; ++k) {
                    if (k == i) swap_perm[k] = j;
                    else if (k == j) swap_perm[k] = i;
                    else swap_perm[k] = k;
                }
                
                if (finder.isAutomorphism(swap_perm)) {
                    result.push_back(swap_perm);
                    std::cout << "Found swap symmetry between sites " << i << " and " << j << std::endl;
                }
            }
        }
    }
    
    // Try composing a new automorphism with existing ones
    void tryCompositions(const HamiltonianAutomorphismFinder& finder, 
                        std::vector<std::vector<int>>& result, 
                        const std::vector<int>& perm,
                        int max_automorphisms) {
        if (result.size() >= max_automorphisms) return;
        
        // Try compositions with existing automorphisms
        size_t current_size = result.size();
        for (size_t i = 0; i < current_size && result.size() < max_automorphisms; ++i) {
            std::vector<int> composed(n_sites_);
            
            // Compose: perm ∘ existing
            for (int j = 0; j < n_sites_; ++j) {
                composed[j] = perm[result[i][j]];
            }
            
            // Check if this composition is new and valid
            if (finder.isAutomorphism(composed) && 
                std::find(result.begin(), result.end(), composed) == result.end()) {
                result.push_back(composed);
            }
            
            // Compose: existing ∘ perm
            for (int j = 0; j < n_sites_; ++j) {
                composed[j] = result[i][perm[j]];
            }
            
            // Check if this composition is new and valid
            if (finder.isAutomorphism(composed) && 
                std::find(result.begin(), result.end(), composed) == result.end()) {
                result.push_back(composed);
            }
        }
    }
};

/**
 * Function to find all automorphisms of a Hamiltonian
 * @param edgesFile Path to file containing edge data
 * @param verticesFile Path to file containing vertex data
 * @param n_sites Number of sites in the system
 * @return Vector of permutations (each a vector of integers)
 */
std::vector<std::vector<int>> generateHamiltonianAutomorphisms(
    const std::string& edgesFile, 
    const std::string& verticesFile, 
    int n_sites) {
    HamiltonianAutomorphismFinder finder(n_sites);
    finder.loadEdgesFromFile(edgesFile);
    finder.loadVerticesFromFile(verticesFile);
    return finder.findAllAutomorphisms();
}

/**
 * AutomorphismCliqueAnalyzer class for finding and visualizing compatible automorphisms
 */
class AutomorphismCliqueAnalyzer {
public:
    AutomorphismCliqueAnalyzer() {}
    
    // Check if two permutations commute
    bool doPermutationsCommute(const std::vector<int>& perm1, const std::vector<int>& perm2) const {
        if (perm1.size() != perm2.size()) {
            throw std::invalid_argument("Permutations must have the same size");
        }
        
        // Check if p1 ∘ p2 = p2 ∘ p1
        size_t n = perm1.size();
        for (size_t i = 0; i < n; ++i) {
            if (perm1[perm2[i]] != perm2[perm1[i]]) {
                return false;
            }
        }
        return true;
    }
    
    // Find the maximum clique using Bron-Kerbosch algorithm
    std::vector<int> findMaximumClique(const std::vector<std::vector<int>>& automorphisms) {
        // Build the graph
        int n = automorphisms.size();
        std::vector<std::vector<int>> graph(n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (doPermutationsCommute(automorphisms[i], automorphisms[j])) {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
            }
        }
        
        // Initialize variables for Bron-Kerbosch
        std::vector<int> maxClique;
        std::vector<int> currentClique;
        std::vector<int> candidates(n);
        std::vector<int> excluded;
        
        // Initialize candidates
        for (int i = 0; i < n; ++i) {
            candidates[i] = i;
        }
        
        // Run Bron-Kerbosch without pivoting
        bronKerbosch(graph, currentClique, candidates, excluded, maxClique);
        
        return maxClique;
    }
    
    // Generate a DOT file to visualize automorphism graph
    void generateAutomorphismGraph(
        const std::vector<std::vector<int>>& automorphisms, 
        const std::string& filename,
        const HamiltonianAutomorphismFinder& finder) {
        
        std::ofstream file(filename+"/automorphism_graph.dot");
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename+"/automorphism_graph.dot");
        }
        
        file << "graph AutomorphismGraph {\n";
        file << "  node [shape=box];\n";
        
        // Add nodes (automorphisms)
        for (size_t i = 0; i < automorphisms.size(); ++i) {
            file << "  A" << i << " [label=\"" << finder.permutationToCycleNotation(automorphisms[i]) << "\"];\n";
        }
        
        // Add edges (commuting relationships)
        for (size_t i = 0; i < automorphisms.size(); ++i) {
            for (size_t j = i + 1; j < automorphisms.size(); ++j) {
                if (doPermutationsCommute(automorphisms[i], automorphisms[j])) {
                    file << "  A" << i << " -- A" << j << ";\n";
                }
            }
        }
        
        file << "}\n";
        
        std::cout << "Generated automorphism graph: " << filename+"/automorphism_graph.dot" << std::endl;
        std::cout << "Visualize with: dot -Tpng " << filename+"/automorphism_graph.dot" << " -o " << filename+"/automorphism_graph.png" << std::endl;
    }
    
    // Highlight a clique in the graph
    void visualizeClique(
        const std::vector<std::vector<int>>& automorphisms,
        const std::vector<int>& clique,
        const std::string& filename,
        const HamiltonianAutomorphismFinder& finder) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        file << "graph AutomorphismClique {\n";
        file << "  node [shape=box];\n";
        
        // Create a set for fast lookup of clique members
        std::set<int> cliqueSet(clique.begin(), clique.end());
        
        // Add nodes (automorphisms)
        for (size_t i = 0; i < automorphisms.size(); ++i) {
            if (cliqueSet.find(i) != cliqueSet.end()) {
                // Node is part of the maximum clique
                file << "  A" << i << " [label=\"" << finder.permutationToCycleNotation(automorphisms[i]) 
                     << "\", style=filled, fillcolor=lightblue];\n";
            } else {
                file << "  A" << i << " [label=\"" << finder.permutationToCycleNotation(automorphisms[i]) << "\"];\n";
            }
        }
        
        // Add edges (commuting relationships)
        for (size_t i = 0; i < automorphisms.size(); ++i) {
            for (size_t j = i + 1; j < automorphisms.size(); ++j) {
                if (doPermutationsCommute(automorphisms[i], automorphisms[j])) {
                    if (cliqueSet.find(i) != cliqueSet.end() && cliqueSet.find(j) != cliqueSet.end()) {
                        // Edge is part of the maximum clique
                        file << "  A" << i << " -- A" << j << " [color=blue, penwidth=2];\n";
                    } else {
                        file << "  A" << i << " -- A" << j << ";\n";
                    }
                }
            }
        }
        
        file << "}\n";
        
        std::cout << "Generated clique visualization: " << filename << std::endl;
        std::cout << "Visualize with: dot -Tpng " << filename << " -o clique_visualization.png" << std::endl;
    }
    
    void saveGraphImage(const std::string& outputFile, const std::string& dotFile) {
        std::string command = "dot -Tpng " + dotFile + " -o " + outputFile;
        int result = system(command.c_str());
        
        if (result != 0) {
            std::cerr << "Failed to generate graph image. Make sure GraphViz is installed." << std::endl;
            return;
        }
        
        std::cout << "Generated graph image: " << outputFile << std::endl;
    }
    
private:
    // Bron-Kerbosch algorithm without pivoting
    void bronKerbosch(
        const std::vector<std::vector<int>>& graph,
        std::vector<int>& currentClique,
        std::vector<int> candidates,
        std::vector<int> excluded,
        std::vector<int>& maxClique) {
        
        if (candidates.empty() && excluded.empty()) {
            // Found a maximal clique
            if (currentClique.size() > maxClique.size()) {
                maxClique = currentClique;
            }
            return;
        }
        
        std::vector<int> candidates_copy = candidates;
        for (int v : candidates_copy) {
            // Add v to current clique
            currentClique.push_back(v);
            
            // Create new candidates and excluded sets
            std::vector<int> new_candidates;
            std::vector<int> new_excluded;
            
            // Intersect candidates with neighbors of v
            for (int u : candidates) {
                if (u != v && std::find(graph[v].begin(), graph[v].end(), u) != graph[v].end()) {
                    new_candidates.push_back(u);
                }
            }
            
            // Intersect excluded with neighbors of v
            for (int u : excluded) {
                if (std::find(graph[v].begin(), graph[v].end(), u) != graph[v].end()) {
                    new_excluded.push_back(u);
                }
            }
            
            // Recursive call
            bronKerbosch(graph, currentClique, new_candidates, new_excluded, maxClique);
            
            // Remove v from current clique
            currentClique.pop_back();
            
            // Move v from candidates to excluded
            candidates.erase(std::remove(candidates.begin(), candidates.end(), v), candidates.end());
            excluded.push_back(v);
        }
    }
};

/**
 * MinimalGeneratorFinder class to find the minimal set of generators for a group of automorphisms
 */
class MinimalGeneratorFinder {
public:
    // Compose two permutations: result(i) = perm1(perm2[i])
    std::vector<int> composePermutations(
        const std::vector<int>& perm1, 
        const std::vector<int>& perm2) {
        std::vector<int> result(perm1.size());
        for (size_t i = 0; i < perm1.size(); ++i) {
            result[i] = perm1[perm2[i]];
        }
        return result;
    }
    
    // Find the inverse of a permutation
    std::vector<int> inversePermutation(const std::vector<int>& perm) {
        std::vector<int> inverse(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            inverse[perm[i]] = i;
        }
        return inverse;
    }
    
    // Find the order of a permutation
    int findOrder(const std::vector<int>& perm) {
        // Create identity permutation
        std::vector<int> identity(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            identity[i] = i;
        }
        
        // Create a working copy of the permutation
        std::vector<int> current = perm;
        int order = 1;
        
        // Keep composing with itself until we get the identity
        while (current != identity) {
            current = composePermutations(current, perm);
            order++;
        }
        
        return order;
    }
    
    // Find minimal generators and their orders
    std::pair<std::vector<std::vector<int>>, std::vector<int>> findMinimalGenerators(
        const std::vector<std::vector<int>>& automorphisms) {
        if (automorphisms.empty()) {
            return {std::vector<std::vector<int>>(), std::vector<int>()};
        }
        
        // Create identity permutation
        std::vector<int> identity(automorphisms[0].size());
        for (size_t i = 0; i < identity.size(); ++i) {
            identity[i] = i;
        }
        
        // Make a copy and ensure the identity is included
        std::set<std::vector<int>> unique_autos(automorphisms.begin(), automorphisms.end());
        unique_autos.insert(identity);
        
        // Convert back to vector and sort for deterministic results
        std::vector<std::vector<int>> sorted_autos(unique_autos.begin(), unique_autos.end());
        std::sort(sorted_autos.begin(), sorted_autos.end());
        
        // Store the generators and their orders
        std::vector<std::vector<int>> generators;
        std::vector<int> orders;
        
        // Set of all elements in the generated subgroup
        std::set<std::vector<int>> generated_elements;
        generated_elements.insert(identity);
        
        // Try each automorphism as a potential generator
        for (const auto& automorphism : sorted_autos) {
            // Skip the identity
            if (automorphism == identity) continue;
            
            // Skip if we can already generate this automorphism
            if (generated_elements.find(automorphism) != generated_elements.end()) continue;
            
            // Add this automorphism as a generator
            generators.push_back(automorphism);
            int order = findOrder(automorphism);
            orders.push_back(order);
            
            // Generate all elements in the subgroup
            generateSubgroup(generators, generated_elements);
            
            // If we've generated the entire group, we're done
            if (generated_elements.size() == unique_autos.size()) {
                break;
            }
        }
        
        return {generators, orders};
    }
    
private:
    // Helper method to generate the subgroup from a set of generators
    void generateSubgroup(
        const std::vector<std::vector<int>>& generators,
        std::set<std::vector<int>>& generated_elements) {
        
        // Start with the identity
        std::vector<int> identity(generators[0].size());
        for (size_t i = 0; i < identity.size(); ++i) {
            identity[i] = i;
        }
        
        generated_elements.clear();
        generated_elements.insert(identity);
        
        // Keep adding new elements until no more can be added
        size_t old_size = 0;
        while (old_size < generated_elements.size()) {
            old_size = generated_elements.size();
            
            // Try composing each element with each generator
            std::vector<std::vector<int>> existing(generated_elements.begin(), generated_elements.end());
            for (const auto& elem : existing) {
                for (const auto& gen : generators) {
                    // Compose in both orders
                    std::vector<int> composed1 = composePermutations(elem, gen);
                    std::vector<int> composed2 = composePermutations(gen, elem);
                    
                    generated_elements.insert(composed1);
                    generated_elements.insert(composed2);
                }
            }
        }
    }
};


/**
 * Represents automorphisms as powers of generators
 */
class AutomorphismPowerRepresentation {
public:
    // Represent an automorphism as powers of generators
    static std::vector<int> representAsGeneratorPowers(
        const std::vector<std::vector<int>>& generators,
        const std::vector<int>& automorphism,
        int maxPower = 5) {
        
        if (generators.empty()) {
            return std::vector<int>();
        }
        
        int numGenerators = generators.size();
        int permSize = automorphism.size();
        
        // If the automorphism is the identity, return all zeros
        bool isIdentity = true;
        for (size_t i = 0; i < permSize; i++) {
            if (automorphism[i] != i) {
                isIdentity = false;
                break;
            }
        }
        
        if (isIdentity) {
            return std::vector<int>(numGenerators, 0);
        }
        
        // Using BFS to find the representation
        struct State {
            std::vector<int> powers;
            std::vector<int> currentPerm;
        };
        
        std::queue<State> queue;
        std::set<std::vector<int>> visited;
        
        // Start with identity permutation
        std::vector<int> identity(permSize);
        for (size_t i = 0; i < permSize; i++) {
            identity[i] = i;
        }
        
        State initialState;
        initialState.powers = std::vector<int>(numGenerators, 0);
        initialState.currentPerm = identity;
        queue.push(initialState);
        visited.insert(identity);
        
        while (!queue.empty()) {
            State current = queue.front();
            queue.pop();
            
            // Try applying each generator
            for (int i = 0; i < numGenerators; i++) {
                // Try both the generator and its inverse
                for (int powerDelta : {1, -1}) {
                    State next = current;
                    next.powers[i] += powerDelta;
                    
                    // if (std::abs(next.powers[i]) > maxPower) {
                    //     continue;
                    // }
                    
                    // Apply the generator or its inverse
                    std::vector<int> genToApply = powerDelta == 1 ? 
                        generators[i] : MinimalGeneratorFinder().inversePermutation(generators[i]);
                    
                    next.currentPerm = MinimalGeneratorFinder().composePermutations(genToApply, next.currentPerm);
                    
                    if (next.currentPerm == automorphism) {
                        return next.powers;
                    }
                    
                    if (visited.find(next.currentPerm) == visited.end()) {
                        visited.insert(next.currentPerm);
                        queue.push(next);
                    }
                }
            }
        }
        
        return std::vector<int>();
    }

    // Represent all automorphisms as powers of generators
    static std::vector<std::vector<int>> representAllAsGeneratorPowers(
        const std::vector<std::vector<int>>& generators,
        const std::vector<std::vector<int>>& automorphisms,
        int maxPower = 5) {
        
        std::vector<std::vector<int>> results;
        results.reserve(automorphisms.size());
        
        for (const auto& automorphism : automorphisms) {
            std::vector<int> powers = representAsGeneratorPowers(generators, automorphism, maxPower);
            results.push_back(powers);
        }
        
        return results;
    }
};


int applyPermutation(int basis, const std::vector<int>& perm) {
    int result = 0;
    for (size_t i = 0; i < perm.size(); ++i) {
        result |= ((basis >> perm[i]) & 1) << i;
    }
    return result;
}


/**
 * Operator class that can represent arbitrary quantum operators
 * through bit flip operations and scalar multiplications
 */
class Operator {
public:
    // Function type for transforming basis states
    using TransformFunction = std::function<std::pair<int, Complex>(int)>;

    std::vector<int> symmetrized_block_ham_sizes;
    // Constructor
    
    Operator(int n_bits) : n_bits_(n_bits) {}

    // Copy assignment operator
    Operator& operator=(const Operator& other) {
        if (this != &other) {  // Check for self-assignment
            n_bits_ = other.n_bits_;
            transforms_ = other.transforms_;
            sparseMatrix_ = other.sparseMatrix_;
            matrixBuilt_ = other.matrixBuilt_;
            symmetrized_block_ham_sizes = other.symmetrized_block_ham_sizes;
        }
        return *this;
    }

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
    Matrix returnSymmetrizedMatrix(const std::string& dir){
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        for (int i = 0; i < dim; ++i) {
            std::vector<Complex> temp_vec_i = read_sym_basis(i, dir);
            for (int j = 0; j < dim; ++j) {
                std::vector<Complex> temp_vec_j = read_sym_basis(j, dir);
                // Apply the operator to the i-th basis vector
                std::vector<Complex> temp_vec_i_F = apply(temp_vec_i);
                Complex res;
                for (int k = 0; k < dim; ++k) {
                    res += temp_vec_i_F[k] * std::conj(temp_vec_j[k]);
                }
                matrix[i][j] = res;
            }
        }
        return matrix;
    }

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
    
    void generateSymmetrizedBasis(const std::string& dir) {
        std::ifstream trans_file(dir + "/Trans.dat");
        if (!trans_file.is_open()) {
            std::cerr << "Error: Cannot open file " << dir + "/Trans.dat" << std::endl;
        }

        // Skip the first line
        std::string dummy_line;
        std::getline(trans_file, dummy_line);

        // Read the second line to get num_site
        std::string dum;
        int num_site;
        trans_file >> dum >> num_site;
        trans_file.close();
        const std::string interactions_file = dir + "/InterAll.dat";
        const std::string site_ops_file = dir + "/Trans.dat";
        const std::string dot_file = dir + "/hamiltonian.dot";
        const std::string png_file = dir + "/hamiltonian.png";
        
        
        
        // Create HamiltonianVisualizer instance
        std::cout << "Initializing HamiltonianVisualizer...\n";
        HamiltonianVisualizer visualizer(num_site);
        
        // Load edges and vertices
        std::cout << "Loading interactions and site operators...\n";
        visualizer.loadEdgesFromFile(interactions_file);
        visualizer.loadVerticesFromFile(site_ops_file);
        
        // Generate DOT file
        std::cout << "Generating DOT file...\n";
        visualizer.generateDotFile(dot_file);
        
        // Generate PNG image
        std::cout << "Generating PNG image...\n";
        visualizer.saveGraphImage(png_file, dot_file);
        
        std::cout << "Visualization complete.\n";
        std::cout << "DOT file: " << dot_file << std::endl;
        std::cout << "PNG file: " << png_file << std::endl;

        HamiltonianAutomorphismFinder finder(num_site);
        finder.loadEdgesFromFile(interactions_file);
        finder.loadVerticesFromFile(site_ops_file);

        std::vector<std::vector<int>> automorphism_groups = finder.findAllAutomorphisms();
        std::cout << "Found " << automorphism_groups.size() << " automorphisms.\n";

        AutomorphismCliqueAnalyzer analyzer;
        auto cliques = analyzer.findMaximumClique(automorphism_groups);
        std::vector<std::vector<int>> max_clique_here;
        for (const auto& clique : cliques) {
            max_clique_here.push_back(automorphism_groups[clique]);
        }
        std::cout << "Maximum clique size: " << max_clique_here.size() << std::endl;
        // std::cout << "Maximum clique:\n";
        // for (const auto& clique : max_clique_here) {
        //     std::cout << "Clique: ";
        //     for (int index : clique) {
        //         std::cout << index << " ";
        //     }
        //     std::cout << std::endl;
        // }

        analyzer.generateAutomorphismGraph(automorphism_groups, dir, finder);

        
        MinimalGeneratorFinder minimal_finder;
        std::pair<std::vector<std::vector<int>>, std::vector<int>> minimal_generators = minimal_finder.findMinimalGenerators(max_clique_here);
        
        std::cout << "Minimal generators:\n";
        int count_temp = 0;
        for (const auto& generator : minimal_generators.first) {
            std::cout << "Generator: ";
            for (int index : generator) {
                std::cout << index << " ";
            }
            std::cout << "with order: ";
            std::cout << minimal_generators.second[count_temp] << " ";
            count_temp++;
            std::cout << std::endl;
        }

        AutomorphismPowerRepresentation automorphism_power_representation;
        std::vector<std::vector<int>> power_representation = automorphism_power_representation.representAllAsGeneratorPowers(minimal_generators.first, max_clique_here);
        std::cout << "Power representation generated.\n";
        // std::cout << "Power representation:\n";
        // for (const auto& powers : power_representation) {
        //     for (int power : powers) {
        //         std::cout << power << " ";
        //     }
        //     std::cout << std::endl;
        // }
        std::cout << "Symmetrized Hamiltonain generated.\n";
        
        // Generate all possible combinations of quantum numbers
        std::vector<std::vector<int>> all_quantum_numbers;
        std::vector<int> current_qnums(minimal_generators.first.size(), 0);

        std::function<void(size_t)> generate_quantum_numbers = [&](size_t position) {
            if (position == minimal_generators.first.size()) {
                all_quantum_numbers.push_back(current_qnums);
                return;
            }
            
            for (int i = 0; i < minimal_generators.second[position]; i++) {
                current_qnums[position] = i;
                generate_quantum_numbers(position + 1);
            }
        };

        generate_quantum_numbers(0);

        std::cout << "Total symmetry sectors: " << all_quantum_numbers.size() << std::endl;

        // Create a directory for the symmetrized basis states
        std::string sym_basis_dir = dir + "/sym_basis";
        std::string mkdir_command = "mkdir -p " + sym_basis_dir;
        system(mkdir_command.c_str());
        std::vector<std::vector<Complex>> unique_sym_basis;

        // Create a filename based on the quantum numbers
        std::string filename = sym_basis_dir + "/sym_basis";

        symmetrized_block_ham_sizes.resize(all_quantum_numbers.size(), 0);
        int count = 0;
        // For each symmetry sector (combination of quantum numbers)
        for (const auto& e_i : all_quantum_numbers) {            
            

            // Total number of basis states in the Hilbert space
            size_t total_basis_states = 1 << n_bits_;
            
            // For each standard basis state
            for (size_t basis = 0; basis < total_basis_states; basis++) {
                // Generate the symmetrized basis vector for this state
                std::vector<Complex> sym_basis_vec = sym_basis_e_(basis, max_clique_here, power_representation, minimal_generators.second, e_i);
                // Check if this symmetrized basis vector is zero (can happen in some symmetry sectors)
                double norm_squared = 0.0;
                for (const auto& val : sym_basis_vec) {
                    norm_squared += std::norm(val);
                }
                
                if (norm_squared < 1e-10) {
                    continue; // Skip zero vectors
                }
                
                // Check if this symmetrized basis vector is already in our collection
                bool is_unique = true;
                for (const auto& existing_vec : unique_sym_basis) {
                    // Calculate overlap between the vectors
                    Complex overlap(0.0, 0.0);
                    for (size_t i = 0; i < total_basis_states; i++) {
                        overlap += std::conj(existing_vec[i]) * sym_basis_vec[i];
                    }
                    
                    // If the absolute value of the overlap is close to 1, the vectors
                    // are the same up to a global phase factor exp(i*θ)
                    if (std::abs(std::abs(overlap) - 1.0) < 1e-10) {
                        is_unique = false;
                        break;
                    }
                }
                
                // If the symmetrized basis vector is unique, add it to our collection
                if (is_unique) {
                    unique_sym_basis.push_back(sym_basis_vec);
                    symmetrized_block_ham_sizes[count]++;
                }
            }
            count++;
        }
        
        // Write the number of unique basis vectors
        std::cout << "Number of unique symmetrized basis vectors: " << unique_sym_basis.size() << std::endl;
        std::cout << "Block sizes: ";
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); i++) {
            std::cout << symmetrized_block_ham_sizes[i] << " ";
        }
        std::cout << std::endl;

        // Write each symmetrized basis vector
        for (size_t i = 0; i < unique_sym_basis.size(); i++) {    
            std::ofstream output_file(filename+std::to_string(i) + ".dat");        
            for (size_t j = 0; j < (1<<num_site); j++) {
                if (std::abs(unique_sym_basis[i][j]) > 1e-8) {
                    output_file << j << " " << unique_sym_basis[i][j].real() << " " << unique_sym_basis[i][j].imag() << std::endl;
                }
            }
            output_file.close();
        }
    }

    std::vector<Complex> read_sym_basis(int index, const std::string& dir){
        std::ifstream file(dir+"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + dir +"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        }
        std::vector<Complex> sym_basis((1<<n_bits_), 0.0);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int index;
            double real, imag;
            if (iss >> index >> real >> imag) {
                sym_basis[index] = Complex(real, imag);
            }
        }
        return sym_basis;
    }

    std::vector<Complex> sym_basis_e_(int basis, std::vector<std::vector<int>> max_clique, 
                                    std::vector<std::vector<int>> power_representation, 
                                    std::vector<int> minimal_generator_orders, 
                                    std::vector<int> e_i) {
        
        std::vector<Complex> sym_basis = std::vector<Complex>(1 << n_bits_, 0.0);
        
        // Recursive function to apply all combinations of generator powers
        std::function<void(int, int, const std::vector<std::vector<int>>&, Complex)> apply_generators = 
            [&](int current_gen, int current_basis, const std::vector<std::vector<int>>& generators, Complex current_phase) {
                
                if (current_gen == generators.size()) {
                    // We've applied all generators, add to the symmetrized basis
                    sym_basis[current_basis] += current_phase;
                    return;
                }
                
                // Apply all powers of the current generator
                for (int power = 0; power < minimal_generator_orders[current_gen]; power++) {
                    // Calculate the phase factor for this power
                    Complex phase_factor = std::exp(Complex(0.0, 2*M_PI*double(power)*double(e_i[current_gen])/
                                                  double(minimal_generator_orders[current_gen])));
                    
                    // Apply the generator 'power' times
                    int transformed_basis = current_basis;
                    for (int p = 0; p < power; p++) {
                        transformed_basis = applyPermutation(transformed_basis, generators[current_gen]);
                    }
                    
                    // Recursively apply the next generator
                    apply_generators(current_gen + 1, transformed_basis, generators, current_phase * phase_factor);
                }
            };
        
        // Start the recursive application with the original basis state
        apply_generators(0, basis, power_representation.empty() ? max_clique : power_representation, Complex(1.0, 0.0));
        
        // Normalize the sym_basis
        double norm = 0.0;
        for (const auto& val : sym_basis) {
            norm += std::norm(val);
        }
        if (norm < 1e-8) {
            return sym_basis; // Return unnormalized basis if essentially zero
        }
        norm = std::sqrt(norm);
        for (auto& val : sym_basis) {
            val /= norm;
        }
        return sym_basis;
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

    // Build symmetrized sparse matrices for each block and save them to files
    void buildAndSaveSymmetrizedBlocks(const std::string& dir) {
        if (symmetrized_block_ham_sizes.empty()) {
            throw std::runtime_error("Symmetrized basis must be generated first with generateSymmetrizedBasis()");
        }
        
        std::cout << "Building and saving symmetrized Hamiltonian blocks..." << std::endl;
        
        // Create a directory for the block matrices
        std::string block_dir = dir + "/sym_blocks";
        std::string mkdir_command = "mkdir -p " + block_dir;
        system(mkdir_command.c_str());
        
        int dim = 1 << n_bits_;
        int block_start = 0;
        
        // Process each symmetry block
        for (size_t block = 0; block < symmetrized_block_ham_sizes.size(); block++) {
            int block_size = symmetrized_block_ham_sizes[block];
            if (block_size == 0) continue;
            
            std::cout << "Processing block " << block << " of size " << block_size << std::endl;
            
            // Create a sparse matrix for this block
            Eigen::SparseMatrix<Complex> blockMatrix(block_size, block_size);
            std::vector<Eigen::Triplet<Complex>> triplets;
            
            // Build the block matrix
            for (int i = 0; i < block_size; i++) {
                std::vector<Complex> basis_i = read_sym_basis(block_start + i, dir);
                std::vector<Complex> transformed_i = apply(basis_i);
                
                for (int j = 0; j < block_size; j++) {
                    std::vector<Complex> basis_j = read_sym_basis(block_start + j, dir);
                    
                    // Calculate matrix element <j|H|i>
                    Complex element(0.0, 0.0);
                    for (int k = 0; k < dim; k++) {
                        element += std::conj(basis_j[k]) * transformed_i[k];
                    }
                    
                    // Add non-zero elements to the sparse matrix
                    if (std::abs(element) > 1e-10) {
                        triplets.emplace_back(j, i, element);
                    }
                }
            }
            
            blockMatrix.setFromTriplets(triplets.begin(), triplets.end());
            
            // Save the block matrix to a file
            std::string filename = block_dir + "/block_" + std::to_string(block) + ".dat";
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }
            
            // Write matrix dimensions
            file << block_size << " " << block_size << std::endl;
            
            // Write non-zero elements (row, col, real, imag)
            for (int k = 0; k < blockMatrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Complex>::InnerIterator it(blockMatrix, k); it; ++it) {
                    file << it.row() << " " << it.col() << " " 
                         << it.value().real() << " " << it.value().imag() << std::endl;
                }
            }
            
            file.close();
            std::cout << "Saved block " << block << " to " << filename << std::endl;
            
            block_start += block_size;
        }
        
        std::cout << "Symmetrized Hamiltonian blocks saved to " << block_dir << std::endl;
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
 * SingleSiteOperator class represents a Pauli operator (X, Y, or Z) acting on a single site
 */
class SingleSiteOperator : public Operator {
public:
    /**
     * Constructor for a single site operator
     * @param num_site Total number of sites/qubits
     * @param op Operator type: 0 for X, 1 for Y, 2 for Z
     * @param site_j Site index to apply the operator to
     */
    SingleSiteOperator(int num_site, int op, int site_j) : Operator(num_site) {
        if (op < 0 || op > 2) {
            throw std::invalid_argument("Invalid operator type. Use 0 for X, 1 for Y, 2 for Z");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Site index out of range");
        }
        
        addTransform([=](int basis) -> std::pair<int, Complex> {
        // Check if all bits match their expected values
            if (op == 2){
                return {basis, 0.5*pow(-1,(basis >> site_j) & 1)};
            }
            else{
                if (((basis >> site_j) & 1) != op) {
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(1.0, 0)};                    
                }
            }
        // Default case: no transformation applies
        return {basis, Complex(0.0, 0.0)};
        });
    }
};


#endif // CONSTRUCT_HAM_H