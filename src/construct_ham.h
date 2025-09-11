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
    
    Operator(int n_bits, float spin_l) : n_bits_(n_bits), spin_l_(spin_l) {}

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

    // Apply the operator to a raw pointer using Eigen sparse matrix operations
    void apply(const Complex* in, Complex* out, size_t size) const {
        int dim = 1 << n_bits_;
        
        if (size != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input/output vector size does not match operator dimension");
        }
        
        // Build the sparse matrix if not already built
        buildSparseMatrix();
        
        // Map the raw pointers to Eigen vectors to avoid copying
        Eigen::Map<const Eigen::VectorXcd> eigenIn(in, dim);
        Eigen::Map<Eigen::VectorXcd> eigenOut(out, dim);
        
        // Perform sparse matrix-vector multiplication
        eigenOut = sparseMatrix_ * eigenIn;
    }

    // Print the operator as a matrix - optimized for memory
    Matrix returnSymmetrizedMatrix(const std::string& dir){
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        
        // Process one basis vector at a time to reduce memory footprint
        std::vector<Complex> temp_vec_i;
        std::vector<Complex> temp_vec_j;
        std::vector<Complex> temp_vec_i_F;
        
        for (int i = 0; i < dim; ++i) {
            temp_vec_i = read_sym_basis(i, dir);
            temp_vec_i_F = apply(temp_vec_i);
            
            for (int j = 0; j < dim; ++j) {
                temp_vec_j = read_sym_basis(j, dir);
                
                Complex res(0.0, 0.0);
                for (int k = 0; k < dim; ++k) {
                    res += temp_vec_i_F[k] * std::conj(temp_vec_j[k]);
                }
                matrix[i][j] = res;
                
                // Clear temp_vec_j after use
                temp_vec_j.clear();
                temp_vec_j.shrink_to_fit();
            }
            
            // Clear vectors after processing each row
            temp_vec_i.clear();
            temp_vec_i.shrink_to_fit();
            temp_vec_i_F.clear();
            temp_vec_i_F.shrink_to_fit();
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
        // Step 1: Load automorphism data
        auto max_clique = loadMaxClique(dir);
        
        // Step 2: Find minimal generators
        auto [generators, orders] = findGenerators(max_clique);
        
        // Step 3: Get power representation
        auto power_representation = computePowerRepresentation(generators, max_clique);
        
        // Step 4: Generate all quantum number combinations
        auto all_quantum_numbers = enumerateQuantumNumbers(generators, orders);
        
        // Step 5: Setup output directory
        prepareOutputDirectory(dir);
        
        // Step 6: Generate basis vectors for each symmetry sector
        generateBasisVectors(dir, max_clique, power_representation, orders, all_quantum_numbers);
        
        // Step 7: Save results
        saveBlockSizes(dir);
        
        std::cout << "Symmetrized basis generation complete." << std::endl;
    }

private:
    std::vector<std::vector<int>> loadMaxClique(const std::string& dir) {
        const std::string filepath = dir + "/automorphism_results/max_clique.json";
        std::cout << "Loading max clique from: " << filepath << std::endl;
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open max_clique.json file: " + filepath);
        }
        
        std::string json_content((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        file.close();
        
        auto max_clique = parseJsonIntArrays(json_content);
        if (max_clique.empty()) {
            throw std::runtime_error("Invalid or empty max_clique.json");
        }
        
        std::cout << "Loaded " << max_clique.size() << " permutations in max clique" << std::endl;
        return max_clique;
    }
    
    std::vector<std::vector<int>> parseJsonIntArrays(const std::string& json) {
        std::vector<std::vector<int>> result;
        size_t pos = json.find('[');
        if (pos == std::string::npos) return result;
        ++pos;
        
        while (pos < json.size()) {
            // Skip whitespace
            while (pos < json.size() && std::isspace(json[pos])) ++pos;
            if (pos >= json.size() || json[pos] == ']') break;
            if (json[pos] != '[') { ++pos; continue; }
            
            // Parse inner array
            ++pos;
            std::vector<int> array;
            std::string number;
            
            while (pos < json.size() && json[pos] != ']') {
                char c = json[pos++];
                if (std::isdigit(c)) {
                    number += c;
                } else if (!number.empty() && (c == ',' || std::isspace(c))) {
                    array.push_back(std::stoi(number));
                    number.clear();
                }
            }
            
            if (!number.empty()) array.push_back(std::stoi(number));
            if (pos < json.size() && json[pos] == ']') ++pos;
            if (!array.empty()) result.push_back(std::move(array));
        }
        
        return result;
    }
    
    std::pair<std::vector<std::vector<int>>, std::vector<int>> 
    findGenerators(const std::vector<std::vector<int>>& max_clique) {
        MinimalGeneratorFinder gen_finder;
        auto [generators, orders] = gen_finder.findMinimalGenerators(max_clique);
        
        std::cout << "Found " << generators.size() << " minimal generators:\n";
        for (size_t i = 0; i < generators.size(); ++i) {
            std::cout << "  Generator " << i << " (order " << orders[i] << "): ";
            for (int v : generators[i]) std::cout << v << " ";
            std::cout << "\n";
        }
        
        return {generators, orders};
    }
    
    std::vector<std::vector<int>> 
    computePowerRepresentation(const std::vector<std::vector<int>>& generators,
                               const std::vector<std::vector<int>>& max_clique) {
        auto power_repr = AutomorphismPowerRepresentation::representAllAsGeneratorPowers(
            generators, max_clique);
        std::cout << "Power representation computed for all automorphisms.\n";
        return power_repr;
    }
    
    std::vector<std::vector<int>> 
    enumerateQuantumNumbers(const std::vector<std::vector<int>>& generators,
                            const std::vector<int>& orders) {
        std::vector<std::vector<int>> quantum_numbers;
        std::vector<int> current(generators.size(), 0);
        
        std::function<void(size_t)> enumerate = [&](size_t pos) {
            if (pos == generators.size()) {
                quantum_numbers.push_back(current);
                return;
            }
            for (int k = 0; k < orders[pos]; ++k) {
                current[pos] = k;
                enumerate(pos + 1);
            }
        };
        
        enumerate(0);
        std::cout << "Total symmetry sectors: " << quantum_numbers.size() << std::endl;
        return quantum_numbers;
    }
    
    void prepareOutputDirectory(const std::string& dir) {
        const std::string sym_basis_dir = dir + "/sym_basis";
        std::string mkdir_cmd = "mkdir -p " + sym_basis_dir;
        system(mkdir_cmd.c_str());
    }
    
    void generateBasisVectors(const std::string& dir,
                             const std::vector<std::vector<int>>& max_clique,
                             const std::vector<std::vector<int>>& power_representation,
                             const std::vector<int>& orders,
                             const std::vector<std::vector<int>>& all_quantum_numbers) {
        const size_t dim = 1ULL << n_bits_;
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(all_quantum_numbers.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < all_quantum_numbers.size(); ++sector_idx) {
            size_t sector_size = processSector(dir, sector_idx, all_quantum_numbers[sector_idx],
                                              max_clique, power_representation, orders,
                                              dim, total_written);
            symmetrized_block_ham_sizes[sector_idx] = sector_size;
            
            std::cout << "Sector " << (sector_idx + 1) << "/" << all_quantum_numbers.size() 
                     << ": " << sector_size << " basis vectors\n";
        }
        
        std::cout << "\nTotal unique symmetrized basis vectors: " << total_written << std::endl;
    }
    
    size_t processSector(const std::string& dir, size_t sector_idx,
                        const std::vector<int>& quantum_nums,
                        const std::vector<std::vector<int>>& max_clique,
                        const std::vector<std::vector<int>>& power_representation,
                        const std::vector<int>& orders,
                        size_t dim, size_t& total_written) {
        std::set<size_t> processed_states;
        std::set<std::string> unique_vectors;
        size_t sector_basis_count = 0;
        
        for (size_t basis = 0; basis < dim; ++basis) {
            if (basis % std::max<size_t>(1, dim / 100) == 0) {
                showProgress(basis, dim, sector_idx);
            }
            
            if (processed_states.count(basis)) continue;
            
            // Mark all states in the orbit as processed
            auto orbit = computeOrbit(basis, max_clique);
            processed_states.insert(orbit.begin(), orbit.end());
            
            // Generate and validate symmetrized vector
            auto vec = createSymmetrizedVector(basis, max_clique, power_representation, 
                                              orders, quantum_nums);
            if (!isValidVector(vec)) continue;
            
            // Normalize phase and check uniqueness
            normalizePhase(vec);
            std::string hash = computeVectorHash(vec);
            if (unique_vectors.count(hash)) continue;
            unique_vectors.insert(hash);
            
            // Save to file
            saveVector(dir, vec, total_written);
            ++sector_basis_count;
            ++total_written;
        }
        
        std::cout << std::endl;
        return sector_basis_count;
    }
    
    std::set<size_t> computeOrbit(size_t basis, const std::vector<std::vector<int>>& max_clique) {
        std::set<size_t> orbit;
        for (const auto& perm : max_clique) {
            orbit.insert(applyPermutation(static_cast<int>(basis), perm));
        }
        return orbit;
    }
    
    std::vector<Complex> createSymmetrizedVector(int basis,
                                                 const std::vector<std::vector<int>>& max_clique,
                                                 const std::vector<std::vector<int>>& power_repr,
                                                 const std::vector<int>& orders,
                                                 const std::vector<int>& quantum_nums) {
        return sym_basis_e_(basis, max_clique, power_repr, orders, quantum_nums);
    }
    
    bool isValidVector(const std::vector<Complex>& vec) {
        double norm_squared = 0.0;
        for (const auto& v : vec) {
            norm_squared += std::norm(v);
        }
        return norm_squared > 1e-10;
    }
    
    void normalizePhase(std::vector<Complex>& vec) {
        // Find first non-zero entry
        size_t first_nonzero = 0;
        while (first_nonzero < vec.size() && std::abs(vec[first_nonzero]) < 1e-10) {
            ++first_nonzero;
        }
        
        if (first_nonzero >= vec.size()) return;
        
        // Normalize by phase of first non-zero entry
        Complex phase = vec[first_nonzero] / std::abs(vec[first_nonzero]);
        for (auto& v : vec) {
            v /= phase;
        }
    }
    
    std::string computeVectorHash(const std::vector<Complex>& vec) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(12);
        
        for (size_t i = 0; i < vec.size(); ++i) {
            if (std::abs(vec[i]) > 1e-10) {
                ss << i << ":" << vec[i].real() << ":" << vec[i].imag() << ";";
            }
        }
        
        return ss.str();
    }
    
    void saveVector(const std::string& dir, const std::vector<Complex>& vec, size_t index) {
        const std::string filepath = dir + "/sym_basis/sym_basis" + std::to_string(index) + ".dat";
        std::ofstream file(filepath);
        
        for (size_t i = 0; i < vec.size(); ++i) {
            if (std::abs(vec[i]) > 1e-10) {
                file << i << " " << vec[i].real() << " " << vec[i].imag() << "\n";
            }
        }
    }
    
    void showProgress(size_t current, size_t total, size_t sector_idx) {
        double percent = (static_cast<double>(current) / total) * 100.0;
        int bar_width = 40;
        int pos = static_cast<int>(bar_width * percent / 100.0);
        
        std::cout << "\rSector " << (sector_idx + 1) << " [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << percent << "%" << std::flush;
    }
    
    void saveBlockSizes(const std::string& dir) {
        std::cout << "Block sizes: ";
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            std::cout << symmetrized_block_ham_sizes[i];
            if (i < symmetrized_block_ham_sizes.size() - 1) std::cout << " ";
        }
        std::cout << std::endl;
        
        std::ofstream file(dir + "/sym_basis/sym_block_sizes.txt");
        if (!file.is_open()) {
            throw std::runtime_error("Could not save block sizes file");
        }
        
        for (int size : symmetrized_block_ham_sizes) {
            file << size << "\n";
        }
    }

public:
    std::vector<Complex> read_sym_basis(int index, const std::string& dir){
        std::ifstream file(dir+"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + dir +"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        }
        std::vector<Complex> sym_basis((1ULL<<n_bits_), 0.0);
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

    // Optimized read function for sparse basis vectors
    void read_sym_basis_sparse(int index, const std::string& dir, 
                              std::map<int, Complex>& sparse_vec) {
        sparse_vec.clear();
        std::ifstream file(dir+"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + dir +"/sym_basis/sym_basis"+std::to_string(index)+".dat");
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int idx;
            double real, imag;
            if (iss >> idx >> real >> imag) {
                Complex val(real, imag);
                if (std::abs(val) > 1e-10) {
                    sparse_vec[idx] = val;
                }
            }
        }
    }

    std::vector<Complex> sym_basis_e_(int basis, std::vector<std::vector<int>> max_clique, std::vector<std::vector<int>> power_representation, std::vector<int> minimal_generators, std::vector<int> e_i){
        
        std::vector<Complex> sym_basis(1 << n_bits_, 0.0);

        // Pre-compute phase factors for all permutations
        std::vector<Complex> phase_factors(max_clique.size(), Complex(1.0, 0.0));
        for(size_t i=0; i<max_clique.size(); i++) {
            double phase = 0.0;
            for(size_t j=0; j<power_representation[i].size(); j++) {
                phase += 2.0*M_PI*power_representation[i][j]*e_i[j]/minimal_generators[j];
            }
            phase_factors[i] = Complex(std::cos(phase), std::sin(phase));
        }

        // Apply permutations and add with phase factors
        for(size_t i=0; i<max_clique.size(); i++) {
            int permuted_state = applyPermutation(basis, max_clique[i]);
            sym_basis[permuted_state] += phase_factors[i];
        }

        // Normalize the sym_basis
        double norm = 0.0;
        for (const auto& val : sym_basis) {
            norm += std::norm(val);
        }
        if (norm < 1e-8) {
            return sym_basis; // Return unnormalized basis
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
                    return {basis, Complex(E,F)*double(spin_l_)*pow(-1,(basis >> indx) & 1)};
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
                    return {basis, Complex(E, F)* double(spin_l_) * double(spin_l_) * pow(-1, bit1) * pow(-1, bit2)};
                } 
                else if (Op1 == 2) {
                    // Op1 is identity with phase, Op2 is bit flip
                    int bit1 = (basis >> indx1) & 1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit2_matches) {
                        int flipped_basis = basis ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F) * double(spin_l_) * pow(-1, bit1)};
                    }
                } 
                else if (Op2 == 2) {
                    // Op2 is identity with phase, Op1 is bit flip
                    int bit2 = (basis >> indx2) & 1;
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    
                    if (bit1_matches) {
                        // Flip the first bit
                        int flipped_basis = basis ^ (1 << indx1);
                        return {flipped_basis, Complex(E, F)* double(spin_l_) * pow(-1, bit2)};
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

    void loadonebodycorrelation(const int Op, const int indx) { 
        addTransform([=](int basis) -> std::pair<int, Complex> {
            // Check if all bits match their expected values
            if (Op == 2){
                return {basis, Complex(1.0,0.0)*double(spin_l_)*pow(-1,(basis >> indx) & 1)};
            }
            else{
                if (((basis >> indx) & 1) != Op) {
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << indx);
                    return {flipped_basis, Complex(1.0 * double(spin_l_*2), 0.0)};                    
                }
            }
            // Default case: no transformation applies
            return {basis, Complex(0.0, 0.0)};
        });
    }

    void loadtwobodycorrelation(const int Op1, const int indx1, const int Op2, const int indx2){
        addTransform([=](int basis) -> std::pair<int, Complex> {
            // Check what type of operators we're dealing with
            if (Op1 == 2 && Op2 == 2) {
                // Both are identity operators with phase factors
                int bit1 = (basis >> indx1) & 1;
                int bit2 = (basis >> indx2) & 1;
                return {basis, Complex(1.0, 0.0)* double(spin_l_) * double(spin_l_) * pow(-1, bit1) * pow(-1, bit2)};
            } 
            else if (Op1 == 2) {
                // Op1 is identity with phase, Op2 is bit flip
                int bit1 = (basis >> indx1) & 1;
                bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                
                if (bit2_matches) {
                    int flipped_basis = basis ^ (1 << indx2);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_) * double(spin_l_*2) * pow(-1, bit1)};
                }
            } 
            else if (Op2 == 2) {
                // Op2 is identity with phase, Op1 is bit flip
                int bit2 = (basis >> indx2) & 1;
                bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                
                if (bit1_matches) {
                    // Flip the first bit
                    int flipped_basis = basis ^ (1 << indx1);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_) * double(spin_l_*2) * pow(-1, bit2)};
                }
            } 
            else {
                // Both are bit flip operators
                bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                
                if (bit1_matches && bit2_matches) {
                    // Flip both bits
                    int flipped_basis = basis ^ (1 << indx1) ^ (1 << indx2);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_*2) * double(spin_l_*2)};
                }
            }
            // Default case: no transformation applies
            return {basis, Complex(0.0, 0.0)};
        });
    }
    // Build symmetrized sparse matrices for each block and save them to files
    void buildAndSaveSymmetrizedBlocks(const std::string& dir) {
        // Step 1: Load block sizes if not already loaded
        loadBlockSizesIfNeeded(dir);
        
        std::cout << "Building and saving symmetrized Hamiltonian blocks..." << std::endl;
        
        // Step 2: Create output directory
        std::string block_dir = createBlockDirectory(dir);
        
        // Step 3: Pre-build the sparse matrix representation once
        buildSparseMatrix();
        
        // Step 4: Process each symmetry block
        processAllSymmetryBlocks(dir, block_dir);
        
        std::cout << "Symmetrized Hamiltonian blocks saved to " << block_dir << std::endl;
    }

private:
    void loadBlockSizesIfNeeded(const std::string& dir) {
        if (!symmetrized_block_ham_sizes.empty()) {
            return;
        }
        
        std::string filepath = dir + "/sym_basis/sym_block_sizes.txt";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error(
                "Symmetrized basis must be generated first with generateSymmetrizedBasis()"
            );
        }
        
        symmetrized_block_ham_sizes.clear();
        int block_size;
        while (file >> block_size) {
            symmetrized_block_ham_sizes.push_back(block_size);
        }
        
        std::cout << "Loaded " << symmetrized_block_ham_sizes.size() 
                  << " symmetrized block sizes." << std::endl;
    }
    
    std::string createBlockDirectory(const std::string& dir) {
        std::string block_dir = dir + "/sym_blocks";
        std::string mkdir_command = "mkdir -p " + block_dir;
        system(mkdir_command.c_str());
        return block_dir;
    }
    
    void processAllSymmetryBlocks(const std::string& dir, const std::string& block_dir) {
        int block_start = 0;
        
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); block_idx++) {
            int block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                continue;
            }
            
            std::cout << "Processing block " << block_idx 
                     << " (size: " << block_size << ")" << std::endl;
            
            processSingleBlockOptimized(dir, block_dir, block_idx, block_start, block_size);
            block_start += block_size;
        }
    }
    
    void processSingleBlockOptimized(const std::string& dir, 
                                    const std::string& block_dir,
                                    size_t block_idx,
                                    int block_start, 
                                    int block_size) {
        // Use column-wise processing for better cache locality
        std::vector<std::vector<std::pair<int, Complex>>> columns(block_size);
        
        // Process columns in parallel
        #pragma omp parallel for schedule(dynamic, 1)
        for (int col = 0; col < block_size; col++) {
            processColumn(dir, block_start, col, block_size, columns[col]);
        }
        
        // Convert to sparse matrix and save
        saveBlockMatrixOptimized(block_dir, block_idx, block_size, columns);
    }
    
    void processColumn(const std::string& dir,
                      int block_start,
                      int col,
                      int block_size,
                      std::vector<std::pair<int, Complex>>& column) {
        // Load basis vector for this column
        std::map<int, Complex> basis_col;
        read_sym_basis_sparse(block_start + col, dir, basis_col);
        
        if (basis_col.empty()) {
            return;
        }
        
        // Apply Hamiltonian using pre-built sparse matrix
        std::map<int, Complex> h_basis_col = applyHamiltonianOptimized(basis_col);
        
        // Pre-allocate space for column
        column.reserve(block_size / 10);  // Estimate sparsity
        
        // Compute matrix elements for this column
        for (int row = 0; row < block_size; row++) {
            Complex element = computeMatrixElementOptimized(dir, block_start + row, h_basis_col);
            
            if (std::abs(element) > 1e-10) {
                column.emplace_back(row, element);
            }
        }
    }
    
    std::map<int, Complex> applyHamiltonianOptimized(const std::map<int, Complex>& basis_vec) {
        std::map<int, Complex> result;
        int dim = 1 << n_bits_;
        
        // Use the pre-built sparse matrix for efficient application
        for (const auto& [idx, val] : basis_vec) {
            if (idx >= dim) continue;
            
            // Apply transforms more efficiently
            for (const auto& transform : transforms_) {
                auto [new_idx, scalar] = transform(idx);
                
                if (new_idx >= 0 && new_idx < dim) {
                    Complex contribution = val * scalar;
                    if (std::abs(contribution) > 1e-10) {
                        result[new_idx] += contribution;
                    }
                }
            }
        }
        
        return result;
    }
    
    Complex computeMatrixElementOptimized(const std::string& dir,
                                         int row_idx,
                                         const std::map<int, Complex>& h_basis_col) {
        // Use static thread-local cache for basis vectors
        static thread_local std::unordered_map<int, std::map<int, Complex>> basis_cache;
        static thread_local int cache_counter = 0;
        
        // Periodically clear cache to prevent memory bloat
        if (++cache_counter > 1000) {
            basis_cache.clear();
            cache_counter = 0;
        }
        
        // Check cache first
        auto cache_it = basis_cache.find(row_idx);
        if (cache_it == basis_cache.end()) {
            // Load and cache basis vector
            std::map<int, Complex> basis_row;
            read_sym_basis_sparse(row_idx, dir, basis_row);
            cache_it = basis_cache.emplace(row_idx, std::move(basis_row)).first;
        }
        
        // Compute inner product
        return computeInnerProductOptimized(cache_it->second, h_basis_col);
    }
    
    Complex computeInnerProductOptimized(const std::map<int, Complex>& basis_row,
                                        const std::map<int, Complex>& h_basis_col) {
        Complex result(0.0, 0.0);
        
        // Use the smaller map for iteration
        if (basis_row.size() < h_basis_col.size()) {
            for (const auto& [idx, val] : basis_row) {
                auto it = h_basis_col.find(idx);
                if (it != h_basis_col.end()) {
                    result += std::conj(val) * it->second;
                }
            }
        } else {
            for (const auto& [idx, val] : h_basis_col) {
                auto it = basis_row.find(idx);
                if (it != basis_row.end()) {
                    result += std::conj(it->second) * val;
                }
            }
        }
        
        return result;
    }
    
    void saveBlockMatrixOptimized(const std::string& block_dir,
                                 size_t block_idx,
                                 int block_size,
                                 const std::vector<std::vector<std::pair<int, Complex>>>& columns) {
        // Count total non-zeros
        size_t nnz = 0;
        for (const auto& col : columns) {
            nnz += col.size();
        }
        
        // Build sparse matrix efficiently
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);
        
        for (int col = 0; col < block_size; col++) {
            for (const auto& [row, val] : columns[col]) {
                triplets.emplace_back(row, col, val);
            }
        }
        
        Eigen::SparseMatrix<Complex> blockMatrix(block_size, block_size);
        blockMatrix.setFromTriplets(triplets.begin(), triplets.end());
        blockMatrix.makeCompressed();
        
        // Save in compressed format
        std::string filename = block_dir + "/block_" + std::to_string(block_idx) + ".dat";
        std::ofstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Write header
        file.write(reinterpret_cast<const char*>(&block_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&block_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&nnz), sizeof(size_t));
        
        // Write non-zero elements in CSC format for better performance
        for (int k = 0; k < blockMatrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<Complex>::InnerIterator it(blockMatrix, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                double real = it.value().real();
                double imag = it.value().imag();
                
                file.write(reinterpret_cast<const char*>(&row), sizeof(int));
                file.write(reinterpret_cast<const char*>(&col), sizeof(int));
                file.write(reinterpret_cast<const char*>(&real), sizeof(double));
                file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
            }
        }
        
        file.close();
        
        std::cout << "  Block " << block_idx << " saved: " 
                  << block_size << "x" << block_size 
                  << " with " << nnz << " non-zeros ("
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * nnz / (block_size * block_size)) << "% fill)"
                  << std::endl;
    }

public:

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

    Eigen::SparseMatrix<Complex> getSparseMatrix() const {
        buildSparseMatrix();
        return sparseMatrix_;
    }

    // Load a symmetrized block matrix from file
    // Supports two formats:
    // 1) ASCII text:
    //      first line: `<rows> <cols>`
    //      subsequent lines: `row col real imag`
    // 2) Binary (written by saveBlockMatrixOptimized):
    //      int rows, int cols, size_t nnz, followed by nnz entries of
    //      {int row, int col, double real, double imag}
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlock(const std::string& filepath) {
        // First, try ASCII text parsing
        {
            std::ifstream tf(filepath);
            if (!tf.is_open()) {
                throw std::runtime_error("Could not open block file: " + filepath);
            }

            std::string firstLine;
            if (std::getline(tf, firstLine)) {
                std::istringstream hdr(firstLine);
                int rows = 0, cols = 0;
                if (hdr >> rows >> cols) {
                    // Looks like ASCII format. Parse the remaining lines.
                    std::vector<Eigen::Triplet<Complex>> triplets;
                    triplets.reserve(1024); // start with a modest reservation

                    int r, c;
                    double real, imag;
                    size_t count = 0;
                    while (tf >> r >> c >> real >> imag) {
                        // Basic sanity: skip out-of-bounds entries if present in file
                        if (r >= 0 && r < rows && c >= 0 && c < cols) {
                            triplets.emplace_back(r, c, Complex(real, imag));
                            ++count;
                        }
                    }

                    Eigen::SparseMatrix<Complex> blockMatrix(rows, cols);
                    blockMatrix.setFromTriplets(triplets.begin(), triplets.end());
                    blockMatrix.makeCompressed();

                    std::cout << "Loaded block matrix (ASCII): " << rows << "x" << cols
                              << " with " << count << " non-zeros ("
                              << std::fixed << std::setprecision(2)
                              << (rows * cols > 0 ? (100.0 * count / (rows * cols)) : 0.0)
                              << "% fill)" << std::endl;
                    return blockMatrix;
                }
            }
            // If first line didn't parse as two ints, fall through to binary path
        }

        // Binary fallback
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open block file: " + filepath);
        }

        int rows = 0, cols = 0;
        size_t nnz = 0;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(size_t));

        if (!file.good()) {
            throw std::runtime_error("Failed to read header from: " + filepath);
        }

        // Sanity checks to avoid pathological reserves from corrupted headers
        if (rows < 0 || cols < 0) {
            throw std::runtime_error("Invalid header (negative dims) in: " + filepath);
        }
        const unsigned long long maxEntries = static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(cols);
        if (nnz > maxEntries) {
            throw std::runtime_error("Invalid header (nnz > rows*cols) in: " + filepath);
        }

        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);

        for (size_t i = 0; i < nnz; ++i) {
            int row = 0, col = 0;
            double real = 0.0, imag = 0.0;

            file.read(reinterpret_cast<char*>(&row), sizeof(int));
            file.read(reinterpret_cast<char*>(&col), sizeof(int));
            file.read(reinterpret_cast<char*>(&real), sizeof(double));
            file.read(reinterpret_cast<char*>(&imag), sizeof(double));

            if (!file.good()) {
                throw std::runtime_error("Failed to read matrix element " +
                                       std::to_string(i) + " from: " + filepath);
            }

            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                triplets.emplace_back(row, col, Complex(real, imag));
            }
        }

        Eigen::SparseMatrix<Complex> blockMatrix(rows, cols);
        blockMatrix.setFromTriplets(triplets.begin(), triplets.end());
        blockMatrix.makeCompressed();

        std::cout << "Loaded block matrix (binary): " << rows << "x" << cols
                  << " with " << triplets.size() << " non-zeros ("
                  << std::fixed << std::setprecision(2)
                  << (rows * cols > 0 ? (100.0 * triplets.size() / (rows * cols)) : 0.0)
                  << "% fill)" << std::endl;

        return blockMatrix;
    }
    
    // Load all symmetrized blocks from a directory
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocks(const std::string& dir) {
        std::string block_dir = dir + "/sym_blocks";
        
        // First, determine how many blocks exist
        loadBlockSizesIfNeeded(dir);
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        std::cout << "Loading symmetrized Hamiltonian blocks from " << block_dir << std::endl;
        
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            if (symmetrized_block_ham_sizes[block_idx] == 0) {
                // Empty block - add an empty matrix
                blocks.emplace_back(0, 0);
                continue;
            }
            
            std::string filename = block_dir + "/block_" + std::to_string(block_idx) + ".dat";
            
            try {
                blocks.push_back(loadSymmetrizedBlock(filename));
                std::cout << "  Loaded block " << block_idx << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load block " << block_idx 
                         << ": " << e.what() << std::endl;
                // Add empty matrix for missing block
                blocks.emplace_back(symmetrized_block_ham_sizes[block_idx], 
                                  symmetrized_block_ham_sizes[block_idx]);
            }
        }
        
        std::cout << "Successfully loaded " << blocks.size() << " blocks" << std::endl;
        return blocks;
    }
    
    // Load a specific symmetrized block by index
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockByIndex(const std::string& dir, size_t block_idx) {
        loadBlockSizesIfNeeded(dir);
        
        if (block_idx >= symmetrized_block_ham_sizes.size()) {
            throw std::out_of_range("Block index " + std::to_string(block_idx) + 
                                   " out of range (max: " + 
                                   std::to_string(symmetrized_block_ham_sizes.size() - 1) + ")");
        }
        
        if (symmetrized_block_ham_sizes[block_idx] == 0) {
            return Eigen::SparseMatrix<Complex>(0, 0);
        }
        
        std::string filepath = dir + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
        return loadSymmetrizedBlock(filepath);
    }

private:
    std::vector<TransformFunction> transforms_;
    int n_bits_; // Number of bits in the basis representation
    float spin_l_;
    const std::array<std::array<double, 4>, 3> operators = {
        {{0, 1, 0, 0}, {0, 0, 1, 0},{1, 0, 0, -1}}
    };

    const std::array<std::array<double, 2>, 2> basis = {
        {{1, 0}, {0, 1}}
    };

    mutable Eigen::SparseMatrix<Complex> sparseMatrix_;
    mutable bool matrixBuilt_ = false;
};


class SingleSiteOperator : public Operator {
public:
    /**
     * Constructor for a single site operator
     * @param num_site Total number of sites/qubits
     * @param op Operator type: 0 for S+, 1 for S-, 2 for Sz, 3 for Sx, 4 for Sy
     * @param site_j Site index to apply the operator to
     */
    SingleSiteOperator(int num_site, float spin_l, int op, int site_j) : Operator(num_site, spin_l) {
        if (op < 0 || op > 4) {
            throw std::invalid_argument("Invalid operator type. Use 0 for S+, 1 for S-, 2 for Sz, 3 for Sx, 4 for Sy");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Site index out of range");
        }
        
        if (op <= 2) {
            // Original S+, S-, Sz operators
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, spin_l * pow(-1, (basis >> site_j) & 1)};
                }
                else if (op == 0 || op == 1) {
                    if (((basis >> site_j) & 1) != op) {
                        int flipped_basis = basis ^ (1 << site_j);
                        return {flipped_basis, Complex(1.0, 0) * double(spin_l * 2)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
        else if (op == 3) {
            // Sx = (S+ + S-) / 2
            // S+ contribution
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (((basis >> site_j) & 1) != 0) {
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(0.5, 0) * double(spin_l * 2)};
                }
                return {basis, Complex(0.0, 0.0)};
            });
            // S- contribution
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (((basis >> site_j) & 1) != 1) {
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(0.5, 0) * double(spin_l * 2)};
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
        else if (op == 4) {
            // Sy = (S+ - S-) / 2i
            // S+ contribution
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (((basis >> site_j) & 1) != 0) {
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(0, -0.5) * double(spin_l * 2)};
                }
                return {basis, Complex(0.0, 0.0)};
            });
            // S- contribution
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (((basis >> site_j) & 1) != 1) {
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(0, 0.5) * double(spin_l * 2)};
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};




class DoubleSiteOperator : public Operator {
public:
    /**
     * Constructor for a double site operator
     * @param num_site Total number of sites/qubits
     * @param op Operator type: 0 for X, 1 for Y, 2 for Z
     * @param site_j Site index to apply the operator to
     */

    DoubleSiteOperator() : Operator(0, 0) {
        // Default constructor
    }


    DoubleSiteOperator(int num_site, float spin_l_, int op_i, int site_i, int op_j, int site_j) : Operator(num_site, spin_l_) {
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            throw std::invalid_argument("Invalid operator type. Use 0 for S+, 1 for S-, 2 for Z");
        }
        if (site_j < 0 || site_j >= num_site || site_i < 0 || site_i >= num_site) {
            throw std::invalid_argument("Site index out of range");
        }
        
        addTransform([=](int basis) -> std::pair<int, Complex> {
            // Check what type of operators we're dealing with
            if (op_i == 2 && op_j == 2) {
                // Both are identity operators with phase factors
                int bit1 = (basis >> site_i) & 1;
                int bit2 = (basis >> site_j) & 1;
                return {basis, Complex(1.0,0.0)*double(spin_l_ * spin_l_) * pow(-1, bit1) * pow(-1, bit2)};
            } 
            else if (op_i == 2) {
                // Op1 is identity with phase, Op2 is bit flip
                int bit1 = (basis >> site_i) & 1;
                bool bit2_matches = ((basis >> site_j) & 1) != op_j;
                
                if (bit2_matches) {
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(1.0,0.0) * double(2*spin_l_) * pow(-1, bit1)};
                }
            } 
            else if (op_j == 2) {
                // Op2 is identity with phase, Op1 is bit flip
                int bit2 = (basis >> site_j) & 1;
                bool bit1_matches = ((basis >> site_i) & 1) != op_i;
                
                if (bit1_matches) {
                    // Flip the first bit
                    int flipped_basis = basis ^ (1 << site_i);
                    return {flipped_basis,  Complex(1.0,0.0)*double(2*spin_l_) * pow(-1, bit2)};
                }
            } 
            else {
                // Both are bit flip operators
                bool bit1_matches = ((basis >> site_i) & 1) != op_i;
                bool bit2_matches = ((basis >> site_j) & 1) != op_j;
                
                if (bit1_matches && bit2_matches) {
                    // Flip both bits
                    int flipped_basis = basis ^ (1 << site_i) ^ (1 << site_j);
                    return {flipped_basis, Complex(1.0,0.0)*double(4*spin_l_*spin_l_)};
                }
            }
            // Default case: no transformation applies
            return {basis, Complex(0.0, 0.0)};
        });
    }
};





/**
 * SumOperator class represents a sum of single-site operators with position-dependent phases
 * S^α = Σᵢ S^α_i e^(iQ·Rᵢ) 1/√N, where α ∈ {+,−,z}, Q is momentum vector, Rᵢ are site positions
 */
class SumOperator : public Operator {
public:
    /**
     * Constructor for sum operator with momentum-dependent phases
     * @param num_site Total number of sites/qubits
     * @param spin_l Spin quantum number
     * @param op Operator type: 0 for S+, 1 for S-, 2 for Sz
     * @param Q_vector Momentum vector [Qx, Qy, Qz]
     * @param positions_file Path to file containing site positions
     */
    SumOperator(int num_site, float spin_l, int op, const std::vector<double>& Q_vector, const std::string& positions_file) 
        : Operator(num_site, spin_l) {
        
        if (op < 0 || op > 2) {
            throw std::invalid_argument("Invalid operator type. Use 0 for S+, 1 for S-, 2 for Sz");
        }
        
        if (Q_vector.size() != 3) {
            throw std::invalid_argument("Q_vector must have 3 components [Qx, Qy, Qz]");
        }
        
        // Read positions from file
        std::vector<std::vector<double>> positions = readPositionsFromFile(positions_file, num_site);
        std::cout << "Loaded positions for " << num_site << " sites from " << positions_file << std::endl;
        // Calculate phase factors for each site
        std::vector<Complex> phase_factors(num_site);
        for (int i = 0; i < num_site; ++i) {
            if (positions[i].size() < 3) {
                throw std::runtime_error("Position vector must have at least 3 components for site " + std::to_string(i));
            }
            
            double phase = Q_vector[0] * positions[i][0] + 
                          Q_vector[1] * positions[i][1] + 
                          Q_vector[2] * positions[i][2];
            phase_factors[i] = Complex(std::cos(phase)/std::sqrt(num_site), std::sin(phase)/std::sqrt(num_site));
        }

        // Debug output for phase factors
        // std::cout << "Phase factors for Q = [" << Q_vector[0] << ", " << Q_vector[1] << ", " << Q_vector[2] << "]:\n";
        // for (int i = 0; i < num_site; ++i) {
        //     std::cout << "  Site " << i << " at position (" 
        //               << positions[i][0] << ", " << positions[i][1] << ", " << positions[i][2] << ")"
        //               << " -> phase factor: " << phase_factors[i].real() 
        //               << " + " << phase_factors[i].imag() << "i"
        //               << " (magnitude: " << std::abs(phase_factors[i]) << ")\n";
        // }

        // Add transforms for each site with appropriate phase factor
        for (int site = 0; site < num_site; ++site) {
            Complex phase_factor = phase_factors[site];
            
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    // Sz operator
                    int bit = (basis >> site) & 1;
                    return {basis, phase_factor * double(spin_l) * pow(-1, bit)};
                } else {
                    // S+ or S- operator
                    if (((basis >> site) & 1) != op) {
                        int flipped_basis = basis ^ (1 << site);
                        return {flipped_basis, phase_factor * double(spin_l * 2)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }

    /**
     * Read site positions from file
     * Expected format: Each line contains x y z coordinates for one site
     * @param filename Path to positions file
     * @param expected_sites Expected number of sites
     * @return Vector of position vectors
     */
    std::vector<std::vector<double>> readPositionsFromFile(const std::string& filename, int expected_sites) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open positions file: " + filename);
        }
        
        std::vector<std::vector<double>> positions(expected_sites);
        std::string line;
        
        // Skip the header lines (comments starting with #)
        while (std::getline(file, line) && line[0] == '#') {
            // Skip header lines
        }
        
        // Process the first non-header line (if we stopped at one)
        bool process_current_line = !line.empty() && line[0] != '#';
        
        do {
            if (process_current_line) {
                std::istringstream iss(line);
                int site_id, matrix_index, sublattice_index;
                double x, y, z;
                
                if (iss >> site_id >> matrix_index >> sublattice_index >> x >> y >> z) {
                    if (site_id >= 0 && site_id < expected_sites) {
                        positions[site_id] = {x, y, z};
                    }
                }
            }
            process_current_line = true;
        } while (std::getline(file, line));
        
        // Verify all positions were loaded
        for (int i = 0; i < expected_sites; ++i) {
            if (positions[i].empty()) {
                throw std::runtime_error("Missing position data for site " + std::to_string(i));
            }
        }
        
        return positions;
    }

};


/**
 * SumOperator class represents a sum of single-site operators with position-dependent phases
 * S^α = Σᵢ S^α_i e^(iQ·Rᵢ) 1/√N, where α ∈ {+,−,z}, Q is momentum vector, Rᵢ are site positions
 */
class SublatticeOperator : public SumOperator {
public:
    /**
     * Constructor for sum operator with momentum-dependent phases
     * @param i Sublattice site index 1
     * @param unit_cel_size Size of the unit cell (number of sublattices)
     * @param num_site Total number of sites/qubits
     * @param spin_l Spin quantum number
     * @param op Operator type: 0 for S+, 1 for S-, 2 for Sz
     * @param Q_vector Momentum vector [Qx, Qy, Qz]
     * @param positions_file Path to file containing site positions
     */
    SublatticeOperator(int i, int unit_cel_size, int num_site, float spin_l, int op, const std::vector<double>& Q_vector, const std::string& positions_file) 
        : SumOperator(num_site, spin_l, op, Q_vector, positions_file) {

        if (op < 0 || op > 2) {
            throw std::invalid_argument("Invalid operator type. Use 0 for S+, 1 for S-, 2 for Sz");
        }
        
        if (Q_vector.size() != 3) {
            throw std::invalid_argument("Q_vector must have 3 components [Qx, Qy, Qz]");
        }
        
        // Read positions from file
        std::vector<std::vector<double>> positions = readPositionsFromFile(positions_file, num_site);
        std::cout << "Loaded positions for " << num_site << " sites from " << positions_file << std::endl;
        // Calculate phase factors for each site
        std::vector<Complex> phase_factors(num_site);
        
        for (int i = 0; i < num_site; ++i) {
            if (positions[i].size() < 3) {
                throw std::runtime_error("Position vector must have at least 3 components for site " + std::to_string(i));
            }
            
            double phase = -Q_vector[0] * positions[i][0] + 
                          -Q_vector[1] * positions[i][1] + 
                          -Q_vector[2] * positions[i][2];
            phase_factors[i] = Complex(std::cos(phase)/std::sqrt(num_site), std::sin(phase)/std::sqrt(num_site));
        }
        // Add transforms for each site with appropriate phase factor
        for (int site = i; site < num_site; site+=unit_cel_size) {
            Complex phase_factor = phase_factors[site];
            
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    // Sz operator
                    int bit = (basis >> site) & 1;
                    return {basis, phase_factor * double(spin_l) * pow(-1, bit)};
                } else {
                    // S+ or S- operator
                    if (((basis >> site) & 1) != op) {
                        int flipped_basis = basis ^ (1 << site);
                        return {flipped_basis, phase_factor * double(spin_l * 2)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

#endif // CONSTRUCT_HAM_H