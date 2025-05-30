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
#include <memory>
#include <omp.h>
#include <unordered_set>

// Define complex number type and matrix type for convenience
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;
using ComplexVector = std::vector<Complex>;

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
    // Get the binary representation of the basis state
    for (size_t i = 0; i < perm.size(); ++i) {
        // Get the bit at position i in the basis state
        int bit = (basis >> i) & 1;
        // Set the bit at the permuted position
        result |= (bit << perm[i]);
    }

    // Print the result for debugging
    // Print in binary representation for debugging
    // std::cout << "applyPermutation: basis = ";
    // for (int i = perm.size() - 1; i >= 0; i--) {
    //     std::cout << ((basis >> i) & 1);
    // }
    // std::cout << " (" << basis << "), perm = [";
    // for (size_t i = 0; i < perm.size(); i++) {
    //     std::cout << perm[i];
    //     if (i < perm.size() - 1) std::cout << " ";
    // }
    // std::cout << "] -> result = ";
    // for (int i = perm.size() - 1; i >= 0; i--) {
    //     std::cout << ((result >> i) & 1);
    // }
    // std::cout << " (" << result << ")" << std::endl;
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
    
    Operator(int n_bits, float spin_l) : n_bits_(n_bits), spin_l_(spin_l), sparseMatrix_(nullptr) {}

    // Copy assignment operator
    Operator& operator=(const Operator& other) {
        if (this != &other) {  // Check for self-assignment
            n_bits_ = other.n_bits_;
            transforms_ = other.transforms_;
            sparseMatrix_ = other.sparseMatrix_; // Shared pointer will automatically handle reference counting
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
    void apply(const std::vector<Complex>& vec, std::vector<Complex>& resultVec) const {
        // Ensure result vector is properly sized
        if (resultVec.size() != vec.size()) {
            resultVec.resize(vec.size());
        }
        // Zero out result vector
        std::fill(resultVec.begin(), resultVec.end(), Complex(0.0, 0.0));
        
        // Use sparse matrix if available for better performance
        if (matrixBuilt_) {
            // Convert input vector to Eigen format
            Eigen::Map<const Eigen::VectorXcd> eigenVec(vec.data(), vec.size());
            // Compute product using optimized sparse matrix multiplication
            Eigen::VectorXcd eigenResult = (*sparseMatrix_) * eigenVec;
            // Copy back to result vector
            std::copy(eigenResult.data(), eigenResult.data() + vec.size(), resultVec.begin());
        } else {
            // Fallback to the original transform-based calculation
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < vec.size(); ++i) {
                if (std::abs(vec[i]) < 1e-15) continue; // Skip near-zero elements
                
                for (const auto& transform : transforms_) {
                    auto [j, scalar] = transform(i);
                    if (j >= 0 && j < vec.size() && std::abs(scalar) > 1e-15) {
                        #pragma omp critical
                        {
                            resultVec[j] += scalar * vec[i];
                        }
                    }
                }
            }
        }
    }

    // Overload that returns a new vector for backward compatibility
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        std::vector<Complex> resultVec(vec.size());
        apply(vec, resultVec);
        return resultVec;
    }

    // Return sparse matrix
    Eigen::SparseMatrix<Complex> getSparseMatrix() const {
        if (!matrixBuilt_) {
            buildSparseMatrix();
        }
        return *sparseMatrix_;
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
        
        // Read max clique from JSON file
        std::string max_clique_path = dir + "/automorphism_results/max_clique.json";
        std::cout << "Loading max clique from: " << max_clique_path << std::endl;
        
        // Read max clique
        std::vector<std::vector<int>> max_clique_here;
        std::ifstream max_clique_file(max_clique_path);
        if (!max_clique_file.is_open()) {
            throw std::runtime_error("Cannot open max_clique.json file: " + max_clique_path);
        }
        
        // Read the entire file content
        std::string json_content((std::istreambuf_iterator<char>(max_clique_file)),
                    std::istreambuf_iterator<char>());
        max_clique_file.close();
        
        // Simple JSON array parser
        size_t pos = 0;
        pos = json_content.find('[', pos);
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid JSON format in max_clique.json");
        }
        pos++;
        
        while (pos < json_content.size()) {
            while (pos < json_content.size() && (json_content[pos] == ' ' || json_content[pos] == '\n' || 
            json_content[pos] == '\r' || json_content[pos] == '\t')) pos++;
            
            if (pos >= json_content.size() || json_content[pos] == ']') break;
            
            if (json_content[pos] != '[') {
                pos++;
                continue;
            }
            
            pos++;
            std::vector<int> permutation;
            
            std::string num_str;
            while (pos < json_content.size() && json_content[pos] != ']') {
                char c = json_content[pos++];
                if (std::isdigit(c)) {
                    num_str += c;
                } else if (!num_str.empty() && (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t')) {
                    permutation.push_back(std::stoi(num_str));
                    num_str.clear();
                }
            }
            
            if (!num_str.empty()) {
                permutation.push_back(std::stoi(num_str));
            }
            
            if (!permutation.empty()) {
                max_clique_here.push_back(permutation);
            }
            
            pos++;
        }
        
        if (max_clique_here.empty()) {
            throw std::runtime_error("Failed to parse max_clique.json or file is empty");
        }
        
        std::cout << "Loaded " << max_clique_here.size() << " permutations in max clique" << std::endl;

        // Trivial case: skip symmetrization if max clique size is 1
        if (max_clique_here.size() == 1) {
            // Same as before...
            std::string sym_basis_dir = dir + "/sym_basis";
            std::string mkdir_command = "mkdir -p " + sym_basis_dir;
            system(mkdir_command.c_str());
            
            int dim = 1 << n_bits_;
            symmetrized_block_ham_sizes = {dim};

            // For extremely large systems, just create a placeholder file
            if (n_bits_ > 30) {
                std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
                if (!block_size_file.is_open()) {
                    throw std::runtime_error("Could not open file: " + sym_basis_dir + "/sym_block_sizes.txt");
                }
                block_size_file << dim << std::endl;
                block_size_file.close();
                
                // Create an index file that explains the implicit representation
                std::ofstream index_file(sym_basis_dir + "/implicit_representation.txt");
                if (index_file.is_open()) {
                    index_file << "System size too large for explicit basis storage.\n";
                    index_file << "Using implicit representation: standard basis with no symmetrization.\n";
                    index_file << "Dimension: " << dim << std::endl;
                    index_file.close();
                }
                
                std::cout << "System too large for explicit basis storage. Using implicit representation." << std::endl;
                return;
            }
            
            // For smaller systems, continue with the original approach
            for (int i = 0; i < dim; i++) {
                std::ofstream file(sym_basis_dir + "/sym_basis" + std::to_string(i) + ".dat");
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open file: " + sym_basis_dir + "/sym_basis" + std::to_string(i) + ".dat");
                }
                file << i << " 1.0 0.0" << std::endl;
                file.close();
            }

            std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
            if (!block_size_file.is_open()) {
                throw std::runtime_error("Could not open file: " + sym_basis_dir + "/sym_block_sizes.txt");
            }
            block_size_file << dim << std::endl;
            block_size_file.close();

            std::cout << "Generated standard basis states as symmetrized basis." << std::endl;
            std::cout << "Block size: " << dim << std::endl;
            return;
        }

        // For systems that require symmetrization
        MinimalGeneratorFinder minimal_finder;
        std::pair<std::vector<std::vector<int>>, std::vector<int>> minimal_generators = minimal_finder.findMinimalGenerators(max_clique_here);
        
        std::cout << "Minimal generators:\n";
        int count_temp = 0;
        for (const auto& generator : minimal_generators.first) {
            std::cout << "Generator: ";
            for (int index : generator) {
                std::cout << index << " ";
            }
            std::cout << "with order: " << minimal_generators.second[count_temp++] << std::endl;
        }

        // Pre-compute power representations - this is relatively lightweight
        AutomorphismPowerRepresentation automorphism_power_representation;
        std::vector<std::vector<int>> power_representation = 
            automorphism_power_representation.representAllAsGeneratorPowers(minimal_generators.first, max_clique_here);
        std::cout << "Power representation generated.\n";

        // Generate all quantum number combinations
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

        // Create directory for symmetrized basis
        std::string sym_basis_dir = dir + "/sym_basis";
        std::string mkdir_command = "mkdir -p " + sym_basis_dir;
        system(mkdir_command.c_str());
        symmetrized_block_ham_sizes.resize(all_quantum_numbers.size(), 0);

        // Check if system is too large for full enumeration
        size_t max_states = static_cast<size_t>(1) << n_bits_;
        bool use_block_enumeration = (n_bits_ > 26);  // Threshold based on memory constraints
        
        // For extremely large systems, directly create a compact representation
        if (n_bits_ > 34) {  // Threshold where even partial enumeration is infeasible
            std::cout << "System size too large for explicit enumeration. Using analytical estimation." << std::endl;
            
            // Estimate block sizes based on group theory
            size_t group_order = max_clique_here.size();
            size_t approx_total_states = max_states / group_order;  // Approximate number of orbits
            
            // Distribute states across sectors roughly evenly (simplified model)
            size_t states_per_sector = approx_total_states / all_quantum_numbers.size();
            
            // Write block sizes to file
            std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
            if (!block_size_file.is_open()) {
                throw std::runtime_error("Could not open file: " + sym_basis_dir + "/sym_block_sizes.txt");
            }
            
            for (size_t i = 0; i < all_quantum_numbers.size(); i++) {
                symmetrized_block_ham_sizes[i] = states_per_sector;
                block_size_file << states_per_sector << std::endl;
            }
            block_size_file.close();
            
            // Create a metadata file explaining the representation
            std::ofstream meta_file(sym_basis_dir + "/large_system_metadata.txt");
            if (meta_file.is_open()) {
                meta_file << "System size: " << n_bits_ << " bits\n";
                meta_file << "Full dimension: " << max_states << "\n";
                meta_file << "Symmetry group order: " << group_order << "\n";
                meta_file << "Estimated states per sector: " << states_per_sector << "\n";
                meta_file << "Total sectors: " << all_quantum_numbers.size() << "\n";
                meta_file << "Note: Using estimated block sizes for large system." << std::endl;
                meta_file.close();
            }
            
            std::cout << "Created estimated block structure for large system." << std::endl;
            return;
        }

        // Process each symmetry sector
        int sector_count = 0;
        for (const auto& e_i : all_quantum_numbers) {
            std::cout << "Processing sector " << sector_count + 1 << "/" << all_quantum_numbers.size() 
                      << " with quantum numbers: ";
            for (int qn : e_i) std::cout << qn << " ";
            std::cout << std::endl;
            
            // Set to track processed orbits
            std::unordered_set<size_t> orbit_representatives;
            size_t sector_basis_count = 0;
            
            // For very large systems, use a different approach
            if (use_block_enumeration) {
                // Process basis states in blocks to control memory usage
                const size_t BLOCK_SIZE = 1ULL << 24;  // Process 16M states at a time
                size_t num_blocks = (max_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                std::cout << "Processing in " << num_blocks << " blocks." << std::endl;
                
                // Hash function for efficient orbit tracking
                std::hash<std::string> hasher;
                
                for (size_t block = 0; block < num_blocks; block++) {
                    size_t block_start = block * BLOCK_SIZE;
                    size_t block_end = std::min((block + 1) * BLOCK_SIZE, max_states);
                    
                    std::cout << "\rProcessing block " << block + 1 << "/" << num_blocks << std::flush;
                    
                    // Temporary file for this block
                    std::string block_file = sym_basis_dir + "/temp_block_" + std::to_string(block) + ".dat";
                    std::ofstream block_out(block_file);
                    
                    // Process states in this block
                    #pragma omp parallel for schedule(dynamic) reduction(+:sector_basis_count)
                    for (size_t basis = block_start; basis < block_end; basis++) {
                        // Skip if this state is already part of a processed orbit
                        bool skip = false;
                        
                        #pragma omp critical
                        {
                            // Check if this state is in an orbit we've already processed
                            if (orbit_representatives.find(basis) != orbit_representatives.end()) {
                                skip = true;
                            }
                        }
                        
                        if (skip) continue;
                        
                        // Get orbit of the current state
                        std::set<size_t> orbit;
                        orbit.insert(basis);
                        
                        // Apply all symmetry operations
                        for (const auto& perm : max_clique_here) {
                            orbit.insert(applyPermutation(basis, perm));
                        }
                        
                        // Mark all states in orbit as processed
                        #pragma omp critical
                        {
                            for (size_t state : orbit) {
                                orbit_representatives.insert(state);
                            }
                        }
                        
                        // Generate symmetrized basis vector (sparse)
                        using SparseVector = std::map<size_t, Complex>;
                        SparseVector sym_vec;
                        
                        // Pre-compute phase factors for efficiency
                        std::vector<Complex> phase_factors(max_clique_here.size());
                        for (size_t i = 0; i < max_clique_here.size(); i++) {
                            double phase = 0.0;
                            for (size_t j = 0; j < power_representation[i].size(); j++) {
                                phase += 2.0 * M_PI * power_representation[i][j] * e_i[j] / minimal_generators.second[j];
                            }
                            phase_factors[i] = Complex(std::cos(phase), std::sin(phase));
                        }
                        
                        // Apply each symmetry with appropriate phase
                        for (size_t i = 0; i < max_clique_here.size(); i++) {
                            size_t permuted_state = applyPermutation(basis, max_clique_here[i]);
                            sym_vec[permuted_state] += phase_factors[i];
                        }
                        
                        // Normalize and check if non-zero
                        double norm_squared = 0.0;
                        for (const auto& [_, val] : sym_vec) {
                            norm_squared += std::norm(val);
                        }
                        
                        if (norm_squared > 1e-10) {
                            // Find first non-zero element for phase normalization
                            size_t first_nonzero = max_states;
                            Complex first_val;
                            
                            for (const auto& [idx, val] : sym_vec) {
                                if (std::abs(val) > 1e-10 && idx < first_nonzero) {
                                    first_nonzero = idx;
                                    first_val = val;
                                }
                            }
                            
                            if (first_nonzero < max_states) {
                                // Normalize phase
                                Complex phase_factor = first_val / std::abs(first_val);
                                
                                // Write to temporary file
                                #pragma omp critical
                                {
                                    block_out << sector_basis_count << "\n";
                                    for (const auto& [idx, val] : sym_vec) {
                                        Complex normalized = val / phase_factor;
                                        if (std::abs(normalized) > 1e-10) {
                                            block_out << idx << " " << normalized.real() << " " << normalized.imag() << "\n";
                                        }
                                    }
                                    block_out << "END\n";
                                    sector_basis_count++;
                                }
                            }
                        }
                    }
                    
                    block_out.close();
                }
                
                std::cout << std::endl;
                
                // Consolidate temporary files
                symmetrized_block_ham_sizes[sector_count] = sector_basis_count;
                
                // Only proceed with file creation if there are any basis states in this sector
                if (sector_basis_count > 0) {
                    // Merge all temporary files into final basis files
                    size_t basis_index = 0;
                    size_t total_index = 0;
                    
                    // Calculate starting index for this sector
                    for (int i = 0; i < sector_count; i++) {
                        total_index += symmetrized_block_ham_sizes[i];
                    }
                    
                    for (size_t block = 0; block < num_blocks; block++) {
                        std::string block_file = sym_basis_dir + "/temp_block_" + std::to_string(block) + ".dat";
                        std::ifstream block_in(block_file);
                        
                        if (!block_in.is_open()) continue;
                        
                        std::string line;
                        std::ofstream* current_file = nullptr;
                        
                        while (std::getline(block_in, line)) {
                            if (isdigit(line[0])) {
                                // New basis vector
                                if (current_file) {
                                    current_file->close();
                                    delete current_file;
                                }
                                
                                current_file = new std::ofstream(sym_basis_dir + "/sym_basis" + 
                                                               std::to_string(total_index + basis_index) + ".dat");
                                basis_index++;
                            } else if (line == "END") {
                                // End of basis vector
                                if (current_file) {
                                    current_file->close();
                                    delete current_file;
                                    current_file = nullptr;
                                }
                            } else if (current_file) {
                                // Write data
                                *current_file << line << "\n";
                            }
                        }
                        
                        if (current_file) {
                            current_file->close();
                            delete current_file;
                        }
                        
                        block_in.close();
                        std::remove(block_file.c_str());
                    }
                }
            } else {
                // Original algorithm for smaller systems, but with optimizations
                std::vector<size_t> representative_states;
                size_t sector_basis_count = 0;
                
                // Use a memory-efficient data structure for tracking processed states
                std::vector<bool> processed(max_states, false);
                
                // Setup progress tracking
                size_t progress_interval = max_states / 100;
                if (progress_interval < 1) progress_interval = 1;
                
                for (size_t basis = 0; basis < max_states; basis++) {
                    // Display progress
                    if (basis % progress_interval == 0 || basis == max_states - 1) {
                        double percentage = static_cast<double>(basis) / max_states * 100.0;
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
                    
                    // Skip if already processed
                    if (processed[basis]) continue;
                    
                    // Get orbit
                    std::set<size_t> orbit;
                    for (const auto& perm : max_clique_here) {
                        orbit.insert(applyPermutation(basis, perm));
                    }
                    
                    // Mark all as processed
                    for (size_t state : orbit) {
                        processed[state] = true;
                    }
                    
                    // Generate sparse symmetrized basis vector
                    using SparseVector = std::map<size_t, Complex>;
                    SparseVector sym_vec;
                    
                    // Apply symmetry operations with phases
                    for (size_t i = 0; i < max_clique_here.size(); i++) {
                        double phase = 0.0;
                        for (size_t j = 0; j < power_representation[i].size(); j++) {
                            phase += 2.0 * M_PI * power_representation[i][j] * e_i[j] / minimal_generators.second[j];
                        }
                        Complex phase_factor(std::cos(phase), std::sin(phase));
                        size_t permuted_state = applyPermutation(basis, max_clique_here[i]);
                        sym_vec[permuted_state] += phase_factor;
                    }
                    
                    // Check norm
                    double norm_squared = 0.0;
                    for (const auto& [_, val] : sym_vec) {
                        norm_squared += std::norm(val);
                    }
                    
                    if (norm_squared > 1e-10) {
                        // Phase normalization
                        size_t first_nonzero = max_states;
                        Complex first_val;
                        
                        for (const auto& [idx, val] : sym_vec) {
                            if (std::abs(val) > 1e-10 && idx < first_nonzero) {
                                first_nonzero = idx;
                                first_val = val;
                            }
                        }
                        
                        if (first_nonzero < max_states) {
                            // Normalize and write directly to file
                            Complex phase_factor = first_val / std::abs(first_val);
                            
                            // Calculate total basis index
                            size_t total_basis_idx = 0;
                            for (int i = 0; i < sector_count; i++) {
                                total_basis_idx += symmetrized_block_ham_sizes[i];
                            }
                            total_basis_idx += sector_basis_count;
                            
                            // Write sparse vector to file
                            std::ofstream vec_file(sym_basis_dir + "/sym_basis" + std::to_string(total_basis_idx) + ".dat");
                            for (const auto& [idx, val] : sym_vec) {
                                Complex normalized = val / phase_factor;
                                if (std::abs(normalized) > 1e-10) {
                                    vec_file << idx << " " << normalized.real() << " " << normalized.imag() << std::endl;
                                }
                            }
                            
                            representative_states.push_back(basis);
                            sector_basis_count++;
                        }
                    }
                }
                
                std::cout << std::endl;
                symmetrized_block_ham_sizes[sector_count] = sector_basis_count;
            }
            
            std::cout << "Sector " << sector_count + 1 << ": Found " << sector_basis_count << " basis vectors" << std::endl;
            sector_count++;
        }

        // Write block sizes
        std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
        if (!block_size_file.is_open()) {
            throw std::runtime_error("Could not open file: " + sym_basis_dir + "/sym_block_sizes.txt");
        }
        
        size_t total_basis_count = 0;
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); i++) {
            block_size_file << symmetrized_block_ham_sizes[i] << std::endl;
            total_basis_count += symmetrized_block_ham_sizes[i];
        }
        block_size_file.close();
        
        std::cout << "Total symmetrized basis vectors: " << total_basis_count << std::endl;
        std::cout << "Block sizes: ";
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); i++) {
            std::cout << symmetrized_block_ham_sizes[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Symmetrized basis generation complete." << std::endl;
    }


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
        if (symmetrized_block_ham_sizes.empty()) {
            // Load symmetrized block sizes from file
            std::ifstream file(dir + "/sym_basis/sym_block_sizes.txt");
            if (!file.is_open()) {
                throw std::runtime_error("Symmetrized basis must be generated first with generateSymmetrizedBasis()");
            }
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int block_size;
                if (iss >> block_size) {
                    symmetrized_block_ham_sizes.push_back(block_size);
                }
            }
            file.close();
            std::cout << "Loaded symmetrized block sizes from file." << std::endl;
        }
        
        std::cout << "Building and saving symmetrized Hamiltonian blocks..." << std::endl;
        
        // Create a directory for the block matrices
        std::string block_dir = dir + "/sym_blocks";
        std::string mkdir_command = "mkdir -p " + block_dir;
        system(mkdir_command.c_str());
        
        int dim = 1 << n_bits_;
        int block_start = 0;
        
        // Use a sparse vector representation to reduce memory usage
        using SparseVector = std::map<int, Complex>;
        
        // Process each symmetry block
        for (size_t block = 0; block < symmetrized_block_ham_sizes.size(); block++) {
            int block_size = symmetrized_block_ham_sizes[block];
            if (block_size == 0) continue;
            
            std::cout << "Processing block " << block << " of size " << block_size << std::endl;

            
            // Create a sparse matrix for this block
            std::vector<Eigen::Triplet<Complex>> triplets;
            triplets.reserve(block_size * 10); // Estimate average 10 non-zeros per row
            
            // Cache for basis vectors to reduce disk I/O
            std::vector<SparseVector> basis_cache(block_size);
            
            // Load all basis vectors into sparse representation
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < block_size; i++) {
                std::string filename = dir + "/sym_basis/sym_basis" + std::to_string(block_start + i) + ".dat";
                std::ifstream file(filename);
                if (!file.is_open()) {
                    #pragma omp critical
                    {
                        std::cerr << "Failed to open: " << filename << std::endl;
                    }
                    continue;
                }
                
                std::string line;
                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    int idx;
                    double real, imag;
                    if (iss >> idx >> real >> imag) {
                        Complex val(real, imag);
                        if (std::abs(val) > 1e-10) {
                            basis_cache[i][idx] = val;
                        }
                    }
                }
            }
            
            // Process basis vectors in batches to improve cache locality
            const int BATCH_SIZE = std::min(50, block_size);
            
            for (int batch_start = 0; batch_start < block_size; batch_start += BATCH_SIZE) {
                int batch_end = std::min(batch_start + BATCH_SIZE, block_size);
                
                #pragma omp parallel for schedule(dynamic)
                for (int i = batch_start; i < batch_end; i++) {
                    // Skip if basis vector was not loaded successfully
                    if (basis_cache[i].empty()) continue;
                    
                    // Apply operator to basis vector i
                    SparseVector transformed_i;
                    for (const auto& [idx, val] : basis_cache[i]) {
                        for (const auto& transform : transforms_) {
                            auto [new_idx, scalar] = transform(idx);
                            if (new_idx >= 0 && new_idx < dim && std::abs(scalar) > 1e-10) {
                                transformed_i[new_idx] += val * scalar;
                            }
                        }
                    }
                    
                    // Calculate matrix elements with each basis vector j
                    std::vector<std::pair<int, Complex>> row_elements;
                    
                    for (int j = 0; j < block_size; j++) {
                        // Skip if basis vector was not loaded successfully
                        if (basis_cache[j].empty()) continue;
                        
                        // Calculate <j|H|i> = sum_k conj(basis_j[k]) * transformed_i[k]
                        Complex element(0.0, 0.0);
                        
                        // Only iterate over non-zero elements of transformed_i
                        for (const auto& [idx, val] : transformed_i) {
                            auto it = basis_cache[j].find(idx);
                            if (it != basis_cache[j].end()) {
                                element += std::conj(it->second) * val;
                            }
                        }
                        
                        if (std::abs(element) > 1e-10) {
                            row_elements.emplace_back(j, element);
                        }
                    }
                    
                    // Add non-zero elements to triplets list
                    #pragma omp critical
                    {
                        for (const auto& [j, element] : row_elements) {
                            triplets.emplace_back(j, i, element);
                        }
                    }
                }
            }
            
            // Create and save the sparse matrix
            Eigen::SparseMatrix<Complex> blockMatrix(block_size, block_size);
            blockMatrix.setFromTriplets(triplets.begin(), triplets.end());
            
            // Save in binary format for efficiency
            std::string filename = block_dir + "/block_" + std::to_string(block) + ".dat";
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }
            
            // Write matrix dimensions
            file << block_size << " " << block_size << std::endl;
            
            // Write non-zero elements (row, col, real, imag) in binary format
            for (int k = 0; k < blockMatrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Complex>::InnerIterator it(blockMatrix, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    double real = it.value().real();
                    double imag = it.value().imag();
                    file << row << " " << col << " " << real << " " << imag << std::endl;
                }
            }
            
            file.close();
            
            block_start += block_size;
        }
        
        std::cout << "Symmetrized Hamiltonian blocks saved to " << block_dir << std::endl;
    }

    
    void buildSparseMatrix() const {
        if (matrixBuilt_) return;  // Already built
        
        int dim = 1 << n_bits_;
        
        // Create a new sparse matrix if it doesn't exist
        if (!sparseMatrix_) {
            sparseMatrix_ = std::make_shared<Eigen::SparseMatrix<Complex>>(dim, dim);
        } else {
            // Resize if dimensions have changed
            sparseMatrix_->resize(dim, dim);
            sparseMatrix_->setZero();
        }
        
        // Build coefficients
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(dim * 5); // Estimate of non-zeros
        
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim && std::abs(scalar) > 1e-10) {
                    triplets.emplace_back(j, i, scalar);
                }
            }
        }
        
        sparseMatrix_->setFromTriplets(triplets.begin(), triplets.end());
        matrixBuilt_ = true;
    }

    void printEntireSymmetrizedMatrix(const std::string& dir) {
        // First, get the total dimension by summing all block sizes
        if (symmetrized_block_ham_sizes.empty()) {
            // Load symmetrized block sizes from file if not already loaded
            std::ifstream file(dir + "/sym_basis/sym_block_sizes.txt");
            if (!file.is_open()) {
                throw std::runtime_error("Symmetrized basis must be generated first with generateSymmetrizedBasis()");
            }
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int block_size;
                if (iss >> block_size) {
                    symmetrized_block_ham_sizes.push_back(block_size);
                }
            }
            file.close();
        }
        
        int total_sym_dim = 0;
        for (int size : symmetrized_block_ham_sizes) {
            total_sym_dim += size;
        }
        
        std::cout << "Total symmetrized dimension: " << total_sym_dim << std::endl;
        
        // Create the full symmetrized matrix
        Matrix sym_matrix(total_sym_dim, std::vector<Complex>(total_sym_dim, 0.0));
        
        // Calculate matrix elements <sym_basis_i|H|sym_basis_j>
        std::cout << "Computing symmetrized matrix elements..." << std::endl;
        
        for (int i = 0; i < total_sym_dim; ++i) {
            // Load basis vector i
            std::vector<Complex> basis_i = read_sym_basis(i, dir);
            
            // Apply Hamiltonian to basis_i
            std::vector<Complex> H_basis_i = apply(basis_i);
            
            for (int j = 0; j < total_sym_dim; ++j) {
                // Load basis vector j
                std::vector<Complex> basis_j = read_sym_basis(j, dir);
                
                // Calculate <basis_j|H|basis_i>
                Complex matrix_element(0.0, 0.0);
                for (int k = 0; k < (1 << n_bits_); ++k) {
                    matrix_element += std::conj(basis_j[k]) * H_basis_i[k];
                }
                
                sym_matrix[i][j] = matrix_element;
            }
            
            // Progress indicator
            if ((i + 1) % 10 == 0 || i == total_sym_dim - 1) {
                std::cout << "\rProgress: " << (i + 1) << "/" << total_sym_dim << " rows computed" << std::flush;
            }
        }
        std::cout << std::endl;
        
        // Print the matrix
        std::cout << "\nFull Symmetrized Hamiltonian Matrix:" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Print with formatting
        std::cout << std::fixed << std::setprecision(6);
        
        // Print column headers
        std::cout << "      ";
        for (int j = 0; j < total_sym_dim; ++j) {
            std::cout << "   sym_" << std::setw(3) << j << "     ";
        }
        std::cout << std::endl;
        
        // Print separator
        std::cout << "      ";
        for (int j = 0; j < total_sym_dim; ++j) {
            std::cout << "------------";
        }
        std::cout << std::endl;
        
        // Print matrix elements
        for (int i = 0; i < total_sym_dim; ++i) {
            std::cout << "sym_" << std::setw(3) << i << " |";
            for (int j = 0; j < total_sym_dim; ++j) {
                if (std::abs(sym_matrix[i][j]) < 1e-10) {
                    std::cout << "     0      ";
                } else {
                    // Print real part
                    if (sym_matrix[i][j].imag() == 0) {
                        std::cout << std::setw(11) << sym_matrix[i][j].real() << " ";
                    } else {
                        // Print complex number
                        std::cout << std::setw(6) << sym_matrix[i][j].real();
                        if (sym_matrix[i][j].imag() >= 0) {
                            std::cout << "+";
                        }
                        std::cout << std::setw(5) << sym_matrix[i][j].imag() << "i";
                    }
                }
            }
            std::cout << std::endl;
        }
        
        // Also save to file for later use
        std::string output_file = dir + "/full_symmetrized_hamiltonian.txt";
        std::ofstream out_file(output_file);
        if (out_file.is_open()) {
            out_file << std::fixed << std::setprecision(10);
            out_file << total_sym_dim << " " << total_sym_dim << std::endl;
            for (int i = 0; i < total_sym_dim; ++i) {
                for (int j = 0; j < total_sym_dim; ++j) {
                    out_file << sym_matrix[i][j].real() << "+1j*" << sym_matrix[i][j].imag() << " ";
                }
                out_file << std::endl;
            }
            out_file.close();
            std::cout << "\nMatrix saved to: " << output_file << std::endl;
        }
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

    mutable std::shared_ptr<Eigen::SparseMatrix<Complex>> sparseMatrix_;
    mutable bool matrixBuilt_ = false;

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
    SingleSiteOperator(int num_site, float spin_l, int op, int site_j) : Operator(num_site, spin_l) {
        if (op < 0 || op > 2) {
            throw std::invalid_argument("Invalid operator type. Use 0 for S+, 1 for S-, 2 for Z");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Site index out of range");
        }
        
        addTransform([=](int basis) -> std::pair<int, Complex> {
        // Check if all bits match their expected values
            if (op == 2){
                return {basis, spin_l*pow(-1,(basis >> site_j) & 1)};
            }
            else{
                if (((basis >> site_j) & 1) != op) {
                    // Flip the A bit
                    int flipped_basis = basis ^ (1 << site_j);
                    return {flipped_basis, Complex(1.0, 0)*double(spin_l*2)};                    
                }
            }
        // Default case: no transformation applies
        return {basis, Complex(0.0, 0.0)};
        });
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

#endif // CONSTRUCT_HAM_H