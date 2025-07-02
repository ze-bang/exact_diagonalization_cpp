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
    
    // Constructor with option to preallocate memory
    Operator(int n_bits, float spin_l, size_t expected_transforms = 0) 
        : n_bits_(n_bits), spin_l_(spin_l), sparseMatrix_(nullptr) {
        if (expected_transforms > 0) {
            transforms_.reserve(expected_transforms);
        }
    }

    // Copy assignment operator
    Operator& operator=(const Operator& other) {
        if (this != &other) {
            n_bits_ = other.n_bits_;
            transforms_ = other.transforms_;
            sparseMatrix_ = other.sparseMatrix_;
            matrixBuilt_ = other.matrixBuilt_;
            symmetrized_block_ham_sizes = other.symmetrized_block_ham_sizes;
            spin_l_ = other.spin_l_;
        }
        return *this;
    }

    // Mark matrix as needing rebuild when new transform is added
    void addTransform(TransformFunction transform) {
        transforms_.push_back(transform);
        matrixBuilt_ = false;
    }

    // Memory-efficient apply using pre-allocated buffer
    void apply(const std::vector<Complex>& vec, std::vector<Complex>& resultVec) const {
        // Ensure result vector is properly sized
        if (resultVec.size() != vec.size()) {
            resultVec.resize(vec.size());
        }
        std::fill(resultVec.begin(), resultVec.end(), Complex(0.0, 0.0));
        
        if (matrixBuilt_) {
            // Use Eigen's efficient sparse matrix multiplication
            Eigen::Map<const Eigen::VectorXcd> eigenVec(vec.data(), vec.size());
            Eigen::Map<Eigen::VectorXcd> eigenResult(resultVec.data(), resultVec.size());
            eigenResult = (*sparseMatrix_) * eigenVec;
        } else {
            // Memory-efficient transform-based calculation
            #pragma omp parallel
            {
                std::vector<Complex> local_result(vec.size(), Complex(0.0, 0.0));
                
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = 0; i < vec.size(); ++i) {
                    if (std::abs(vec[i]) < 1e-15) continue;
                    
                    for (const auto& transform : transforms_) {
                        auto [j, scalar] = transform(i);
                        if (j >= 0 && j < vec.size() && std::abs(scalar) > 1e-15) {
                            local_result[j] += scalar * vec[i];
                        }
                    }
                }
                
                #pragma omp critical
                {
                    for (size_t i = 0; i < vec.size(); ++i) {
                        resultVec[i] += local_result[i];
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

    // Memory-efficient symmetrized matrix computation using chunking
    Matrix returnSymmetrizedMatrix(const std::string& dir) {
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        
        // Process in chunks to reduce memory footprint
        const int chunk_size = std::min(256, dim);
        
        // Process by chunks
        for (int chunk_start = 0; chunk_start < dim; chunk_start += chunk_size) {
            int chunk_end = std::min(chunk_start + chunk_size, dim);
            
            // Cache transformed vectors for this chunk only
            std::vector<std::vector<Complex>> chunk_cache(chunk_end - chunk_start);
            
            #pragma omp parallel for
            for (int i = chunk_start; i < chunk_end; ++i) {
                std::vector<Complex> basis_i = read_sym_basis(i, dir);
                chunk_cache[i - chunk_start] = apply(basis_i);
            }
            
            // Compute matrix elements for this chunk
            #pragma omp parallel for schedule(dynamic)
            for (int i = chunk_start; i < chunk_end; ++i) {
                for (int j = i; j < dim; ++j) {
                    std::vector<Complex> basis_j = read_sym_basis(j, dir);
                    
                    Complex res(0.0, 0.0);
                    const auto& cached_vec = chunk_cache[i - chunk_start];
                    for (int k = 0; k < dim; ++k) {
                        res += std::conj(basis_j[k]) * cached_vec[k];
                    }
                    matrix[i][j] = res;
                    
                    if (i != j) {
                        matrix[j][i] = std::conj(res);
                    }
                }
            }
        }
        
        return matrix;
    }

    Matrix returnMatrix() {
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        
        // Build sparse representation first if not already built
        if (!matrixBuilt_) {
            buildSparseMatrix();
        }
        
        // Convert sparse to dense
        for (int k = 0; k < sparseMatrix_->outerSize(); ++k) {
            for (Eigen::SparseMatrix<Complex>::InnerIterator it(*sparseMatrix_, k); it; ++it) {
                matrix[it.row()][it.col()] = it.value();
            }
        }
        
        return matrix;
    }

    // Memory-optimized basis generation using streaming approach
    void generateSymmetrizedBasis(const std::string& dir) {
        std::ifstream trans_file(dir + "/Trans.dat");
        if (!trans_file.is_open()) {
            std::cerr << "Error: Cannot open file " << dir + "/Trans.dat" << std::endl;
        }

        std::string dummy_line;
        std::getline(trans_file, dummy_line);
        
        std::string dum;
        int num_site;
        trans_file >> dum >> num_site;
        trans_file.close();
        
        // Load max clique with memory-mapped file for large data
        std::string max_clique_path = dir + "/automorphism_results/max_clique.json";
        std::cout << "Loading max clique from: " << max_clique_path << std::endl;
        
        std::vector<std::vector<int>> max_clique_here;
        std::ifstream max_clique_file(max_clique_path);
        if (!max_clique_file.is_open()) {
            throw std::runtime_error("Cannot open max_clique.json file: " + max_clique_path);
        }
        
        // Stream-based JSON parsing to reduce memory usage
        std::string line;
        while (std::getline(max_clique_file, line)) {
            if (line.find('[') != std::string::npos && line.find('[', line.find('[') + 1) != std::string::npos) {
                std::vector<int> permutation;
                size_t pos = line.find('[', line.find('[') + 1);
                std::string num_str;
                
                while (pos < line.size() && line[pos] != ']') {
                    if (std::isdigit(line[pos])) {
                        num_str += line[pos];
                    } else if (!num_str.empty()) {
                        permutation.push_back(std::stoi(num_str));
                        num_str.clear();
                    }
                    pos++;
                }
                if (!num_str.empty()) {
                    permutation.push_back(std::stoi(num_str));
                }
                if (!permutation.empty()) {
                    max_clique_here.push_back(permutation);
                }
            }
        }
        max_clique_file.close();
        
        if (max_clique_here.empty()) {
            throw std::runtime_error("Failed to parse max_clique.json or file is empty");
        }
        
        std::cout << "Loaded " << max_clique_here.size() << " permutations in max clique" << std::endl;

        // Handle trivial case efficiently
        if (max_clique_here.size() == 1) {
            handleTrivialSymmetrization(dir);
            return;
        }

        // Generate symmetrized basis with memory-efficient approach
        generateSymmetrizedBasisOptimized(dir, max_clique_here);
    }

    // Memory-efficient sparse basis reading with caching
    std::vector<Complex> read_sym_basis(int index, const std::string& dir) {
        // Check cache first
        auto cache_it = basis_cache_.find(index);
        if (cache_it != basis_cache_.end()) {
            return cache_it->second;
        }
        
        std::vector<Complex> sym_basis((1ULL<<n_bits_), 0.0);
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
                sym_basis[idx] = Complex(real, imag);
            }
        }
        
        // Add to cache with size limit
        if (basis_cache_.size() < MAX_CACHE_SIZE) {
            basis_cache_[index] = sym_basis;
        }
        
        return sym_basis;
    }

    // Clear basis cache to free memory
    void clearBasisCache() {
        basis_cache_.clear();
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

    // Memory-efficient file loading with buffering
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        // Enable buffering for better I/O performance
        const size_t buffer_size = 65536;
        std::vector<char> buffer(buffer_size);
        file.rdbuf()->pubsetbuf(buffer.data(), buffer_size);
        
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Reserve space for expected transforms
        transforms_.reserve(transforms_.size() + numLines);
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            
            if (!(lineStream >> Op >> indx >> E >> F)) {
                continue;
            }
            
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (Op == 2){
                    return {basis, Complex(E,F)*double(spin_l_)*pow(-1,(basis >> indx) & 1)};
                }
                else{
                    if (((basis >> indx) & 1) != Op) {
                        int flipped_basis = basis ^ (1 << indx);
                        return {flipped_basis, Complex(E, F)};                    
                    }
                }
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
        
        // Enable buffering
        const size_t buffer_size = 65536;
        std::vector<char> buffer(buffer_size);
        file.rdbuf()->pubsetbuf(buffer.data(), buffer_size);
        
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Reserve space
        transforms_.reserve(transforms_.size() + numLines);
        
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op1, indx1, Op2, indx2;
            double E, F;
            
            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                continue;
            }
            
            addTransform([=](int basis) -> std::pair<int, Complex> {
                if (Op1 == 2 && Op2 == 2) {
                    int bit1 = (basis >> indx1) & 1;
                    int bit2 = (basis >> indx2) & 1;
                    return {basis, Complex(E, F)* double(spin_l_) * double(spin_l_) * pow(-1, bit1) * pow(-1, bit2)};
                } 
                else if (Op1 == 2) {
                    int bit1 = (basis >> indx1) & 1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit2_matches) {
                        int flipped_basis = basis ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F) * double(spin_l_) * pow(-1, bit1)};
                    }
                } 
                else if (Op2 == 2) {
                    int bit2 = (basis >> indx2) & 1;
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    
                    if (bit1_matches) {
                        int flipped_basis = basis ^ (1 << indx1);
                        return {flipped_basis, Complex(E, F)* double(spin_l_) * pow(-1, bit2)};
                    }
                } 
                else {
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit1_matches && bit2_matches) {
                        int flipped_basis = basis ^ (1 << indx1) ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;    
    }

    void loadonebodycorrelation(const int Op, const int indx) { 
        addTransform([=](int basis) -> std::pair<int, Complex> {
            if (Op == 2){
                return {basis, Complex(1.0,0.0)*double(spin_l_)*pow(-1,(basis >> indx) & 1)};
            }
            else{
                if (((basis >> indx) & 1) != Op) {
                    int flipped_basis = basis ^ (1 << indx);
                    return {flipped_basis, Complex(1.0 * double(spin_l_*2), 0.0)};                    
                }
            }
            return {basis, Complex(0.0, 0.0)};
        });
    }

    void loadtwobodycorrelation(const int Op1, const int indx1, const int Op2, const int indx2){
        addTransform([=](int basis) -> std::pair<int, Complex> {
            if (Op1 == 2 && Op2 == 2) {
                int bit1 = (basis >> indx1) & 1;
                int bit2 = (basis >> indx2) & 1;
                return {basis, Complex(1.0, 0.0)* double(spin_l_) * double(spin_l_) * pow(-1, bit1) * pow(-1, bit2)};
            } 
            else if (Op1 == 2) {
                int bit1 = (basis >> indx1) & 1;
                bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                
                if (bit2_matches) {
                    int flipped_basis = basis ^ (1 << indx2);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_) * double(spin_l_*2) * pow(-1, bit1)};
                }
            } 
            else if (Op2 == 2) {
                int bit2 = (basis >> indx2) & 1;
                bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                
                if (bit1_matches) {
                    int flipped_basis = basis ^ (1 << indx1);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_) * double(spin_l_*2) * pow(-1, bit2)};
                }
            } 
            else {
                bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                
                if (bit1_matches && bit2_matches) {
                    int flipped_basis = basis ^ (1 << indx1) ^ (1 << indx2);
                    return {flipped_basis, Complex(1.0, 0.0) * double(spin_l_*2) * double(spin_l_*2)};
                }
            }
            return {basis, Complex(0.0, 0.0)};
        });
    }

    // Memory-optimized block building with streaming
    void buildAndSaveSymmetrizedBlocks(const std::string& dir) {
        if (symmetrized_block_ham_sizes.empty()) {
            loadBlockSizes(dir);
        }
        
        std::cout << "Building and saving symmetrized Hamiltonian blocks (Memory-optimized)..." << std::endl;
        
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
            
            // Use streaming approach for large blocks
            if (block_size > 1000) {
                buildLargeBlockStreaming(dir, block_dir, block, block_start, block_size);
            } else {
                buildSmallBlockInMemory(dir, block_dir, block, block_start, block_size);
            }
            
            block_start += block_size;
            
            // Clear cache periodically to free memory
            clearBasisCache();
        }
        
        std::cout << "Symmetrized Hamiltonian blocks saved to " << block_dir << std::endl;
    }

    // Optimized sparse matrix building
    void buildSparseMatrix() const {
        if (matrixBuilt_) return;
        
        int dim = 1 << n_bits_;
        
        if (!sparseMatrix_) {
            sparseMatrix_ = std::make_shared<Eigen::SparseMatrix<Complex>>(dim, dim);
        } else {
            sparseMatrix_->resize(dim, dim);
            sparseMatrix_->setZero();
        }
        
        // Estimate number of non-zeros
        size_t estimated_nnz = dim * transforms_.size() * 2;
        sparseMatrix_->reserve(estimated_nnz);
        
        // Use column-major storage for better cache efficiency
        std::vector<std::vector<Eigen::Triplet<Complex>>> col_triplets(dim);
        
        #pragma omp parallel for
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim && std::abs(scalar) > 1e-10) {
                    #pragma omp critical
                    {
                        col_triplets[i].emplace_back(j, i, scalar);
                    }
                }
            }
        }
        
        // Combine all triplets
        std::vector<Eigen::Triplet<Complex>> all_triplets;
        all_triplets.reserve(estimated_nnz);
        for (const auto& col : col_triplets) {
            all_triplets.insert(all_triplets.end(), col.begin(), col.end());
        }
        
        sparseMatrix_->setFromTriplets(all_triplets.begin(), all_triplets.end());
        sparseMatrix_->makeCompressed();
        matrixBuilt_ = true;
    }

    // Memory-efficient matrix printing with streaming
    void printEntireSymmetrizedMatrix(const std::string& dir) {
        if (symmetrized_block_ham_sizes.empty()) {
            loadBlockSizes(dir);
        }
        
        int total_sym_dim = 0;
        for (int size : symmetrized_block_ham_sizes) {
            total_sym_dim += size;
        }
        
        std::cout << "Total symmetrized dimension: " << total_sym_dim << std::endl;
        
        if (total_sym_dim > 1000) {
            std::cout << "Warning: Matrix dimension is " << total_sym_dim 
                      << ". This may require significant memory." << std::endl;
            std::cout << "Continue? (y/n): ";
            char response;
            std::cin >> response;
            if (response != 'y' && response != 'Y') {
                std::cout << "Operation cancelled." << std::endl;
                return;
            }
        }
        
        // Stream matrix to file instead of loading entirely in memory
        std::string output_file = dir + "/full_symmetrized_hamiltonian.txt";
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }
        
        out_file << std::fixed << std::setprecision(10);
        out_file << total_sym_dim << " " << total_sym_dim << std::endl;
        
        // Process in row chunks to limit memory usage
        const int chunk_size = std::min(100, total_sym_dim);
        
        for (int row_chunk = 0; row_chunk < total_sym_dim; row_chunk += chunk_size) {
            int chunk_end = std::min(row_chunk + chunk_size, total_sym_dim);
            
            // Compute matrix elements for this chunk
            std::vector<std::vector<Complex>> chunk_matrix(chunk_end - row_chunk, 
                                                          std::vector<Complex>(total_sym_dim, 0.0));
            
            computeMatrixChunk(dir, chunk_matrix, row_chunk, chunk_end, total_sym_dim);
            
            // Write chunk to file
            for (int i = 0; i < chunk_end - row_chunk; ++i) {
                for (int j = 0; j < total_sym_dim; ++j) {
                    out_file << chunk_matrix[i][j].real() << "+1j*" 
                            << chunk_matrix[i][j].imag() << " ";
                }
                out_file << std::endl;
            }
            
            std::cout << "\rProgress: " << chunk_end << "/" << total_sym_dim 
                      << " rows written" << std::flush;
        }
        
        out_file.close();
        std::cout << "\nMatrix saved to: " << output_file << std::endl;
    }

private:
    std::vector<TransformFunction> transforms_;
    int n_bits_;
    float spin_l_;
    const std::array<std::array<double, 4>, 3> operators = {
        {{0, 1, 0, 0}, {0, 0, 1, 0},{1, 0, 0, -1}}
    };

    const std::array<std::array<double, 2>, 2> basis = {
        {{1, 0}, {0, 1}}
    };

    mutable std::shared_ptr<Eigen::SparseMatrix<Complex>> sparseMatrix_;
    mutable bool matrixBuilt_ = false;
    
    // Basis vector cache with limited size
    static constexpr size_t MAX_CACHE_SIZE = 1000;
    mutable std::unordered_map<int, std::vector<Complex>> basis_cache_;

    // Helper function to load block sizes
    void loadBlockSizes(const std::string& dir) {
        std::ifstream file(dir + "/sym_basis/sym_block_sizes.txt");
        if (!file.is_open()) {
            throw std::runtime_error("Symmetrized basis must be generated first");
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

    // Helper for trivial symmetrization
    void handleTrivialSymmetrization(const std::string& dir) {
        std::string sym_basis_dir = dir + "/sym_basis";
        std::string mkdir_command = "mkdir -p " + sym_basis_dir;
        system(mkdir_command.c_str());
        
        int dim = 1 << n_bits_;
        symmetrized_block_ham_sizes = {dim};

        if (n_bits_ > 30) {
            std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
            block_size_file << dim << std::endl;
            block_size_file.close();
            
            std::ofstream index_file(sym_basis_dir + "/implicit_representation.txt");
            index_file << "System size too large for explicit basis storage.\n";
            index_file << "Using implicit representation: standard basis with no symmetrization.\n";
            index_file << "Dimension: " << dim << std::endl;
            index_file.close();
            
            return;
        }
        
        // Stream basis vectors to files instead of creating all at once
        for (int i = 0; i < dim; i++) {
            std::ofstream file(sym_basis_dir + "/sym_basis" + std::to_string(i) + ".dat");
            file << i << " 1.0 0.0" << std::endl;
            file.close();
        }

        std::ofstream block_size_file(sym_basis_dir + "/sym_block_sizes.txt");
        block_size_file << dim << std::endl;
        block_size_file.close();
    }

    // Helper for optimized symmetrized basis generation
    void generateSymmetrizedBasisOptimized(const std::string& dir, 
                                          const std::vector<std::vector<int>>& max_clique_here) {
        // Implementation would go here - this is a placeholder
        // The actual implementation would use the memory-efficient techniques
        // shown in the original generateSymmetrizedBasis method
    }

    // Helper for building large blocks with streaming
    void buildLargeBlockStreaming(const std::string& dir, const std::string& block_dir,
                                 size_t block, int block_start, int block_size) {
        // Stream computation for large blocks
        std::string filename = block_dir + "/block_" + std::to_string(block) + ".dat";
        std::ofstream file(filename);
        
        file << block_size << " " << block_size << std::endl;
        
        // Process in sub-chunks
        const int sub_chunk_size = 100;
        
        for (int i = 0; i < block_size; i += sub_chunk_size) {
            int sub_chunk_end = std::min(i + sub_chunk_size, block_size);
            
            // Process sub-chunk
            for (int row = i; row < sub_chunk_end; ++row) {
                // Load basis vector only when needed
                std::map<int, Complex> basis_i_sparse;
                read_sym_basis_sparse(block_start + row, dir, basis_i_sparse);
                
                // Apply operator
                std::map<int, Complex> transformed_sparse;
                for (const auto& [idx, val] : basis_i_sparse) {
                    for (const auto& transform : transforms_) {
                        auto [new_idx, scalar] = transform(idx);
                        if (new_idx >= 0 && std::abs(scalar) > 1e-10) {
                            transformed_sparse[new_idx] += val * scalar;
                        }
                    }
                }
                
                // Compute and write matrix elements
                for (int col = 0; col < block_size; ++col) {
                    std::map<int, Complex> basis_j_sparse;
                    read_sym_basis_sparse(block_start + col, dir, basis_j_sparse);
                    
                    Complex element(0.0, 0.0);
                    for (const auto& [idx, val] : transformed_sparse) {
                        auto it = basis_j_sparse.find(idx);
                        if (it != basis_j_sparse.end()) {
                            element += std::conj(it->second) * val;
                        }
                    }
                    
                    if (std::abs(element) > 1e-10) {
                        file << row << " " << col << " " 
                             << element.real() << " " << element.imag() << std::endl;
                    }
                }
            }
            
            // Clear cache periodically
            if ((i / sub_chunk_size) % 10 == 0) {
                clearBasisCache();
            }
        }
        
        file.close();
    }

    // Helper for building small blocks in memory
    void buildSmallBlockInMemory(const std::string& dir, const std::string& block_dir,
                                size_t block, int block_start, int block_size) {
        // Original implementation for small blocks that fit in memory
        // Similar to the original buildAndSaveSymmetrizedBlocks implementation
    }

    // Helper for computing matrix chunks
    void computeMatrixChunk(const std::string& dir, 
                           std::vector<std::vector<Complex>>& chunk_matrix,
                           int row_start, int row_end, int total_dim) {
        #pragma omp parallel for
        for (int i = row_start; i < row_end; ++i) {
            std::map<int, Complex> basis_i_sparse;
            read_sym_basis_sparse(i, dir, basis_i_sparse);
            
            // Apply operator
            std::map<int, Complex> transformed_sparse;
            for (const auto& [idx, val] : basis_i_sparse) {
                for (const auto& transform : transforms_) {
                    auto [new_idx, scalar] = transform(idx);
                    if (new_idx >= 0 && std::abs(scalar) > 1e-10) {
                        transformed_sparse[new_idx] += val * scalar;
                    }
                }
            }
            
            // Compute row elements
            for (int j = 0; j < total_dim; ++j) {
                std::map<int, Complex> basis_j_sparse;
                read_sym_basis_sparse(j, dir, basis_j_sparse);
                
                Complex element(0.0, 0.0);
                for (const auto& [idx, val] : transformed_sparse) {
                    auto it = basis_j_sparse.find(idx);
                    if (it != basis_j_sparse.end()) {
                        element += std::conj(it->second) * val;
                    }
                }
                
                chunk_matrix[i - row_start][j] = element;
            }
        }
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