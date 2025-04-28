import numpy as np
import networkx as nx
from collections import defaultdict, deque
import itertools
import time
import os
import argparse
from typing import List, Tuple, Dict, Set, Optional

#!/usr/bin/env python3
"""
This module implements an efficient algorithm for finding automorphisms of quantum Hamiltonians.
It provides a significant speed improvement over brute force approaches that check all permutations.
"""



class HamiltonianGraph:
    """
    Represents a quantum Hamiltonian as a colored graph to find its automorphisms.
    """
    
    def __init__(self, n_sites: int):
        """
        Initialize a Hamiltonian graph with n_sites vertices.
        
        Args:
            n_sites: Number of sites in the quantum system
        """
        self.n_sites = n_sites
        self.graph = nx.Graph()
        self.site_operators = []  # List of (site, op, weight) tuples
        self.interactions = []    # List of (site1, op1, site2, op2, weight) tuples
        
        # Add all vertices
        for i in range(n_sites):
            self.graph.add_node(i, type='site')
    
    def load_edges_from_file(self, filename: str) -> None:
        """
        Load interaction terms from a file.
        
        Args:
            filename: Path to the interaction file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Parse num_interactions from second line
            num_interactions = int(lines[1].split()[1])
            
            # Skip header lines
            data_lines = lines[4:4+num_interactions]
            
            for line in data_lines:
                tokens = line.strip().split()
                if len(tokens) >= 6:
                    op1 = int(tokens[0])
                    site1 = int(tokens[1])
                    op2 = int(tokens[2])
                    site2 = int(tokens[3])
                    real = float(tokens[4])
                    imag = float(tokens[5])
                    weight = complex(real, imag)
                    
                    self.interactions.append((site1, op1, site2, op2, weight))
                    
                    # Add edge to graph with attributes
                    if site1 != site2:  # Avoid self-loops in the graph
                        self.graph.add_edge(site1, site2, 
                                           op1=op1, op2=op2, 
                                           weight=weight,
                                           type='interaction')
    
    def load_vertices_from_file(self, filename: str) -> None:
        """
        Load site operators from a file.
        
        Args:
            filename: Path to the vertex operator file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Parse num_vertices from second line
            num_vertices = int(lines[1].split()[1])
            
            # Skip header lines
            data_lines = lines[4:4+num_vertices]
            
            for line in data_lines:
                tokens = line.strip().split()
                if len(tokens) >= 4:
                    op = int(tokens[0])
                    site = int(tokens[1])
                    real = float(tokens[2])
                    imag = float(tokens[3])
                    weight = complex(real, imag)
                    
                    self.site_operators.append((site, op, weight))
                    
                    # Add attributes to existing node
                    self.graph.nodes[site]['op'] = op
                    self.graph.nodes[site]['weight'] = weight
    
    def get_initial_vertex_coloring(self) -> Dict[int, Tuple]:
        """
        Generate initial colors for vertices based on their properties.
        
        Returns:
            Dictionary mapping vertex indices to color tuples
        """
        colors = {}
        for i in range(self.n_sites):
            node = self.graph.nodes[i]
            
            # Get all site operators for this vertex
            site_ops = []
            for site, op, weight in self.site_operators:
                if site == i:
                    site_ops.append((op, weight))
            
            # Sort for consistent representation
            site_ops.sort()
            
            # Get all interactions for this vertex
            interactions = []
            for site1, op1, site2, op2, weight in self.interactions:
                if site1 == i:
                    interactions.append((op1, site2, op2, weight))
                elif site2 == i:
                    interactions.append((op2, site1, op1, weight))
            
            # Sort for consistent representation
            interactions.sort()
            
            # Degree is also an invariant property
            degree = self.graph.degree(i)
            
            # Color tuple includes all invariant properties
            colors[i] = (tuple(site_ops), tuple(interactions), degree)
            
        return colors
    
    def refine_coloring(self, coloring: Dict[int, Tuple]) -> Dict[int, Tuple]:
        """
        Refine vertex coloring by incorporating neighbor information.
        
        Args:
            coloring: Current vertex coloring
            
        Returns:
            Refined vertex coloring
        """
        new_coloring = {}
        
        for i in range(self.n_sites):
            # Get multiset of neighbor colors
            neighbor_colors = []
            for neighbor in self.graph.neighbors(i):
                # Include edge properties in the neighborhood color
                edge = self.graph[i][neighbor]
                neighbor_colors.append((coloring[neighbor], edge.get('op1'), edge.get('op2'), edge.get('weight')))
            
            # Sort for consistent representation
            neighbor_colors.sort()
            
            # New color combines old color with neighbor information
            new_coloring[i] = (coloring[i], tuple(neighbor_colors))
            
        return new_coloring
    
    def get_stable_coloring(self) -> Dict[int, int]:
        """
        Compute a stable coloring of vertices through iterative refinement.
        
        Returns:
            Dictionary mapping vertices to their color classes (integers)
        """
        # Get initial coloring
        coloring = self.get_initial_vertex_coloring()
        
        # Iteratively refine until stable
        while True:
            new_coloring = self.refine_coloring(coloring)
            
            # Check if coloring has stabilized
            if set(new_coloring.values()) == set(coloring.values()):
                break
                
            coloring = new_coloring
        
        # Map complex color tuples to simple integers
        color_to_int = {}
        final_coloring = {}
        
        for vertex, color in coloring.items():
            if color not in color_to_int:
                color_to_int[color] = len(color_to_int)
            final_coloring[vertex] = color_to_int[color]
            
        return final_coloring
    
    def is_automorphism(self, permutation: List[int]) -> bool:
        """
        Check if a permutation is an automorphism of the Hamiltonian.
        
        Args:
            permutation: List representing the permutation mapping
            
        Returns:
            True if the permutation is an automorphism, False otherwise
        """
        # Check site operators
        for site, op, weight in self.site_operators:
            permuted_site = permutation[site]
            
            # Find matching operator on permuted site
            found = False
            for other_site, other_op, other_weight in self.site_operators:
                if other_site == permuted_site and other_op == op and abs(other_weight - weight) < 1e-10:
                    found = True
                    break
            
            if not found:
                return False
        
        # Check interactions
        for site1, op1, site2, op2, weight in self.interactions:
            permuted_site1 = permutation[site1]
            permuted_site2 = permutation[site2]
            
            # Find matching interaction on permuted sites
            found = False
            for other_site1, other_op1, other_site2, other_op2, other_weight in self.interactions:
                # Check both orientations since interactions can be undirected
                matches1 = (other_site1 == permuted_site1 and 
                            other_site2 == permuted_site2 and
                            other_op1 == op1 and 
                            other_op2 == op2 and
                            abs(other_weight - weight) < 1e-10)
                            
                matches2 = (other_site1 == permuted_site2 and 
                            other_site2 == permuted_site1 and
                            other_op1 == op2 and 
                            other_op2 == op1 and
                            abs(other_weight - weight) < 1e-10)
                            
                if matches1 or matches2:
                    found = True
                    break
            
            if not found:
                return False
        
        return True
    
    def find_automorphisms(self) -> List[List[int]]:
        """
        Find all automorphisms of the Hamiltonian using an efficient algorithm.
        
        Returns:
            List of automorphisms, where each automorphism is a permutation list
        """
        # Get stable coloring
        coloring = self.get_stable_coloring()
        
        # Group vertices by color
        color_classes = defaultdict(list)
        for vertex, color in coloring.items():
            color_classes[color].append(vertex)
        
        # Generate candidate permutations that respect coloring
        candidates = []
        
        # Use backtracking to generate valid permutations
        def backtrack(perm, used, level):
            if level == self.n_sites:
                # Complete permutation
                candidates.append(perm.copy())
                return
            
            # Get color of current position
            current_color = coloring[level]
            
            # Try all vertices with matching color
            for v in color_classes[current_color]:
                if not used[v]:
                    perm[level] = v
                    used[v] = True
                    backtrack(perm, used, level + 1)
                    used[v] = False
        
        perm = [0] * self.n_sites
        used = [False] * self.n_sites
        backtrack(perm, used, 0)
        
        # Filter candidates to find actual automorphisms
        automorphisms = []
        for perm in candidates:
            if self.is_automorphism(perm):
                automorphisms.append(perm)
        
        return automorphisms


def permutation_to_cycle_notation(perm: List[int]) -> str:
    """
    Convert a permutation to cycle notation.
    
    Args:
        perm: Permutation list
        
    Returns:
        String representation in cycle notation
    """
    visited = [False] * len(perm)
    result = []
    
    for i in range(len(perm)):
        if visited[i] or perm[i] == i:
            continue
        
        cycle = []
        j = i
        while not visited[j]:
            cycle.append(j)
            visited[j] = True
            j = perm[j]
            if j == i:
                break
        
        if cycle:
            result.append('(' + ' '.join(map(str, cycle)) + ')')
    
    # Add fixed points
    for i in range(len(perm)):
        if perm[i] == i:
            result.append(f'({i})')
    
    if not result:
        return '()'  # Identity permutation
        
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description='Find automorphisms of a quantum Hamiltonian')
    parser.add_argument('--interactions', '-i', type=str, required=True,
                        help='Path to file containing interaction terms')
    parser.add_argument('--vertices', '-v', type=str, required=True,
                        help='Path to file containing site operators')
    parser.add_argument('--nsites', '-n', type=int, required=True,
                        help='Number of sites in the system')
    parser.add_argument('--output', '-o', type=str, default='automorphisms.txt',
                        help='Output file for automorphisms')
    
    args = parser.parse_args()
    
    # Create Hamiltonian graph
    ham_graph = HamiltonianGraph(args.nsites)
    
    # Load data from files
    print(f"Loading interaction terms from {args.interactions}")
    ham_graph.load_edges_from_file(args.interactions)
    
    print(f"Loading site operators from {args.vertices}")
    ham_graph.load_vertices_from_file(args.vertices)
    
    # Find automorphisms
    print("Finding automorphisms...")
    start_time = time.time()
    automorphisms = ham_graph.find_automorphisms()
    end_time = time.time()
    
    print(f"Found {len(automorphisms)} automorphisms in {end_time - start_time:.2f} seconds")
    
    # Save results
    with open(args.output, 'w') as f:
        f.write(f"# Automorphisms of a Hamiltonian with {args.nsites} sites\n")
        f.write(f"# Total automorphisms found: {len(automorphisms)}\n\n")
        
        for i, auto in enumerate(automorphisms):
            cycle_notation = permutation_to_cycle_notation(auto)
            f.write(f"Automorphism {i+1}: {auto} = {cycle_notation}\n")
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()