import numpy as np
from pynauty import Graph, autgrp
from collections import defaultdict
import argparse
import os
import subprocess
import json
import networkx as nx
from itertools import combinations


def read_trans_file(filename):
    """Read vertex weights from Trans.dat file."""
    vertex_weights = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip header lines
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('num'):
                parts = line.split()
                if len(parts) >= 4:
                    vertex_type = int(parts[0])
                    vertex_id = int(parts[1])
                    weight_real = float(parts[2])
                    weight_imag = float(parts[3])
                    vertex_weights[vertex_id] = (vertex_type, weight_real, weight_imag)
    return vertex_weights

def read_interall_file(filename):
    """Read edge information from InterAll.dat file."""
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip header lines
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('num'):
                parts = line.split()
                if len(parts) >= 6:
                    type1 = int(parts[0])
                    vertex1 = int(parts[1])
                    type2 = int(parts[2])
                    vertex2 = int(parts[3])
                    weight_real = float(parts[4])
                    weight_imag = float(parts[5])
                    edges.append({
                        'vertex1': vertex1,
                        'vertex2': vertex2,
                        'type1': type1,
                        'type2': type2,
                        'weight': (weight_real, weight_imag)
                    })
    return edges

def compute_vertex_colors(vertex_weights, edges):
    """Compute vertex colors based on local environment."""
    # Create adjacency list with edge weights
    adj_list = defaultdict(list)
    for edge in edges:
        v1, v2 = edge['vertex1'], edge['vertex2']
        weight = edge['weight']
        adj_list[v1].append((v2, weight))
        adj_list[v2].append((v1, weight))
    
    # Compute color for each vertex based on its local environment
    vertex_environments = {}
    for vertex in vertex_weights:
        # Get vertex's own properties
        own_props = vertex_weights[vertex]
        
        # Get properties of neighbors
        neighbor_props = []
        for neighbor, edge_weight in sorted(adj_list[vertex]):
            if neighbor in vertex_weights:
                neighbor_props.append((vertex_weights[neighbor], edge_weight))
        
        # Create a hashable representation of the local environment
        env = (own_props, tuple(sorted(neighbor_props)))
        vertex_environments[vertex] = env
    
    # Assign colors to unique environments
    unique_envs = list(set(vertex_environments.values()))
    env_to_color = {env: i for i, env in enumerate(unique_envs)}
    
    vertex_colors = {}
    for vertex, env in vertex_environments.items():
        vertex_colors[vertex] = env_to_color[env]
    
    return vertex_colors

def construct_colored_graph(vertex_weights, edges):
    """Construct a colored undirected graph for pynauty."""
    # Get vertex colors
    vertex_colors = compute_vertex_colors(vertex_weights, edges)
    
    # Find the number of vertices
    num_vertices = len(vertex_weights)
    
    # Create adjacency dictionary
    adjacency_dict = defaultdict(set)
    for edge in edges:
        v1, v2 = edge['vertex1'], edge['vertex2']
        if v1 != v2:  # Avoid self-loops
            adjacency_dict[v1].add(v2)
            adjacency_dict[v2].add(v1)
    
    # Convert to pynauty format
    adjacency_dict = {v: list(neighbors) for v, neighbors in adjacency_dict.items()}
    
    # Create coloring list
    coloring = []
    color_to_vertices = defaultdict(list)
    for vertex, color in vertex_colors.items():
        color_to_vertices[color].append(vertex)
    
    for color in sorted(color_to_vertices.keys()):
        coloring.append(set(color_to_vertices[color]))
    
    # Create pynauty graph
    g = Graph(num_vertices, directed=False, adjacency_dict=adjacency_dict, vertex_coloring=coloring)
    
    return g, vertex_colors


class AutomorphismCliqueAnalyzer:
    """Class for analyzing cliques of compatible automorphisms"""
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def do_permutations_commute(self, perm1, perm2):
        """Check if two permutations commute"""
        if len(perm1) != len(perm2):
            return False
        
        # Check if perm1 ∘ perm2 = perm2 ∘ perm1
        n = len(perm1)
        
        for i in range(n):
            # perm1(perm2[i]) should equal perm2(perm1[i])
            if perm1[perm2[i]] != perm2[perm1[i]]:
                return False
                
        return True
    
    def build_commutation_graph(self, automorphisms):
        """Build a graph where nodes are automorphisms and edges connect commuting pairs"""
        G = nx.Graph()
        
        # Add nodes for each automorphism
        for i, auto in enumerate(automorphisms):
            G.add_node(i, perm=auto)
        
        # Add edges between commuting automorphisms
        for i in range(len(automorphisms)):
            for j in range(i+1, len(automorphisms)):
                if self.do_permutations_commute(automorphisms[i], automorphisms[j]):
                    G.add_edge(i, j)
        
        return G
    
    def find_maximum_clique(self, automorphisms):
        """Find the maximum clique of commuting automorphisms using NetworkX"""
        # Build the commutation graph
        comm_graph = self.build_commutation_graph(automorphisms)
        
        # Use NetworkX to find the maximum clique
        max_clique_indices = list(nx.find_cliques(comm_graph))
        
        # Get the maximum clique by size
        if max_clique_indices:
            max_clique = max(max_clique_indices, key=len)
            return max_clique
        else:
            return []
    
    def generate_automorphism_graph(self, automorphisms, filename, finder):
        """Generate a DOT file visualizing the automorphism commutation graph"""
        comm_graph = self.build_commutation_graph(automorphisms)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Generate DOT file
        dot_file = f"{filename}/automorphism_graph.dot"
        with open(dot_file, 'w') as f:
            f.write("graph AutomorphismGraph {\n")
            f.write("  node [shape=box];\n")
            
            # Add nodes
            for i, auto in enumerate(automorphisms):
                cycles = finder.permutation_to_cycle_notation(auto)
                label = ' '.join(str(c) for c in cycles)
                f.write(f'  {i} [label="{label}"];\n')
            
            # Add edges
            for i, j in comm_graph.edges():
                f.write(f"  {i} -- {j};\n")
                
            f.write("}\n")
        
        print(f"Generated automorphism graph: {dot_file}")
        print(f"Visualize with: dot -Tpng {dot_file} -o {filename}/automorphism_graph.png")
        
        # Try to generate the image directly
        self.save_graph_image(f"{filename}/automorphism_graph.png", dot_file)
        
    def visualize_clique(self, automorphisms, clique, filename, finder):
        """Generate a visualization highlighting a specific clique in the graph"""
        comm_graph = self.build_commutation_graph(automorphisms)
        
        # Generate DOT file
        with open(filename, 'w') as f:
            f.write("graph AutomorphismClique {\n")
            f.write("  node [shape=box];\n")
            
            # Create set for fast lookup
            clique_set = set(clique)
            
            # Add nodes
            for i, auto in enumerate(automorphisms):
                cycles = finder.permutation_to_cycle_notation(auto)
                label = ' '.join(str(c) for c in cycles)
                
                if i in clique_set:
                    # Highlight clique members
                    f.write(f'  {i} [label="{label}", style=filled, fillcolor=lightblue];\n')
                else:
                    f.write(f'  {i} [label="{label}"];\n')
            
            # Add edges
            for i, j in comm_graph.edges():
                if i in clique_set and j in clique_set:
                    # Highlight edges within the clique
                    f.write(f"  {i} -- {j} [color=blue, penwidth=2.0];\n")
                else:
                    f.write(f"  {i} -- {j};\n")
                
            f.write("}\n")
        
        print(f"Generated clique visualization: {filename}")
        print(f"Visualize with: dot -Tpng {filename} -o clique_visualization.png")
        
        # Try to generate the image directly
        self.save_graph_image("clique_visualization.png", filename)
        
    def save_graph_image(self, output_file, dot_file):
        """Generate PNG image from DOT file using Graphviz"""
        try:
            subprocess.run(['dot', '-Tpng', dot_file, '-o', output_file], check=True)
            print(f"Generated graph image: {output_file}")
        except subprocess.CalledProcessError:
            print(f"Failed to generate image. Make sure Graphviz is installed.")
        except FileNotFoundError:
            print(f"Graphviz command 'dot' not found. Please install Graphviz.")

class AutomorphismFinder:
    """Class for finding and analyzing automorphisms"""
    
    def __init__(self):
        """Initialize the finder"""
        pass
    
    def generate_all_automorphisms(self, generators):
        """Generate all automorphisms from the given generators"""
        if not generators:
            return [list(range(len(generators[0]) if generators else 0))]
        
        n = len(generators[0])
        identity = list(range(n))
        
        # Start with identity and generators
        automorphisms = [identity]
        automorphisms.extend([list(gen) for gen in generators])
        
        # Generate all possible combinations
        queue = list(automorphisms)
        seen = {tuple(perm) for perm in automorphisms}
        
        while queue:
            perm1 = queue.pop(0)
            
            for gen in generators:
                # Compose perm1 with generator
                new_perm = [perm1[gen[i]] for i in range(n)]
                
                if tuple(new_perm) not in seen:
                    seen.add(tuple(new_perm))
                    automorphisms.append(new_perm)
                    queue.append(new_perm)
        
        return automorphisms
    
    def permutation_to_cycle_notation(self, perm):
        """Convert permutation to cycle notation"""
        n = len(perm)
        visited = [False] * n
        cycles = []
        
        for i in range(n):
            if not visited[i] and perm[i] != i:
                cycle = [i]
                j = perm[i]
                visited[i] = True
                
                while j != i:
                    cycle.append(j)
                    visited[j] = True
                    j = perm[j]
                
                if len(cycle) > 1:
                    cycles.append(tuple(cycle))
        
        return cycles if cycles else [(0,)]
    
    def save_automorphisms_json(self, automorphisms, filename):
        """Save automorphisms to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(automorphisms, f, indent=2)
        
        print(f"Saved {len(automorphisms)} automorphisms to {filename}")


class MaximalAbelianSubgroupFinder:
    """Class for finding minimal generators of abelian subgroups"""
    
    def __init__(self):
        """Initialize the finder"""
        pass
    
    def permutation_order(self, perm):
        """Calculate the order of a permutation"""
        n = len(perm)
        visited = [False] * n
        lcm = 1
        
        for i in range(n):
            if not visited[i]:
                cycle_length = 0
                j = i
                
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_length += 1
                
                if cycle_length > 1:
                    lcm = self._lcm(lcm, cycle_length)
        
        return lcm
    
    def _gcd(self, a, b):
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def _lcm(self, a, b):
        """Calculate least common multiple"""
        return abs(a * b) // self._gcd(a, b)
    
    def compose_permutations(self, perm1, perm2):
        """Compose two permutations: result[i] = perm1[perm2[i]]"""
        return [perm1[perm2[i]] for i in range(len(perm1))]
    
    def is_generated_by(self, element, generators):
        """Check if element can be generated by the given generators"""
        if not generators:
            return element == list(range(len(element)))
        
        n = len(element)
        identity = list(range(n))
        
        # Start with identity
        generated = {tuple(identity)}
        queue = [identity]
        
        while queue:
            current = queue.pop(0)
            
            for gen in generators:
                # Compose with generator
                new_perm = self.compose_permutations(current, gen)
                
                if tuple(new_perm) == tuple(element):
                    return True
                
                if tuple(new_perm) not in generated:
                    generated.add(tuple(new_perm))
                    queue.append(new_perm)
        
        return False
    
    def find_minimal_generators(self, permutations):
        """Find a minimal set of generators for the given abelian group"""
        if not permutations:
            return []
        
        n = len(permutations[0])
        identity = list(range(n))
        
        # Remove identity if present
        non_identity_perms = [p for p in permutations if p != identity]
        
        if not non_identity_perms:
            return []
        
        # Sort by order (smallest first) to prefer simpler generators
        sorted_perms = sorted(non_identity_perms, key=lambda p: self.permutation_order(p))
        
        generators = []
        
        for perm in sorted_perms:
            # Check if this permutation is already generated by current generators
            if not self.is_generated_by(perm, generators):
                generators.append(perm)
        
        # Compute orders of generators
        generator_info = []
        for gen in generators:
            order = self.permutation_order(gen)
            generator_info.append({
                'permutation': gen,
                'order': order,
                'cycles': AutomorphismFinder().permutation_to_cycle_notation(gen)
            })
        
        return generator_info


def main():
    parser = argparse.ArgumentParser(description='Hamiltonian Automorphism Finder')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    args = parser.parse_args()

    inter_all_file = os.path.join(args.data_dir, "InterAll.dat")
    trans_file = os.path.join(args.data_dir, "Trans.dat")

    # Read data files
    vertex_weights = read_trans_file(trans_file)
    edges = read_interall_file(inter_all_file)

    # Construct colored graph
    graph, vertex_colors = construct_colored_graph(vertex_weights, edges)
    
    # Find automorphism group
    aut_group = autgrp(graph)
    
    # Print results
    print("Number of vertices:", len(vertex_weights))
    print("Number of edges:", len(edges))
    print("\nVertex colors:")
    for vertex in sorted(vertex_colors.keys()):
        print(f"  Vertex {vertex}: Color {vertex_colors[vertex]}")
    
    print("\nAutomorphism group information:")
    print(f"  Number of generators: {len(aut_group[0])}")
    print(f"  Order of automorphism group: {aut_group[1]}")
    print(f"  Orbit partition: {aut_group[2]}")
    output_dir = os.path.join(args.data_dir, "automorphism_results")
    os.makedirs(output_dir, exist_ok=True)
    
    finder = AutomorphismFinder()
    automorphisms_file = os.path.join(output_dir, "automorphisms.json")
    # Generate all automorphisms from generators
    all_automorphisms = finder.generate_all_automorphisms(aut_group[0])
    
    # Save all automorphisms to JSON
    finder.save_automorphisms_json(all_automorphisms, automorphisms_file)

    # Initialize the automorphism finder and analyzer
    analyzer = AutomorphismCliqueAnalyzer()

    # Find maximum clique of commuting automorphisms
    max_clique_indices = analyzer.find_maximum_clique(all_automorphisms)
    
    # Save maximum clique to JSON
    max_clique_file = os.path.join(output_dir, "max_clique.json")
    with open(max_clique_file, 'w') as f:
        json.dump([all_automorphisms[i] for i in max_clique_indices], f, indent=2)

    print(f"\nSaved maximum clique of {len(max_clique_indices)} commuting automorphisms to {max_clique_file}")

    # Generate visualization of the automorphism graph
    analyzer.generate_automorphism_graph(all_automorphisms, output_dir, finder)
    
    # Generate visualization highlighting the maximum clique
    clique_dot_file = os.path.join(output_dir, "max_clique_visualization.dot")
    analyzer.visualize_clique(all_automorphisms, max_clique_indices, clique_dot_file, finder)

    max_clique = [all_automorphisms[i] for i in max_clique_indices]

    MaximalAbelianSubgroupFinder().find_minimal_generators(max_clique)
    print("Maximal abelian subgroup generators (if any):")
    generators = MaximalAbelianSubgroupFinder().find_minimal_generators(max_clique)
    for gen in generators:
        print(f"  Generator: {gen}")
    
    # Save generators to JSON
    generators_file = os.path.join(output_dir, "minimal_generators.json")
    with open(generators_file, 'w') as f:
        json.dump(generators, f, indent=2)
    print(f"Saved minimal generators to {generators_file}")

if __name__ == "__main__":
    main()