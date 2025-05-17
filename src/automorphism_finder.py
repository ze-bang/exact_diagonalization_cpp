import numpy as np
import networkx as nx
from itertools import permutations
import os
import subprocess
from collections import deque
import json
import argparse
from networkx.algorithms import isomorphism

#!/usr/bin/env python3
# automorphism_finder.py
# A Python implementation of Hamiltonian symmetry analysis tools

import matplotlib.pyplot as plt


class HamiltonianAutomorphismFinder:
    """Class to find automorphisms of a Hamiltonian represented as a graph"""
    
    def __init__(self, n_sites):
        """Initialize with the number of sites"""
        self.n_sites = n_sites
        self.edges = []
        self.vertices = []
        self.hamiltonian_graph = nx.Graph()
        
    def load_edges_from_file(self, filename):
        """Load edge data from InterAll.dat file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Parse number of interactions
            num_line = lines[1].strip()
            num_interactions = int(num_line.split()[1])
            
            # Skip header lines
            start_idx = 5
            
            # Read each interaction
            for i in range(start_idx, start_idx + num_interactions):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split()
                if len(parts) >= 6:
                    op1 = int(parts[0])
                    site1 = int(parts[1])
                    op2 = int(parts[2])
                    site2 = int(parts[3])
                    real_weight = float(parts[4])
                    imag_weight = float(parts[5])
                    
                    self.edges.append({
                        'site1': site1,
                        'site2': site2,
                        'op1': op1,
                        'op2': op2,
                        'weight': complex(real_weight, imag_weight)
                    })
                    
    def load_vertices_from_file(self, filename):
        """Load vertex data from Trans.dat file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Parse number of vertices
            num_line = lines[1].strip()
            num_vertices = int(num_line.split()[1])
            
            # Skip header lines
            start_idx = 5
            
            # Read each vertex
            for i in range(start_idx, start_idx + num_vertices):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    op = int(parts[0])
                    site = int(parts[1])
                    real_weight = float(parts[2])
                    imag_weight = float(parts[3])
                    
                    self.vertices.append({
                        'site': site,
                        'op': op,
                        'weight': complex(real_weight, imag_weight)
                    })
    
    def build_hamiltonian_graph(self):
        """Build a colored graph representing the Hamiltonian structure"""
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes for each site with attributes
        for i in range(self.n_sites):
            G.add_node(i, type='site')
            
        # Add vertex attributes (on-site terms)
        for vertex in self.vertices:
            site = vertex['site']
            op = vertex['op']
            weight = vertex['weight']
            
            # Store on-site operation as node attribute
            if 'ops' not in G.nodes[site]:
                G.nodes[site]['ops'] = {}
            
            G.nodes[site]['ops'][op] = weight
            
        # Add edges with attributes
        for edge in self.edges:
            site1 = edge['site1']
            site2 = edge['site2']
            op1 = edge['op1']
            op2 = edge['op2']
            weight = edge['weight']
            
            # Only add edge if sites are different
            if site1 != site2:
                # Create unique edge key for this interaction
                edge_key = f"{op1}_{op2}"
                
                if not G.has_edge(site1, site2):
                    G.add_edge(site1, site2, interactions={})
                
                G[site1][site2]['interactions'][edge_key] = weight
        
        self.hamiltonian_graph = G
        return G
    
    def is_automorphism(self, permutation):
        """Check if a permutation is an automorphism of the Hamiltonian"""
        # Create mapping dictionary from the permutation
        mapping = {i: permutation[i] for i in range(self.n_sites)}
        
        # Create a copy of the Hamiltonian graph
        H = self.hamiltonian_graph.copy()
        
        # Relabel the nodes according to the permutation
        H = nx.relabel_nodes(H, mapping)
        
        # Use graph isomorphism with custom comparison for node and edge attributes
        return self._graph_isomorphic_with_attributes(self.hamiltonian_graph, H)
    
    def _graph_isomorphic_with_attributes(self, G1, G2):
        """Custom isomorphism checker that accounts for our specific node and edge attributes"""
        # Check basic graph structure
        if not nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get('type') == n2.get('type')):
            return False
        
        # Check node attributes (on-site operations)
        for node in G1.nodes():
            # Find corresponding node in G2
            for node2 in G2.nodes():
                if G1.nodes[node].get('type') == G2.nodes[node2].get('type'):
                    # Compare on-site operations
                    ops1 = G1.nodes[node].get('ops', {})
                    ops2 = G2.nodes[node2].get('ops', {})
                    if ops1 != ops2:
                        return False
        
        # Check edge attributes (interactions)
        for u, v in G1.edges():
            # Find corresponding edge in G2
            found_match = False
            for u2, v2 in G2.edges():
                interactions1 = G1[u][v].get('interactions', {})
                interactions2 = G2[u2][v2].get('interactions', {})
                if interactions1 == interactions2:
                    found_match = True
                    break
            
            if not found_match:
                return False
                
        return True
    
    def find_all_automorphisms(self):
        """Find all automorphisms of the Hamiltonian using NetworkX"""
        # First build the Hamiltonian graph if not already built
        if not self.hamiltonian_graph.nodes():
            self.build_hamiltonian_graph()
        
        # Define custom node and edge match functions for our specific attributes
        def node_match(n1, n2):
            # Check node type and operations
            if n1.get('type') != n2.get('type'):
                return False
            # Compare on-site operations
            return n1.get('ops', {}) == n2.get('ops', {})
        
        def edge_match(e1, e2):
            # Compare interaction attributes
            return e1.get('interactions', {}) == e2.get('interactions', {})
        
        # Use NetworkX's isomorphism utilities
        GM = isomorphism.GraphMatcher(
            self.hamiltonian_graph, 
            self.hamiltonian_graph, 
            node_match=node_match,
            edge_match=edge_match
        )
        
        # Find all automorphisms
        automorphisms = []
        print("Finding automorphisms using NetworkX...")
        for mapping in GM.isomorphisms_iter():
            # Convert mapping to permutation form
            perm = [mapping[i] for i in range(self.n_sites)]
            automorphisms.append(perm)
        
        return automorphisms
    
    def permutation_to_cycle_notation(self, permutation):
        """Convert a permutation to cycle notation"""
        n = len(permutation)
        visited = [False] * n
        cycles = []
        
        for i in range(n):
            if not visited[i]:
                cycle = []
                j = i
                
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = permutation[j]
                    
                if j == i and len(cycle) > 1:
                    cycles.append(tuple(cycle))
        
        # Add fixed points (cycles of length 1)
        for i in range(n):
            if permutation[i] == i:
                cycles.append((i,))
                
        return cycles


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


class MinimalGeneratorFinder:
    """Class to find minimal generators for a group of automorphisms using an improved algorithm"""
    
    def __init__(self):
        """Initialize the finder"""
        pass
    
    def compose_permutations(self, perm1, perm2):
        """Compose two permutations: result(i) = perm1(perm2[i])"""
        return [perm1[perm2[i]] for i in range(len(perm1))]
    
    def find_order(self, perm):
        """Find the order of a permutation (how many times to compose to get identity)"""
        n = len(perm)
        identity = list(range(n))
        current = perm.copy()
        order = 1
        
        while current != identity:
            current = self.compose_permutations(perm, current)
            order += 1
            
        return order
    
    def incrementally_add_to_group(self, group, new_element):
        """Efficiently add a new element and all its combinations to the group"""
        if tuple(new_element) in group:
            return False

        # Use BFS to add all new elements generated by the new generator
        queue = [new_element]
        added = False
        
        while queue:
            elem = queue.pop(0)
            elem_tuple = tuple(elem)
            
            if elem_tuple in group:
                continue
                
            # Add the new element
            group.add(elem_tuple)
            added = True
            
            # Generate new elements by combining with existing ones
            for existing in list(group):
                # Skip identity and self
                if existing == tuple(range(len(new_element))):
                    continue
                    
                # Compose in both directions
                comp1 = self.compose_permutations(elem, list(existing))
                comp2 = self.compose_permutations(list(existing), elem)
                
                if tuple(comp1) not in group:
                    queue.append(comp1)
                if tuple(comp2) not in group:
                    queue.append(comp2)
                    
        return added
    
    def find_minimal_generators(self, automorphisms):
        """Find a minimal set of generators for the group of automorphisms using an improved algorithm"""
        if not automorphisms:
            return [], []
            
        # Initialize
        n_sites = len(automorphisms[0])
        identity = list(range(n_sites))
        
        # Get unique automorphisms
        unique_autos = set(map(tuple, automorphisms))
        unique_autos.add(tuple(identity))
        
        # Sort automorphisms by order (smaller orders first)
        auto_with_orders = [(list(auto), self.find_order(list(auto))) for auto in unique_autos if list(auto) != identity]
        sorted_autos = sorted(auto_with_orders, key=lambda x: x[1])
        
        # Results
        generators = []
        orders = []
        
        # Generated group (start with identity)
        generated_group = {tuple(identity)}
        
        # Try each automorphism as potential generator
        for auto, order in sorted_autos:
            # Try to add the automorphism to the group
            if self.incrementally_add_to_group(generated_group, auto):
                generators.append(auto)
                orders.append(order)
            
            # Check if we've generated the entire group
            if len(generated_group) == len(unique_autos):
                break
        
        return generators, orders

class AutomorphismPowerRepresentation:
    """Class to represent automorphisms as powers of generators"""
    
    @staticmethod
    def represent_as_generator_powers(generators, automorphism, max_power=5):
        """Represent an automorphism as powers of generators using BFS"""
        if not generators:
            return []
            
        num_generators = len(generators)
        perm_size = len(automorphism)
        
        # Check if the automorphism is the identity
        identity = list(range(perm_size))
        if automorphism == identity:
            return [0] * num_generators
            
        # BFS queue
        queue = deque()
        visited = set()
        
        # Store state as (powers, current_perm)
        initial_state = ([0] * num_generators, identity)
        queue.append(initial_state)
        visited.add(tuple(identity))
        
        # Helper for composition
        def compose(perm1, perm2):
            return [perm1[perm2[i]] for i in range(len(perm1))]
        
        while queue:
            powers, current_perm = queue.popleft()
            
            for gen_idx, gen in enumerate(generators):
                for direction in [-1, 1]:  # Try both positive and negative powers
                    # Don't exceed max power
                    new_powers = powers.copy()
                    new_powers[gen_idx] += direction
                    
                    if abs(new_powers[gen_idx]) > max_power:
                        continue
                    
                    # Apply generator
                    if direction == 1:
                        new_perm = compose(gen, current_perm)
                    else:
                        # Use inverse
                        inv_gen = [0] * len(gen)
                        for i, p in enumerate(gen):
                            inv_gen[p] = i
                        new_perm = compose(inv_gen, current_perm)
                    
                    # Check if this matches the target automorphism
                    if new_perm == automorphism:
                        return new_powers
                    
                    # Add to queue if not visited
                    if tuple(new_perm) not in visited:
                        visited.add(tuple(new_perm))
                        queue.append((new_powers, new_perm))
        
        # No representation found within max_power
        return []
    
    @staticmethod
    def represent_all_as_generator_powers(generators, automorphisms, max_power=5):
        """Represent all automorphisms as powers of generators"""
        results = []
        
        for auto in automorphisms:
            representation = AutomorphismPowerRepresentation.represent_as_generator_powers(
                generators, auto, max_power
            )
            results.append(representation)
            
        return results


class HamiltonianVisualizer:
    """Class for visualizing Hamiltonian structure"""
    
    def __init__(self, n_sites):
        """Initialize with number of sites"""
        self.n_sites = n_sites
        self.edges = []
        self.vertices = []
        
    def load_edges_from_file(self, filename):
        """Load edge data from InterAll.dat file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Parse number of interactions
            num_line = lines[1].strip()
            num_interactions = int(num_line.split()[1])
            
            # Skip header lines
            start_idx = 5
            
            # Read each interaction
            for i in range(start_idx, start_idx + num_interactions):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split()
                if len(parts) >= 6:
                    op1 = int(parts[0])
                    site1 = int(parts[1])
                    op2 = int(parts[2])
                    site2 = int(parts[3])
                    real_weight = float(parts[4])
                    imag_weight = float(parts[5])
                    
                    self.edges.append({
                        'site1': site1,
                        'site2': site2,
                        'op1': op1,
                        'op2': op2,
                        'weight': complex(real_weight, imag_weight)
                    })
                    
    def load_vertices_from_file(self, filename):
        """Load vertex data from Trans.dat file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Parse number of vertices
            num_line = lines[1].strip()
            num_vertices = int(num_line.split()[1])
            
            # Skip header lines
            start_idx = 5
            
            # Read each vertex
            for i in range(start_idx, start_idx + num_vertices):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    op = int(parts[0])
                    site = int(parts[1])
                    real_weight = float(parts[2])
                    imag_weight = float(parts[3])
                    
                    self.vertices.append({
                        'site': site,
                        'op': op,
                        'weight': complex(real_weight, imag_weight)
                    })
    
    def get_operator_name(self, op):
        """Get the name of an operator based on its index"""
        operators = {0: "X", 1: "Y", 2: "Z"}
        return operators.get(op, "?")
        
    def generate_dot_file(self, filename):
        """Generate DOT file for visualizing the Hamiltonian"""
        with open(filename, 'w') as file:
            file.write("graph Hamiltonian {\n")
            file.write("  node [shape=circle];\n")
            
            # Add nodes for sites
            for i in range(self.n_sites):
                file.write(f"  {i} [label=\"{i}\"];\n")
            
            # Add edges for interactions
            edge_id = 0
            for edge in self.edges:
                site1 = edge['site1']
                site2 = edge['site2']
                op1 = self.get_operator_name(edge['op1'])
                op2 = self.get_operator_name(edge['op2'])
                weight = edge['weight']
                
                # Only add edge if sites are different
                if site1 != site2:
                    label = f"{op1}{site1}⊗{op2}{site2}: {weight.real:.4f}"
                    if weight.imag != 0:
                        label += f" + {weight.imag:.4f}i"
                    
                    file.write(f"  {site1} -- {site2} [label=\"{label}\", id=\"edge_{edge_id}\"];\n")
                    edge_id += 1
            
            # Add vertex labels
            for vertex in self.vertices:
                site = vertex['site']
                op = self.get_operator_name(vertex['op'])
                weight = vertex['weight']
                
                if weight.real != 0 or weight.imag != 0:
                    label = f"{op}{site}: {weight.real:.4f}"
                    if weight.imag != 0:
                        label += f" + {weight.imag:.4f}i"
                    
                    file.write(f"  site_{site}_op_{op} [label=\"{label}\", shape=box];\n")
                    file.write(f"  {site} -- site_{site}_op_{op} [style=dashed];\n")
            
            file.write("}\n")
        
        print(f"Generated DOT file: {filename}")
        print(f"Visualize with: dot -Tpng {filename} -o graph.png")
        
    def save_graph_image(self, output_file, dot_file="temp_hamiltonian.dot"):
        """Generate PNG image from DOT file using Graphviz"""
        self.generate_dot_file(dot_file)
        
        try:
            subprocess.run(['dot', '-Tpng', dot_file, '-o', output_file], check=True)
            print(f"Generated graph image: {output_file}")
        except subprocess.CalledProcessError:
            print(f"Failed to generate image. Make sure Graphviz is installed.")
        except FileNotFoundError:
            print(f"Graphviz command 'dot' not found. Please install Graphviz.")


def main():
    # Set paths to data files
    parser = argparse.ArgumentParser(description='Hamiltonian Automorphism Finder')

    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    args = parser.parse_args()


    inter_all_file = os.path.join(args.data_dir, "InterAll.dat")
    trans_file = os.path.join(args.data_dir, "Trans.dat")
    
    # Determine number of sites from Trans.dat file
    with open(trans_file, 'r') as file:
        max_site_index = -1
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
                site_index = int(parts[1])
                max_site_index = max(max_site_index, site_index)
        
        n_sites = max_site_index + 1 if max_site_index >= 0 else 0
    print(f"Number of sites: {n_sites}")
    
    # Create output directory
    output_dir = os.path.join(args.data_dir, "automorphism_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Hamiltonian finder and load data
    ham_finder = HamiltonianAutomorphismFinder(n_sites)
    ham_finder.load_edges_from_file(inter_all_file)
    ham_finder.load_vertices_from_file(trans_file)
    ham_finder.build_hamiltonian_graph()
    
    # Find automorphisms
    print("Finding Hamiltonian automorphisms...")
    automorphisms = ham_finder.find_all_automorphisms()
    print(f"Found {len(automorphisms)} automorphisms")
    
    # Save automorphisms to file
    with open(os.path.join(output_dir, "automorphisms.json"), 'w') as f:
        json.dump(automorphisms, f, indent=2)
    
    # Find maximal commuting set
    clique_analyzer = AutomorphismCliqueAnalyzer()
    print("Finding maximal commuting set...")
    max_clique = clique_analyzer.find_maximum_clique(automorphisms)
    print(f"Maximal commuting set has {len(max_clique)} elements")
    
    # Save maximal commuting set
    with open(os.path.join(output_dir, "max_clique.json"), 'w') as f:
        json.dump([automorphisms[i] for i in max_clique], f, indent=2)
    
    # max_clique_autos = [automorphisms[i] for i in max_clique]    
    # # Find minimal generators
    # generator_finder = MinimalGeneratorFinder()
    # print("Finding minimal generators...")
    # generators, orders = generator_finder.find_minimal_generators(max_clique_autos)
    # print(f"Found {len(generators)} minimal generators with orders: {orders}")
    
    # # Save generators to file
    # with open(os.path.join(output_dir, "generators.json"), 'w') as f:
    #     json.dump({
    #         "generators": generators,
    #         "orders": orders
    #     }, f, indent=2)
    
    # # Represent automorphisms as powers of generators
    # print("Representing automorphisms as powers of generators...")
    # power_representations = AutomorphismPowerRepresentation.represent_all_as_generator_powers(
    #     generators, max_clique_autos
    # )
    
    # # Save power representations
    # with open(os.path.join(output_dir, "power_representations.json"), 'w') as f:
    #     json.dump(power_representations, f, indent=2)
    
    # print(f"All results saved to {output_dir} directory")


if __name__ == "__main__":
    main()