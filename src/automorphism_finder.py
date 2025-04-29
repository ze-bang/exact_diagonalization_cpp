import numpy as np
import networkx as nx
import itertools
import os
import math
from collections import deque
import argparse
from typing import List, Tuple, Set, Dict, Optional
import sys

#!/usr/bin/env python3
import matplotlib.pyplot as plt


class HamiltonianGraph:
    """Class to represent a Hamiltonian as a graph for automorphism analysis"""
    
    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.edges = []
        self.vertices = []
        self.graph = nx.Graph()
        
    def load_edges_from_file(self, filename: str) -> None:
        """Load interaction terms from file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Extract number of interactions from second line
            num_interactions = int(lines[1].split()[1])
            
            # Skip header lines
            data_lines = lines[4:4+num_interactions]
            
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    op1, site1, op2, site2 = map(int, parts[:4])
                    weight = complex(float(parts[4]), float(parts[5]))
                    
                    self.edges.append({
                        'site1': site1,
                        'site2': site2,
                        'op1': op1,
                        'op2': op2,
                        'weight': weight
                    })
    
    def load_vertices_from_file(self, filename: str) -> None:
        """Load site operators from file"""
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Extract number of vertices from second line
            num_vertices = int(lines[1].split()[1])
            
            # Skip header lines
            data_lines = lines[4:4+num_vertices]
            
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    op, site = map(int, parts[:2])
                    weight = complex(float(parts[2]), float(parts[3]))
                    
                    self.vertices.append({
                        'site': site,
                        'op': op,
                        'weight': weight
                    })
    
    def build_graph(self) -> None:
        """Construct a networkx graph from the loaded edges and vertices"""
        self.graph.clear()
        
        # Add nodes with attributes
        for i in range(self.n_sites):
            node_attrs = {'type': 'site', 'index': i}
            
            # Add vertex operators as node attributes
            for vertex in self.vertices:
                if vertex['site'] == i:
                    operator_type = self.get_operator_name(vertex['op'])
                    node_attrs[f'op_{operator_type}'] = vertex['weight']
            
            self.graph.add_node(i, **node_attrs)
        
        # Add edges with attributes
        for edge in self.edges:
            if edge['site1'] != edge['site2']:
                op1_name = self.get_operator_name(edge['op1'])
                op2_name = self.get_operator_name(edge['op2'])
                
                self.graph.add_edge(
                    edge['site1'], 
                    edge['site2'], 
                    op1=edge['op1'],
                    op2=edge['op2'],
                    op_type=f"{op1_name}-{op2_name}",
                    weight=edge['weight']
                )
    
    def get_operator_name(self, op: int) -> str:
        """Convert operator code to name"""
        operators = {0: "X", 1: "Y", 2: "Z"}
        return operators.get(op, "?")
    

class AutomorphismFinder:
    """Find automorphisms of a Hamiltonian using graph isomorphism"""
    
    def __init__(self, hamiltonian_graph: HamiltonianGraph):
        self.ham_graph = hamiltonian_graph
        self.automorphisms = []
    
    def is_valid_automorphism(self, perm: List[int]) -> bool:
        """Check if a permutation is a valid automorphism"""
        if len(perm) != self.ham_graph.n_sites:
            return False
            
        # Create a mapping dictionary
        mapping = {i: perm[i] for i in range(len(perm))}
        
        # Check if applying the permutation preserves the graph structure
        for edge in self.ham_graph.edges:
            site1, site2 = edge['site1'], edge['site2']
            op1, op2 = edge['op1'], edge['op2']
            weight = edge['weight']
            
            if site1 == site2:
                continue  # Skip self-loops
                
            # Check if permuted edge exists with same attributes
            perm_site1, perm_site2 = perm[site1], perm[site2]
            
            found = False
            for other_edge in self.ham_graph.edges:
                # Check both orientations for undirected graph
                matches1 = (other_edge['site1'] == perm_site1 and 
                           other_edge['site2'] == perm_site2 and
                           other_edge['op1'] == op1 and 
                           other_edge['op2'] == op2 and
                           abs(other_edge['weight'] - weight) < 1e-10)
                           
                matches2 = (other_edge['site1'] == perm_site2 and 
                           other_edge['site2'] == perm_site1 and
                           other_edge['op1'] == op2 and 
                           other_edge['op2'] == op1 and
                           abs(other_edge['weight'] - weight) < 1e-10)
                
                if matches1 or matches2:
                    found = True
                    break
            
            if not found:
                return False
        
        # Check vertices
        for vertex in self.ham_graph.vertices:
            site, op = vertex['site'], vertex['op']
            weight = vertex['weight']
            
            perm_site = perm[site]
            
            found = False
            for other_vertex in self.ham_graph.vertices:
                if (other_vertex['site'] == perm_site and 
                    other_vertex['op'] == op and
                    abs(other_vertex['weight'] - weight) < 1e-10):
                    found = True
                    break
            
            if not found:
                return False
        
        return True
    
    def find_automorphisms_networkx(self) -> List[List[int]]:
        """Find all automorphisms using NetworkX"""
        self.ham_graph.build_graph()
        
        # Use NetworkX's graph isomorphism to find automorphisms
        gm = nx.algorithms.isomorphism.GraphMatcher(
            self.ham_graph.graph, 
            self.ham_graph.graph,
            node_match=lambda n1, n2: self._node_match(n1, n2),
            edge_match=lambda e1, e2: self._edge_match(e1, e2)
        )
        
        automorphisms = []
        for iso in gm.isomorphisms_iter():
            # Convert dictionary to permutation list
            perm = [iso.get(i, i) for i in range(self.ham_graph.n_sites)]
            if self.is_valid_automorphism(perm):
                automorphisms.append(perm)
        
        self.automorphisms = automorphisms
        return automorphisms
    
    def _node_match(self, n1, n2) -> bool:
        """Check if two nodes match for isomorphism"""
        # Nodes should have same operator types and weights
        for key in n1:
            if key.startswith('op_'):
                if key not in n2 or abs(n1[key] - n2[key]) > 1e-10:
                    return False
        return True
    
    def _edge_match(self, e1, e2) -> bool:
        """Check if two edges match for isomorphism"""
        if 'op1' in e1 and 'op1' in e2:
            if e1['op1'] != e2['op1'] or e1['op2'] != e2['op2']:
                return False
            if abs(e1['weight'] - e2['weight']) > 1e-10:
                return False
        return True
    
    def permutation_to_cycle_notation(self, perm: List[int]) -> str:
        """Convert a permutation to cycle notation"""
        visited = [False] * len(perm)
        result = ""
        
        for i in range(len(perm)):
            if visited[i] or perm[i] == i:
                continue
            
            result += "("
            j = i
            while not visited[j]:
                result += str(j)
                visited[j] = True
                j = perm[j]
                if j != i and not visited[j]:
                    result += " "
            result += ")"
        
        # Add fixed points
        for i in range(len(perm)):
            if perm[i] == i:
                result += f"({i})"
        
        if not result:
            result = "()"  # Identity permutation
        
        return result

class AutomorphismAnalyzer:
    """Analyze automorphisms, find cliques and minimal generators"""
    
    def __init__(self, automorphisms: List[List[int]]):
        self.automorphisms = automorphisms
        self.compatibility_graph = nx.Graph()
    
    def do_permutations_commute(self, perm1: List[int], perm2: List[int]) -> bool:
        """Check if two permutations commute"""
        if len(perm1) != len(perm2):
            return False
        
        # Check if p1 ∘ p2 = p2 ∘ p1
        for i in range(len(perm1)):
            if perm1[perm2[i]] != perm2[perm1[i]]:
                return False
        return True
    
    def build_compatibility_graph(self) -> nx.Graph:
        """Build a graph where nodes are automorphisms and edges indicate that they commute"""
        self.compatibility_graph.clear()
        
        # Add automorphisms as nodes
        for i, auto in enumerate(self.automorphisms):
            self.compatibility_graph.add_node(i, permutation=auto)
        
        # Add edges between commuting automorphisms
        for i in range(len(self.automorphisms)):
            for j in range(i+1, len(self.automorphisms)):
                if self.do_permutations_commute(self.automorphisms[i], self.automorphisms[j]):
                    self.compatibility_graph.add_edge(i, j)
        
        return self.compatibility_graph
    
    def find_maximum_clique(self) -> List[int]:
        """Find the maximum clique in the compatibility graph"""
        if not self.compatibility_graph:
            self.build_compatibility_graph()
        
        # Use NetworkX's maximum clique finder
        max_clique = max(nx.find_cliques(self.compatibility_graph), key=len)
        return max_clique
    
    def compose_permutations(self, perm1: List[int], perm2: List[int]) -> List[int]:
        """Compose two permutations: result(i) = perm1(perm2[i])"""
        return [perm1[perm2[i]] for i in range(len(perm1))]
    
    def inverse_permutation(self, perm: List[int]) -> List[int]:
        """Find the inverse of a permutation"""
        inverse = [0] * len(perm)
        for i in range(len(perm)):
            inverse[perm[i]] = i
        return inverse
    
    def find_order(self, perm: List[int]) -> int:
        """Find the order of a permutation"""
        # Create identity permutation
        identity = list(range(len(perm)))
        
        # Create a working copy of the permutation
        current = perm.copy()
        order = 1
        
        # Keep composing with itself until we get the identity
        while current != identity:
            current = self.compose_permutations(current, perm)
            order += 1
        
        return order
    
    def find_minimal_generators(self, clique_indices: List[int]) -> Tuple[List[List[int]], List[int]]:
        """Find minimal generators and their orders for the group in the clique"""
        # Get the permutations in the clique
        clique_perms = [self.automorphisms[idx] for idx in clique_indices]
        
        # Create identity permutation
        n = len(self.automorphisms[0])
        identity = list(range(n))
        
        # Make sure we include the identity permutation
        unique_perms = set(tuple(perm) for perm in clique_perms)
        unique_perms.add(tuple(identity))
        
        # Convert back to list and sort for deterministic results
        sorted_perms = sorted(list(unique_perms))
        
        # Store the generators and their orders
        generators = []
        orders = []
        
        # Generated elements
        generated_elements = {tuple(identity)}
        
        # Try each permutation as a potential generator
        for perm_tuple in sorted_perms:
            perm = list(perm_tuple)
            # Skip the identity
            if perm == identity:
                continue
            
            # Skip if we can already generate this permutation
            if tuple(perm) in generated_elements:
                continue
            
            # Add this permutation as a generator
            generators.append(perm)
            order = self.find_order(perm)
            orders.append(order)
            
            # Generate all elements from the current set of generators
            self._generate_subgroup(generators, generated_elements)
            
            # If we've generated the entire group, we're done
            if len(generated_elements) == len(unique_perms):
                break
        
        return generators, orders
    
    def _generate_subgroup(self, generators: List[List[int]], generated_elements: Set[Tuple[int, ...]]) -> None:
        """Generate all elements in the subgroup from a set of generators"""
        # Start with the identity
        n = len(generators[0])
        identity = list(range(n))
        
        generated_elements.clear()
        generated_elements.add(tuple(identity))
        
        # Keep adding new elements until no more can be added
        old_size = 0
        while old_size < len(generated_elements):
            old_size = len(generated_elements)
            
            # Try composing each element with each generator
            existing = list(generated_elements)
            for elem_tuple in existing:
                elem = list(elem_tuple)
                for gen in generators:
                    # Compose in both orders
                    composed1 = self.compose_permutations(elem, gen)
                    composed2 = self.compose_permutations(gen, elem)
                    
                    generated_elements.add(tuple(composed1))
                    generated_elements.add(tuple(composed2))
    
    def find_power_representation(self, automorphism: List[int], generators: List[List[int]]) -> List[int]:
        """
        Express an automorphism as a product of powers of generators.
        Returns an array of powers for each generator.
        """
        n = len(automorphism)
        identity = list(range(n))
        
        # Handle the identity case
        if automorphism == identity:
            return [0] * len(generators)
        
        # Find the orders of generators
        orders = [self.find_order(gen) for gen in generators]
        
        # Generate all possible combinations of powers
        power_ranges = [range(order) for order in orders]
        
        for powers in itertools.product(*power_ranges):
            # Construct permutation from powers
            result = identity.copy()
            for i, power in enumerate(powers):
                # Apply generator i, power times
                gen = generators[i]
                for _ in range(power):
                    result = self.compose_permutations(result, gen)
            
            # Check if we've found our automorphism
            if result == automorphism:
                return list(powers)
        
        # If we can't represent it, return None
        return None

    def find_all_power_representations(self, automorphisms: List[List[int]], generators: List[List[int]]) -> List[List[int]]:
        """
        Find power representations for all given automorphisms.
        Returns a list of power representations in the same order as the input automorphisms.
        If an automorphism can't be represented, its entry will be all zeros.
        """
        representations = []
        
        for auto in automorphisms:
            power_rep = self.find_power_representation(auto, generators)
            if power_rep is None:
                # If not representable, use all zeros
                power_rep = [0] * len(generators)
            representations.append(power_rep)
        
        return representations

class SymmetrizedBasisGenerator:
    """Generate symmetrized basis vectors using automorphisms and save in sparse format"""
    
    def __init__(self, automorphisms: List[List[int]], generators: List[List[int]], 
                    generator_orders: List[int], n_sites: int):
        self.automorphisms = automorphisms
        self.generators = generators
        self.generator_orders = generator_orders
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.analyzer = AutomorphismAnalyzer(automorphisms)
        self.power_representations = self.analyzer.find_all_power_representations(
            automorphisms, generators)
        self.symmetry_sectors = []
        self.block_sizes = []
    
    def apply_permutation(self, basis: int, perm: List[int]) -> int:
        """Apply a permutation to a basis state represented as an integer"""
        result = 0
        for i, p in enumerate(perm):
            # Get bit from position p in the original basis and place it at position i
            bit = (basis >> p) & 1
            result |= (bit << i)
        return result
    
    def generate_quantum_numbers(self) -> List[List[int]]:
        """Generate all possible combinations of quantum numbers"""
        qnums = []
        
        def generate_recursive(position: int, current: List[int]):
            if position == len(self.generator_orders):
                qnums.append(current.copy())
                return
            
            for i in range(self.generator_orders[position]):
                current[position] = i
                generate_recursive(position + 1, current)
        
        current_qnums = [0] * len(self.generators)
        generate_recursive(0, current_qnums)
        
        self.symmetry_sectors = qnums
        return qnums
    
    def symmetrize_basis(self, basis_state: int, qnums: List[int]) -> np.ndarray:
        """Create a symmetrized basis vector for a standard basis state and quantum numbers"""
        sym_basis = np.zeros(self.dim, dtype=np.complex128)
        
        for i, auto in enumerate(self.automorphisms):
            # Apply permutation to get new basis state
            permuted_state = self.apply_permutation(basis_state, auto)
            
            # Calculate phase factor based on quantum numbers and power representation
            phase = 0.0
            for j, power in enumerate(self.power_representations[i]):
                phase += 2 * np.pi * power * qnums[j] / self.generator_orders[j]
            
            # Add to symmetrized basis with appropriate phase
            sym_basis[permuted_state] += np.exp(1j * phase)
        
        # Normalize
        norm = np.linalg.norm(sym_basis)
        if norm > 1e-10:
            sym_basis /= norm
            
        return sym_basis
    
    def generate_all_symmetrized_bases(self, output_dir: str) -> None:
        """Generate all symmetrized basis vectors and save them in sparse format"""
        # Make sure output directory exists
        symm_basis_dir = f"{output_dir}/sym_basis"
        os.makedirs(symm_basis_dir, exist_ok=True)
        
        # First generate all quantum number combinations
        qnum_combinations = self.generate_quantum_numbers()
        print(f"Generated {len(qnum_combinations)} symmetry sectors")
        
        # Initialize block sizes
        self.block_sizes = [0] * len(qnum_combinations)
        basis_count = 0
        
        # For each symmetry sector
        for sector_idx, qnums in enumerate(qnum_combinations):
            print(f"Processing symmetry sector {sector_idx+1}/{len(qnum_combinations)}")
            unique_bases = []
            
            # Try symmetrizing each standard basis state
            for basis in range(self.dim):
                sym_vec = self.symmetrize_basis(basis, qnums)
                
                # Skip if this produces a zero vector
                if np.linalg.norm(sym_vec) < 1e-10:
                    continue
                
                # Check if this is linearly independent from existing vectors
                is_unique = True
                for existing_vec in unique_bases:
                    # Calculate overlap
                    overlap = np.abs(np.vdot(existing_vec, sym_vec))
                    if np.abs(overlap - 1.0) < 1e-10:
                        is_unique = False
                        break
                
                if is_unique:
                    unique_bases.append(sym_vec)
                    
                    # Save this basis vector in sparse format
                    with open(f"{symm_basis_dir}/sym_basis{basis_count}.dat", 'w') as f:
                        # Write non-zero elements only: index, real part, imaginary part
                        for idx in range(self.dim):
                            if abs(sym_vec[idx]) > 1e-10:
                                f.write(f"{idx} {sym_vec[idx].real} {sym_vec[idx].imag}\n")
                    
                    basis_count += 1
                    self.block_sizes[sector_idx] += 1
        
        # Save block sizes information
        with open(f"{output_dir}/sym_block_sizes.txt", 'w') as f:
            f.write("Symmetry block sizes:\n")
            for i, size in enumerate(self.block_sizes):
                qnums_str = ' '.join(map(str, qnum_combinations[i]))
                f.write(f"Block {i}: size = {size}, quantum numbers = [{qnums_str}]\n")
        
        print(f"Generated {basis_count} symmetrized basis vectors")
        print(f"Block sizes: {self.block_sizes}")
        print(f"Basis vectors saved in {symm_basis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Find and analyze Hamiltonian automorphisms')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory')
    parser.add_argument('--output', type=str, default='automorphisms', help='Output directory')

    interactions_file = os.path.join(parser.parse_args().work_dir, 'InterAll.dat')
    trans_file = os.path.join(parser.parse_args().work_dir, 'Trans.dat')

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Read the number of sites from Trans.dat
    with open(trans_file, 'r') as f:
        _ = f.readline()  # Skip header
        n_sites = int(f.readline().split()[1])
    
    print(f"System with {n_sites} sites")


    # Create Hamiltonian graph
    ham_graph = HamiltonianGraph(n_sites)
    ham_graph.load_edges_from_file(interactions_file)
    ham_graph.load_vertices_from_file(trans_file)
    ham_graph.build_graph()
    

    # Find automorphisms
    finder = AutomorphismFinder(ham_graph)
    automorphisms = finder.find_automorphisms_networkx()
    
    print(f"Found {len(automorphisms)} automorphisms")
    
    # Save automorphisms to file
    with open(f"{args.output}/automorphisms.txt", 'w') as f:
        for i, auto in enumerate(automorphisms):
            cycle_notation = finder.permutation_to_cycle_notation(auto)
            f.write(f"Auto {i}: {auto} => {cycle_notation}\n")
    
    # Analyze automorphisms
    analyzer = AutomorphismAnalyzer(automorphisms)
    compatibility_graph = analyzer.build_compatibility_graph()
    
    # Find maximum clique
    max_clique = analyzer.find_maximum_clique()
    print(f"Maximum clique size: {len(max_clique)}")
    
    # Find minimal generators
    generators, orders = analyzer.find_minimal_generators(max_clique)
    
    print("Minimal generators:")
    for i, gen in enumerate(generators):
        cycle_notation = finder.permutation_to_cycle_notation(gen)
        print(f"Generator {i}: {gen} => {cycle_notation}, Order: {orders[i]}")
    
    # Save generators to file
    with open(f"{args.output}/generators.txt", 'w') as f:
        for i, gen in enumerate(generators):
            cycle_notation = finder.permutation_to_cycle_notation(gen)
            f.write(f"Generator {i}: {gen} => {cycle_notation}, Order: {orders[i]}\n")
    
    # Find power representations
    power_representations = analyzer.find_all_power_representations(automorphisms, generators)
    print("Power representations:")
    for i, auto in enumerate(automorphisms):
        power_rep = power_representations[i]
        print(f"Automorphism {i}: {auto} => {power_rep}")
    
    # Save power representations to file
    with open(f"{args.output}/power_representations.txt", 'w') as f:
        for i, auto in enumerate(automorphisms):
            power_rep = power_representations[i]
            f.write(f"Automorphism {i}: {auto} => {power_rep}\n")


    # Generate symmetrized basis
    symm_gen = SymmetrizedBasisGenerator(automorphisms, generators, orders, n_sites)
    symm_gen.generate_all_symmetrized_bases(args.output)
    print("Symmetrized basis generation completed")
    # Generate quantum numbers
    qnums = symm_gen.generate_quantum_numbers()
    print(f"Generated {len(qnums)} quantum numbers")
    # Save quantum numbers to file
    with open(f"{args.output}/quantum_numbers.txt", 'w') as f:
        for i, qnum in enumerate(qnums):
            f.write(f"Quantum numbers {i}: {qnum}\n")
    

    # Visualize compatibility graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(compatibility_graph, seed=42)
    
    # Draw all nodes
    nx.draw_networkx_nodes(compatibility_graph, pos, node_size=300, alpha=0.8)
    
    # Highlight clique nodes
    nx.draw_networkx_nodes(compatibility_graph, pos, nodelist=max_clique, 
                          node_color='lightblue', node_size=400)
    
    # Draw edges
    nx.draw_networkx_edges(compatibility_graph, pos, alpha=0.3)
    
    # Draw clique edges (highlighted)
    clique_edges = [(i, j) for i in max_clique for j in max_clique if i < j and compatibility_graph.has_edge(i, j)]
    nx.draw_networkx_edges(compatibility_graph, pos, edgelist=clique_edges, width=2, edge_color='blue')
    
    # Draw labels
    labels = {i: f"{i}" for i in compatibility_graph.nodes()}
    nx.draw_networkx_labels(compatibility_graph, pos, labels, font_size=10)
    
    plt.title(f"Automorphism Compatibility Graph (Max Clique Size: {len(max_clique)})")
    plt.savefig(f"{args.output}/compatibility_graph.png", dpi=300)
    plt.close()
    
    print(f"Results saved in {args.output}/")

if __name__ == "__main__":
    main()