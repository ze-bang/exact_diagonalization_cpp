import argparse
import numpy as np
import networkx as nx
import itertools
import sys
from collections import defaultdict
import os
import collections

#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a pyrochlore lattice for
numerical linked cluster expansion (NLCE) calculations.
"""
try:
    import matplotlib.pyplot as plt  # optional; only needed for --visualize
except Exception:
    plt = None

# ---------------- New Multiplicity Logic (root-anchored embeddings) ---------------- #
def _cluster_automorphisms_fixing_root(pattern: nx.Graph, root):
    """Count automorphisms of pattern that fix the chosen root.

    Uses a full isomorphism iterator; clusters are small in typical NLCE orders
    so this is acceptable. For larger clusters, consider nauty/bliss.
    """
    gm = nx.algorithms.isomorphism.GraphMatcher(pattern, pattern)
    count = 0
    for iso in gm.isomorphisms_iter():
        if iso[root] == root:
            count += 1
    return max(count, 1)


def _select_anchor_tetrahedron(tet_graph: nx.Graph, required_degree: int):
    """Pick an interior tetrahedron: choose node with maximum degree (>= required_degree).

    This avoids boundary effects when counting root-anchored embeddings.
    """
    # Compute maximum degree in the lattice graph
    deg_items = list(tet_graph.degree())
    if not deg_items:
        return next(iter(tet_graph.nodes()))
    max_deg = max(d for _, d in deg_items)
    # Prefer nodes with max_deg; fallback to any >= required_degree, else any node
    candidates = [n for n, d in deg_items if d == max_deg]
    if candidates:
        return candidates[0]
    for n, d in deg_items:
        if d >= required_degree:
            return n
    return next(iter(tet_graph.nodes()))


def _build_local_patch(tet_graph: nx.Graph, center, radius: int):
    """Return induced subgraph of nodes within BFS distance <= radius from center."""
    if radius <= 0:
        return tet_graph.subgraph([center]).copy()
    visited = {center}
    frontier = {center}
    for _ in range(radius):
        nxt = set()
        for u in frontier:
            nxt.update(tet_graph.neighbors(u))
        nxt -= visited
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return tet_graph.subgraph(visited).copy()


def _pattern_diameter(pattern: nx.Graph):
    try:
        return nx.diameter(pattern)
    except Exception:
        return 0


def _root_anchored_embedding_count(pattern: nx.Graph, root, lattice_patch: nx.Graph, anchor):
    """Count monomorphism embeddings of pattern into lattice_patch with root fixed to anchor.

    Returns raw count (includes automorphisms that fix the root).
    """
    lattice_adj = {n: set(lattice_patch.neighbors(n)) for n in lattice_patch.nodes()}
    pattern_adj = {n: set(pattern.neighbors(n)) for n in pattern.nodes()}

    order = [root] + sorted([n for n in pattern.nodes() if n != root], key=lambda x: (-pattern.degree[x], x))
    mapping = {root: anchor}
    used = {anchor}
    raw = 0

    if pattern.degree[root] > lattice_patch.degree(anchor):
        return 0

    def backtrack(idx: int):
        nonlocal raw
        if idx == len(order):
            raw += 1
            return
        u = order[idx]
        mapped_neighbors = [v for v in pattern_adj[u] if v in mapping]
        if not mapped_neighbors:
            return  # enforce connectivity ordering
        candidate_sets = [lattice_adj[mapping[v]] for v in mapped_neighbors]
        candidates = set.intersection(*candidate_sets)
        required_deg = pattern.degree[u]
        for c in list(candidates):
            if c in used:
                continue
            if lattice_patch.degree(c) < required_deg:
                continue
            mapping[u] = c
            used.add(c)
            backtrack(idx + 1)
            used.remove(c)
            del mapping[u]

    backtrack(1)
    return raw


def compute_cluster_multiplicity(rep_nodes, tet_graph):
    """Compute per-site multiplicity for a representative cluster (list of tetrahedron indices).

    multiplicity = raw_embeddings / (2 * |Aut_root|)
    where raw_embeddings counts embeddings with root fixed; pyrochlore has N_tet = N_site/2.
    """
    if len(rep_nodes) == 1:
        return 0.5

    pattern_orig = tet_graph.subgraph(rep_nodes).copy()
    relabel_map = {old: i for i, old in enumerate(pattern_orig.nodes())}
    pattern = nx.relabel_nodes(pattern_orig, relabel_map, copy=True)

    root = max(pattern.nodes(), key=lambda n: (pattern.degree[n], -n))
    aut_fix_root = _cluster_automorphisms_fixing_root(pattern, root)

    anchor_required_degree = pattern.degree[root]
    anchor = _select_anchor_tetrahedron(tet_graph, anchor_required_degree)

    diam = _pattern_diameter(pattern)
    patch = _build_local_patch(tet_graph, anchor, diam)
    raw = _root_anchored_embedding_count(pattern, root, patch, anchor)
    if raw == 0:
        patch2 = _build_local_patch(tet_graph, anchor, diam + 1)
        raw = _root_anchored_embedding_count(pattern, root, patch2, anchor)

    return raw / (2.0 * aut_fix_root)
# ---------------------------------------------------------------------------------- #

def extract_cluster_info(lattice, pos, tetrahedra, cluster):
    """
    Extract detailed information about a cluster.
    """
    # Get all vertices in the cluster
    vertices = set()
    for tet_idx in cluster:
        vertices.update(tetrahedra[tet_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Get vertex positions
    vertex_positions = {v: pos[v] for v in vertices}
    
    # Get edges in the cluster
    edges = list(subgraph.edges())
    
    # Get the tetrahedra that make up the cluster
    cluster_tetrahedra = [tetrahedra[tet_idx] for tet_idx in cluster]
    
    # Create adjacency matrix
    nodes = sorted(list(vertices))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
    for u, v in edges:
        adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
        adj_matrix[node_to_idx[v], node_to_idx[u]] = 1
    
    return {
        'vertices': list(vertices),
        'vertex_positions': vertex_positions,
        'edges': edges,
        'tetrahedra': cluster_tetrahedra,
        'adjacency_matrix': adj_matrix,
        'node_mapping': node_to_idx
    }

def save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir='.'):
    """
    Save detailed information about a cluster to a file.
    """
    filename = f"{output_dir}/cluster_{cluster_id}_order_{order}.dat"
    
    with open(filename, 'w') as f:
        f.write(f"# Cluster ID: {cluster_id}\n")
        f.write(f"# Order (number of tetrahedra): {order}\n")
        f.write(f"# Multiplicity: {multiplicity}\n")
        f.write(f"# Number of vertices: {len(cluster_info['vertices'])}\n")
        f.write(f"# Number of edges: {len(cluster_info['edges'])}\n\n")
        
        f.write("# Vertices (index, x, y, z):\n")
        for v in cluster_info['vertices']:
            pos = cluster_info['vertex_positions'][v]
            f.write(f"{v}, {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}\n")
        
        f.write("\n# Edges (vertex1, vertex2):\n")
        for u, v in cluster_info['edges']:
            f.write(f"{u}, {v}\n")
        
        f.write("\n# Tetrahedra (vertex1, vertex2, vertex3, vertex4):\n")
        for tet in cluster_info['tetrahedra']:
            f.write(f"{', '.join(map(str, tet))}\n")
        
        f.write("\n# Adjacency Matrix:\n")
        for row in cluster_info['adjacency_matrix']:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\n# Node Mapping (original_id: matrix_index):\n")
        for node, idx in cluster_info['node_mapping'].items():
            f.write(f"{node}: {idx}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate topologically distinct clusters on a pyrochlore lattice.')
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--visualize', action='store_true', help='Visualize each cluster')
    parser.add_argument('--lattice_size', type=int, default=0, help='Size of finite lattice (default: 2*max_order)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for cluster information')
    return parser.parse_args()

def create_pyrochlore_lattice(L):
    """
    Create a pyrochlore lattice of size L×L×L unit cells.
    
    Returns:
    - G: NetworkX graph representing the lattice
    - pos: Dictionary mapping node IDs to 3D positions
    - tetrahedra: List of tetrahedra, each represented as a list of 4 node IDs
    """
    # FCC lattice vectors
    a1 = np.array([0, 0.5, 0.5])
    a2 = np.array([0.5, 0, 0.5])
    a3 = np.array([0.5, 0.5, 0])
    
    # Positions within unit cell
    basis_pos = np.array(
        [[0.125,0.125,0.125],
         [0.125,-0.125,-0.125],
         [-0.125,0.125,-0.125],
         [-0.125,-0.125,0.125]])


    G = nx.Graph()
    pos = {}
    tetrahedra = []
    
    # Generate lattice sites
    site_id = 0
    site_mapping = {}
    
    for i, j, k in itertools.product(range(L), repeat=3):
        cell_origin = i*a1 + j*a2 + k*a3
        
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, k, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1
    
    # Generate tetrahedra (two types per unit cell)
    for i, j, k in itertools.product(range(L), repeat=3):
        # First tetrahedron in the unit cell
        if all(0 <= x < L for x in [i, j, k]):
            tet1 = [
                site_mapping.get((i, j, k, 0)),
                site_mapping.get((i, j, k, 1)),
                site_mapping.get((i, j, k, 2)),
                site_mapping.get((i, j, k, 3))
            ]
            if None not in tet1:
                tetrahedra.append(tet1)
                # Add edges within tetrahedron
                for v1, v2 in itertools.combinations(tet1, 2):
                    G.add_edge(v1, v2)
        
        # Second tetrahedron (spans across unit cells)
        if all(0 <= x < L for x in [i, j, k, i+1, j+1, k+1]):
            tet2 = [
                site_mapping.get((i, j, k, 0)),
                site_mapping.get((i+1, j, k, 1)),
                site_mapping.get((i, j+1, k, 2)),
                site_mapping.get((i, j, k+1, 3))
            ]
            if None not in tet2:
                tetrahedra.append(tet2)
                # Add edges within tetrahedron
                for v1, v2 in itertools.combinations(tet2, 2):
                    G.add_edge(v1, v2)
    
    return G, pos, tetrahedra

def build_tetrahedron_graph(tetrahedra):
    """Build a graph where nodes are tetrahedra and edges represent shared vertices."""
    tet_graph = nx.Graph()
    
    for i, tet1 in enumerate(tetrahedra):
        tet_graph.add_node(i)
        for j, tet2 in enumerate(tetrahedra):
            if i < j:  # Avoid duplicate checks
                # Check if tetrahedra share a vertex
                if set(tet1).intersection(set(tet2)):
                    tet_graph.add_edge(i, j)
    
    return tet_graph

def generate_clusters(tet_graph, max_order):
    """
    Generate all topologically distinct clusters up to max_order and their multiplicities.
    
    Args:
        tet_graph: NetworkX graph where nodes are tetrahedra and edges connect adjacent tetrahedra
        max_order: Maximum number of tetrahedra in a cluster
        
    Returns:
        distinct_clusters: List of topologically distinct clusters
        multiplicities: List of multiplicities for each distinct cluster
    """
    
    distinct_clusters = []
    multiplicities = []
    
    # Process each order
    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order}...")
        
        # For order 1, all tetrahedra are equivalent in a uniform lattice
        if order == 1:
            # Take one representative tetrahedron
            first_tet = list(tet_graph.nodes())[0]
            distinct_clusters.append([first_tet])
            multiplicities.append(0.5)
            continue
        
        # For higher orders, generate all possible connected subgraphs
        all_subgraphs = []
        
        # Start from each tetrahedron and grow clusters
        for start_tet in tet_graph.nodes():
            # Use BFS to systematically grow clusters
            queue = collections.deque([(frozenset([start_tet]), set(tet_graph.neighbors(start_tet)))])
            visited_configurations = set()
            
            while queue:
                current, frontier = queue.popleft()
                
                # Skip if we've seen this configuration before
                if current in visited_configurations:
                    continue
                visited_configurations.add(current)
                
                if len(current) == order:
                    all_subgraphs.append(current)
                    continue
                
                if len(current) > order:
                    continue
                
                # Try adding each frontier tetrahedron
                for next_tet in frontier:
                    new_set = current | {next_tet}
                    
                    # Update frontier with neighbors of the new tetrahedron
                    new_frontier = frontier | set(tet_graph.neighbors(next_tet))
                    new_frontier -= new_set  # Remove tetrahedra already in the set
                    
                    queue.append((new_set, new_frontier))
        
        # Remove duplicates
        unique_subgraphs = set(all_subgraphs)
        
        # First, group by isomorphism class
        isomorphism_groups = []

        for subgraph_nodes in unique_subgraphs:
            subgraph = tet_graph.subgraph(subgraph_nodes)
            
            found_match = False
            for idx, group in enumerate(isomorphism_groups):
                rep_nodes = group[0]  # Use first graph as temporary representative
                rep_subgraph = tet_graph.subgraph(rep_nodes)
                
                if nx.is_isomorphic(subgraph, rep_subgraph):
                    isomorphism_groups[idx].append(subgraph_nodes)
                    found_match = True
                    break
            
            if not found_match:
                isomorphism_groups.append([subgraph_nodes])

        # Now select the most symmetric representative for each group
        isomorphism_classes = []

        for group in isomorphism_groups:
            # Find the graph with the most automorphisms
            max_automorphisms = 0
            most_symmetric = None
            
            for nodes in group:
                subgraph = tet_graph.subgraph(nodes)
                max_clique_size = nx.algorithms.clique.node_clique_number(subgraph)
                # Get the largest clique size across all nodes as a measure of symmetry
                num_automorphisms = max(max_clique_size.values()) if max_clique_size else 0

                if num_automorphisms > max_automorphisms:
                    max_automorphisms = num_automorphisms
                    most_symmetric = nodes
            
            # Use the most symmetric graph as the representative
            isomorphism_classes.append((most_symmetric, group))

        # Add to results and compute multiplicities independent of L
        for rep_nodes, _embeddings in isomorphism_classes:
            cluster = list(rep_nodes)
            distinct_clusters.append(cluster)
            multiplicities.append(compute_cluster_multiplicity(cluster, tet_graph))
            
        print(f"Found {len(isomorphism_classes)} distinct clusters of order {order}")
    
    return distinct_clusters, multiplicities


def visualize_cluster(lattice, pos, tetrahedra, cluster, cluster_index):
    """Visualize a single cluster in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all vertices in the cluster
    vertices = set()
    for tet_idx in cluster:
        vertices.update(tetrahedra[tet_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Draw vertices
    xs = [pos[v][0] for v in subgraph.nodes()]
    ys = [pos[v][1] for v in subgraph.nodes()]
    zs = [pos[v][2] for v in subgraph.nodes()]
    ax.scatter(xs, ys, zs, c='r', s=100, label='Vertices')
    
    # Draw edges
    for u, v in subgraph.edges():
        ax.plot([pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]], 'k-', lw=1)
    
    # Draw tetrahedra
    for tet_idx in cluster:
        tet = tetrahedra[tet_idx]
        # Draw faces of tetrahedron
        for face in itertools.combinations(tet, 3):
            triangle = np.array([pos[v] for v in face])
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                          color='b', alpha=0.2)
    
    ax.set_title(f'Cluster {cluster_index} - {len(cluster)} tetrahedra')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster_index}_order_{len(cluster)}.png')
    plt.close()

def identify_subclusters(distinct_clusters, tet_graph):
    """
    Identify all topological subclusters for each distinct cluster and their multiplicities.
    
    Args:
    - distinct_clusters: List of topologically distinct clusters
    - tet_graph: NetworkX graph where nodes are tetrahedra
    
    Returns:
    - subclusters_info: Dictionary mapping cluster indices to their subclusters info
    """
    # Group distinct clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i, cluster))
    
    # Initialize results
    subclusters_info = {}
    
    # Process each distinct cluster
    for i, cluster in enumerate(distinct_clusters):
        cluster_order = len(cluster)
        subclusters_info[i] = []
        
        # Skip if cluster_order is 1 (no subclusters)
        if cluster_order == 1:
            continue
        
        # Find subclusters for each lower order
        for order in range(1, cluster_order):
            # Get candidate distinct clusters of this order
            candidates = clusters_by_order[order]
            
            # Track subcluster counts for this order
            subcluster_counts = defaultdict(int)
            
            # Generate all subclusters of the current order
            for subcluster_set in itertools.combinations(cluster, order):
                subcluster = frozenset(subcluster_set)
                subgraph = tet_graph.subgraph(subcluster)
                
                # Find matching distinct cluster
                for cand_idx, cand_cluster in candidates:
                    cand_subgraph = tet_graph.subgraph(cand_cluster)
                    if nx.is_isomorphic(subgraph, cand_subgraph):
                        subcluster_counts[cand_idx] += 1
                        break
            
            # Add to results
            for subcluster_idx, count in subcluster_counts.items():
                subclusters_info[i].append((subcluster_idx, count))
        
        # Sort by order
        subclusters_info[i].sort(key=lambda x: len(distinct_clusters[x[0]]))
    
    return subclusters_info

def save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir):
    """
    Save information about subclusters of each distinct cluster to a file.
    
    Args:
    - subclusters_info: Dictionary mapping cluster indices to their subclusters info
    - distinct_clusters: List of topologically distinct clusters
    - multiplicities: List of multiplicities for each cluster
    - output_dir: Directory to save the output file
    """
    with open(f"{output_dir}/subclusters_info.txt", 'w') as f:
        f.write("# Subclusters information for each topologically distinct cluster\n")
        f.write("# Format: Cluster_ID, Order, Multiplicity, Subclusters[(ID, Multiplicity), ...]\n\n")
        
        for i, cluster in enumerate(distinct_clusters):
            cluster_id = i + 1
            order = len(cluster)
            multiplicity = multiplicities[i]
            
            subclusters = subclusters_info.get(i, [])
            subcluster_str = ", ".join([f"({j+1}, {count})" for j, count in subclusters])
            
            f.write(f"Cluster {cluster_id} (Order {order}):\n")
            if subclusters:
                f.write(f"  Subclusters: {subcluster_str}\n")
            else:
                f.write("  No subclusters (order 1 cluster)\n")
            f.write("\n")


def main():
    args = parse_arguments()
    max_order = args.max_order
    
    # Set lattice size
    L = args.lattice_size if args.lattice_size > 0 else max(10, 6*max_order)
    
    print(f"Generating pyrochlore lattice of size {L}×{L}×{L}...")
    lattice, pos, tetrahedra = create_pyrochlore_lattice(L)
    print(f"Generated lattice with {lattice.number_of_nodes()} sites and {len(tetrahedra)} tetrahedra")
    
    print("Building tetrahedron adjacency graph...")
    tet_graph = build_tetrahedron_graph(tetrahedra)
    
    print(f"Generating clusters up to order {max_order}...")
    distinct_clusters, multiplicities = generate_clusters(tet_graph, max_order)
    
    # Organize clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        clusters_by_order[order].append((i, cluster, multiplicities[i]))
    
    # Print results
    print("\nCluster statistics:")
    for order in sorted(clusters_by_order.keys()):
        print(f"  Order {order}: {len(clusters_by_order[order])} distinct clusters")
    
    # Create output directory for cluster info
    output_dir = args.output_dir + f"/cluster_info_order_{max_order}"
    os.makedirs(output_dir, exist_ok=True)


    save_subclusters_info(identify_subclusters(distinct_clusters, tet_graph), distinct_clusters, multiplicities, output_dir)
    print(f"Subclusters information saved to {output_dir}/subclusters_info.txt")
    
    # Extract and save detailed information for each cluster
    print("\nExtracting and saving detailed cluster information...")
    for i, (cluster, multiplicity) in enumerate(zip(distinct_clusters, multiplicities)):
        cluster_id = i + 1
        order = len(cluster)
        print(f"  Processing cluster {cluster_id} (order {order})...")
        
        # Extract detailed information
        cluster_info = extract_cluster_info(lattice, pos, tetrahedra, cluster)
        
        # Save to file
        save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir)

        print(f"  Saved to {output_dir}/cluster_{cluster_id}_order_{order}.dat")

    
    print(f"Detailed cluster information saved to {output_dir}/ directory")
    
    # Visualize clusters if requested
    if args.visualize:
        print("\nVisualizing clusters...")
        for i, cluster in enumerate(distinct_clusters):
            visualize_cluster(lattice, pos, tetrahedra, cluster, i+1)
        print(f"Visualization images saved as cluster_*.png")

if __name__ == "__main__":
    main()