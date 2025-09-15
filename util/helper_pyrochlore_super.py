import numpy as np
import sys
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

import matplotlib.pyplot as plt

def generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc=False):
    """
    Generate a pyrochlore lattice cluster with tetrahedra stacked in a tetrahedral fashion
    
    Tetrahedron centers are at:
    - (0, 0, 0)
    - (0, 1/2, 1/2)
    - (1/2, 0, 1/2)
    - (1/2, 1/2, 0)
    
    Each tetrahedron has 4 sites at offsets:
    - (0, 0, 0)
    - (0, 0.25, 0.25)
    - (0.25, 0, 0.25)
    - (0.25, 0.25, 0)
    
    Args:
        dim1, dim2, dim3: Dimensions of the lattice (number of unit cells)
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        tetrahedra: List of (v1, v2, v3, v4) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertex_to_cell: Map vertex_id to (i, j, k, tet_idx, site_idx)
    """
    # Tetrahedron centers in the unit cell
    tetrahedron_centers = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Site offsets within each tetrahedron
    site_offsets = np.array([
        [0, 0, 0],
        [0, 0.25, 0.25],
        [0.25, 0, 0.25],
        [0.25, 0.25, 0]
    ])

    # Unit cell lattice vectors
    lattice_vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Generate vertices
    vertices = {}
    vertex_id = 0
    vertex_to_cell = {}  # Map vertex_id to (i, j, k, tet_idx, site_idx)
    cell_to_vertex = {}  # Map (i, j, k, tet_idx, site_idx) to vertex_id
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                # Unit cell position
                unit_cell_pos = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                
                # For each tetrahedron in the unit cell
                for tet_idx in range(4):
                    tet_center = unit_cell_pos + tetrahedron_centers[tet_idx]
                    
                    # For each site in the tetrahedron
                    for site_idx in range(4):
                        position = tet_center + site_offsets[site_idx]
                        vertices[vertex_id] = tuple(position)
                        vertex_to_cell[vertex_id] = (i, j, k, tet_idx, site_idx)
                        cell_to_vertex[(i, j, k, tet_idx, site_idx)] = vertex_id
                        vertex_id += 1
    
    # Helper function to get vertex id with PBC
    def get_vertex_with_pbc(i, j, k, tet_idx, site_idx):
        if use_pbc:
            # Apply periodic boundary conditions
            i = i % dim1
            j = j % dim2
            k = k % dim3
        else:
            # Check if out of bounds for open boundary conditions
            if i < 0 or i >= dim1 or j < 0 or j >= dim2 or k < 0 or k >= dim3:
                return None
        
        key = (i, j, k, tet_idx, site_idx)
        return cell_to_vertex.get(key, None)
    
    # Generate edges based on nearest neighbors
    edges = set()  # Use set to avoid duplicates
    
    for v1_id, v1_info in vertex_to_cell.items():
        i1, j1, k1, tet1, site1 = v1_info
        pos1 = np.array(vertices[v1_id])
        
        # Check neighbors within same unit cell and adjacent cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    # Check all tetrahedra
                    for tet2 in range(4):
                        # Check all sites in the tetrahedron
                        for site2 in range(4):
                            v2_id = get_vertex_with_pbc(i1+di, j1+dj, k1+dk, tet2, site2)
                            
                            if v2_id is None or v2_id == v1_id:
                                continue
                            
                            # Calculate actual distance (considering PBC if needed)
                            pos2 = np.array(vertices[v2_id])
                            
                            if use_pbc:
                                # Calculate minimum image distance
                                delta = pos2 - pos1
                                # Apply minimum image convention
                                for dim_idx in range(3):
                                    box_size = [dim1, dim2, dim3][dim_idx]
                                    if delta[dim_idx] > box_size/2:
                                        delta[dim_idx] -= box_size
                                    elif delta[dim_idx] < -box_size/2:
                                        delta[dim_idx] += box_size
                                dist = np.linalg.norm(delta)
                            else:
                                dist = np.linalg.norm(pos2 - pos1)
                            
                            # Check if they are nearest neighbors
                            # Threshold for nearest neighbors in pyrochlore
                            if dist < 0.4 and dist > 0.01:  # Avoid self-connections
                                edge = tuple(sorted([v1_id, v2_id]))
                                edges.add(edge)
    
    # Convert edges set to list
    edges = list(edges)
    
    # Generate tetrahedra list
    tetrahedra = []
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for tet_idx in range(4):
                    # Get the 4 vertices of this tetrahedron
                    tet_vertices = []
                    for site_idx in range(4):
                        v_id = cell_to_vertex[(i, j, k, tet_idx, site_idx)]
                        tet_vertices.append(v_id)
                    tetrahedra.append(tuple(tet_vertices))
    
    # Create node mapping
    node_mapping = {i: i for i in range(len(vertices))}
    
    return vertices, edges, tetrahedra, node_mapping, vertex_to_cell

def get_sublattice_index(vertex_id, vertex_to_cell):
    """
    Return the sublattice index based on which site in the tetrahedron
    """
    if vertex_id in vertex_to_cell:
        _, _, _, _, site_idx = vertex_to_cell[vertex_id]
        return site_idx
    return vertex_id % 4

def create_nn_lists(edges, node_mapping, vertices, vertex_to_cell):
    """
    Create nearest neighbor lists from the edge information
    
    Args:
        edges: List of (vertex1, vertex2) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y, z)}
        vertex_to_cell: Map vertex_id to (i, j, k, tet_idx, site_idx)
        
    Returns:
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
    """
    nn_list = {}
    positions = {}
    sublattice_indices = {}
    
    # Initialize empty lists for all vertices
    for vertex_id in vertices:
        nn_list[vertex_id] = []
        positions[vertex_id] = vertices[vertex_id]
        sublattice_indices[vertex_id] = get_sublattice_index(vertex_id, vertex_to_cell)
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    return nn_list, positions, sublattice_indices

def write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping):
    """
    Write nearest neighbor list, positions, and sublattice indices to a file
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Write nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_nn_list.dat", 'w') as f:
        f.write("# Nearest neighbor list for cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list.keys()):
            neighbors = nn_list[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in neighbors:
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions and sublattice indices
    with open(f"{output_dir}/positions.dat", 'w') as f:
        f.write("# Site information for cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index, x, y, z\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    # Write lattice parameters
    with open(f"{output_dir}/{cluster_name}_lattice_parameters.dat", 'w') as f:
        f.write("# Pyrochlore super lattice parameters\n\n")
        
        # Tetrahedron centers
        tetrahedron_centers = np.array([
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ])
        
        # Site offsets
        site_offsets = np.array([
            [0, 0, 0],
            [0, 0.25, 0.25],
            [0.25, 0, 0.25],
            [0.25, 0.25, 0]
        ])

        
        # Write tetrahedron centers
        f.write("# Tetrahedron centers in unit cell\n")
        f.write("# tet_index, x, y, z\n")
        for i, center in enumerate(tetrahedron_centers):
            f.write(f"{i} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")
        
        f.write("\n")
        
        # Write site offsets
        f.write("# Site offsets within each tetrahedron\n")
        f.write("# site_index, x, y, z\n")
        for i, offset in enumerate(site_offsets):
            f.write(f"{i} {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

def prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, 
                                  node_mapping, Jxx, Jyy, Jzz, h, field_dir):
    """
    Prepare Hamiltonian parameters for exact diagonalization
    """
    # Prepare Hamiltonian parameters
    Jpm = -(Jxx+Jyy)/4
    Jpmpm = (Jxx-Jyy)/4
    
    # Define local z-axes for pyrochlore lattice
    z_local = np.array([
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3)
    ])
    
    # Normalize field direction
    field_dir = np.array(field_dir)
    field_dir = field_dir / np.linalg.norm(field_dir)
    
    interALL = []
    transfer = []

    gamma = np.exp(1j*2*np.pi/3)
    # Define non-kramer pyrochlore factor
    non_kramer_factor = np.array([[0, 1, gamma, gamma**2],
                                  [1, 0, gamma**2, gamma],
                                  [gamma, gamma**2, 0, 1],
                                  [gamma**2, gamma, 1, 0]])
    
    # Generate Heisenberg interactions
    for site_id in sorted(nn_list.keys()):
        # Site index (for the Hamiltonian)
        i = site_id
        
        # Zeeman term
        sub_idx = sublattice_indices[site_id]
        local_field = h * np.dot(field_dir, z_local[sub_idx])
        transfer.append([2, node_mapping[i], -local_field, 0])
        
        # Exchange interactions
        for neighbor_id in nn_list[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                
                # Heisenberg interactions
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz, 0])  # Sz-Sz
                interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])   # S+-S-
                interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])   # S--S+
                if non_kramer:
                    Jpmpm_ = Jpmpm * non_kramer_factor[sublattice_indices[i], sublattice_indices[j]]
                else:
                    Jpmpm_ = Jpmpm
                interALL.append([1, node_mapping[i], 1, node_mapping[j], np.real(Jpmpm_), np.imag(Jpmpm_)])  # S--S-
                interALL.append([0, node_mapping[i], 0, node_mapping[j], np.real(Jpmpm_), -np.imag(Jpmpm_)])  # S+-S+
    
    # Convert to arrays
    interALL = np.array(interALL) if interALL else np.empty((0, 6))
    transfer = np.array(transfer) if transfer else np.empty((0, 4))
    
    # Write interaction and transfer files
    write_interALL(output_dir, interALL, f"InterAll.dat")
    write_transfer(output_dir, transfer, f"Trans.dat")
    
    # Write field strength
    fstrength = np.array([[h]])
    np.savetxt(f"{output_dir}/field_strength.dat", fstrength)
    
    # Write correlation functions
    max_site = len(nn_list)
    opname = ['S+', 'S-', 'Sz']
    
    for i in range(3):
        write_one_body_correlations(output_dir, i, max_site, f"one_body_correlations{opname[i]}.dat")
        for j in range(3):
            write_two_body_correlations(output_dir, i, j, max_site, f"two_body_correlations{opname[i]}{opname[j]}.dat")

def write_interALL(output_dir, interALL, file_name):
    """Write interaction parameters to a file"""
    num_param = len(interALL)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(num_param):
            f.write(f" {int(interALL[i,0]):8d} " \
                   f" {int(interALL[i,1]):8d}   " \
                   f" {int(interALL[i,2]):8d}   " \
                   f" {int(interALL[i,3]):8d}   " \
                   f" {interALL[i,4]:8f}   " \
                   f" {interALL[i,5]:8f}   " \
                   f"\n")

def write_transfer(output_dir, transfer, file_name):
    """Write transfer (field) parameters to a file"""
    num_param = len(transfer)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(num_param):
            f.write(f" {int(transfer[i,0]):8d} " \
                   f" {int(transfer[i,1]):8d}   " \
                   f" {transfer[i,2]:8f}   " \
                   f" {transfer[i,3]:8f}" \
                   f"\n")

def write_one_body_correlations(output_dir, Op, N, file_name):
    """Write one-body correlation parameters to a file"""
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {N:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            f.write(f" {Op:8d} " \
                   f" {i:8d}   " \
                   f" {1:8f}   " \
                   f" {0:8f}" \
                   f"\n")

def write_two_body_correlations(output_dir, Op1, Op2, N, file_name):
    """Write two-body correlation parameters to a file"""
    num_green_two = N * N
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_two:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            for j in range(N):
                f.write(f" {Op1:8d} " \
                       f" {i:8d}   " \
                       f" {Op2:8d}   " \
                       f" {j:8d}   " \
                       f" {1:8f}   " \
                       f" {0:8f}   " \
                       f"\n")

def plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the cluster showing sites and their nearest neighbor connections
    Publication-quality plot for Physical Review Letters
    """
    try:
        import matplotlib.pyplot as plt
        
        # Set publication-quality parameters
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['xtick.major.width'] = 0.8
        mpl.rcParams['ytick.major.width'] = 0.8
        
        # Create figure with golden ratio
        fig = plt.figure(figsize=(7, 7/1.618))
        ax = fig.add_subplot(111, projection='3d')
        
        # Modern color palette (colorblind-friendly)
        colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
        
        # Calculate positions for cleaner plotting
        positions = np.array([vertices[v] for v in vertices])
        
        # Plot vertices by sublattice
        for sub_idx in range(4):
            mask = [sublattice_indices.get(v, v % 4) == sub_idx for v in vertices]
            sub_positions = positions[mask]
            if len(sub_positions) > 0:
                ax.scatter(sub_positions[:, 0], sub_positions[:, 1], sub_positions[:, 2], 
                          s=80, c=colors[sub_idx], marker='o', alpha=0.9,
                          edgecolors='black', linewidth=0.5,
                          label=f'Sublattice {sub_idx}', depthshade=True)
        
        # Plot edges with gradient effect
        for v1, v2 in edges:
            pos1 = np.array(vertices[v1])
            pos2 = np.array(vertices[v2])
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                   'k-', alpha=0.2, linewidth=0.4, zorder=1)
        
        # Set labels with LaTeX formatting
        ax.set_xlabel(r'$x$ (a.u.)', fontsize=12, labelpad=8)
        ax.set_ylabel(r'$y$ (a.u.)', fontsize=12, labelpad=8)
        ax.set_zlabel(r'$z$ (a.u.)', fontsize=12, labelpad=8)
        
        # Remove title for cleaner publication look
        # Title information should be in figure caption
        
        # Set equal aspect ratio and limits
        positions_array = np.array(list(vertices.values()))
        max_range = np.max(positions_array)
        min_range = np.min(positions_array)
        margin = 0.1 * (max_range - min_range)
        
        ax.set_xlim([min_range - margin, max_range + margin])
        ax.set_ylim([min_range - margin, max_range + margin])
        ax.set_zlim([min_range - margin, max_range + margin])
        
        # Set viewing angle for best visualization
        ax.view_init(elev=20, azim=45)
        
        # Clean up the plot
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Add minimalist legend
        legend = ax.legend(loc='upper left', frameon=True, 
                          fancybox=False, shadow=False,
                          framealpha=0.9, edgecolor='black',
                          borderpad=0.5, columnspacing=1.0,
                          handlelength=1.5, handletextpad=0.5)
        legend.get_frame().set_linewidth(0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        # Save in multiple formats
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        # High-quality PNG
        plt.savefig(f"{output_dir}/{cluster_name}_plot.png", 
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        
        # Vector format for publication
        plt.savefig(f"{output_dir}/{cluster_name}_plot.pdf", 
                   bbox_inches='tight', pad_inches=0.02)
        
        # EPS format (some journals prefer this)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.eps", 
                   bbox_inches='tight', pad_inches=0.02)
        
        plt.close()
        
        return True
        
    except ImportError:
        print("Warning: matplotlib not installed, skipping cluster plot")
        return False

def main():
    """Main function to process command line arguments and run the program"""
    if len(sys.argv) < 13:
        print("Usage: python helper_pyrochlore_super.py Jxx Jyy Jzz h fieldx fieldy fieldz output_dir dim1 dim2 dim3 pbc [non_kramer]")
        sys.exit(1)
    
    # Parse command line arguments
    Jxx = float(sys.argv[1])
    Jyy = float(sys.argv[2])
    Jzz = float(sys.argv[3])
    h = float(sys.argv[4])
    field_dir = [float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])]
    output_dir = sys.argv[8]
    dim1 = int(sys.argv[9])
    dim2 = int(sys.argv[10])
    dim3 = int(sys.argv[11])
    use_pbc = bool(int(sys.argv[12]))
    non_kramer = bool(int(sys.argv[13])) if len(sys.argv) > 13 else False
    
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Create cluster name
    pbc_str = "pbc" if use_pbc else "obc"
    non_kramer_str = "non_kramer" if non_kramer else "kramer"
    cluster_name = f"pyrochlore_super_{dim1}x{dim2}x{dim3}_{pbc_str}_{non_kramer_str}"

    # Generate cluster
    vertices, edges, tetrahedra, node_mapping, vertex_to_cell = generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc)
    
    # Create nearest neighbor lists
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices, vertex_to_cell)
    
    # Write nearest neighbor list and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping)
    
    # Prepare Hamiltonian parameters
    prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, 
                                  node_mapping, Jxx, Jyy, Jzz, h, field_dir)

    # Plot cluster
    plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices)
    
    print(f"Generated pyrochlore super lattice cluster with dimensions {dim1}x{dim2}x{dim3}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of bonds: {len(edges)}")
    print(f"Number of tetrahedra: {len(tetrahedra)}")
    print(f"Sites per unit cell: 16 (4 tetrahedra × 4 sites)")
    print(f"Output written to: {output_dir}")

if __name__ == "__main__":
    main()