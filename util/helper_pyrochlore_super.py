import numpy as np
import sys
import os
import re
from mpl_toolkits.mplot3d import Axes3D

# filepath: /home/pc_linux/exact_diagonalization_cpp/util/helper_pyrochlore_super.py
import matplotlib.pyplot as plt

def generate_super_pyrochlore_cluster(dim1, dim2, dim3, use_pbc=False):
    """
    Generate a super pyrochlore lattice cluster
    
    Args:
        dim1, dim2, dim3: Dimensions of the lattice
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        tetrahedra: List of (v1, v2, v3, v4) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
    """
    # Tetrahedron centers
    tetra_centers = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Site positions relative to tetrahedron center
    site_offsets = np.array([
        [0.125, 0.125, 0.125],
        [0.125, -0.125, -0.125],
        [-0.125, 0.125, -0.125],
        [-0.125, -0.125, 0.125]
    ])
    
    # Lattice basis vectors
    basis = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Generate vertices
    vertices = {}
    vertex_id = 0
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for t in range(4):  # For each tetrahedron center
                    for s in range(4):  # For each site in the tetrahedron
                        # Calculate position in Cartesian coordinates
                        unit_cell_pos = i*basis[0] + j*basis[1] + k*basis[2]
                        tetra_pos = tetra_centers[t]
                        site_pos = site_offsets[s]
                        position = unit_cell_pos + tetra_pos + site_pos
                        
                        vertices[vertex_id] = tuple(position)
                        vertex_id += 1
    
    # Generate edges based on nearest neighbors
    edges = []
    tetrahedra = []
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for t in range(4):  # For each tetrahedron
                    # Calculate base index for this tetrahedron
                    base_idx = (i*dim2*dim3*16 + j*dim3*16 + k*16) + (t*4)
                    
                    # Connect sites within the same tetrahedron
                    tetra_sites = [base_idx, base_idx+1, base_idx+2, base_idx+3]
                    tetrahedra.append(tuple(tetra_sites))
                    
                    # Connect each site with the other three in the tetrahedron
                    for s1 in range(4):
                        for s2 in range(s1+1, 4):
                            edges.append((tetra_sites[s1], tetra_sites[s2]))
                    
                    # Connect to neighboring tetrahedra
                    for s in range(4):
                        site_idx = base_idx + s
                        
                        # Find nearest neighbors in other tetrahedra
                        neighbors = get_neighbors(i, j, k, t, s, dim1, dim2, dim3, use_pbc)
                        
                        for neighbor_idx in neighbors:
                            if 0 <= neighbor_idx < len(vertices) and site_idx < neighbor_idx:
                                edges.append((site_idx, neighbor_idx))
    
    # Create node mapping (identity mapping)
    node_mapping = {i: i for i in range(len(vertices))}
    
    return vertices, edges, tetrahedra, node_mapping

def get_neighbors(i, j, k, t, s, dim1, dim2, dim3, use_pbc=False):
    """
    Get nearest neighbors for a site in the super pyrochlore lattice
    
    Args:
        i, j, k: Unit cell coordinates
        t: Tetrahedron index (0-3)
        s: Site index within tetrahedron (0-3)
        dim1, dim2, dim3: Lattice dimensions
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        List of neighbor indices
    """
    neighbors = []
    
    # Define the connectivity in the super pyrochlore lattice
    # Each site is connected to sites in neighboring tetrahedra
    
    # Tetrahedron centers
    tetra_centers = np.array([
        [0, 0, 0],
        [0, -0.5, -0.5],
        [-0.5, 0, -0.5],
        [-0.5, -0.5, 0]
    ])
    
    # Site positions relative to tetrahedron center
    site_offsets = np.array([
        [0.125, 0.125, 0.125],
        [0.125, -0.125, -0.125],
        [-0.125, 0.125, -0.125],
        [-0.125, -0.125, 0.125]
    ])
    
    basis = np.array([
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])

    # Define the connectivity based on the super pyrochlore lattice structure
    # Each site is connected to 6 sites - 3 in its own tetrahedron and 3 in neighboring tetrahedra
    
    # Use position-based search for nearest neighbors
    current_pos = np.array([i, j, k]) + tetra_centers[t] + site_offsets[s]
    
    # Define the expected nearest neighbor offsets for each site
    # These are the relative positions from current site to its neighbors
    nn_offsets = []
    
    if s == 0:  # Site (1/8, 1/8, 1/8)
        # Within tetrahedron: connect to sites 1, 2, 3
        # Outside tetrahedron: specific connections based on geometry
        nn_offsets = [
            site_offsets[1] - site_offsets[0],  # To site 1 in same tetra
            site_offsets[2] - site_offsets[0],  # To site 2 in same tetra
            site_offsets[3] - site_offsets[0],  # To site 3 in same tetra
            # Connections to neighboring tetrahedra
            basis[0] + site_offsets[1] - site_offsets[0],
            basis[1] + site_offsets[2] - site_offsets[0],
            basis[2] + site_offsets[3] - site_offsets[0]
        ]
    elif s == 1:  # Site (1/8, -1/8, -1/8)
        nn_offsets = [
            site_offsets[0] - site_offsets[1],  # To site 0 in same tetra
            site_offsets[2] - site_offsets[1],  # To site 2 in same tetra
            site_offsets[3] - site_offsets[1],  # To site 3 in same tetra
            # Connections to neighboring tetrahedra
            -basis[0] + site_offsets[0] - site_offsets[1],
            -basis[0] + basis[1] + site_offsets[2] - site_offsets[1],
            -basis[0] + basis[2] + site_offsets[3] - site_offsets[1]
        ]
    elif s == 2:  # Site (-1/8, 1/8, -1/8)
        nn_offsets = [
            site_offsets[0] - site_offsets[2],  # To site 0 in same tetra
            site_offsets[1] - site_offsets[2],  # To site 1 in same tetra
            site_offsets[3] - site_offsets[2],  # To site 3 in same tetra
            # Connections to neighboring tetrahedra
            -basis[1] + site_offsets[0] - site_offsets[2],
            basis[0] - basis[1] + site_offsets[1] - site_offsets[2],
            basis[1] - basis[2] + site_offsets[3] - site_offsets[2]
        ]
    elif s == 3:  # Site (-1/8, -1/8, 1/8)
        nn_offsets = [
            site_offsets[0] - site_offsets[3],  # To site 0 in same tetra
            site_offsets[1] - site_offsets[3],  # To site 1 in same tetra
            site_offsets[2] - site_offsets[3],  # To site 2 in same tetra
            # Connections to neighboring tetrahedra
            -basis[2] + site_offsets[0] - site_offsets[3],
            basis[0] - basis[2] + site_offsets[1] - site_offsets[3],
            -basis[1] + basis[2] + site_offsets[2] - site_offsets[3]
        ]

    print(nn_offsets)

    # Search for sites at the expected neighbor positions
    tolerance = 1e-6
    for offset in nn_offsets:
        target_pos = current_pos + offset
        
        # Apply periodic boundary conditions if needed
        if use_pbc:
            # Wrap target position back into the unit cell
            wrapped_pos = target_pos.copy()
            wrapped_pos[0] = wrapped_pos[0] % dim1
            wrapped_pos[1] = wrapped_pos[1] % dim2
            wrapped_pos[2] = wrapped_pos[2] % dim3
        else:
            wrapped_pos = target_pos
        
        # Search through all vertices to find one at this position
        for i2 in range(dim1):
            for j2 in range(dim2):
                for k2 in range(dim3):
                    for t2 in range(4):
                        for s2 in range(4):
                            candidate_pos = np.array([i2, j2, k2]) + tetra_centers[t2] + site_offsets[s2]
                            
                            # Check if positions match within tolerance
                            if np.linalg.norm(candidate_pos - wrapped_pos) < tolerance:
                                neighbor_idx = (i2*dim2*dim3*16 + j2*dim3*16 + k2*16) + (t2*4) + s2
                                if 0 <= neighbor_idx < dim1*dim2*dim3*16:
                                    neighbors.append(neighbor_idx)
                                break
    # Similar patterns for other sites (1-3)
    # This is simplified - a complete implementation would include all connections
    # for all sites (0-3) in all tetrahedra (0-3)
    
    # Filter out invalid indices
    valid_neighbors = [n for n in neighbors if 0 <= n < dim1*dim2*dim3*16]
    
    return valid_neighbors

def get_sublattice_index(vertex_id):
    """Return the sublattice index within the unit cell (0-15)"""
    return vertex_id % 4

def create_nn_lists(edges, node_mapping, vertices):
    """
    Create nearest neighbor lists from the edge information
    
    Args:
        edges: List of (vertex1, vertex2) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y, z)}
        
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
        sublattice_indices[vertex_id] = get_sublattice_index(vertex_id)
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    return nn_list, positions, sublattice_indices

def write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping):
    """
    Write nearest neighbor list, positions, and sublattice indices to a file
    
    Args:
        output_dir: Directory to write output files
        cluster_name: Name of the cluster
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
        node_mapping: Dictionary mapping original IDs to matrix indices
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
            for neighbor in sorted(neighbors):
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions and sublattice indices
    with open(f"{output_dir}/{cluster_name}_site_info.dat", 'w') as f:
        f.write("# Site information for cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index, x, y, z\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    # Write lattice parameters
    with open(f"{output_dir}/{cluster_name}_lattice_parameters.dat", 'w') as f:
        f.write("# Super pyrochlore lattice parameters\n\n")
        
        # Tetrahedron centers
        tetra_centers = np.array([
            [0, 0, 0],
            [0, -0.5, -0.5],
            [-0.5, 0, -0.5],
            [-0.5, -0.5, 0]
        ])
        
        # Site positions relative to tetrahedron center
        site_offsets = np.array([
            [0.125, 0.125, 0.125],
            [0.125, -0.125, -0.125],
            [-0.125, 0.125, -0.125],
            [-0.125, -0.125, 0.125]
        ])
        
        # Lattice basis vectors
        basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Write tetrahedron centers
        f.write("# Tetrahedron centers (4 centers)\n")
        f.write("# tetra_index, x, y, z\n")
        for i, tetra in enumerate(tetra_centers):
            f.write(f"{i} {tetra[0]:.6f} {tetra[1]:.6f} {tetra[2]:.6f}\n")
        
        f.write("\n")
        
        # Write site offsets
        f.write("# Site offsets relative to tetrahedron centers (4 sites per tetrahedron)\n")
        f.write("# site_index, x, y, z\n")
        for i, site in enumerate(site_offsets):
            f.write(f"{i} {site[0]:.6f} {site[1]:.6f} {site[2]:.6f}\n")
        
        f.write("\n")
        
        # Write lattice vectors
        f.write("# Lattice vectors\n")
        f.write("# vector_index, x, y, z\n")
        for i, vector in enumerate(basis):
            f.write(f"{i} {vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")

def prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, node_mapping, Jxx, Jyy, Jzz, h, field_dir):
    """
    Prepare Hamiltonian parameters for exact diagonalization
    
    Args:
        output_dir: Directory to write output files
        non_kramer: Whether to use non-Kramers doublets model
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
        node_mapping: Dictionary mapping original IDs to matrix indices
        Jxx, Jyy, Jzz: Exchange couplings
        h: Field strength
        field_dir: Field direction (3-vector)
    """
    # Prepare Hamiltonian parameters
    Jpm = -(Jxx+Jyy)/4
    Jpmpm = (Jxx-Jyy)/4
    
    # Define local z-axes for pyrochlore lattice (repeating for each tetrahedron)
    z_local = np.array([
        # For tetrahedron 0
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3),
        # For tetrahedron 1
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3),
        # For tetrahedron 2
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3),
        # For tetrahedron 3
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
    # Define non-kramer pyrochlore factor (extended for super pyrochlore)
    non_kramer_factor = np.zeros((16, 16), dtype=complex)
    for t1 in range(4):
        for t2 in range(4):
            for s1 in range(4):
                for s2 in range(4):
                    i = t1*4 + s1
                    j = t2*4 + s2
                    if t1 == t2:  # Same tetrahedron
                        if s1 != s2:  # Different sites
                            if (s1, s2) in [(0,1), (1,0), (2,3), (3,2)]:
                                non_kramer_factor[i, j] = 1
                            elif (s1, s2) in [(0,2), (2,0), (1,3), (3,1)]:
                                non_kramer_factor[i, j] = gamma
                            elif (s1, s2) in [(0,3), (3,0), (1,2), (2,1)]:
                                non_kramer_factor[i, j] = gamma**2
                    # Add connections between tetrahedra if needed
    
    # Generate Heisenberg interactions
    for site_id in sorted(nn_list.keys()):
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
                    i_sub = i % 16
                    j_sub = j % 16
                    Jpmpm_ = Jpmpm * non_kramer_factor[i_sub, j_sub]
                else:
                    Jpmpm_ = Jpmpm
                interALL.append([1, node_mapping[i], 1, node_mapping[j], np.real(Jpmpm_), np.imag(Jpmpm_)])  # S--S-
                interALL.append([0, node_mapping[i], 0, node_mapping[j], np.real(Jpmpm_), -np.imag(Jpmpm_)])  # S+-S+
    
    # Convert to arrays
    interALL = np.array(interALL)
    transfer = np.array(transfer)
    
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
            
    # Write spin operators at specific k-points
    write_spin_operators(output_dir, 1, [0, 0, 0], "observables_S-_Gamma.dat", max_site, positions)
    write_spin_operators(output_dir, 2, [0, 0, 0], "observables_Sz_Gamma.dat", max_site, positions)
    write_spin_operators(output_dir, 1, [0, 0, 4*np.pi], "observables_S-_Gammap.dat", max_site, positions)
    write_spin_operators(output_dir, 2, [0, 0, 4*np.pi], "observables_Sz_Gammap.dat", max_site, positions)

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

def write_spin_operators(output_dir, Op, Q, file_name, N, positions):
    """Write spin operator parameters for specific k-points"""
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {N:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for site_id in range(N):
            pos = positions[site_id]
            factor = np.exp(1j*Q[0]*pos[0] + 1j*Q[1]*pos[1] + 1j*Q[2]*pos[2])
            
            f.write(f" {Op:8d} " \
                   f" {site_id:8d}   " \
                   f" {np.real(factor):8f}   " \
                   f" {np.imag(factor):8f}" \
                   f"\n")

def plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the cluster showing sites and their nearest neighbor connections
    
    Args:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        output_dir: Directory to save the plot
        cluster_name: Name of the cluster
        sublattice_indices: Dictionary mapping each site to its sublattice index
    """
    try:
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for each tetrahedron
        tetra_colors = ['red', 'green', 'blue', 'purple']
        
        # Plot vertices
        for vertex_id, pos in vertices.items():
            if sublattice_indices is None:
                sub_idx = get_sublattice_index(vertex_id)
            else:
                sub_idx = sublattice_indices[vertex_id]
            
            # Which tetrahedron this site belongs to
            tetra_idx = (sub_idx // 4) % 4
            site_idx = sub_idx % 4
            
            # Different markers for different sites within a tetrahedron
            markers = ['o', 's', '^', 'D']
            
            ax.scatter(pos[0], pos[1], pos[2], s=80, c=tetra_colors[tetra_idx], 
                      marker=markers[site_idx], edgecolors='black')
            
            # Add vertex label
            ax.text(pos[0], pos[1], pos[2], str(vertex_id), fontsize=8)
        
        # Plot edges
        for v1, v2 in edges:
            pos1 = vertices[v1]
            pos2 = vertices[v2]
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Super Pyrochlore Cluster: {cluster_name}')
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add a legend
        tetra_handles = []
        for i in range(4):
            tetra_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=f'Tetra {i}',
                          markerfacecolor=tetra_colors[i], markersize=10)
            )
        ax.legend(handles=tetra_handles, loc='upper right')
        
        # Save the figure
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.png", dpi=300, bbox_inches='tight')
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
    cluster_name = f"super_pyrochlore_{dim1}x{dim2}x{dim3}_{pbc_str}_{non_kramer_str}"

    # Generate cluster
    vertices, edges, tetrahedra, node_mapping = generate_super_pyrochlore_cluster(dim1, dim2, dim3, use_pbc)
    
    # Create nearest neighbor lists
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices)
    
    # Write nearest neighbor list and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping)
    
    # Prepare Hamiltonian parameters
    prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, node_mapping, Jxx, Jyy, Jzz, h, field_dir)

    # Plot cluster
    plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices)
    
    print(f"Generated super pyrochlore lattice cluster with dimensions {dim1}x{dim2}x{dim3}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of bonds: {len(edges)}")
    print(f"Number of tetrahedra: {len(tetrahedra)}")
    print(f"Output written to: {output_dir}")

if __name__ == "__main__":
    main()