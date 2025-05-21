import numpy as np
import sys
import os
import argparse

# filepath: /home/pc_linux/exact_diagonalization_cpp/util/helper_honeycomb.py
import matplotlib.pyplot as plt

# Honeycomb lattice parameters
# 2 sites per unit cell (A and B sublattices)
site_basis = np.array([[0, 0], [0, 1/np.sqrt(3)]])
# Lattice vectors
basis = np.array([[1, 0], [0.5, np.sqrt(3)/2]])
# Bond types (x, y, z) for each nearest neighbor direction
bond_types = ['x', 'y', 'z']
# Unit vectors for the three bond directions from sublattice A
bond_vectors = [
    np.array([0, 1/np.sqrt(3)]),              # z-bond
    np.array([np.sqrt(3)/4, -3/4/np.sqrt(3)]), # x-bond
    np.array([-np.sqrt(3)/4, -3/4/np.sqrt(3)]) # y-bond
]

#J1=-6.54, Jzp=-3.5, Jpmpm=0.15, J2=-0.21, J3=2, Delta1=0.36, Delta2=0, Delta3=0.03
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate honeycomb lattice Kitaev Hamiltonian')
    parser.add_argument('--dim1', type=int, default=4, help='Number of unit cells in the x-direction')
    parser.add_argument('--dim2', type=int, default=4, help='Number of unit cells in the y-direction')
    parser.add_argument('--pbc', type=int, default=1, help='Use periodic boundary conditions (1) or open boundary conditions (0)')
    parser.add_argument('--J1', type=float, default=-6.54, help='J1 coupling constant')
    parser.add_argument('--delta1', type=float, default=0.36, help='Delta1 coupling constant')
    parser.add_argument('--Jpmpm', type=float, default=0.15, help='Jpmpm coupling constant')
    parser.add_argument('--Jzpm', type=float, default=-3.76, help='Jzpm coupling constant')
    parser.add_argument('--J2', type=float, default=-0.21, help='J2 coupling constant')
    parser.add_argument('--J3', type=float, default=1.7, help='J3 coupling constant')
    parser.add_argument('--delta3', type=float, default=0.03, help='Delta3 coupling constant')
    parser.add_argument('--h', type=float, default=0.0, help='Field strength')
    parser.add_argument('--hx', type=float, default=1.0, help='Field strength in x-direction')
    parser.add_argument('--hy', type=float, default=0.0, help='Field strength in y-direction')
    parser.add_argument('--hz', type=float, default=0.0, help='Field strength in z-direction')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory')
    
    return parser.parse_args()

def indices_periodic_BC(i, j, u, d1, d2):
    """Generate neighbor indices with periodic boundary conditions"""
    neighbors = []

    if u == 0:  # Sublattice A
        # Z-bond neighbor
        neighbors.append([i, j, 1])
        # X-bond neighbor
        neighbors.append([np.mod(i, d1), np.mod(j-1, d2), 1])
        # Y-bond neighbor
        neighbors.append([np.mod(i+1, d1), np.mod(j-1, d2), 1])
    else:  # Sublattice B (u == 1)
        # Z-bond neighbor
        neighbors.append([i, j, 0])
        # X-bond neighbor
        neighbors.append([np.mod(i, d1), np.mod(j+1, d2), 0])
        # Y-bond neighbor
        neighbors.append([np.mod(i-1, d1), np.mod(j+1, d2), 0])
    
    return neighbors

def indices_open_BC(i, j, u, d1, d2):
    """Generate neighbor indices with open boundary conditions"""
    neighbors = []

    if u == 0:  # Sublattice A
        # Z-bond neighbor
        neighbors.append([i, j, 1])
        # X-bond neighbor
        neighbors.append([i, j-1, 1])
        # Y-bond neighbor
        neighbors.append([i+1, j-1, 1])
    else:  # Sublattice B (u == 1)
        # Z-bond neighbor
        neighbors.append([i, j, 0])
        # X-bond neighbor
        neighbors.append([i, j+1, 0])
        # Y-bond neighbor
        neighbors.append([i-1, j+1, 0])
    
    return neighbors

def indices_open_BC_NNN(i, j, u, d1, d2):
    """Generate next-nearest neighbor indices with periodic boundary conditions"""
    neighbors = []

    neighbors.append([i+1, j, u])
    neighbors.append([i, j+1, u])
    neighbors.append([i+1, j-1, u])
    neighbors.append([i-1, j, u])
    neighbors.append([i, j-1, u])
    neighbors.append([i-1, j+1, u])
    return neighbors

def indices_periodic_BC_NNN(i, j, u, d1, d2):
    """Generate next-nearest neighbor indices with periodic boundary conditions"""
    neighbors = []

    neighbors.append([np.mod(i+1, d1), j, u])
    neighbors.append([i, np.mod(j+1, d2), u])
    neighbors.append([np.mod(i+1, d1), np.mod(j-1, d2), u])
    neighbors.append([np.mod(i-1, d1), j, u])
    neighbors.append([i, np.mod(j-1, d2), u])
    neighbors.append([np.mod(i-1, d1), np.mod(j+1, d2), u])
    return neighbors

def indices_periodic_BC_NNNN(i, j, u, d1, d2):
    """Generate neighbor indices with periodic boundary conditions"""
    neighbors = []

    if u == 0:  # Sublattice A
        neighbors.append([np.mod(i+1, d1), j, 1])
        neighbors.append([np.mod(i-1, d1), j, 1])
        neighbors.append([np.mod(i+1, d1), np.mod(j-2, d2), 1])
    else:  # Sublattice B (u == 1)
        neighbors.append([np.mod(i-1, d1), j, 0])
        neighbors.append([np.mod(i+1, d1), j, 0])
        neighbors.append([np.mod(i-1, d1), np.mod(j+2, d2), 0])
    
    return neighbors

def indices_open_BC_NNNN(i, j, u, d1, d2):
    """Generate neighbor indices with open boundary conditions"""
    neighbors = []

    if u == 0:  # Sublattice A
        neighbors.append([i+1, j, 1])
        neighbors.append([i-1, j, 1])
        neighbors.append([i+1, j-2, 1])
    else:  # Sublattice B (u == 1)
        neighbors.append([i-1, j, 0])
        neighbors.append([i+1, j, 0])
        neighbors.append([i-1, j+2, 0])
    
    return neighbors


def flattenIndex(indices, dim2):
    """Convert 3D indices [i, j, u] to flattened 1D index"""
    flattened = []
    for idx in indices:
        i, j, u = idx
        flattened.append(i*dim2*2 + j*2 + u)
    return flattened

def genNN_list(d1, d2, PBC):
    """Generate nearest neighbor list with either PBC or open BC"""
    NN_list = []
    bond_list = []  # To track bond types (x, y, z)
    
    for i in range(d1):
        for j in range(d2):
            for u in range(2):  # 2 sublattices
                site_idx = i*d2*2 + j*2 + u
                
                if PBC:
                    neighbor_indices = indices_periodic_BC(i, j, u, d1, d2)
                else:
                    neighbor_indices = indices_open_BC(i, j, u, d1, d2)
                
                # Get flattened indices and corresponding bond types
                nn_flat = flattenIndex(neighbor_indices, d2)
                
                # Define bond types
                if u == 0:  # Sublattice A
                    bond_types_for_site = ['z', 'x', 'y'][:len(nn_flat)]
                else:  # Sublattice B
                    bond_types_for_site = ['z', 'x', 'y'][:len(nn_flat)]
                
                NN_list.append(nn_flat)
                bond_list.append(bond_types_for_site)
    
    return NN_list, bond_list

def genNNN_list(d1, d2, PBC):
    NN_list = []

    for i in range(d1):
        for j in range(d2):
            for u in range(2):
                site_idx = i*d2*2 + j*2 + u
                if PBC:
                    neighbor_indices = indices_periodic_BC_NNN(i, j, u, d1, d2)
                else:
                    neighbor_indices = indices_open_BC_NNN(i, j, u, d1, d2)
                
                nn_flat = flattenIndex(neighbor_indices, d2)
                NN_list.append(nn_flat)

    return NN_list

def genNNNN_list(d1, d2, PBC):
    NN_list = []

    for i in range(d1):
        for j in range(d2):
            for u in range(2):
                site_idx = i*d2*2 + j*2 + u
                if PBC:
                    neighbor_indices = indices_periodic_BC_NNNN(i, j, u, d1, d2)
                else:
                    neighbor_indices = indices_open_BC_NNNN(i, j, u, d1, d2)
                
                nn_flat = flattenIndex(neighbor_indices, d2)
                NN_list.append(nn_flat)

    return NN_list


def KitaevNN(J_values, indx1, indx2, bond_type, max_site):
    """Generate Kitaev interaction terms for a given bond"""
    J1, delta1, Jpmpm, Jzpm, J2, J3, delta3 = J_values
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        alpha = 0
        if bond_type == 'x':
            alpha = np.pi/3*2
        elif bond_type == 'y':
            alpha = -np.pi/3*2
        elif bond_type == 'z':
            alpha = 0
        return np.array([[2, indx1, 2, indx2, J1*delta1, 0],
                    [1, indx1, 0, indx2, J1/2, 0],
                    [0, indx1, 1, indx2, J1/2, 0],
                    [1, indx1, 2, indx2, Jzpm/2*np.cos(alpha), -Jzpm/2*np.sin(alpha)],
                    [2, indx1, 1, indx2, Jzpm/2*np.cos(alpha), -Jzpm/2*np.sin(alpha)],
                    [0, indx1, 2, indx2, Jzpm/2*np.cos(alpha), Jzpm/2*np.sin(alpha)],
                    [2, indx1, 0, indx2, Jzpm/2*np.cos(alpha), Jzpm/2*np.sin(alpha)],
                    [0, indx1, 0, indx2, Jpmpm*np.cos(alpha), Jpmpm*np.sin(alpha)],
                    [1, indx1, 1, indx2, Jpmpm*np.cos(alpha), -Jpmpm*np.sin(alpha)]])
    return np.array([])

def J2NNN(J_values, indx1, indx2, bond_type, max_site):
    """Generate next-nearest neighbor interaction terms"""
    J1, delta1, Jpmpm, Jzpm, J2, J3, delta3 = J_values
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        return np.array([[1, indx1, 0, indx2, J2/2, 0],[0, indx1, 1, indx2, J2/2, 0]])
    return np.array([])

def J3NNNN(J_values, indx1, indx2, bond_type, max_site):
    J1, delta1, Jpmpm, Jzpm, J2, J3, delta3 = J_values
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        return np.array([[2, indx1, 2, indx2, J3*delta3, 0],
                    [1, indx1, 0, indx2, J3/2, 0],
                    [0, indx1, 1, indx2, J3/2, 0]])
    return np.array([])

def Zeeman(h, hx, hy, hz):
    """Generate Zeeman interaction terms"""
    # Normalize field direction
    h_dir = np.array([hx, hy, hz])
    if np.linalg.norm(h_dir) > 0:
        h_dir = h_dir / np.linalg.norm(h_dir)
    
    # Scale by field strength
    hx, hy, hz = h * h_dir
    
    return np.array([
        [0, 0, -hx, 0],  # σx term
        [1, 0, -hy, 0],  # σy term
        [2, 0, -hz, 0]   # σz term
    ])

def write_interALL(interALL, output_dir, file_name):
    """Write interaction terms to file"""
    num_param = len(interALL)
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(num_param):
            f.write(f" {int(interALL[i,0]):8d} " \
                  + f" {int(interALL[i,1]):8d}   " \
                  + f" {int(interALL[i,2]):8d}   " \
                  + f" {int(interALL[i,3]):8d}   " \
                  + f" {interALL[i,4]:8f}   " \
                  + f" {interALL[i,5]:8f}   " \
                  + "\n")

def write_transfer(interALL, output_dir, file_name):
    """Write transfer terms to file"""
    num_param = len(interALL)
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(num_param):
            f.write(f" {int(interALL[i,0]):8d} " \
                  + f" {int(interALL[i,1]):8d}   " \
                  + f" {interALL[i,2]:8f}   " \
                  + f" {interALL[i,3]:8f}" \
                  + "\n")

def write_site_positions(output_dir, dim1, dim2):
    """Write site positions to a file in the format: index, x, y"""
    with open(output_dir+"site_positions.dat", 'wt') as f:
        f.write("# index, x, y\n")
        
        for i in range(dim1):
            for j in range(dim2):
                for u in range(2):  # 2 sublattices
                    # Calculate site index
                    site_index = i*dim2*2 + j*2 + u
                    
                    # Calculate position in Cartesian coordinates
                    unit_cell_pos = i*basis[0] + j*basis[1]
                    position = unit_cell_pos + site_basis[u]
                    
                    # Write to file
                    f.write(f"{site_index} {position[0]:.6f} {position[1]:.6f}\n")

def write_lattice_parameters(output_dir):
    """Write the unit cell site basis and lattice vectors to a file"""
    with open(output_dir+"lattice_parameters.dat", 'wt') as f:
        f.write("# Honeycomb lattice parameters\n\n")
        
        # Write site basis
        f.write("# Unit cell site basis (2 sites per unit cell)\n")
        f.write("# site_index, x, y\n")
        for i, site in enumerate(site_basis):
            f.write(f"{i} {site[0]:.6f} {site[1]:.6f}\n")
        
        f.write("\n")
        
        # Write lattice vectors
        f.write("# Lattice vectors\n")
        f.write("# vector_index, x, y\n")
        for i, vector in enumerate(basis):
            f.write(f"{i} {vector[0]:.6f} {vector[1]:.6f}\n")

def lattice_pos(site_indx, dim1, dim2):
    """Convert site index to 2D lattice position"""
    i  = site_indx // (dim2 * 2)
    j = (site_indx // 2) % dim2
    u = site_indx % 2

    position = i * basis[0] + j * basis[1] + site_basis[u]

    return position

def plot_honeycomb_lattice(output_dir, dim1, dim2, use_pbc, NN_list, bond_list):
    """Plot the honeycomb lattice showing sites and their bond-dependent interactions"""
    # Calculate all site positions
    positions = []
    sublattice_indices = []
    
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                # Calculate position in Cartesian coordinates
                unit_cell_pos = i*basis[0] + j*basis[1]
                position = unit_cell_pos + site_basis[u]
                positions.append(position)
                sublattice_indices.append(u)
    
    # Convert to numpy array for easier handling
    positions = np.array(positions)
    sublattice_indices = np.array(sublattice_indices)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Colors for each sublattice
    colors = ['r', 'b']
    
    # Plot sites as scatter points, color by sublattice
    for u in range(2):
        mask = sublattice_indices == u
        plt.scatter(
            positions[mask, 0], 
            positions[mask, 1],
            s=80,  # Point size
            c=colors[u],
            marker='o',
            label=f'Sublattice {u}'
        )
    
    # Connect nearest neighbors with colored lines based on bond type
    bond_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    
    for site_idx in range(len(NN_list)):
        site_pos = positions[site_idx]
        for idx, neighbor_idx in enumerate(NN_list[site_idx]):
            if idx < len(bond_list[site_idx]):  # Ensure we have a bond type for this neighbor
                bond_type = bond_list[site_idx][idx]
                neighbor_pos = positions[neighbor_idx]
                
                # Only draw each connection once (from smaller to larger index)
                if site_idx < neighbor_idx:
                    plt.plot(
                        [site_pos[0], neighbor_pos[0]],
                        [site_pos[1], neighbor_pos[1]],
                        color=bond_colors[bond_type],
                        alpha=0.7,
                        linewidth=2,
                        label=f'{bond_type}-bond' if (site_idx == 0 and idx == 0) else ""
                    )
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    bc_type = "PBC" if use_pbc else "Open BC"
    plt.title(f'Honeycomb Lattice ({dim1}x{dim2} unit cells, {bc_type})')
    
    # Add a legend (with unique entries)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Equal aspect ratio
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(f"{output_dir}honeycomb_lattice.png", dpi=300, bbox_inches='tight')
    plt.close()

def one_body_correlations(file_name, All_N, output_dir):
    num_green_one = All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_one:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(3):  # x, y, z components
            for all_i in range(0, All_N):
                f.write(f" {i:8d} {all_i:8d}   1.000000   0.000000\n")

def two_body_correlations(file_name, All_N, output_dir):
    num_green_two = All_N*All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_two:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(3):  # x, y, z components for first site
            for j in range(3):  # x, y, z components for second site
                for all_i in range(0, All_N):
                    for all_j in range(0, All_N):
                        f.write(f" {i:8d} {all_i:8d}   {j:8d}   {all_j:8d}   1.000000   0.000000\n")

def spin_operators(Op, Q, file_name, All_N, dim1, dim2, output_dir):
    num_green_one = All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_one:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for all_i in range(0, All_N):
            pos = lattice_pos(all_i, dim1, dim2)
            factor = np.exp(1j*Q[0]*pos[0]+1j*Q[1]*pos[1])
            
            f.write(f" {Op:8d} {all_i:8d}   {np.real(factor):8f}   {np.imag(factor):8f}\n")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Make global variables accessible in functions
    global dim1, dim2
    dim1, dim2 = args.dim1, args.dim2
    
    # Extract parameters
    use_pbc = args.pbc == 1
    J1, delta1, Jpmpm, Jzpm, J2, J3, delta3 = args.J1, args.delta1, args.Jpmpm, args.Jzpm, args.J2, args.J3, args.delta3
    J_values = [J1, delta1, Jpmpm, Jzpm, J2, J3, delta3]

    
    # Field parameters
    h = args.h
    hx, hy, hz = args.hx, args.hy, args.hz
    
    # Output directory
    output_dir = args.outdir + "/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Generate nearest neighbor list with specified boundary condition
    NN_list, bond_list = genNN_list(dim1, dim2, use_pbc)
    NNN_list = genNNN_list(dim1, dim2, use_pbc)
    NNNN_list = genNNNN_list(dim1, dim2, use_pbc)

    max_site = dim1*dim2*2  # 2 sites per unit cell


    # Generate all interactions
    interALL = []
    transfer = []
    
    # Generate Zeeman terms for all sites
    zeeman_term = Zeeman(h, hx, hy, hz)
    for i in range(max_site):
        for term in zeeman_term:
            transfer.append(np.array([[term[0], i, term[2], term[3]]]))
    
    # Generate Kitaev interaction terms
    for site_idx in range(max_site):
        for j, neighbor_idx in enumerate(NN_list[site_idx]):
            if j < len(bond_list[site_idx]):  # Ensure we have a bond type for this neighbor
                bond_type = bond_list[site_idx][j]
                if site_idx < neighbor_idx:  # Only add each bond once
                    term = KitaevNN(J_values, site_idx, neighbor_idx, bond_type, max_site)
                    if term.size > 0:
                        interALL.append(term)
        
        # Generate next-nearest neighbor interaction terms
        for j, neighbor_idx in enumerate(NNN_list[site_idx]):
            if site_idx < neighbor_idx:
                term = J2NNN(J_values, site_idx, neighbor_idx, bond_type, max_site)
                if term.size > 0:
                    interALL.append(term)
        # Generate next-next-nearest neighbor interaction terms
        for j, neighbor_idx in enumerate(NNNN_list[site_idx]):
            if site_idx < neighbor_idx:
                term = J3NNNN(J_values, site_idx, neighbor_idx, bond_type, max_site)
                if term.size > 0:
                    interALL.append(term)
    
    interALL = np.vstack([arr for arr in interALL if arr.size > 0]) if interALL else np.array([])
    transfer = np.vstack([arr for arr in transfer if arr.size > 0]) if transfer else np.array([])
    
    # Write field strength
    fstrength = np.zeros((1,1))
    fstrength[0,0] = h
    np.savetxt(output_dir+"field_strength.dat", fstrength)
    
    # Write boundary condition info
    with open(output_dir+"boundary_condition.dat", 'w') as f:
        f.write(f"PBC: {'True' if use_pbc else 'False'}\n")
        f.write(f"Dimensions: {dim1}x{dim2}\n")
    
    # Write output files
    write_interALL(interALL, output_dir, "InterAll.dat")
    write_transfer(transfer, output_dir, "Trans.dat")
    write_site_positions(output_dir, dim1, dim2)
    write_lattice_parameters(output_dir)
    plot_honeycomb_lattice(output_dir, dim1, dim2, use_pbc, NN_list, bond_list)

    # Write one-body and two-body correlation functions
    one_body_correlations("one_body_correlations.dat", max_site, output_dir)
    two_body_correlations("two_body_correlations.dat", max_site, output_dir)

    # Write spin operators for different k-points
    spin_operators(0, [0, 0], "observables_S+_Gamma.dat", max_site, dim1, dim2, output_dir)
    spin_operators(1, [0, 0], "observables_S-_Gamma.dat", max_site, dim1, dim2, output_dir)
    spin_operators(2, [0, 0], "observables_Sz_Gamma.dat", max_site, dim1, dim2, output_dir)

    spin_operators(0, [np.pi, 0], "observables_S+_M.dat", max_site, dim1, dim2, output_dir)
    spin_operators(1, [np.pi, 0], "observables_S-_M.dat", max_site, dim1, dim2, output_dir)
    spin_operators(2, [np.pi, 0], "observables_Sz_M.dat", max_site, dim1, dim2, output_dir)

    print(f"Generated honeycomb lattice Kitaev Hamiltonian with dimensions {dim1}x{dim2}")
    print(f"Boundary conditions: {'Periodic' if use_pbc else 'Open'}")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()