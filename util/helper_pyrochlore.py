import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse

# Constants and configurations
# Local z-axes for the four sublattices of pyrochlore
z = np.array([np.array([1,1,1])/np.sqrt(3), 
             np.array([1,-1,-1])/np.sqrt(3), 
             np.array([-1,1,-1])/np.sqrt(3), 
             np.array([-1,-1,1])/np.sqrt(3)])

# Pyrochlore lattice parameters
site_basis = np.array([[0.125,0.125,0.125], [0.125,-0.125,-0.125], 
                        [-0.125,0.125,-0.125], [-0.125,-0.125,0.125]])
basis = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate pyrochlore lattice Hamiltonian')
    parser.add_argument('--Jxx', type=float, required=True, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, required=True, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, required=True, help='Jzz coupling')
    parser.add_argument('--h', type=float, required=True, help='Field strength')
    parser.add_argument('--hx', type=float, required=True, help='Field x component')
    parser.add_argument('--hy', type=float, required=True, help='Field y component')
    parser.add_argument('--hz', type=float, required=True, help='Field z component')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--dim1', type=int, required=True, help='Lattice dimension 1')
    parser.add_argument('--dim2', type=int, required=True, help='Lattice dimension 2')
    parser.add_argument('--dim3', type=int, required=True, help='Lattice dimension 3')
    parser.add_argument('--pbc', type=int, default=0, help='Periodic boundary conditions (1=on, 0=off)')
    
    return parser.parse_args()

def indices_periodic_BC(i, j, k, u, d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], 
                          [np.mod(i-1, d1), j, k, 1], [i, np.mod(j-1, d2), k, 2], [i, j, np.mod(k-1, d3), 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], 
                          [np.mod(i+1, d1), j, k, 0], [np.mod(i+1, d1), np.mod(j-1, d2), k, 2], [np.mod(i+1, d1), j, np.mod(k-1, d3), 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], 
                          [i, np.mod(j+1, d2), k, 0], [np.mod(i-1, d1), np.mod(j+1, d2), k, 1], [i, np.mod(j+1, d2), np.mod(k-1, d3), 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], 
                          [i, j, np.mod(k+1, d3), 0], [np.mod(i-1, d1), j, np.mod(k+1, d3), 1], [i, np.mod(j-1, d2), np.mod(k+1, d3), 2]])

def indices_open_BC(i, j, k, u, d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], 
                          [i-1, j, k, 1], [i, j-1, k, 2], [i, j, k-1, 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], 
                          [i+1, j, k, 0], [i+1, j-1, k, 2], [i+1, j, k-1, 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], 
                          [i, j+1, k, 0], [i-1, j+1, k, 1], [i, j+1, k-1, 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], 
                          [i, j, k+1, 0], [i-1, j, k+1, 1], [i, j-1, k+1, 2]])

def flattenIndex(Indx):
    """Convert 4D indices to flattened 1D indices"""
    temp = np.zeros(len(Indx))
    for i in range(6):
        temp[i] = Indx[i][0]*dim2*dim3*4 + Indx[i][1]*dim3*4 + Indx[i][2]*4 + Indx[i][3]
    return temp

def genNN_list(d1, d2, d3, PBC):
    """Generate nearest neighbor list with either PBC or open BC"""
    NN_list = np.zeros((d1*d2*d3*4, 6))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for u in range(4):
                    site_idx = i*d2*d3*4 + j*d3*4 + k*4 + u
                    if PBC:
                        indices = indices_periodic_BC(i, j, k, u, d1, d2, d3)
                    else:
                        indices = indices_open_BC(i, j, k, u, d1, d2, d3)
                    NN_list[site_idx] = flattenIndex(indices)
    return NN_list

def HeisenbergNN(Jzz, Jpm, Jpmpm, indx1, indx2, max_site):
    """Generate Heisenberg interaction terms"""
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        Jzz = Jzz/2 
        Jpm = Jpm/2
        Jpmpm = Jpmpm/2
        return np.array([
            [2, indx1, 2, indx2, Jzz, 0],
            [0, indx1, 1, indx2, -Jpm, 0],
            [1, indx1, 0, indx2, -Jpm, 0],
            [1, indx1, 1, indx2, Jpmpm, 0],
            [0, indx1, 0, indx2, Jpmpm, 0]
        ])
    return np.array([])

def Zeeman(h, indx):
    """Generate Zeeman interaction terms"""
    here = h[indx % 4]
    return np.array([[2, indx, -here, 0]])

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

def write_site_positions(output_dir, dim1, dim2, dim3):
    """Write site positions to a file in the format: index, x, y, z"""
    with open(output_dir+"site_positions.dat", 'wt') as f:
        f.write("# index, x, y, z\n")
        
        for i in range(dim1):
            for j in range(dim2):
                for k in range(dim3):
                    for u in range(4):
                        # Calculate site index
                        site_index = i*dim2*dim3*4 + j*dim3*4 + k*4 + u
                        
                        # Calculate position in Cartesian coordinates
                        unit_cell_pos = i*basis[0] + j*basis[1] + k*basis[2]
                        position = unit_cell_pos + site_basis[u]
                        
                        # Write to file
                        f.write(f"{site_index} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n")

def write_lattice_parameters(output_dir):
    """Write the unit cell site basis and lattice vectors to a file"""
    with open(output_dir+"lattice_parameters.dat", 'wt') as f:
        f.write("# Pyrochlore lattice parameters\n\n")
        
        # Write site basis
        f.write("# Unit cell site basis (4 sites per unit cell)\n")
        f.write("# site_index, x, y, z\n")
        for i, site in enumerate(site_basis):
            f.write(f"{i} {site[0]:.6f} {site[1]:.6f} {site[2]:.6f}\n")
        
        f.write("\n")
        
        # Write lattice vectors
        f.write("# Lattice vectors\n")
        f.write("# vector_index, x, y, z\n")
        for i, vector in enumerate(basis):
            f.write(f"{i} {vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")

def plot_pyrochlore_lattice(output_dir, dim1, dim2, dim3, use_pbc, look_up_table):
    """Plot the pyrochlore lattice showing sites and their nearest neighbor connections"""
    # Calculate all site positions
    positions = []
    sublattice_indices = []
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for u in range(4):
                    # Calculate position in Cartesian coordinates
                    unit_cell_pos = i*basis[0] + j*basis[1] + k*basis[2]
                    position = unit_cell_pos + site_basis[u]
                    positions.append(position)
                    sublattice_indices.append(u)
    
    # Convert to numpy array for easier handling
    positions = np.array(positions)
    sublattice_indices = np.array(sublattice_indices)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for each sublattice
    colors = ['r', 'g', 'b', 'purple']
    
    # Plot sites as scatter points, color by sublattice
    for u in range(4):
        mask = sublattice_indices == u
        ax.scatter(
            positions[mask, 0], 
            positions[mask, 1], 
            positions[mask, 2],
            s=80,  # Point size
            c=colors[u],
            marker='o',
            label=f'Sublattice {u}'
        )
    
    # Connect nearest neighbors
    for site_idx in range(len(look_up_table)):
        site_pos = positions[site_idx]
        
        for neighbor_idx in look_up_table[site_idx]:
            # Convert to integer and check bounds
            neighbor_idx = int(neighbor_idx)
            if 0 <= neighbor_idx < len(positions):
                # Only draw each connection once to avoid duplicate lines
                if site_idx < neighbor_idx:
                    neighbor_pos = positions[neighbor_idx]
                    
                    ax.plot(
                        [site_pos[0], neighbor_pos[0]],
                        [site_pos[1], neighbor_pos[1]],
                        [site_pos[2], neighbor_pos[2]],
                        'k-', alpha=0.3, linewidth=1
                    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    bc_type = "PBC" if use_pbc else "Open BC"
    ax.set_title(f'Pyrochlore Lattice ({dim1}x{dim2}x{dim3} unit cells, {bc_type})')
    
    # Add a legend
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure
    plt.savefig(f"{output_dir}pyrochlore_lattice.png", dpi=300, bbox_inches='tight')
    plt.close()


def one_body_correlations(file_name, All_N, output_dir):
    num_green_one  = All_N
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("loc "+"{0:8d}".format(num_green_one)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(3):
        for all_i in range(0,All_N):
            f.write(" {0:8d} ".format(i) \
            +" {0:8d}   ".format(all_i)     \
            +" {0:8f}   ".format(1)     \
            +" {0:8f}   ".format(0)
            +"\n")
    f.close()


def two_body_correlations(file_name, All_N, output_dir):
    num_green_two  = All_N*All_N
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("loc "+"{0:8d}".format(num_green_two)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(3):
        for j in range(3):
            for all_i in range(0,All_N):
                for all_j in range(0,All_N):
                    f.write(" {0:8d} ".format(i) \
                    +" {0:8d}   ".format(all_i)     \
                    +" {0:8d}   ".format(j)     \
                    +" {0:8d}   ".format(all_j)     \
                    +" {0:8f}   ".format(1) \
                    +" {0:8f}   ".format(0) \
                    +"\n")
    f.close()



def main():
    # Parse arguments
    args = parse_arguments()
    
    # Make global variables accessible in functions
    global dim1, dim2, dim3
    dim1, dim2, dim3 = args.dim1, args.dim2, args.dim3
    
    # Extract parameters
    use_pbc = args.pbc == 1
    Jxx, Jyy, Jzz = args.Jxx, args.Jyy, args.Jzz
    Jpm = -(Jxx+Jyy)/4  # J±
    Jpmpm = (Jxx-Jyy)/4  # J±±
    
    # Field parameters
    h = args.h
    fielddir = np.array([args.hx, args.hy, args.hz])
    fielddir = fielddir/np.linalg.norm(fielddir)
    B = np.einsum('r, ir->i', h*fielddir, z)
    
    # Output directory
    output_dir = "./" + args.outdir + "/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Generate nearest neighbor list with specified boundary condition
    look_up_table = genNN_list(dim1, dim2, dim3, use_pbc)
    max_site = dim1*dim2*dim3*4
    
    # Generate all interactions
    interALL = []
    transfer = []
    
    for i in range(max_site):
        transfer.append(Zeeman(B, i))
        for j in range(len(look_up_table[i])):
            neighbor_idx = int(look_up_table[i][j])
            if 0 <= neighbor_idx < max_site:  # Ensure valid neighbor index
                interALL.append(HeisenbergNN(Jzz, Jpm, Jpmpm, i, neighbor_idx, max_site))
    
    interALL = np.vstack([arr for arr in interALL if arr.size > 0])
    transfer = np.vstack([arr for arr in transfer if arr.size > 0])
    
    # Write field strength
    fstrength = np.zeros((1,1))
    fstrength[0,0] = h
    np.savetxt(output_dir+"field_strength.dat", fstrength)
    
    # Write boundary condition info
    with open(output_dir+"boundary_condition.dat", 'w') as f:
        f.write(f"PBC: {'True' if use_pbc else 'False'}\n")
        f.write(f"Dimensions: {dim1}x{dim2}x{dim3}\n")
    
    # Write output files
    write_interALL(interALL, output_dir, "InterAll.dat")
    write_transfer(transfer, output_dir, "Trans.dat")
    write_site_positions(output_dir, dim1, dim2, dim3)
    write_lattice_parameters(output_dir)
    plot_pyrochlore_lattice(output_dir, dim1, dim2, dim3, use_pbc, look_up_table)

    # Write one-body and two-body correlation functions
    one_body_correlations(f"one_body_correlations.dat", dim1*dim2*dim3*4, output_dir)
    two_body_correlations(f"two_body_correlations.dat", dim1*dim2*dim3*4, output_dir)

    
    print(f"Generated pyrochlore lattice Hamiltonian with dimensions {dim1}x{dim2}x{dim3}")
    print(f"Boundary conditions: {'Periodic' if use_pbc else 'Open'}")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
