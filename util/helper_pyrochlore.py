import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])

def indices_periodic_BC(i,j,k,u,d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], [np.mod(i-1, d1), j, k, 1], [i, np.mod(j-1, d2), k, 2], [i, j, np.mod(k-1, d3), 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], [np.mod(i+1, d1), j, k, 0],[np.mod(i+1, d1), np.mod(j-1, d2), k, 2], [np.mod(i+1, d1), j, np.mod(k-1, d3), 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], [i, np.mod(j+1, d2), k, 0], [np.mod(i-1, d1), np.mod(j+1, d2), k, 1], [i, np.mod(j+1, d2), np.mod(k-1, d3), 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], [i, j, np.mod(k+1, d3), 0], [np.mod(i-1, d1), j, np.mod(k+1, d3), 1], [i, np.mod(j-1, d2), np.mod(k+1, d3), 2]])

def indices_open_BC(i,j,k,u, d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], [i-1, j, k, 1], [i, j-1, k, 2], [i, j, k-1, 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], [i+1, j, k, 0],[i+1, j-1, k, 2], [i+1, j, k-1, 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], [i, j+1, k, 0], [i-1, j+1, k, 1], [i, j+1, k-1, 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], [i, j, k+1, 0], [i-1, j, k+1, 1], [i, j-1, k+1, 2]])

site_basis = np.array([[.125,0.125,0.125],[0.125,-0.125,-0.125],[-0.125,0.125,-0.125],[-0.125,-0.125,0.125]])
basis = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])

dim1 = int(sys.argv[9])
dim2 = int(sys.argv[10])
dim3 = int(sys.argv[11])


con = np.zeros((dim1, dim2, dim3, 4))

Jxx, Jyy, Jzz = float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3])

Jpm = -(Jxx+Jyy)/4
Jpmpm = (Jxx-Jyy)/4
h = float(sys.argv[4])
fielddir = np.array([float(sys.argv[5]),float(sys.argv[6]), float(sys.argv[7])])
fielddir = fielddir/np.linalg.norm(fielddir)
B = np.einsum('r, ir->i', h*fielddir,z)


def flattenIndex(Indx):
    temp = np.zeros(len(Indx))
    for i in range(6):
        temp[i] = Indx[i][0]*dim2*dim3*4 + Indx[i][1]*dim3*4 + Indx[i][2]*4 + Indx[i][3]
    return temp

def genNN_list(d1,d2,d3, PBC = True):
    NN_list = np.zeros((d1*d2*d3*4, 6))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for u in range(4):
                    if PBC:
                        NN_list[i*d2*d3*4+j*d3*4+k*4+u] = flattenIndex(indices_periodic_BC(i,j,k,u,d1,d2,d3))
                    else:
                        NN_list[i*d2*d3*4+j*d3*4+k*4+u] = flattenIndex(indices_open_BC(i,j,k,u,d1,d2,d3))
    return NN_list

look_up_table = genNN_list(dim1, dim2, dim3)
#Sz = 2, Sp = 0, Sm = 1
def HeisenbergNN(Jzz, Jpm, Jpmpm, indx1, indx2):
    if indx1 <= dim1*dim2*dim3*4 and indx2 <= dim1*dim2*dim3*4 and indx1 >= 0 and indx2 >= 0:
        Jzz = Jzz/2 
        Jpm = Jpm/2
        Jpmpm = Jpmpm/2
        return np.array([[2, indx1, 2, indx2, Jzz, 0],
                        
                        [0, indx1, 1, indx2, -Jpm, 0],
                        [1, indx1, 0, indx2, -Jpm, 0],

                        [1, indx1, 1, indx2, Jpmpm, 0],
                        [0, indx1, 0, indx2, Jpmpm, 0]])

def Zeeman(h, indx):
    here = h[indx % 4]
    return np.array([[2, indx, -here, 0]])   


interALL = []
transfer = []


for i in range(len(look_up_table)):
    transfer.append(Zeeman(B, i))
    for j in range(len(look_up_table[i])):
        interALL.append(HeisenbergNN(Jzz, Jpm, Jpmpm, i, look_up_table[i][j]))

interALL = np.array(interALL).reshape(-1, 6)
transfer = np.array(transfer).reshape(-1, 4)




output_dir = "./" + sys.argv[8] + "/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

max_site = dim1*dim2*dim3*4
All_N = max_site
exct = 1

fstrength = np.zeros((1,1))
fstrength[0,0] = h
np.savetxt(output_dir+"field_strength.dat", fstrength)


def write_interALL(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8d}   ".format(int(interALL[i,2]))     \
        +" {0:8d}   ".format(int(interALL[i,3]))     \
        +" {0:8f}   ".format(interALL[i,4]) \
        +" {0:8f}   ".format(interALL[i,5]) \
        +"\n")
    f.close()

def write_transfer(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8f}   ".format(interALL[i,2])     \
        +" {0:8f}   ".format(interALL[i,3])
        +"\n")
    f.close()

def one_body_correlations(Op, file_name):
    num_green_one  = All_N
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("loc "+"{0:8d}".format(num_green_one)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for all_i in range(0,All_N):
        f.write(" {0:8d} ".format(Op) \
        +" {0:8d}   ".format(all_i)     \
        +" {0:8f}   ".format(1)     \
        +" {0:8f}   ".format(0)
        +"\n")
    f.close()


def two_body_correlations(Op1, Op2, file_name):
    num_green_two  = All_N*All_N
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("loc "+"{0:8d}".format(num_green_two)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for all_i in range(0,All_N):
        for all_j in range(0,All_N):
            f.write(" {0:8d} ".format(Op1) \
            +" {0:8d}   ".format(all_i)     \
            +" {0:8d}   ".format(Op2)     \
            +" {0:8d}   ".format(all_j)     \
            +" {0:8f}   ".format(1) \
            +" {0:8f}   ".format(0) \
            +"\n")
    f.close()



write_interALL(interALL, "InterAll.dat")
write_transfer(transfer, "Trans.dat")

opname = ['S+', 'S-', 'Sz']

for i in range(3):
    one_body_correlations(i, "one_body_correlations"+opname[i]+".dat")
    for j in range(3):
        two_body_correlations(i, j, "two_body_correlations"+opname[i]+opname[j]+".dat")



def write_site_positions(output_dir, dim1, dim2, dim3):
    """Write site positions to a file in the format: index, x, y, z"""
    f = open(output_dir+"site_positions.dat", 'wt')
    f.write("# index, x, y, z\n")
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for u in range(4):
                    # Calculate site index
                    site_index = i*dim2*dim3*4 + j*dim3*4 + k*4 + u
                    
                    # Calculate position in Cartesian coordinates
                    # Unit cell position + basis position
                    unit_cell_pos = i*basis[0] + j*basis[1] + k*basis[2]
                    position = unit_cell_pos + site_basis[u]
                    
                    # Write to file
                    f.write(f"{site_index} {position[0]:.6f} {position[1]:.6f}, {position[2]:.6f}\n")
    
    f.close()

# Call the function to write site positions
write_site_positions(output_dir, dim1, dim2, dim3)


def plot_pyrochlore_lattice(output_dir, dim1, dim2, dim3):
    """Plot the pyrochlore lattice showing sites and their nearest neighbor connections"""
    import matplotlib.pyplot as plt
    
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
    # Each row in look_up_table contains the 6 nearest neighbor indices for a site
    look_up_table_open = genNN_list(dim1, dim2, dim3, False)
    for site_idx in range(len(look_up_table_open)):
        site_pos = positions[site_idx]
        
        for neighbor_idx in look_up_table_open[site_idx]:
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
    ax.set_title(f'Pyrochlore Lattice ({dim1}x{dim2}x{dim3} unit cells)')
    
    # Add a legend
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure
    plt.savefig(f"{output_dir}pyrochlore_lattice.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Plot the lattice
plot_pyrochlore_lattice(output_dir, dim1, dim2, dim3)

def write_lattice_parameters(output_dir):
    """Write the unit cell site basis and lattice vectors to a file"""
    f = open(output_dir+"lattice_parameters.dat", 'wt')
    
    # Write header
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
    
    f.close()

# Call the function to write lattice parameters
write_lattice_parameters(output_dir)