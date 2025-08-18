import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

def read_eigenvector(filename):
    """Read eigenvector from file (complex values)."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                real = float(parts[0])
                imag = float(parts[1])
                data.append(complex(real, imag))
    return np.array(data)

def get_num_sites(dim):
    """Determine number of sites from dimension."""
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(f"Dimension {dim} is not a power of 2")
    return n

def create_spin_operators(n_sites):
    """Create Sx, Sy, Sz operators for each site."""
    dim = 2**n_sites
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    
    # Identity
    I = np.eye(2, dtype=complex)
    
    # Create operators for each site
    Sx_ops = []
    Sy_ops = []
    Sz_ops = []
    
    for site in range(n_sites):
        # Build tensor product
        ops_x = [I.copy() for _ in range(n_sites)]
        ops_y = [I.copy() for _ in range(n_sites)]
        ops_z = [I.copy() for _ in range(n_sites)]
        
        ops_x[site] = sx
        ops_y[site] = sy
        ops_z[site] = sz
        
        # Compute tensor products
        Sx_i = ops_x[0]
        Sy_i = ops_y[0]
        Sz_i = ops_z[0]
        
        for j in range(1, n_sites):
            Sx_i = np.kron(Sx_i, ops_x[j])
            Sy_i = np.kron(Sy_i, ops_y[j])
            Sz_i = np.kron(Sz_i, ops_z[j])
        
        Sx_ops.append(Sx_i)
        Sy_ops.append(Sy_i)
        Sz_ops.append(Sz_i)
    
    return Sx_ops, Sy_ops, Sz_ops

def compute_expectation_values(psi, Sx_ops, Sy_ops, Sz_ops):
    """Compute expectation values <psi|S_i|psi> for each site."""
    n_sites = len(Sx_ops)
    
    sx_exp = np.zeros(n_sites)
    sy_exp = np.zeros(n_sites)
    sz_exp = np.zeros(n_sites)
    
    for i in range(n_sites):
        # <psi|Sx_i|psi>
        sx_exp[i] = np.real(np.vdot(psi, Sx_ops[i] @ psi))
        # <psi|Sy_i|psi>
        sy_exp[i] = np.real(np.vdot(psi, Sy_ops[i] @ psi))
        # <psi|Sz_i|psi>
        sz_exp[i] = np.real(np.vdot(psi, Sz_ops[i] @ psi))
    
    return sx_exp, sy_exp, sz_exp

def plot_spin_configuration(sx, sy, sz, title="Spin Configuration"):
    """Plot the spin configuration."""
    n_sites = len(sx)
    sites = np.arange(n_sites)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot Sx
    axes[0].bar(sites, sx, color='red', alpha=0.7)
    axes[0].set_ylabel(r'$\langle S_x \rangle$')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot Sy
    axes[1].bar(sites, sy, color='green', alpha=0.7)
    axes[1].set_ylabel(r'$\langle S_y \rangle$')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot Sz
    axes[2].bar(sites, sz, color='blue', alpha=0.7)
    axes[2].set_ylabel(r'$\langle S_z \rangle$')
    axes[2].set_xlabel('Site')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_spin_vectors_3d(sx, sy, sz, title="3D Spin Vectors"):
    """Plot spin vectors in 3D."""
    
    n_sites = len(sx)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot spin vectors
    for i in range(n_sites):
        # Starting point (site position on x-axis)
        x_start = i
        y_start = 0
        z_start = 0
        
        # Vector components
        dx = 0
        dy = 0
        dz = 0.5  # Offset vectors vertically for visibility
        
        # Plot vector
        ax.quiver(x_start, y_start, z_start, 
                 sx[i], sy[i], sz[i],
                 color='black', arrow_length_ratio=0.1, linewidth=2)
        
        # Add site label
        ax.text(x_start, y_start, z_start-0.1, f'Site {i}', fontsize=8)
    
    ax.set_xlabel('Site')
    ax.set_ylabel(r'$S_y$')
    ax.set_zlabel(r'$S_z$')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(-0.5, n_sites-0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    
    return fig

def main():
    # Read eigenvector
    import argparse

    parser = argparse.ArgumentParser(description="Process spin configuration.")
    parser.add_argument("--filename", type=str, default="eigenvector_0.dat",
                        help="Path to the eigenvector file.")
    args = parser.parse_args()

    filename = args.filename
    print(f"Reading eigenvector from {filename}...")
    psi = read_eigenvector(filename)
    
    # Normalize
    norm = np.linalg.norm(psi)
    psi = psi / norm
    print(f"Eigenvector dimension: {len(psi)}")
    print(f"Norm after normalization: {np.linalg.norm(psi):.10f}")
    
    # Determine number of sites
    n_sites = get_num_sites(len(psi))
    print(f"Number of sites: {n_sites}")
    
    # Create spin operators
    print("Creating spin operators...")
    Sx_ops, Sy_ops, Sz_ops = create_spin_operators(n_sites)
    
    # Compute expectation values
    print("Computing expectation values...")
    sx_exp, sy_exp, sz_exp = compute_expectation_values(psi, Sx_ops, Sy_ops, Sz_ops)
    
    # Print results
    print("\nSpin configuration:")
    print("Site |  <Sx>    |  <Sy>    |  <Sz>")
    print("-" * 40)
    for i in range(n_sites):
        print(f"{i:4d} | {sx_exp[i]:8.5f} | {sy_exp[i]:8.5f} | {sz_exp[i]:8.5f}")
    
    # Calculate total spin
    total_sx = np.sum(sx_exp)
    total_sy = np.sum(sy_exp)
    total_sz = np.sum(sz_exp)
    total_s = np.sqrt(total_sx**2 + total_sy**2 + total_sz**2)
    
    print(f"\nTotal <Sx>: {total_sx:.5f}")
    print(f"Total <Sy>: {total_sy:.5f}")
    print(f"Total <Sz>: {total_sz:.5f}")
    print(f"Total |S|: {total_s:.5f}")
    
    # Plot results
    fig1 = plot_spin_configuration(sx_exp, sy_exp, sz_exp, 
                                   title="Spin Expectation Values per Site")
    plt.savefig("spin_configuration.png", dpi=150, bbox_inches='tight')
    
    # 3D plot
    fig2 = plot_spin_vectors_3d(sx_exp, sy_exp, sz_exp,
                                title="3D Spin Vector Visualization")
    plt.savefig("spin_vectors_3d.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Save results to file
    with open("spin_expectation_values.txt", "w") as f:
        f.write("# Site  <Sx>  <Sy>  <Sz>\n")
        for i in range(n_sites):
            f.write(f"{i} {sx_exp[i]:.8e} {sy_exp[i]:.8e} {sz_exp[i]:.8e}\n")
    
    print("\nResults saved to spin_expectation_values.txt")
    print("Plots saved to spin_configuration.png and spin_vectors_3d.png")

if __name__ == "__main__":
    main()