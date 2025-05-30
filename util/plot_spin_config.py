import numpy as np

import matplotlib.pyplot as plt

def read_data(filename):
    """Read data from file, skipping comment lines."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                continue
            data.append(line.strip().split())
    return np.array(data)

# Read data files
spin_data = read_data('./Sasha_16sites/output/spin_expectations/spin_expectations_T0.dat')
position_data = read_data('./Sasha_16sites/site_positions.dat')

# Extract site positions
site_positions = {}
for row in position_data:
    site_idx = int(row[0])
    x, y = float(row[1]), float(row[2])
    site_positions[site_idx] = (x, y)

# Extract spin values
site_spins = {}
for row in spin_data:
    site_idx = int(row[0])
    sp = float(row[1]) + 1j* float(row[2])
    sm = float(row[3]) + 1j* float(row[4])
    sx = np.real(sp + sm)/2
    sy = np.real((sp - sm)/(2j))
    sz = float(row[5])  # Sz_real column
    site_spins[site_idx] = np.array([sx, sy, sz])

# Create the plot
plt.figure(figsize=(10, 8))

# Prepare data for quiver plot
x_positions = []
y_positions = []
spin_x = []
spin_y = []
spin_z = []  # For coloring

# Sort by site index to ensure consistent ordering
sorted_indices = sorted(site_positions.keys())

for idx in sorted_indices:
    if idx in site_spins:  # Make sure the site has spin data
        x, y = site_positions[idx]
        sx, sy, sz = site_spins[idx]
        
        # Convert complex numbers to real if needed
        if isinstance(sx, complex):
            sx = sx.real
        if isinstance(sy, complex):
            sy = sy.real
            
        x_positions.append(x)
        y_positions.append(y)
        spin_x.append(sx)
        spin_y.append(sy)
        spin_z.append(sz)

# Create quiver plot colored by the z-component of spin
quiv = plt.quiver(x_positions, y_positions, spin_x, spin_y, spin_z, 
                 scale=15, width=0.005, cmap='coolwarm', pivot='mid')

# Add colorbar for z-component
cbar = plt.colorbar(quiv, label='Sz Component')

# Add scatter plot for site positions
plt.scatter(x_positions, y_positions, color='black', s=30, zorder=2)

# Add site indices as labels
for i, idx in enumerate(sorted_indices):
    if idx in site_spins:
        plt.text(x_positions[i], y_positions[i], str(idx), 
                fontsize=8, ha='center', va='bottom')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Spin Configuration')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig('spin_configuration.png', dpi=300)
plt.show()