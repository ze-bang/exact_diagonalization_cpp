import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data_file = '/home/pc_linux/exact_diagonalization_cpp/ED_test_thermodynamics_full.dat'
data = np.loadtxt(data_file, comments='#')

# Extract temperature and specific heat columns
temperature = data[:, 0]
specific_heat = data[:, 2]

def gaussian(x, mu, sigma):
    """Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def lorentzian(x, mu, gamma):
    """Lorentzian function."""
    return (gamma / np.pi) / ((x - mu) ** 2 + gamma ** 2)

# specific_heat = specific_heat * (0.6*lorentzian(temperature, mu=0.32, gamma=0.1)+1)*0.8

# Create figure and axes
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(temperature, specific_heat, '-', color='blue', linewidth=2, label='Specific Heat')

# Set x-axis to log scale since temperature values span several orders of magnitude
plt.xscale('log')

# Add labels and title
plt.xlabel('Temperature (log scale)', fontsize=14)
plt.ylabel('Specific Heat', fontsize=14)
plt.title('Specific Heat vs Temperature', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)


# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/pc_linux/exact_diagonalization_cpp/specific_heat_plot.png', dpi=300)

# Show the plot
plt.show()

# Extract energy values 
energy = data[:, 1]

# Create figure for energy vs temperature plot
plt.figure(figsize=(10, 6))

# Plot energy vs temperature
plt.plot(temperature, energy, '-', color='red', linewidth=2, label='Energy')

# Set x-axis to log scale since temperature values span several orders of magnitude
plt.xscale('log')

# Add labels and title
plt.xlabel('Temperature (log scale)', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.title('Energy vs Temperature', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

Emin = np.loadtxt('/home/pc_linux/exact_diagonalization_cpp/ED_test_full_spectrum.dat')

plt.text(0.1, 0.8, r'$E_{min} = '+ str(Emin[0]) +'$', fontsize=16, color='red', transform=plt.gca().transAxes)
plt.xscale('log')
# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/pc_linux/exact_diagonalization_cpp/energy_plot.png', dpi=300)

# Show the plot
plt.show()

# Extract entropy values 
energy = data[:, 3]

# Create figure for energy vs temperature plot
plt.figure(figsize=(10, 6))

# Plot energy vs temperature
plt.plot(temperature, energy, '-', color='red', linewidth=2, label='Entropy')

# Set x-axis to log scale since temperature values span several orders of magnitude
plt.xscale('log')

# Add labels and title
plt.xlabel('Temperature (log scale)', fontsize=14)
plt.ylabel('Entropy', fontsize=14)
plt.title('Entropy vs Temperature', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/pc_linux/exact_diagonalization_cpp/entropy_plot.png', dpi=300)

# Show the plot
plt.show()

# Extract entropy values 
energy = (data[:,1] - data[:, 4])/temperature

# Create figure for energy vs temperature plot
plt.figure(figsize=(10, 6))

# Plot energy vs temperature
plt.plot(temperature, energy, '-', color='red', linewidth=2, label='Energy')
plt.text(0.1, 0.8, r'$E_{min} = '+ str(data[0,4]) +'$', fontsize=16, color='red', transform=plt.gca().transAxes)

# Set x-axis to log scale since temperature values span several orders of magnitude
plt.xscale('log')

# Add labels and title
plt.xlabel('Temperature (log scale)', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.title('Energy vs Temperature', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/pc_linux/exact_diagonalization_cpp/free_energy_plot.png', dpi=300)

# Show the plot
plt.show()



