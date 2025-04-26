import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data_file = './NLC_test/cluster_3_order_3/thermo/thermo_data.txt'
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
plt.savefig('./specific_heat_plot.png', dpi=300)

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
plt.xscale('log')
# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('./energy_plot.png', dpi=300)

# Show the plot
plt.show()

# Calculate the derivative of energy with respect to temperature
energy_derivative = np.gradient(energy, temperature)

# Create figure for energy derivative plot
plt.figure(figsize=(10, 6))

# Plot energy derivative vs temperature
plt.plot(temperature, energy_derivative, '-', color='green', linewidth=2, label='Energy Derivative')

# Set x-axis to log scale since temperature values span several orders of magnitude
plt.xscale('log')

# Add labels and title
plt.xlabel('Temperature (log scale)', fontsize=14)
plt.ylabel('dE/dT', fontsize=14)
plt.title('Energy Derivative vs Temperature', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('./energy_derivative_plot.png', dpi=300)

# Show the plot
plt.show()
