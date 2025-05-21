import numpy as np
import matplotlib.pyplot as plt

directory = './small_Jpm=0.03/output/'

# Load data from file
data_file = directory + '/eigenvalues.txt'
data = np.loadtxt(data_file, comments='#')


def calculate_thermodynamic_quantities(temperatures, eigenvalues, SI=False):
    results = {
        'energy': np.zeros_like(temperatures),
        'specific_heat': np.zeros_like(temperatures),
        'entropy': np.zeros_like(temperatures)
    }
    
    for i, temp in enumerate(temperatures):
        # For extremely large beta (low T), use ground state approximation
        if temp < 1e-5:
            ground_state_energy = np.min(eigenvalues)
            results['energy'][i] = ground_state_energy
            results['specific_heat'][i] = 0.0  # Specific heat approaches 0 as T->0
            results['entropy'][i] = 0.0  # Entropy approaches 0 as T->0 (third law)
            continue
            
        # Find ground state energy (minimum eigenvalue)
        ground_state_energy = np.min(eigenvalues)
        
        # Shift eigenvalues by ground state energy for numerical stability
        shifted_eigenvalues = eigenvalues - ground_state_energy
        
        # Calculate exponential terms with shifted eigenvalues
        # exp(-β(Ei-E0)) instead of exp(-βEi) to prevent underflow
        exp_terms = np.exp(-shifted_eigenvalues / temp)
        Z_shifted = np.sum(exp_terms)
        
        # Calculate energy using original eigenvalues but stable exponentials
        energy = np.sum(eigenvalues * exp_terms) / Z_shifted
        
        # Calculate energy squared
        energy_squared = np.sum(eigenvalues**2 * exp_terms) / Z_shifted
        
        # Specific heat = β² * (⟨E²⟩ - ⟨E⟩²)
        specific_heat = (energy_squared - energy**2) / (temp * temp)
        
        # Calculate entropy, accounting for shifted partition function
        # S = kB * [ln(Z) + βE]
        # where ln(Z) = ln(Z_shifted) + β*ground_state_energy
        entropy = (np.log(Z_shifted) + (energy - ground_state_energy) / (temp))

        # Convert to per site
        energy /= 16
        specific_heat /= 16
        entropy /= 16

        if SI:
            specific_heat *= (6.02214076e23  * 1.380649e-23)  # Convert to SI units (J/K)
            entropy *= (6.02214076e23 * 1.380649e-23)
            energy *= (6.02214076e23 * 1.380649e-23)
        results['energy'][i] = energy
        results['specific_heat'][i] = specific_heat 
        results['entropy'][i] = entropy
        
    return results


temperature = np.logspace(-3, 2, 100)  # Temperature range from 1e-5 to 100

results = calculate_thermodynamic_quantities(temperature, data)


# Create figure and axes
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(temperature, results['specific_heat'], '-', color='blue', linewidth=2, label='Specific Heat')

exp_data = np.loadtxt('./specific_heat_Pr2Zr2O7.txt', comments='#')
# Plot experimental data
# plt.plot(exp_data[:, 0], exp_data[:, 1], 'o', color='red', markersize=5, label='Experimental Data')
plt.plot(temperature, results['energy'], '-', color='orange', linewidth=2, label='Specific Heat')


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
plt.savefig(directory+'/specific_heat_plot.png', dpi=300)

# Show the plot
plt.show()
