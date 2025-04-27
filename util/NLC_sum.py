import os
import numpy as np
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import argparse

#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility.
Calculates thermodynamic properties of a lattice using cluster expansion.
"""

import matplotlib.pyplot as plt

class NLCExpansion:
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, SI_units=False):
        """
        Initialize the NLC expansion calculator.
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_values: Array of inverse temperature values to evaluate
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        
        if not SI_units:
            self.kB = 1.0
            self.SI = False
        else:
            self.kB = 0.086173  # Boltzmann constant in meV/K
            self.SI = True

        self.temp_values = np.logspace(np.log(temp_min)/np.log(10), np.log(temp_max)/np.log(10), num_temps)  # Default temperature range

        

        self.clusters = {}  # Will store {cluster_id: {order, multiplicity, eigenvalues, etc.}}
        self.weights = {}   # Will store calculated weights for each cluster and property
        
    def read_clusters(self):
        """Read all cluster information from files in the cluster directory."""
        pattern = os.path.join(self.cluster_dir, "cluster_*_order_*.dat")
        cluster_files = glob.glob(pattern)
        
        for file_path in cluster_files:
            # Extract cluster ID and order from filename
            match = re.search(r'cluster_(\d+)_order_(\d+)', file_path)
            if not match:
                continue
                
            cluster_id, order = int(match.group(1)), int(match.group(2))
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            multiplicity = None
            num_vertices = None
            for line in lines:
                if line.startswith("# Multiplicity:"):
                    multiplicity = int(line.split(":")[1].strip())
                elif line.startswith("# Number of vertices:"):
                    num_vertices = int(line.split(":")[1].strip())
                
                # Break if we found both values
                if multiplicity is not None and num_vertices is not None:
                    break
            
            if multiplicity is None:
                print(f"Warning: Multiplicity not found for cluster {cluster_id}")
                continue
                
            if num_vertices is None:
                print(f"Warning: Number of vertices not found for cluster {cluster_id}")
                continue
                
            self.clusters[cluster_id] = {
                'order': order,
                'multiplicity': multiplicity,
                'num_vertices': num_vertices,
                'file_path': file_path,
                'eigenvalues': None  # Will be loaded later
            }
    def read_eigenvalues(self):
        """Read eigenvalues for each cluster from ED output files."""
        for cluster_id in self.clusters:
            eigenvalue_file = os.path.join(
                self.eigenvalue_dir, 
                f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output/eigenvalues.txt"
            )
            
            if not os.path.exists(eigenvalue_file):
                print(f"Warning: Eigenvalue file not found for cluster {cluster_id}")
                continue
                
            with open(eigenvalue_file, 'r') as f:
                eigenvalues = [float(line.strip()) for line in f if line.strip()]
                
            self.clusters[cluster_id]['eigenvalues'] = np.array(eigenvalues)
    
    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster.
        For this demo, we'll just use a simple heuristic based on order.
        
        In a full implementation, this would use the actual topology data.
        """
        order = self.clusters[cluster_id]['order']
        return [cid for cid, data in self.clusters.items() 
                if data['order'] < order]
    
    def calculate_thermodynamic_quantities(self, eigenvalues):
        """
        Calculate thermodynamic quantities from eigenvalues.
        Uses a numerically stable approach to handle large beta values (low temperatures).
        
        Returns:
            Dictionary with 'energy', 'specific_heat', and 'entropy' as keys
        """
        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values)
        }
        
        for i, temp in enumerate(self.temp_values):
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
            exp_terms = np.exp(-shifted_eigenvalues / (temp * self.kB) )
            Z_shifted = np.sum(exp_terms)
            
            # Calculate energy using original eigenvalues but stable exponentials
            energy = np.sum(eigenvalues * exp_terms) / Z_shifted
            
            # Calculate energy squared
            energy_squared = np.sum(eigenvalues**2 * exp_terms) / Z_shifted
            
            # Specific heat = β² * (⟨E²⟩ - ⟨E⟩²)
            specific_heat = (energy_squared - energy**2) / (temp * temp * self.kB)
            
            # Calculate entropy, accounting for shifted partition function
            # S = kB * [ln(Z) + βE]
            # where ln(Z) = ln(Z_shifted) + β*ground_state_energy
            entropy = self.kB * (np.log(Z_shifted) + (energy - ground_state_energy) / (self.kB * temp))

            results['energy'][i] = energy
            results['specific_heat'][i] = specific_heat 
            results['entropy'][i] = entropy
            
        return results
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
        # Sort clusters by order
        sorted_clusters = sorted(
            [(cid, data['order']) for cid, data in self.clusters.items()], 
            key=lambda x: x[1]
        )
        
        # Initialize weights dictionary
        self.weights = {
            'energy': {},
            'specific_heat': {},
            'entropy': {}   
        }
        
        # Calculate weights for each cluster
        for cluster_id, _ in sorted_clusters:
            if self.clusters[cluster_id]['eigenvalues'] is None:
                continue
                
            # Get thermodynamic quantities for this cluster
            quantities = self.calculate_thermodynamic_quantities(
                self.clusters[cluster_id]['eigenvalues'],
            )
            
            # Get subclusters
            subclusters = self.get_subclusters(cluster_id)
            
            # Calculate weights for energy and specific heat
            for prop in ['energy', 'specific_heat', 'entropy']:
                # Property of the cluster
                property_value = quantities[prop]
                
                # Subtract contributions from all subclusters
                for subcluster_id in subclusters:
                    if subcluster_id in self.weights[prop]:
                        property_value -= self.weights[prop][subcluster_id] * \
                                          self.clusters[subcluster_id]['multiplicity']
                
                # Store the weight
                self.weights[prop][cluster_id] = property_value / self.clusters[cluster_id]['multiplicity']
    
    def sum_nlc(self, euler_resum=False, order_cutoff=None):
        """
        Perform the NLC summation with optional Euler resummation.
        
        Args:
            euler_resum: If True, apply Euler resummation
            order_cutoff: Maximum order to include in the summation
            
        Returns:
            Dictionary with summed properties
        """
        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values)
        }
        
        # Calculate the NLC sum for each property
        for prop in ['energy', 'specific_heat', 'entropy']:
            # Sum by order
            sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))
            
            for cluster_id, weight in self.weights[prop].items():
                order = self.clusters[cluster_id]['order']
                if order_cutoff is not None and order > order_cutoff:
                    continue
                    
                sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']
            
            if not euler_resum:
                # Regular summation
                for order, contribution in sum_by_order.items():
                    results[prop] += contribution
            else:
                # Euler resummation
                max_order = max(sum_by_order.keys()) if sum_by_order else 0
                
                # Initialize partial sums array
                partial_sums = np.zeros((max_order + 1, len(self.temp_values)))
                
                # Calculate partial sums
                for order in range(max_order + 1):
                    if order > 0:
                        partial_sums[order] = partial_sums[order-1]
                    if order in sum_by_order:
                        partial_sums[order] += sum_by_order[order]
                
                # Apply Euler transformation
                euler_sums = np.zeros_like(partial_sums)
                euler_sums[0] = partial_sums[0]
                
                for k in range(1, max_order + 1):
                    for j in range(k, max_order + 1):
                        binomial = 1
                        for l in range(j-k+1, j+1):
                            binomial *= l
                        for l in range(1, k+1):
                            binomial //= l
                        
                        euler_sums[k] += binomial * (-1)**(j-k) * partial_sums[j]
                
                # Use the highest order Euler sum
                results[prop] = euler_sums[max_order]
        
        return results
    
    def run(self, euler_resum=False, order_cutoff=None):
        """Run the full NLC calculation."""
        print("Reading cluster information...")
        self.read_clusters()
        
        print("Reading eigenvalues...")
        self.read_eigenvalues()
        
        print("Calculating weights...")
        self.calculate_weights()
        
        print("Performing NLC summation...")
        results = self.sum_nlc(euler_resum, order_cutoff)
        
        return results
    
    def plot_results(self, results, save_path=None):
        """Plot energy and specific heat vs temperature."""
        temperatures = 1.0 / self.temp_values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot energy
        ax1.plot(temperatures, results['energy'], 'b-')
        ax1.set_xlabel('Temperature (T)')
        ax1.set_ylabel('Energy per site')
        ax1.set_title('Energy vs Temperature')
        ax1.set_xscale('log')
        
        # Plot specific heat
        ax2.plot(temperatures, results['specific_heat'], 'r-')
        ax2.set_xlabel('Temperature (T)')
        ax2.set_ylabel('Specific Heat per site')
        ax2.set_title('Specific Heat vs Temperature')
        ax2.set_xscale('log')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NLC calculation for lattice models')
    parser.add_argument('--cluster_dir', required=True, help='Directory containing cluster information files')
    parser.add_argument('--eigenvalue_dir', required=True, help='Directory containing eigenvalue files from ED calculations')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files')
    parser.add_argument('--euler_resum', action='store_true', help='Use Euler resummation')
    parser.add_argument('--order_cutoff', type=int, help='Maximum order to include in summation')
    parser.add_argument('--plot', action='store_true', help='Generate plot of results')
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')
    parser.add_argument('--temp_min', type=float, default=1e-4, help='Minimum temperature for calculations')
    parser.add_argument('--temp_max', type=float, default=1.0, help='Maximum temperature for calculations')
    parser.add_argument('--temp_bins', type=int, default=200, help='Number of temperature points to calculate')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create NLC instance
    nlc = NLCExpansion(args.cluster_dir, args.eigenvalue_dir, args.temp_min, args.temp_max, args.temp_bins, args.SI_units)
    
    # Run NLC calculation
    results = nlc.run(euler_resum=args.euler_resum, order_cutoff=args.order_cutoff)
    
    # Save results in separate files for each quantity
    energy_file = os.path.join(args.output_dir, "nlc_energy.txt")
    specific_heat_file = os.path.join(args.output_dir, "nlc_specific_heat.txt")
    entropy_file = os.path.join(args.output_dir, "nlc_entropy.txt")

    # Save energy data
    with open(energy_file, 'w') as f:
        f.write("# Temperature\tEnergy\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['energy'][i]:.8e}\n")

    # Save specific heat data
    with open(specific_heat_file, 'w') as f:
        f.write("# Temperature\tSpecific_Heat\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['specific_heat'][i]:.8e}\n")

    # Save entropy data
    with open(entropy_file, 'w') as f:
        f.write("# Temperature\tEntropy\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['entropy'][i]:.8e}\n")

    # Plot results if requested
    if args.plot:
        temperatures = nlc.temp_values
        
        # Plot energy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['energy'], 'b-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Energy per site')
        plt.title('Energy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        energy_plot_path = os.path.join(args.output_dir, "nlc_energy.png")
        plt.savefig(energy_plot_path)
        plt.close()
        
        # Plot specific heat
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['specific_heat'], 'r-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Specific Heat per site')
        plt.title('Specific Heat vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        specific_heat_plot_path = os.path.join(args.output_dir, "nlc_specific_heat.png")
        plt.savefig(specific_heat_plot_path)
        plt.close()
        
        # Plot entropy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['entropy'], 'g-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Entropy per site')
        plt.title('Entropy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        entropy_plot_path = os.path.join(args.output_dir, "nlc_entropy.png")
        plt.savefig(entropy_plot_path)
        plt.close()
    
    print(f"NLC calculation completed! Results saved to {args.output_dir}")
    
