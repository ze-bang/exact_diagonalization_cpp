import os
import numpy as np
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import argparse
from scipy import interpolate

#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility.
Calculates thermodynamic properties of a lattice using cluster expansion.
"""

import matplotlib.pyplot as plt

class NLCExpansion:
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, measure_spin, SI_units=False):
        """
        Initialize the NLC expansion calculator.
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_values: Array of inverse temperature values to evaluate
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        
        self.SI = SI_units  # Flag for SI units
        self.measure_spin = measure_spin  # Flag for measuring spin expectation values
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
                    multiplicity = float(line.split(":")[1].strip())
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
                'eigenvalues': None,  # Will be loaded later
                'sp': None,  # Will be loaded later
                'sm': None,  # Will be loaded later
                'sp': None   # Will be loaded later    
            }

    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster.
        For this demo, we'll just use a simple heuristic based on order.
        
        In a full implementation, this would use the actual topology data.
        """
        order = self.clusters[cluster_id]['order']
        return [cid for cid, data in self.clusters.items() 
                if data['order'] < order]
    


    def read_thermodynamic_quantities(self, cluster_id):
        """
        Read thermodynamic quantities from the eigenvalue file for a given cluster.
        
        Args:
            cluster_id: ID of the cluster to read
        
        Returns:
            Dictionary with 'energy', 'specific_heat', and 'entropy' as keys
        """
        data_file = os.path.join(self.eigenvalue_dir, f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output/SS_rand0.dat")
        
        results = {
            'energy': None,
            'specific_heat': None,
        }

        if not os.path.exists(data_file):
            print(f"Eigenvalue file not found for cluster {cluster_id}: {data_file}")
            return None
            
        # Read eigenvalues from file
        try:
            data = np.loadtxt(data_file)
        except Exception as e:
            print(f"Error reading eigenvalues for cluster {cluster_id}: {str(e)}")
            return None
        
        # Interpolate data to get energy and specific heat at self.temp_values
        # Assuming the first column is inverse temperature and the second column is energy
        # and the third column is specific heat

        # Check if the data has at least 3 columns
        if data.shape[1] < 3:
            print(f"Warning: Data for cluster {cluster_id} has fewer than 3 columns")
            return None

        # Extract inverse temperatures (beta), energy, and specific heat
        temp_values = 1/data[2:, 0]  # First column is inverse temperature
        energy_values = data[2:, 1]  # Second column is energy
        specific_heat_values = data[2:, 2] * data[2:, 0] * data[2:, 0]  # Third column is specific heat

        # Interpolate energy and specific heat to the desired temperature points
        try:
            energy_interp = interpolate.interp1d(temp_values, energy_values, 
                                                bounds_error=False, fill_value="extrapolate", kind='cubic')
            specific_heat_interp = interpolate.interp1d(temp_values, specific_heat_values, 
                                                      bounds_error=False, fill_value="extrapolate", kind='cubic')

            # Evaluate at the requested temperature points
            results['energy'] = energy_interp(self.temp_values)
            results['specific_heat'] = specific_heat_interp(self.temp_values)

            if self.SI:
                # Convert to SI units if needed
                results['energy'] *= (6.02214076e23 * 1.380649e-23)
                results['specific_heat'] *= (6.02214076e23 * 1.380649e-23)

            return results
        
        except Exception as e:
            print(f"Error interpolating data for cluster {cluster_id}: {str(e)}")
            return None


    def read_subcluster_info(self):
        """Read subcluster information from the provided file."""

        self.subcluster_info = {}
        
        subcluster_file = 'subclusters_info.txt'  # Default subcluster info file

        # Check if file exists in the specified directory
        filepath = os.path.join(self.cluster_dir, subcluster_file)
        print(f"Looking for subcluster info file at: {filepath}")
        if not os.path.exists(filepath):
            filepath = subcluster_file  # Try relative path
            if not os.path.exists(filepath):
                print(f"Warning: Subcluster info file not found at {subcluster_file}")
                return
                
        with open(filepath, 'r') as f:
            current_cluster = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('Cluster'):
                    # Parse cluster header: "Cluster X (Order Y, Multiplicity Z):"
                    match = re.match(r'Cluster (\d+) \(Order (\d+)\):', line)
                    if match:
                        current_cluster = int(match.group(1))
                        self.subcluster_info[current_cluster] = {'subclusters': {}}
                        
                elif line.startswith('Subclusters:'):
                    # Parse subclusters: "(1, 2), (3, 4), ..."
                    subclusters_str = line.replace('  Subclusters:', '').strip()
                    if subclusters_str == "No subclusters" or not subclusters_str:
                        continue
                        
                    # Extract all (id, multiplicity) pairs
                    pairs = re.findall(r'\((\d+),\s*(\d+)\)', subclusters_str)
                    for subcluster_id, multiplicity in pairs:
                        print(f"Adding subcluster {subcluster_id} with multiplicity {multiplicity} to cluster {current_cluster}")
                        self.subcluster_info[current_cluster]['subclusters'][int(subcluster_id)] = int(multiplicity)
    
    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster with their multiplicities.
        
        Returns:
            Dictionary mapping subcluster_id to its multiplicity in this cluster
        """
        # Try to use subcluster info if available
        if hasattr(self, 'subcluster_info') and cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]['subclusters']
        
        # Fallback to simple heuristic based on order
        subclusters = {}
        order = self.clusters[cluster_id]['order']
        for cid, data in self.clusters.items():
            if data['order'] < order:
                subclusters[cid] = 1  # Assume multiplicity 1 as a fallback
                
        return subclusters
    

    def print_subcluster_info(self):
        """Print out the subcluster information for all clusters."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()
            
        if not hasattr(self, 'subcluster_info') or not self.subcluster_info:
            print("No subcluster information available.")
            return
            
        print("Subcluster Information:")
        print("----------------------")
        
        for cluster_id, info in sorted(self.subcluster_info.items()):
            print(f"Cluster {cluster_id}:")
            if 'subclusters' in info and info['subclusters']:
                print("  Subclusters (ID, Multiplicity):")
                for subcluster_id, multiplicity in sorted(info['subclusters'].items()):
                    print(f"    {subcluster_id}: {multiplicity}")
            else:
                print("  No subclusters")
            print()
    
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()

        self.print_subcluster_info()
            
        # Sort clusters by order
        sorted_clusters = sorted(
            [(cid, data['order']) for cid, data in self.clusters.items()], 
            key=lambda x: x[1]
        )
        
        # Initialize weights dictionary
        if self.measure_spin:
            self.weights = {
                'energy': {},
                'specific_heat': {},
                'sp': {},
                'sm': {},
                'sz': {}
            }
        else:
            self.weights = {
                'energy': {},
                'specific_heat': {},
            }
        
        # Calculate weights for each cluster
        for cluster_id, _ in sorted_clusters:   
            # Get thermodynamic quantities for this cluster
            quantities = self.read_thermodynamic_quantities(cluster_id)

            # Get subclusters with their multiplicities
            subclusters = self.get_subclusters(cluster_id)
            
            # Calculate weights for energy and specific heat
            for prop in ['energy', 'specific_heat']:
                # Property of the cluster
                property_value = quantities[prop]
                # Subtract contributions from all subclusters with correct multiplicities
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        property_value -= self.weights[prop][subcluster_id] * multiplicity
                # Store the weight
                self.weights[prop][cluster_id] = property_value

            



    def sum_nlc(self, euler_resum=False, order_cutoff=None):
        """
        Perform the NLC summation with optional Euler resummation.
        
        Args:
            euler_resum: If True, apply Euler resummation
            order_cutoff: Maximum order to include in the summation
            
        Returns:
            Dictionary with summed properties
        """

        if self.measure_spin:
            # Initialize results for spin expectation values
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'sp': np.zeros_like(self.temp_values),
                'sm': np.zeros_like(self.temp_values),
                'sz': np.zeros_like(self.temp_values)
            }
        else:
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
            }
    
        # Calculate the NLC sum for each property
        for prop in ['energy', 'specific_heat']:
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
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create NLC instance
    nlc = NLCExpansion(args.cluster_dir, args.eigenvalue_dir, args.temp_min, args.temp_max, args.temp_bins, args.measure_spin, args.SI_units)
    
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

    # Save spin expectation values if requested
    if args.measure_spin:
        spin_file = os.path.join(args.output_dir, "nlc_spin_expectations.txt")
        with open(spin_file, 'w') as f:
            f.write("# Temperature\tsp\tsp_imag\tsm\tsm_imag\tsz\tsz_imag\n")
            for i, temp in enumerate(nlc.temp_values):
                sp = results['sp'][i]
                sm = results['sm'][i]
                sz = results['sz'][i]
                f.write(f"{temp:.8e}\t{sp.real:.8e}\t{sp.imag:.8e}\t{sm.real:.8e}\t{sm.imag:.8e}\t{sz.real:.8e}\t{sz.imag:.8e}\n")

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
        
    
    print(f"NLC calculation completed! Results saved to {args.output_dir}")
    
