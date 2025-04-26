#!/usr/bin/env python3
"""
Numerical Linked Cluster Expansion (NLCe) for thermodynamic properties
of quantum spin systems on the pyrochlore lattice.

This module provides functions to calculate specific heat capacity, 
energy, entropy, and other thermodynamic properties using the NLCe method
with data from exact diagonalization of finite clusters.
"""

import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import argparse

def calculate_boltzmann_weights(energies, temperatures):
    """
    Calculate Boltzmann weights exp(-beta*E)/Z for a set of energy levels and temperatures.
    
    Args:
        energies: Array of energy eigenvalues
        temperatures: Array of temperatures
    
    Returns:
        weights: 2D array of Boltzmann weights for each energy and temperature
        partition_functions: Array of partition functions for each temperature
    """
    # Convert to numpy arrays if they aren't already
    energies = np.asarray(energies)
    temperatures = np.asarray(temperatures)
    
    # Calculate β = 1/(kB*T) for each temperature (kB=1 in natural units)
    betas = 1.0 / temperatures
    
    # Calculate e^(-βE) for all energies and temperatures
    # Subtract minimum energy to avoid numerical overflow
    E_min = np.min(energies)
    exp_terms = np.exp(-np.outer(betas, energies - E_min))
    
    # Calculate partition function Z = ∑e^(-βE) for each temperature
    partition_functions = np.sum(exp_terms, axis=1)
    
    # Calculate Boltzmann weights w = e^(-βE)/Z
    weights = exp_terms / partition_functions[:, np.newaxis]
    
    return weights, partition_functions


def calculate_thermodynamics_from_spectrum(energies, T_min, T_max, num_points, ground_state_degeneracy=1):
    """
    Calculate thermodynamic properties from an energy spectrum.
    
    Args:
        energies: Array of energy eigenvalues
        T_min: Minimum temperature
        T_max: Maximum temperature
        num_points: Number of temperature points
        ground_state_degeneracy: Degeneracy of the ground state
    
    Returns:
        ThermodynamicData object with calculated properties
    """
    # Generate temperature array (can use logarithmic spacing if desired)
    temperatures = np.linspace(T_min, T_max, num_points)
    
    # Calculate Boltzmann weights and partition function
    weights, Z = calculate_boltzmann_weights(energies, temperatures)
    
    # Calculate β = 1/(kB*T) for each temperature
    betas = 1.0 / temperatures
    
    # Calculate internal energy: U = ∑ E_i * w_i
    energy = np.sum(weights * energies, axis=1)
    
    # Calculate energy squared: <E²> = ∑ E_i² * w_i
    energy_squared = np.sum(weights * energies**2, axis=1)
    
    # Calculate specific heat: C = β² * (<E²> - <E>²)
    specific_heat = betas**2 * (energy_squared - energy**2)
    
    # Calculate entropy: S = log(Z) + β*U
    # Adjust for ground state degeneracy
    entropy = np.log(Z) + betas * energy
    if ground_state_degeneracy > 1:
        entropy += np.log(ground_state_degeneracy)
    
    # Calculate free energy: F = -T*log(Z)
    free_energy = -temperatures * np.log(Z)
    if ground_state_degeneracy > 1:
        free_energy -= temperatures * np.log(ground_state_degeneracy)
    
    # Return data as a dictionary-like object
    class ThermodynamicData:
        def __init__(self):
            self.temperatures = temperatures
            self.energy = energy
            self.specific_heat = specific_heat
            self.entropy = entropy
            self.free_energy = free_energy
            self.partition_function = Z
    
    return ThermodynamicData()


def read_cluster_info(cluster_dir):
    """
    Read cluster information from files in the given directory.
    
    Args:
        cluster_dir: Directory containing cluster info files
    
    Returns:
        Dictionary with cluster information
    """
    # Find site info file
    site_info_files = glob.glob(os.path.join(cluster_dir, "*site_info.dat"))
    if not site_info_files:
        raise FileNotFoundError(f"No site info file found in {cluster_dir}")
    
    site_info_file = site_info_files[0]
    
    # Extract cluster name
    cluster_name = os.path.basename(site_info_file).replace("_site_info.dat", "")
    
    # Read site info
    sites = []
    with open(site_info_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                site_id = int(parts[0])
                matrix_idx = int(parts[1])
                sublattice = int(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                sites.append({
                    'site_id': site_id,
                    'matrix_idx': matrix_idx,
                    'sublattice': sublattice,
                    'position': (x, y, z)
                })
    
    # Read eigenvalues if available
    eigenvalues_file = os.path.join(cluster_dir, "output", "eigenvalues.txt")
    eigenvalues = []
    if os.path.exists(eigenvalues_file):
        try:
            eigenvalues = np.loadtxt(eigenvalues_file)/len(sites)
        except:
            print(f"Warning: Could not read eigenvalues from {eigenvalues_file}")
    
    # Return cluster info
    return {
        'cluster_name': cluster_name,
        'num_sites': len(sites),
        'sites': sites,
        'eigenvalues': eigenvalues,
        'directory': cluster_dir
    }


def read_thermo_data(cluster_dir):
    """
    Read pre-computed thermodynamic data from a file.
    
    Args:
        cluster_dir: Directory containing thermodynamic data
    
    Returns:
        ThermodynamicData object or None if data not found
    """
    thermo_file = os.path.join(cluster_dir, "thermo", "thermo_data.txt")
    
    if not os.path.exists(thermo_file):
        return None
    
    try:
        # Read data (skip header line)
        data = np.loadtxt(thermo_file, skiprows=1)
        
        # Create ThermodynamicData object
        thermo = type('ThermodynamicData', (), {})()
        thermo.temperatures = data[:, 0]
        thermo.energy = data[:, 1]
        thermo.specific_heat = data[:, 2]
        thermo.entropy = data[:, 3]
        thermo.free_energy = data[:, 4]
        
        return thermo
    except Exception as e:
        print(f"Warning: Error reading thermo data from {thermo_file}: {e}")
        return None


def extract_cluster_order_id(cluster_name):
    """
    Extract order and ID from a cluster name like 'cluster_2_order_3'
    
    Returns:
        (cluster_id, order) tuple or (None, None) if name doesn't match pattern
    """
    match = re.search(r'cluster_(\d+)_order_(\d+)', cluster_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def find_clusters(base_dir, max_order=None):
    """
    Find all cluster directories and organize them by order.
    
    Args:
        base_dir: Base directory to search for clusters
        max_order: Maximum cluster order to include (None for all)
    
    Returns:
        Dictionary mapping cluster order to list of cluster directories
    """
    clusters_by_order = defaultdict(list)
    
    # Look for cluster directories
    for root, dirs, files in os.walk(base_dir):
        site_info_files = [f for f in files if f.endswith('_site_info.dat')]
        
        for file in site_info_files:
            cluster_name = file.replace('_site_info.dat', '')
            cluster_id, order = extract_cluster_order_id(cluster_name)
            
            if cluster_id is not None and order is not None:
                if max_order is None or order <= max_order:
                    clusters_by_order[order].append(root)
    
    return clusters_by_order


def calculate_nlce_weights(clusters_by_order):
    """
    Calculate NLCe weights for each cluster.
    
    Args:
        clusters_by_order: Dictionary mapping orders to lists of cluster info
    
    Returns:
        Dictionary mapping cluster directories to their NLCe weights
    """
    weights = {}
    
    # Process clusters by order, starting from lowest
    for order in sorted(clusters_by_order.keys()):
        for cluster_dir in clusters_by_order[order]:
            # Read cluster info
            cluster_info = read_cluster_info(cluster_dir)
            cluster_name = cluster_info['cluster_name']
            
            # For order 1, weight is always 1
            if order == 1:
                weights[cluster_dir] = 1.0
                continue
            
            # Initialize weight to 1
            weight = 1.0
            
            # Subtract weights of all subclusters
            for sub_order in range(1, order):
                for sub_cluster_dir in clusters_by_order[sub_order]:
                    sub_cluster_info = read_cluster_info(sub_cluster_dir)
                    
                    # Check if this is a subcluster (based on embedding factor)
                    # For simplicity, we use a placeholder embedding calculation
                    # In a real implementation, you would need to compute actual embedding factors
                    embedding_factor = calculate_embedding_factor(cluster_info, sub_cluster_info)
                    
                    if embedding_factor > 0:
                        weight -= embedding_factor * weights[sub_cluster_dir]
            
            weights[cluster_dir] = weight
    
    return weights


def calculate_embedding_factor(cluster_info, subcluster_info):
    """
    Calculate the number of ways a subcluster can be embedded in a cluster.
    
    This is a placeholder implementation. In a real application, this would
    require proper graph isomorphism checking.
    
    Args:
        cluster_info: Dictionary with information about the larger cluster
        subcluster_info: Dictionary with information about the potential subcluster
    
    Returns:
        Number of ways subcluster can be embedded in cluster (0 if not a subcluster)
    """
    # Extract cluster IDs and orders
    cluster_name = cluster_info['cluster_name']
    subcluster_name = subcluster_info['cluster_name']
    
    cluster_id, cluster_order = extract_cluster_order_id(cluster_name)
    subcluster_id, subcluster_order = extract_cluster_order_id(subcluster_name)
    
    if cluster_order <= subcluster_order:
        return 0  # Cannot embed a larger cluster in a smaller one
    
    # In a real implementation, you would use the lattice structure to determine
    # the actual embedding factor. For demonstration, we use a simple formula.
    # This should be replaced with actual embedding calculations.
    
    # For a pyrochlore lattice with tetrahedral clusters, a rough approximation
    if cluster_order == subcluster_order + 1:
        # Each order-n cluster contains n subclusters of order n-1
        return cluster_order
    elif cluster_order > subcluster_order + 1:
        # Combinatorial factor (rough approximation)
        import math
        return math.comb(cluster_order, subcluster_order)
    
    return 0


def combine_nlce_results(clusters_by_order, weights, property_name, max_order=None, T_min=0.001, T_max=10, num_points=1000):
    """
    Combine NLCe results up to a maximum order.
    
    Args:
        clusters_by_order: Dictionary mapping orders to lists of cluster directories
        weights: Dictionary mapping cluster directories to NLCe weights
        property_name: Name of the property to combine ('energy', 'specific_heat', etc.)
        max_order: Maximum order to include (None for all available)
        T_min, T_max, num_points: Temperature range and number of points
    
    Returns:
        Dictionary with combined results for each order up to max_order
    """
    # Create temperature array
    temperatures = np.logspace(np.log(T_min)/np.log(10), np.log(T_max)/np.log(10), num_points)
    
    # Initialize results dictionary
    results = {
        'temperatures': temperatures,
        'by_order': {}
    }
    
    # Initialize arrays to accumulate results for each order
    available_orders = sorted(clusters_by_order.keys())
    if max_order is None:
        max_order = max(available_orders)
    
    for order in range(1, max_order + 1):
        results['by_order'][order] = np.zeros(num_points)
    
    # Process each cluster and accumulate its contribution
    for order in available_orders:
        if order > max_order:
            continue
            
        order_contribution = np.zeros(num_points)
        
        for cluster_dir in clusters_by_order[order]:
            # Read thermodynamic data
            thermo_data = read_thermo_data(cluster_dir)
            # If no pre-computed data, try to calculate from eigenvalues
            if thermo_data is None:
                cluster_info = read_cluster_info(cluster_dir)
                if len(cluster_info['eigenvalues']) > 0:
                    thermo_data = calculate_thermodynamics_from_spectrum(
                        cluster_info['eigenvalues'],
                        T_min, T_max, num_points
                    )
            
            # Skip this cluster if we couldn't get thermodynamic data
            if thermo_data is None:
                print(f"Warning: No thermodynamic data available for {cluster_dir}")
                continue
            
            # Get the property values (interpolate if necessary)
            if hasattr(thermo_data, property_name):
                prop_values = getattr(thermo_data, property_name)
                
                # Interpolate if temperature points don't match
                if len(thermo_data.temperatures) != len(temperatures) or not np.allclose(thermo_data.temperatures, temperatures):
                    prop_values = np.interp(temperatures, thermo_data.temperatures, prop_values)
                
                # Add weighted contribution
                weight = weights.get(cluster_dir, 0.0)
                order_contribution += weight * prop_values
            else:
                print(f"Warning: Property '{property_name}' not found in thermodynamic data for {cluster_dir}")
        
        # Accumulate results up to this order
        for o in range(1, order + 1):
            results['by_order'][o] = results['by_order'][o-1] if o > 1 else np.zeros(num_points)
            if o == order:
                results['by_order'][o] += order_contribution
    
    return results


def visualize_nlce_results(results, property_name, output_file=None, show_orders=None):
    """
    Visualize NLCe results for a thermodynamic property.
    
    Args:
        results: Dictionary with NLCe results from combine_nlce_results
        property_name: Name of the property being visualized
        output_file: Path to save the figure (None to display only)
        show_orders: List of orders to show (None for all)
    """
    temperatures = results['temperatures']
    all_orders = sorted(results['by_order'].keys())
    
    if show_orders is None:
        show_orders = all_orders
    else:
        show_orders = [o for o in show_orders if o in all_orders]
    
    plt.figure(figsize=(10, 6))
    
    for order in show_orders:
        plt.plot(temperatures, results['by_order'][order], 
                 label=f'Order {order}', linewidth=2)
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel(property_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'NLCe Results for {property_name.replace("_", " ").title()}', fontsize=14)
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def run_nlce(base_dir, max_order=None, T_min=0.01, T_max=10, num_points=100, output_dir=None):
    """
    Run a complete NLCe calculation.
    
    Args:
        base_dir: Directory containing cluster data
        max_order: Maximum cluster order to include
        T_min, T_max, num_points: Temperature range and number of points
        output_dir: Directory to save results (None to use base_dir/nlce_results)
    
    Returns:
        Dictionary with NLCe results for different properties
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'nlce_results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all clusters
    print(f"Finding clusters in {base_dir}...")
    clusters_by_order = find_clusters(base_dir, max_order)
    
    # Check if clusters were found
    if not clusters_by_order:
        raise ValueError(f"No clusters found in {base_dir}")
    
    # Print cluster statistics
    print("\nCluster statistics:")
    for order in sorted(clusters_by_order.keys()):
        print(f"  Order {order}: {len(clusters_by_order[order])} clusters")
    
    # Calculate NLCe weights
    print("\nCalculating NLCe weights...")
    weights = calculate_nlce_weights(clusters_by_order)
    
    # Save weights to file
    weights_file = os.path.join(output_dir, 'nlce_weights.txt')
    with open(weights_file, 'w') as f:
        f.write("# NLCe weights\n")
        f.write("# cluster_directory weight\n")
        for cluster_dir, weight in weights.items():
            f.write(f"{cluster_dir} {weight}\n")
    
    # Properties to calculate
    properties = ['energy', 'specific_heat', 'entropy', 'free_energy']
    
    # Calculate and save NLCe results for each property
    results = {}
    for prop in properties:
        print(f"\nCalculating NLCe results for {prop}...")
        prop_results = combine_nlce_results(
            clusters_by_order, weights, prop, max_order,
            T_min, T_max, num_points
        )
        results[prop] = prop_results
        
        # Save results to file
        for order in prop_results['by_order']:
            result_file = os.path.join(output_dir, f'{prop}_order_{order}.txt')
            with open(result_file, 'w') as f:
                f.write(f"# NLCe results for {prop}, order {order}\n")
                f.write("# temperature value\n")
                for t, val in zip(prop_results['temperatures'], prop_results['by_order'][order]):
                    f.write(f"{t} {val}\n")
        
        # Visualize results
        print(f"Generating plot for {prop}...")
        output_file = os.path.join(output_dir, f'{prop}_nlce.png')
        visualize_nlce_results(prop_results, prop, output_file)
    
    print(f"\nNLCe calculation complete. Results saved to {output_dir}")
    return results


def main():
    """Main function to run NLCe from command line"""
    parser = argparse.ArgumentParser(description='Run NLCe calculations for thermodynamic properties')
    parser.add_argument('base_dir', help='Directory containing cluster data')
    parser.add_argument('--max-order', type=int, help='Maximum cluster order to include')
    parser.add_argument('--t-min', type=float, default=0.01, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=10.0, help='Maximum temperature')
    parser.add_argument('--num-points', type=int, default=100, help='Number of temperature points')
    parser.add_argument('--output-dir', help='Directory to save results')
    
    args = parser.parse_args()
    
    run_nlce(
        args.base_dir,
        max_order=args.max_order,
        T_min=args.t_min,
        T_max=args.t_max,
        num_points=args.num_points,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()