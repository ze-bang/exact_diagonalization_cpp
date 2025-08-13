import os
import numpy as np
import glob
import re
import argparse
from collections import defaultdict
from scipy.optimize import curve_fit

#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility.
Calculates thermodynamic properties of a lattice using cluster expansion.
"""

try:  # Optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting not essential for core logic
    plt = None

# ================= Resummation Utilities ======================

def _forward_differences(a):
    """Compute successive forward differences until exhausted."""
    diffs = []
    current = list(a)
    while current:
        diffs.append(current[0])
        if len(current) == 1:
            break
        current = [current[i] - current[i+1] for i in range(len(current)-1)]
    return diffs

def euler_alternating_sum(terms):
    """Euler transform for an alternating series sum (-1)^n a_n with a_n>=0.
    terms: list of original alternating series terms t_n (including sign).
    Returns accelerated sum estimate.
    """
    if not terms:
        return 0.0
    # Extract a_n with sign pattern relative to first term
    sign0 = np.sign(terms[0]) if terms[0] != 0 else 1
    a = []
    for n, t in enumerate(terms):
        expected = sign0 * ((-1)**n)
        if np.sign(t) == 0:
            a.append(0.0)
        elif np.sign(t) != expected:
            # Not strictly alternating – abort to simple sum
            return sum(terms)
        a.append(abs(t))
    diffs = _forward_differences(a)
    # Euler transformed partial sum
    s = 0.0
    factor = 0.5
    for k, d in enumerate(diffs):
        s += factor * d
        factor *= 0.5
    return s * sign0  # restore leading sign convention

def shanks_transform(last_three_partial):
    """Apply single Shanks transform using last three partial sums S_{n-2}, S_{n-1}, S_n."""
    if len(last_three_partial) < 3:
        return last_three_partial[-1]
    S2, S1, S0 = last_three_partial  # note ordering passed should be [S_{n-2}, S_{n-1}, S_n]
    denom = S0 - 2*S1 + S2
    if denom == 0:
        return S0
    return (S0*S2 - S1*S1) / denom

def wynn_epsilon(partial_sums):
    """Basic Wynn epsilon algorithm implementation.
    Returns (accelerated_value, table) where table is list of lists.
    """
    S = list(partial_sums)
    n = len(S)
    # epsilon table stored as list of rows; eps[-1][k] corresponds to epsilon_{k}^{(n)}
    eps = [[0.0]*(n+1)]  # row 0 (k=-1) zeros
    eps.append(S + [0.0])  # row 1 (k=0) initial sequence
    best = S[-1]
    for m in range(2, 2*n+1):
        row = [0.0]*(n+1)
        for k in range(n - (m//2)):
            a = eps[m-2][k+1]
            b = eps[m-1][k]
            if a == b:
                row[k] = np.inf
            else:
                row[k] = b + 1.0/(a - b)
        eps.append(row)
        # Even m rows give improved estimates at element 0
        if m % 2 == 0:
            val = row[0]
            if np.isfinite(val):
                best = val
    return best, eps

def analyze_sequence(terms):
    """Analyze order contribution sequence.
    Returns dict with diagnostics: alt_fraction, monotone_fraction, last_rel_change.
    """
    if len(terms) < 2:
        return {
            'alt_fraction': 0.0,
            'monotone_fraction': 1.0,
            'last_rel_change': 0.0
        }
    signs = np.sign(terms)
    sign_changes = sum(1 for i in range(1,len(signs)) if signs[i]*signs[i-1] < 0)
    alt_fraction = sign_changes / (len(signs)-1)
    mags = [abs(x) for x in terms]
    monotone = sum(1 for i in range(1,len(mags)) if mags[i] <= mags[i-1]) / (len(mags)-1)
    partial_sums = np.cumsum(terms)
    last_rel_change = abs(partial_sums[-1]-partial_sums[-2])/(abs(partial_sums[-2])+1e-12) if len(partial_sums)>1 else 0.0
    return {
        'alt_fraction': alt_fraction,
        'monotone_fraction': monotone,
        'last_rel_change': last_rel_change
    }

def auto_resum_sequence(order_terms, absolute_tolerance=1e-8, relative_tolerance=1e-4):
    """Apply heuristic to choose Euler, Wynn (Shanks), or none.
    order_terms: list of per-order contributions (NOT partial sums).
    Returns dict with keys: value, method, diagnostics
    """
    if not order_terms:
        return {'value':0.0,'method':'none','diagnostics':{}}
    diagnostics = analyze_sequence(order_terms)
    terms = order_terms
    partial = np.cumsum(terms)
    # Convergence check
    if len(partial) >= 2:
        delta = abs(partial[-1]-partial[-2])
        if delta < absolute_tolerance or delta/(abs(partial[-2])+1e-12) < relative_tolerance:
            return {'value': partial[-1], 'method':'none', 'diagnostics':diagnostics}
    # Alternating & smooth decay? -> Euler
    if diagnostics['alt_fraction'] > 0.6 and diagnostics['monotone_fraction'] > 0.7:
        val = euler_alternating_sum(terms)
        return {'value': val, 'method':'euler', 'diagnostics':diagnostics}
    # Otherwise try Wynn / Shanks
    if len(partial) >= 3:
        shanks_val = shanks_transform(partial[-3:])
    else:
        shanks_val = partial[-1]
    wynn_val, _ = wynn_epsilon(partial)
    # Choose stabilized accelerated value if reasonable
    candidates = []
    if np.isfinite(wynn_val):
        candidates.append(('wynn', wynn_val, abs(wynn_val-partial[-1])))
    candidates.append(('shanks', shanks_val, abs(shanks_val-partial[-1])))
    # Pick candidate with minimal deviation but some improvement (delta reduction)
    candidates.sort(key=lambda x: x[2])
    method, val, _ = candidates[0]
    return {'value': val, 'method': method, 'diagnostics': diagnostics}

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
        self.resum_diagnostics = {}

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

            if self.SI:
                specific_heat *= (6.02214076e23  * 1.380649e-23)  # Convert to SI units (J/K)
                entropy *= (6.02214076e23 * 1.380649e-23)
                energy *= (6.02214076e23 * 1.380649e-23)
            results['energy'][i] = energy
            results['specific_heat'][i] = specific_heat 
            results['entropy'][i] = entropy
            
        return results
    
    def read_spin_expectations(self, spin_exp_dir):
        """
        Read spin expectation values from files in the specified directory.
        
        Args:
            spin_exp_dir: Directory containing spin expectation files
        
        Returns:
            Dictionary mapping temperature to spin expectation values
        """

        spin_expectations = {
            'sp': np.zeros_like(self.temp_values),
            'sm': np.zeros_like(self.temp_values),
            'sz': np.zeros_like(self.temp_values)
        }
        # Find all spin expectation files
        pattern = os.path.join(spin_exp_dir, "spin_expectations_T*.dat")
        spin_files = glob.glob(pattern)
        
        if not spin_files:
            print(f"No spin expectation files found in {spin_exp_dir}")
            return
            
        print(f"Found {len(spin_files)} spin expectation files")
        
        # Process each file
        for file_path in spin_files:
            # Extract temperature from filename
            match = re.search(r'T(\d+\.\d+)\.dat', file_path)
            if not match:
                print(f"Could not extract temperature from filename: {file_path}")
                continue
                
            temperature = float(match.group(1))
            print(f"Processing spin expectations for T = {temperature}")
            
            # Read file data
            try:
                data = np.loadtxt(file_path, skiprows=1)
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            # Extract site indices and spin expectations
            sites = data[:, 0].astype(int)
            sp_real = np.mean(data[:, 1])
            sp_imag = np.mean(data[:, 2])
            sm_real = np.mean(data[:, 3])
            sm_imag = np.mean(data[:, 4])
            sz_real = np.mean(data[:, 5])
            sz_imag = np.mean(data[:, 6])

            # Find the index corresponding to the temperature
            temp_index = np.argmin(np.abs(self.temp_values - temperature))
            if temp_index >= len(spin_expectations['sp']):
                print(f"Warning: Temperature index {temp_index} out of bounds for spin expectations")
                continue
            # Store the spin expectation values
            
            # Store as complex numbers in a dictionary
            spin_expectations['sp'][temp_index] = sp_real + 1j * sp_imag
            spin_expectations['sm'][temp_index] = sm_real + 1j * sm_imag
            spin_expectations['sz'][temp_index] = sz_real + 1j * sz_imag
            
        return spin_expectations


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
                'entropy': {},
                'sp': {},
                'sm': {},
                'sz': {}
            }
        else:
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

            # Get subclusters with their multiplicities
            subclusters = self.get_subclusters(cluster_id)
            
            # Calculate weights for energy and specific heat
            for prop in ['energy', 'specific_heat', 'entropy']:
                # Property of the cluster
                property_value = quantities[prop]
                # Subtract contributions from all subclusters with correct multiplicities
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        property_value -= self.weights[prop][subcluster_id] * multiplicity
                # Store the weight
                self.weights[prop][cluster_id] = property_value


            if self.measure_spin:
                # Read spin expectation values if needed
                spin_exp_dir = os.path.join(self.eigenvalue_dir, f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output/spin_expectations")
                quantities_spin =  self.read_spin_expectations(spin_exp_dir)
                
                print(f"Spin expectation values for cluster {cluster_id}: {quantities_spin}")

                for prop in ['sp', 'sm', 'sz']:
                    # Get the spin expectation value
                    spin_value = quantities_spin[prop]
                    # Subtract contributions from all subclusters with correct multiplicities
                    for subcluster_id, multiplicity in subclusters.items():
                        if subcluster_id in self.weights[prop]:
                            spin_value -= self.weights[prop][subcluster_id] * multiplicity
                    # Store the weight
                    self.weights[prop][cluster_id] = spin_value
            



    def sum_nlc(self, euler_resum=False, order_cutoff=None, auto_resum=True):
        """Perform the NLC summation with optional Euler or automatic (Euler/Wynn/none) resummation.

        The internal weight storage uses a property-first mapping: self.weights[prop][cluster_id] = weight_array.
        This function reorganizes weights per order, then (optionally) applies a resummation heuristic
        independently at each temperature.
        """
        # Initialize results container with zeros
        props = ['energy', 'specific_heat', 'entropy'] + (['sp', 'sm', 'sz'] if self.measure_spin else [])
        results = {p: np.zeros_like(self.temp_values) for p in props}

        for prop in props:
            if prop not in self.weights:
                continue
            # Map: order -> list of weight arrays (clusters of that order)
            order_map = defaultdict(list)
            for cid, warr in self.weights[prop].items():
                order = self.clusters.get(cid, {}).get('order')
                if order is None:
                    continue
                if order_cutoff is not None and order > order_cutoff:
                    continue
                order_map[order].append(np.array(warr))
            if not order_map:
                continue
            # Aggregate contributions per order
            order_sequences = []  # list of (order, contrib_array)
            for o in sorted(order_map.keys()):
                contrib = np.sum(order_map[o], axis=0)
                order_sequences.append((o, contrib))
            if not order_sequences:
                continue
            nT = len(self.temp_values)
            out = np.zeros(nT)
            diag_list = []
            for ti in range(nT):
                per_order_terms = [seq[1][ti] for seq in order_sequences]
                if auto_resum:
                    diag = auto_resum_sequence(per_order_terms)
                    out[ti] = diag['value']
                    diag['order_terms'] = per_order_terms
                    diag_list.append(diag)
                elif euler_resum:
                    out[ti] = euler_alternating_sum(per_order_terms)
                else:
                    out[ti] = np.sum(per_order_terms)
            results[prop] = out
            if auto_resum:
                self.resum_diagnostics[prop] = diag_list
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
    parser.add_argument('--euler_resum', action='store_true', help='Force Euler resummation (overrides auto)')
    parser.add_argument('--auto_resum', action='store_true', help='Automatically choose resummation (Euler/Wynn/none) per temperature')
    parser.add_argument('--order_cutoff', type=int, help='Maximum order to include in summation')
    parser.add_argument('--plot', action='store_true', help='Generate plot of results')
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')
    parser.add_argument('--temp_min', type=float, default=1e-4, help='Minimum temperature for calculations')
    parser.add_argument('--temp_max', type=float, default=1.0, help='Maximum temperature for calculations')
    parser.add_argument('--temp_bins', type=int, default=200, help='Number of temperature points to calculate')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    nlc = NLCExpansion(args.cluster_dir, args.eigenvalue_dir, args.temp_min, args.temp_max, args.temp_bins, args.measure_spin, args.SI_units)
    results = nlc.run(euler_resum=args.euler_resum, order_cutoff=args.order_cutoff) if not args.auto_resum else nlc.sum_nlc(euler_resum=args.euler_resum, order_cutoff=args.order_cutoff, auto_resum=True)
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

        # Plot spin expectation values if requested
        if args.measure_spin:
            plt.figure(figsize=(8, 6))
            plt.plot(temperatures, results['sp'], 'm-', label='sp')
            plt.plot(temperatures, results['sm'], 'c-', label='sm')
            plt.plot(temperatures, results['sz'], 'y-', label='sz')
            plt.xlabel('Temperature (T)')
            plt.ylabel('Spin Expectation Values')
            plt.title('Spin Expectation Values vs Temperature')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()
            spin_plot_path = os.path.join(args.output_dir, "nlc_spin_expectations.png")
            plt.savefig(spin_plot_path)
            plt.close()
    
    print(f"NLC calculation completed! Results saved to {args.output_dir}")

