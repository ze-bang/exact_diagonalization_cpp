#!/usr/bin/env python3
"""
Post-processing script for pedantic_modes method results.
Combines eigenmode correlations into global structure factor.

Usage:
    python combine_modes.py <modes_directory> <output_file> [options]

Example:
    python combine_modes.py ./structure_factor_results/beta_1.0/modes/ global_SpSm_Q001.dat \
           --spin-combo SpSm --Q 0 0 1 --beta 1.0 --sample 1
"""

import numpy as np
import argparse
import os
import glob
from pathlib import Path

def read_eigenvalues(eigenval_file):
    """Read eigenvalue file and return list of eigenvalues."""
    eigenvalues = []
    try:
        with open(eigenval_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    mode_k = int(parts[0])
                    eigenval = float(parts[1])
                    eigenvalues.append(eigenval)
        return eigenvalues
    except FileNotFoundError:
        raise FileNotFoundError(f"Eigenvalue file not found: {eigenval_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading eigenvalue file {eigenval_file}: {e}")

def read_time_correlation(corr_file):
    """Read time correlation file and return (time, correlation) arrays."""
    times = []
    correlations = []
    try:
        with open(corr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    t = float(parts[0])
                    real_part = float(parts[1])
                    imag_part = float(parts[2])
                    times.append(t)
                    correlations.append(complex(real_part, imag_part))
        return np.array(times), np.array(correlations)
    except FileNotFoundError:
        raise FileNotFoundError(f"Correlation file not found: {corr_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading correlation file {corr_file}: {e}")

def combine_modes_to_global(modes_dir, spin_combo, Q, beta, sample, output_file):
    """
    Combine eigenmode correlations into global structure factor.
    
    Args:
        modes_dir: Directory containing mode results
        spin_combo: Spin combination name (e.g., "SpSm", "SzSz")
        Q: Momentum vector [Qx, Qy, Qz]
        beta: Inverse temperature
        sample: Sample index
        output_file: Output file path
    """
    modes_path = Path(modes_dir)
    
    # Construct eigenvalue filename
    Q_str = f"_Qx{Q[0]}_Qy{Q[1]}_Qz{Q[2]}"
    eigenval_file = modes_path / f"eigenvalues_{spin_combo}_q{Q_str}_beta={beta}.dat"
    
    print(f"Reading eigenvalues from: {eigenval_file}")
    eigenvalues = read_eigenvalues(eigenval_file)
    print(f"Found {len(eigenvalues)} eigenmodes with eigenvalues: {eigenvalues}")
    
    # Read and combine mode correlations
    combined_times = None
    combined_corr = None
    
    for k, eigenval in enumerate(eigenvalues):
        mode_name = f"{spin_combo}_mode{k}_q{Q_str}"
        corr_file = modes_path / f"time_corr_rand{sample}_{mode_name}_beta={beta}.dat"
        
        print(f"Reading mode {k} from: {corr_file}")
        
        try:
            times, correlations = read_time_correlation(corr_file)
            
            if combined_times is None:
                # Initialize combined result
                combined_times = times
                combined_corr = eigenval * correlations
                print(f"  Mode {k}: eigenvalue={eigenval:.6e}, contribution initialized")
            else:
                # Add weighted contribution
                if len(times) != len(combined_times):
                    print(f"  Warning: Mode {k} has different time grid size. Skipping.")
                    continue
                combined_corr += eigenval * correlations
                print(f"  Mode {k}: eigenvalue={eigenval:.6e}, contribution added")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Error processing mode {k}: {e}")
            continue
    
    if combined_corr is None:
        raise RuntimeError("No valid mode correlations found!")
    
    # Write combined result
    print(f"\nWriting global structure factor to: {output_file}")
    with open(output_file, 'w') as f:
        f.write("# t global_structure_factor_real global_structure_factor_imag\n")
        for t, corr in zip(combined_times, combined_corr):
            f.write(f"{t:.16e} {corr.real:.16e} {corr.imag:.16e}\n")
    
    print(f"Success! Combined {len(eigenvalues)} eigenmodes into global structure factor.")
    
    # Print some statistics
    max_corr = np.max(np.abs(combined_corr))
    print(f"Statistics:")
    print(f"  Time range: {combined_times[0]:.3f} to {combined_times[-1]:.3f}")
    print(f"  Time points: {len(combined_times)}")
    print(f"  Max |S(t)|: {max_corr:.6e}")
    print(f"  Final |S(t)|: {abs(combined_corr[-1]):.6e}")

def find_mode_files(modes_dir, pattern_prefix):
    """Find all mode files matching a pattern."""
    modes_path = Path(modes_dir)
    pattern = f"{pattern_prefix}*.dat"
    files = list(modes_path.glob(pattern))
    return sorted(files)

def auto_detect_parameters(modes_dir):
    """Auto-detect available parameters from directory contents."""
    modes_path = Path(modes_dir)
    
    # Find eigenvalue files to extract parameters
    eigenval_files = list(modes_path.glob("eigenvalues_*.dat"))
    
    if not eigenval_files:
        return [], []
    
    combinations = []
    for f in eigenval_files:
        # Parse filename: eigenvalues_SpSm_q_Qx0_Qy0_Qz1_beta=1.0.dat
        name = f.name
        if name.startswith("eigenvalues_") and name.endswith(".dat"):
            # Extract combination and Q
            parts = name[12:-4].split("_q_")  # Remove "eigenvalues_" and ".dat"
            if len(parts) == 2:
                spin_combo = parts[0]
                q_beta_part = parts[1]
                
                # Extract Q coordinates and beta
                import re
                q_match = re.search(r'Qx([^_]+)_Qy([^_]+)_Qz([^_]+)_beta=([^_]+)', q_beta_part)
                if q_match:
                    Qx, Qy, Qz, beta = q_match.groups()
                    combinations.append({
                        'spin_combo': spin_combo,
                        'Q': [float(Qx), float(Qy), float(Qz)],
                        'beta': float(beta),
                        'file': f
                    })
    
    return combinations

def main():
    parser = argparse.ArgumentParser(description="Combine eigenmode correlations into global structure factor")
    parser.add_argument("modes_dir", help="Directory containing mode results")
    parser.add_argument("output_file", help="Output file for global structure factor")
    
    parser.add_argument("--spin-combo", default="SpSm", help="Spin combination (e.g., SpSm, SzSz)")
    parser.add_argument("--Q", nargs=3, type=float, default=[0, 0, 1], help="Momentum vector [Qx Qy Qz]")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--sample", type=int, default=1, help="Sample index")
    
    parser.add_argument("--list", action="store_true", help="List available combinations and exit")
    parser.add_argument("--auto", action="store_true", help="Process all available combinations")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.modes_dir):
        print(f"Error: Directory not found: {args.modes_dir}")
        return 1
    
    if args.list:
        print("Available combinations in directory:")
        combinations = auto_detect_parameters(args.modes_dir)
        for i, combo in enumerate(combinations):
            print(f"  {i+1}. {combo['spin_combo']} Q=({combo['Q'][0]}, {combo['Q'][1]}, {combo['Q'][2]}) beta={combo['beta']}")
        return 0
    
    if args.auto:
        print("Auto-processing all available combinations...")
        combinations = auto_detect_parameters(args.modes_dir)
        
        for combo in combinations:
            output_name = f"global_{combo['spin_combo']}_Q{combo['Q'][0]}{combo['Q'][1]}{combo['Q'][2]}_beta{combo['beta']}.dat"
            output_path = Path(args.modes_dir).parent / output_name
            
            print(f"\n=== Processing {combo['spin_combo']} Q=({combo['Q'][0]}, {combo['Q'][1]}, {combo['Q'][2]}) beta={combo['beta']} ===")
            try:
                combine_modes_to_global(
                    args.modes_dir, 
                    combo['spin_combo'], 
                    combo['Q'], 
                    combo['beta'], 
                    args.sample, 
                    output_path
                )
            except Exception as e:
                print(f"Error processing combination: {e}")
                continue
    else:
        # Process single combination
        print(f"Processing single combination: {args.spin_combo} Q=({args.Q[0]}, {args.Q[1]}, {args.Q[2]}) beta={args.beta}")
        try:
            combine_modes_to_global(
                args.modes_dir, 
                args.spin_combo, 
                args.Q, 
                args.beta, 
                args.sample, 
                args.output_file
            )
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())