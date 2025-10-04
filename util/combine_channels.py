#!/usr/bin/env python3
"""
Script to combine SF/NSF channels and SpSm/SmSp correlations for both taylor and global modes.

For taylor (local frame) mode:
- Combines SpSm and SmSp to produce (SpSm + SmSp)/2

For global (global frame) mode:
- Combines SpSm_SF and SmSp_SF to produce (SpSm_SF + SmSp_SF)/2
- Combines SpSm_NSF and SmSp_NSF to produce (SpSm_NSF + SmSp_NSF)/2
- Sums SF and NSF channels: (SpSm_SF + SmSp_SF)/2 + (SpSm_NSF + SmSp_NSF)/2

Usage:
    python combine_channels.py <structure_factor_results_dir> [--beta BETA]
    
Example:
    python combine_channels.py ./structure_factor_results
    python combine_channels.py ./structure_factor_results --beta 1.0
"""

import os
import sys
import glob
import numpy as np
import re
from collections import defaultdict
import argparse


def parse_filename(filename):
    """
    Extract information from time correlation filenames.
    
    Expected format: time_corr_rand<N>_<SpinCombo>_q_Qx<x>_Qy<y>_Qz<z>[_SF|_NSF]_beta=<beta>.dat
    
    Returns:
        dict with keys: sample_idx, spin_combo, Qx, Qy, Qz, channel (SF/NSF/None), beta
    """
    basename = os.path.basename(filename)
    
    # Pattern to match the filename structure
    pattern = r'time_corr_(?:rand|sample)(\d+)_(\w+)_q_Qx([-\d.]+)_Qy([-\d.]+)_Qz([-\d.]+)(?:_(SF|NSF))?_beta=([\d.]+|inf)\.dat'
    
    match = re.match(pattern, basename)
    if not match:
        return None
    
    sample_idx = int(match.group(1))
    spin_combo = match.group(2)
    Qx = float(match.group(3))
    Qy = float(match.group(4))
    Qz = float(match.group(5))
    channel = match.group(6)  # SF, NSF, or None
    beta_str = match.group(7)
    beta = float('inf') if beta_str == 'inf' else float(beta_str)
    
    return {
        'sample_idx': sample_idx,
        'spin_combo': spin_combo,
        'Qx': Qx,
        'Qy': Qy,
        'Qz': Qz,
        'channel': channel,
        'beta': beta,
        'basename': basename
    }


def load_correlation_data(filepath):
    """Load time correlation data from file."""
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        time = data[:, 0]
        
        if data.shape[1] >= 3:
            real_part = data[:, 1]
            imag_part = data[:, 2]
        elif data.shape[1] == 2:
            real_part = data[:, 1]
            imag_part = np.zeros_like(real_part)
        else:
            return None, None
        
        corr = real_part + 1j * imag_part
        return time, corr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def save_correlation_data(filepath, time, corr):
    """Save time correlation data to file."""
    real_part = np.real(corr)
    imag_part = np.imag(corr)
    
    data = np.column_stack((time, real_part, imag_part))
    header = "t correlation_real correlation_imag"
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, header=header, fmt='%.8e')
    print(f"Saved: {filepath}")


def process_taylor_mode(beta_dir, beta_value):
    """
    Process taylor mode data: combine SpSm and SmSp to get (SpSm + SmSp)/2
    """
    taylor_dir = os.path.join(beta_dir, 'taylor')
    if not os.path.isdir(taylor_dir):
        return
    
    print(f"\nProcessing taylor mode for beta={beta_value}")
    
    # Find all correlation files
    files = glob.glob(os.path.join(taylor_dir, 'time_corr_*.dat'))
    
    # Organize files by momentum and sample
    file_info = defaultdict(lambda: defaultdict(dict))
    
    for filepath in files:
        info = parse_filename(filepath)
        if info is None or info['channel'] is not None:
            continue  # Skip if parsing failed or if it has SF/NSF suffix
        
        Q_key = (info['Qx'], info['Qy'], info['Qz'])
        sample = info['sample_idx']
        spin_combo = info['spin_combo']
        
        file_info[Q_key][sample][spin_combo] = filepath
    
    # Process each momentum point
    output_dir = os.path.join(beta_dir, 'taylor_combined')
    
    for Q_key, sample_data in file_info.items():
        Qx, Qy, Qz = Q_key
        
        for sample, spin_files in sample_data.items():
            # Check if we have both SpSm and SmSp
            if 'SpSm' not in spin_files or 'SmSp' not in spin_files:
                continue
            
            # Load data
            time_spsm, corr_spsm = load_correlation_data(spin_files['SpSm'])
            time_smsp, corr_smsp = load_correlation_data(spin_files['SmSp'])
            
            if time_spsm is None or time_smsp is None:
                continue
            
            # Check if time arrays match
            if not np.allclose(time_spsm, time_smsp):
                print(f"Warning: Time arrays don't match for sample {sample}, Q={Q_key}")
                continue
            
            # Compute average
            corr_avg = (corr_spsm + corr_smsp) / 2.0
            
            # Save result
            output_filename = f'time_corr_rand{sample}_SpSm+SmSp_q_Qx{Qx}_Qy{Qy}_Qz{Qz}_beta={beta_value}.dat'
            output_path = os.path.join(output_dir, output_filename)
            
            save_correlation_data(output_path, time_spsm, corr_avg)


def process_global_mode(beta_dir, beta_value):
    """
    Process global mode data:
    1. Combine SpSm_SF and SmSp_SF to get (SpSm_SF + SmSp_SF)/2
    2. Combine SpSm_NSF and SmSp_NSF to get (SpSm_NSF + SmSp_NSF)/2
    3. Sum SF and NSF: (SpSm_SF + SmSp_SF)/2 + (SpSm_NSF + SmSp_NSF)/2
    """
    global_dir = os.path.join(beta_dir, 'global')
    if not os.path.isdir(global_dir):
        return
    
    print(f"\nProcessing global mode for beta={beta_value}")
    
    # Find all correlation files
    files = glob.glob(os.path.join(global_dir, 'time_corr_*.dat'))
    
    # Organize files by momentum, channel, and sample
    file_info = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for filepath in files:
        info = parse_filename(filepath)
        if info is None or info['channel'] is None:
            continue  # Skip if parsing failed or if it doesn't have SF/NSF
        
        Q_key = (info['Qx'], info['Qy'], info['Qz'])
        channel = info['channel']
        sample = info['sample_idx']
        spin_combo = info['spin_combo']
        
        file_info[Q_key][channel][sample][spin_combo] = filepath
    
    # Process each momentum point
    output_dir = os.path.join(beta_dir, 'global_combined')
    
    for Q_key, channel_data in file_info.items():
        Qx, Qy, Qz = Q_key
        
        # Check if we have both SF and NSF channels
        if 'SF' not in channel_data or 'NSF' not in channel_data:
            print(f"Warning: Missing SF or NSF channel for Q={Q_key}")
            continue
        
        # Process each sample
        sf_samples = set(channel_data['SF'].keys())
        nsf_samples = set(channel_data['NSF'].keys())
        common_samples = sf_samples & nsf_samples
        
        for sample in common_samples:
            sf_files = channel_data['SF'][sample]
            nsf_files = channel_data['NSF'][sample]
            
            # Check if we have SpSm and SmSp for both channels
            if 'SpSm' not in sf_files or 'SmSp' not in sf_files:
                print(f"Warning: Missing SpSm/SmSp in SF channel for sample {sample}, Q={Q_key}")
                continue
            
            if 'SpSm' not in nsf_files or 'SmSp' not in nsf_files:
                print(f"Warning: Missing SpSm/SmSp in NSF channel for sample {sample}, Q={Q_key}")
                continue
            
            # Load SF channel data
            time_sf_spsm, corr_sf_spsm = load_correlation_data(sf_files['SpSm'])
            time_sf_smsp, corr_sf_smsp = load_correlation_data(sf_files['SmSp'])
            
            # Load NSF channel data
            time_nsf_spsm, corr_nsf_spsm = load_correlation_data(nsf_files['SpSm'])
            time_nsf_smsp, corr_nsf_smsp = load_correlation_data(nsf_files['SmSp'])
            
            if any(x is None for x in [time_sf_spsm, time_sf_smsp, time_nsf_spsm, time_nsf_smsp]):
                continue
            
            # Check time arrays match
            if not (np.allclose(time_sf_spsm, time_sf_smsp) and 
                    np.allclose(time_sf_spsm, time_nsf_spsm) and
                    np.allclose(time_sf_spsm, time_nsf_smsp)):
                print(f"Warning: Time arrays don't match for sample {sample}, Q={Q_key}")
                continue
            
            time = time_sf_spsm
            
            # Compute averages for each channel
            corr_sf_avg = (corr_sf_spsm + corr_sf_smsp) / 2.0
            corr_nsf_avg = (corr_nsf_spsm + corr_nsf_smsp) / 2.0
            
            # Save SF channel average
            output_sf = f'time_corr_rand{sample}_SpSm+SmSp_q_Qx{Qx}_Qy{Qy}_Qz{Qz}_SF_beta={beta_value}.dat'
            save_correlation_data(os.path.join(output_dir, output_sf), time, corr_sf_avg)
            
            # Save NSF channel average
            output_nsf = f'time_corr_rand{sample}_SpSm+SmSp_q_Qx{Qx}_Qy{Qy}_Qz{Qz}_NSF_beta={beta_value}.dat'
            save_correlation_data(os.path.join(output_dir, output_nsf), time, corr_nsf_avg)
            
            # Compute total (SF + NSF)
            corr_total = corr_sf_avg + corr_nsf_avg
            
            # Save total
            output_total = f'time_corr_rand{sample}_SpSm+SmSp_q_Qx{Qx}_Qy{Qy}_Qz{Qz}_SF+NSF_beta={beta_value}.dat'
            save_correlation_data(os.path.join(output_dir, output_total), time, corr_total)


def process_structure_factor_results(base_dir, target_beta=None):
    """
    Process all beta directories in the structure_factor_results directory.
    """
    # Find all beta directories
    beta_dirs = glob.glob(os.path.join(base_dir, 'beta_*'))
    
    if not beta_dirs:
        print(f"No beta directories found in {base_dir}")
        return
    
    print(f"Found {len(beta_dirs)} beta directories")
    
    for beta_dir in sorted(beta_dirs):
        # Extract beta value
        beta_match = re.search(r'beta_([\d\.]+|inf)', os.path.basename(beta_dir))
        if not beta_match:
            continue
        
        beta_str = beta_match.group(1)
        beta_value = float('inf') if beta_str == 'inf' else float(beta_str)
        
        # Skip if target_beta is specified and doesn't match
        if target_beta is not None and not np.isclose(beta_value, target_beta, rtol=1e-6):
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing beta directory: {os.path.basename(beta_dir)}")
        print(f"{'='*60}")
        
        # Process taylor mode
        process_taylor_mode(beta_dir, beta_str)
        
        # Process global mode
        process_global_mode(beta_dir, beta_str)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Combine SF/NSF channels and SpSm/SmSp correlations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'base_dir',
        help='Path to structure_factor_results directory'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Process only a specific beta value (optional)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.base_dir):
        print(f"Error: Directory not found: {args.base_dir}")
        sys.exit(1)
    
    process_structure_factor_results(args.base_dir, args.beta)


if __name__ == '__main__':
    main()
