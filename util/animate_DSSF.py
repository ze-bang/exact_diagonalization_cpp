#!/usr/bin/env python3
"""
Script to read spectral_beta_inf.dat files across all h=# directories
and create animated/stacked plots for each species as a function of h.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import glob
from pathlib import Path
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Configuration
BASE_DIR = "/scratch/zhouzb79/DSSF_PCD_mag_field_sweep_0_flux"
OUTPUT_DIR = os.path.join(BASE_DIR, "spectral_animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Conversion factors
# For h values (magnetic field)
H_CONVERSION_FACTOR = 0.063 / (2.5 * 0.0578)
# For omega values (energy/frequency): Jzz = 0.063 meV in real material
ENERGY_CONVERSION_FACTOR = 0.063  # converts from Jzz units to meV

# Frequency range limits (in Jzz units)
FREQ_MIN = -0.5
FREQ_MAX = 4.0

# Factor to apply to SmSp when calculating total
SMSP_FACTOR = 0.5

def extract_h_value(h_dir):
    """Extract numerical h value from directory name like 'h=0.1'"""
    match = re.search(r'h=([0-9.]+)', h_dir)
    if match:
        return float(match.group(1))
    return None

def find_all_h_directories():
    """Find all h=# directories and sort them by h value"""
    h_dirs = glob.glob(os.path.join(BASE_DIR, "h=*"))
    h_dirs = [d for d in h_dirs if os.path.isdir(d)]
    
    # Extract h values and sort
    h_data = []
    for d in h_dirs:
        h_val = extract_h_value(d)
        if h_val is not None:
            h_data.append((h_val, d))
    
    h_data.sort(key=lambda x: x[0])
    return h_data

def find_all_species(h_dir):
    """Find all species directories in a given h directory"""
    processed_data_dir = os.path.join(h_dir, "structure_factor_results", "processed_data")
    if not os.path.exists(processed_data_dir):
        return []
    
    species_dirs = glob.glob(os.path.join(processed_data_dir, "*"))
    species_dirs = [d for d in species_dirs if os.path.isdir(d)]
    species_names = [os.path.basename(d) for d in species_dirs]
    
    return sorted(species_names)

def read_spectral_data(h_dir, species):
    """Read spectral_beta_inf.dat file for a given h and species"""
    file_path = os.path.join(h_dir, "structure_factor_results", "processed_data", 
                              species, "spectral_beta_inf.dat")
    
    if not os.path.exists(file_path):
        return None, None
    
    try:
        data = np.loadtxt(file_path)
        freq = data[:, 0]
        spectral = data[:, 1]
        return freq, spectral
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def create_do_channel(h_values, h_dirs, base_species):
    """
    Create DO (Difference Orbital) channel by summing SF and NSF channels.
    Returns freq_data and spectral_data dictionaries for the DO channel.
    """
    freq_data_do = {}
    spectral_data_do = {}
    
    sf_species = base_species + "_SF"
    nsf_species = base_species + "_NSF"
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Read SF channel
        freq_sf, spectral_sf = read_spectral_data(h_dir, sf_species)
        # Read NSF channel
        freq_nsf, spectral_nsf = read_spectral_data(h_dir, nsf_species)
        
        if freq_sf is not None and spectral_sf is not None and \
           freq_nsf is not None and spectral_nsf is not None:
            
            # Check if frequencies match
            if len(freq_sf) == len(freq_nsf) and np.allclose(freq_sf, freq_nsf):
                # Direct sum if frequencies match
                freq_data_do[h] = freq_sf
                spectral_data_do[h] = spectral_sf + spectral_nsf
            else:
                # Interpolate NSF to SF grid if they don't match
                spectral_nsf_interp = np.interp(freq_sf, freq_nsf, spectral_nsf)
                freq_data_do[h] = freq_sf
                spectral_data_do[h] = spectral_sf + spectral_nsf_interp
    
    return freq_data_do, spectral_data_do

def create_stacked_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 2D stacked plot showing spectral function vs frequency for different h values"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a colormap
    colors = cm.viridis(np.linspace(0, 1, len(h_values)))
    
    # Plot each h value with an offset for visibility
    offset = 0
    offset_step = np.abs(np.mean([np.max(spec) - np.min(spec) for spec in spectral_data.values()])) * 1.2
    
    for i, h in enumerate(h_values):
        if h in spectral_data and h in freq_data:
            # Filter data to frequency range
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            # Convert frequency to meV
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            h_converted = h * H_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask] + offset, 
                   label=f'h={h_converted:.2f}', color=colors[i], linewidth=1.5, alpha=0.8)
            offset += offset_step
    
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function (offset for clarity)', fontsize=12)
    ax.set_title(f'Spectral Function vs Energy - {species}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved stacked plot: {output_file}")
    plt.close()

def create_3d_surface_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 3D surface plot showing spectral function vs frequency and h"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a common frequency grid (use the first one as reference)
    if not h_values or h_values[0] not in freq_data:
        print(f"No data available for 3D plot of {species}")
        return
    
    # Filter frequency range to [-3, 6] (in Jzz units)
    freq_ref_full = freq_data[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    # Convert to meV
    freq_ref_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Convert h values for plotting
    h_values_converted = [h * H_CONVERSION_FACTOR for h in h_values]
    
    # Create meshgrid
    H, FREQ = np.meshgrid(h_values_converted, freq_ref_meV)
    SPECTRAL = np.zeros_like(H)
    
    for i, h in enumerate(h_values):
        if h in spectral_data:
            # Filter data to frequency range
            freq_h = freq_data[h]
            spec_h = spectral_data[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            freq_h_filtered = freq_h[mask_h]
            spec_h_filtered = spec_h[mask_h]
            
            # Interpolate to common grid
            SPECTRAL[:, i] = np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
    
    # Create surface plot
    surf = ax.plot_surface(H, FREQ, SPECTRAL, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Energy (meV)', fontsize=12)
    ax.set_zlabel('Spectral Function', fontsize=12)
    ax.set_title(f'Spectral Function Surface - {species}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved 3D surface plot: {output_file}")
    plt.close()

def create_heatmap_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 2D heatmap showing spectral function vs frequency and h"""
    # Get a common frequency grid
    if not h_values or h_values[0] not in freq_data:
        print(f"No data available for heatmap of {species}")
        return
    
    # Filter frequency range to [-3, 6] (in Jzz units)
    freq_ref_full = freq_data[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Create data matrix
    spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        if h in spectral_data:
            # Filter data to frequency range
            freq_h = freq_data[h]
            spec_h = spectral_data[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            freq_h_filtered = freq_h[mask_h]
            spec_h_filtered = spec_h[mask_h]
            
            # Interpolate to common grid
            spectral_matrix[:, i] = np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert h values for plotting
    h_min_converted = min(h_values) * H_CONVERSION_FACTOR
    h_max_converted = max(h_values) * H_CONVERSION_FACTOR
    
    # Convert frequency to meV
    freq_min_meV = min(freq_ref) * ENERGY_CONVERSION_FACTOR
    freq_max_meV = max(freq_ref) * ENERGY_CONVERSION_FACTOR
    
    # Create heatmap
    im = ax.imshow(spectral_matrix, aspect='auto', origin='lower', 
                   cmap='viridis', interpolation='bilinear',
                   extent=[h_min_converted, h_max_converted, freq_min_meV, freq_max_meV])
    
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Energy (meV)', fontsize=12)
    ax.set_title(f'Spectral Function Heatmap - {species}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Spectral Function', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()

def create_animation(h_values, freq_data, spectral_data, species, output_file):
    """Create an animation showing spectral function evolving with h"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get y-axis limits
    all_spectral = [spectral_data[h] for h in h_values if h in spectral_data]
    if not all_spectral:
        print(f"No data available for animation of {species}")
        return
    
    ymin = min([np.min(s) for s in all_spectral])
    ymax = max([np.max(s) for s in all_spectral])
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Set x-axis limits to [-3, 6] in Jzz units, converted to meV
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line.set_data([], [])
        return line, title
    
    def animate(frame):
        h = h_values[frame]
        if h in freq_data and h in spectral_data:
            # Filter data to frequency range [-3, 6] (in Jzz units)
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            # Convert frequency to meV
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            line.set_data(freq_meV, spec[mask])
            h_converted = h * H_CONVERSION_FACTOR
            title.set_text(f'Spectral Function - {species} - h={h_converted:.3f} T')
        return line, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=2)
    anim.save(output_file, writer=writer)
    print(f"Saved animation: {output_file}")
    plt.close()

def create_combined_component_plot_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create a plot showing SmSp, SzSz, and their sum (Total) for given species.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name (e.g., "SmSp_q_Qx0_Qy0_Qz0_SF")
    - szsz_species: full SzSz species name (e.g., "SzSz_q_Qx0_Qy0_Qz0_SF")
    - display_name: name to display in plot title
    - output_file: path to save the plot
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined plot: missing {smsp_species} or {szsz_species}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot for each h value (show up to 4 representative h values)
    num_plots = min(4, len(h_values))
    plot_indices = np.linspace(0, len(h_values)-1, num_plots, dtype=int)
    
    for plot_idx, h_idx in enumerate(plot_indices):
        h = h_values[h_idx]
        ax = axes[plot_idx]
        
        # Get SmSp data
        if h in all_freq_data[smsp_species] and h in all_spectral_data[smsp_species]:
            freq_spm = all_freq_data[smsp_species][h]
            spec_spm = all_spectral_data[smsp_species][h]/2  # Apply factor to SmSp
            mask_spm = (freq_spm >= FREQ_MIN) & (freq_spm <= FREQ_MAX)
            freq_spm_meV = freq_spm[mask_spm] * ENERGY_CONVERSION_FACTOR
            spec_spm_filtered = spec_spm[mask_spm]
        else:
            freq_spm_meV = None
            spec_spm_filtered = None
        
        # Get SzSz data
        if h in all_freq_data[szsz_species] and h in all_spectral_data[szsz_species]:
            freq_szz = all_freq_data[szsz_species][h]
            spec_szz = all_spectral_data[szsz_species][h]
            mask_szz = (freq_szz >= FREQ_MIN) & (freq_szz <= FREQ_MAX)
            freq_szz_meV = freq_szz[mask_szz] * ENERGY_CONVERSION_FACTOR
            spec_szz_filtered = spec_szz[mask_szz]
        else:
            freq_szz_meV = None
            spec_szz_filtered = None
        
        # Plot individual components
        if freq_spm_meV is not None:
            ax.plot(freq_spm_meV, spec_spm_filtered, 'b-', label='SmSp', linewidth=2, alpha=0.7)
        if freq_szz_meV is not None:
            ax.plot(freq_szz_meV, spec_szz_filtered, 'r-', label='SzSz', linewidth=2, alpha=0.7)
        
        # Calculate and plot sum if both exist
        if freq_spm_meV is not None and freq_szz_meV is not None:
            # Interpolate to common grid if needed
            if len(freq_spm_meV) == len(freq_szz_meV) and np.allclose(freq_spm_meV, freq_szz_meV):
                spec_total = spec_spm_filtered + spec_szz_filtered
                freq_total = freq_spm_meV
            else:
                # Use Spm grid as reference
                spec_szz_interp = np.interp(freq_spm[mask_spm], freq_szz[mask_szz], spec_szz_filtered)
                spec_total = spec_spm_filtered + spec_szz_interp
                freq_total = freq_spm_meV
            
            ax.plot(freq_total, spec_total, 'g-', label='Total (SmSp+SzSz)', linewidth=2.5, alpha=0.9)
        
        h_converted = h * H_CONVERSION_FACTOR
        ax.set_xlabel('Energy (meV)', fontsize=11)
        ax.set_ylabel('Spectral Function', fontsize=11)
        ax.set_title(f'{display_name} - h={h_converted:.3f} T', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    
    plt.suptitle(f'Spectral Components Comparison - {display_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved combined component plot: {output_file}")
    plt.close()

def create_combined_heatmap_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create a 3-panel heatmap showing SmSp, SzSz, and Total side by side.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name
    - szsz_species: full SzSz species name
    - display_name: name to display in plot title
    - output_file: path to save the plot
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined heatmap: missing {smsp_species} or {szsz_species}")
        return
    
    # Get a common frequency grid
    if not h_values or h_values[0] not in all_freq_data[smsp_species]:
        print(f"  ⚠ No data available for combined heatmap of {display_name}")
        return
    
    freq_ref_full = all_freq_data[smsp_species][h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Create data matrices for SmSp, SzSz, and Total
    spm_matrix = np.zeros((len(freq_ref), len(h_values)))
    szz_matrix = np.zeros((len(freq_ref), len(h_values)))
    total_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        # Process SmSp
        if h in all_spectral_data[smsp_species]:
            freq_h = all_freq_data[smsp_species][h]
            spec_h = all_spectral_data[smsp_species][h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            spm_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])/2
        
        # Process SzSz
        if h in all_spectral_data[szsz_species]:
            freq_h = all_freq_data[szsz_species][h]
            spec_h = all_spectral_data[szsz_species][h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            szz_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])
        
        # Calculate total
        total_matrix[:, i] = spm_matrix[:, i] + szz_matrix[:, i]
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    h_min_converted = min(h_values) * H_CONVERSION_FACTOR
    h_max_converted = max(h_values) * H_CONVERSION_FACTOR
    freq_min_meV = min(freq_ref) * ENERGY_CONVERSION_FACTOR
    freq_max_meV = max(freq_ref) * ENERGY_CONVERSION_FACTOR
    
    # Determine common color scale for consistency
    vmin = min(spm_matrix.min(), szz_matrix.min(), total_matrix.min())
    vmax = max(spm_matrix.max(), szz_matrix.max(), total_matrix.max())
    
    # Plot SmSp
    im1 = axes[0].imshow(spm_matrix, aspect='auto', origin='lower', 
                         cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax,
                         extent=[h_min_converted, h_max_converted, freq_min_meV, freq_max_meV])
    axes[0].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[0].set_ylabel('Energy (meV)', fontsize=11)
    axes[0].set_title('SmSp', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot SzSz
    im2 = axes[1].imshow(szz_matrix, aspect='auto', origin='lower', 
                         cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax,
                         extent=[h_min_converted, h_max_converted, freq_min_meV, freq_max_meV])
    axes[1].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[1].set_ylabel('Energy (meV)', fontsize=11)
    axes[1].set_title('SzSz', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[1])
    
    # Plot Total
    im3 = axes[2].imshow(total_matrix, aspect='auto', origin='lower', 
                         cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax,
                         extent=[h_min_converted, h_max_converted, freq_min_meV, freq_max_meV])
    axes[2].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[2].set_ylabel('Energy (meV)', fontsize=11)
    axes[2].set_title('Total (SmSp+SzSz)', fontsize=12, fontweight='bold')
    fig.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Spectral Components Heatmap - {display_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved combined heatmap: {output_file}")
    plt.close()

def create_combined_animation_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create an animation showing SmSp, SzSz, and Total evolving with h.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name
    - szsz_species: full SzSz species name
    - display_name: name to display in plot title
    - output_file: path to save the animation
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined animation: missing {smsp_species} or {szsz_species}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get y-axis limits by checking all data
    all_smsp = [all_spectral_data[smsp_species][h] for h in h_values if h in all_spectral_data[smsp_species]]
    all_szsz = [all_spectral_data[szsz_species][h] for h in h_values if h in all_spectral_data[szsz_species]]
    
    if not all_smsp or not all_szsz:
        print(f"  ⚠ No data available for animation of {display_name}")
        return
    
    # Calculate y-limits considering both components and their sum
    ymin = min(min([np.min(s) for s in all_smsp]), min([np.min(s) for s in all_szsz]))
    ymax_smsp = max([np.max(s) for s in all_smsp])/2
    ymax_szsz = max([np.max(s) for s in all_szsz])
    ymax_total = ymax_smsp + ymax_szsz  # Approximate maximum of sum
    ymax = ymax_total
    
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Create line objects
    line_smsp, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='SmSp')
    line_szsz, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='SzSz')
    line_total, = ax.plot([], [], 'g-', linewidth=2.5, alpha=0.9, label='Total (SmSp+SzSz)')
    
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line_smsp.set_data([], [])
        line_szsz.set_data([], [])
        line_total.set_data([], [])
        return line_smsp, line_szsz, line_total, title
    
    def animate(frame):
        h = h_values[frame]
        h_converted = h * H_CONVERSION_FACTOR
        
        # Get SmSp data
        if h in all_freq_data[smsp_species] and h in all_spectral_data[smsp_species]:
            freq_smsp = all_freq_data[smsp_species][h]
            spec_smsp = all_spectral_data[smsp_species][h]/2
            mask_smsp = (freq_smsp >= FREQ_MIN) & (freq_smsp <= FREQ_MAX)
            freq_smsp_meV = freq_smsp[mask_smsp] * ENERGY_CONVERSION_FACTOR
            spec_smsp_filtered = spec_smsp[mask_smsp]
            line_smsp.set_data(freq_smsp_meV, spec_smsp_filtered)
        else:
            line_smsp.set_data([], [])
            freq_smsp_meV = None
            spec_smsp_filtered = None
        
        # Get SzSz data
        if h in all_freq_data[szsz_species] and h in all_spectral_data[szsz_species]:
            freq_szsz = all_freq_data[szsz_species][h]
            spec_szsz = all_spectral_data[szsz_species][h]
            mask_szsz = (freq_szsz >= FREQ_MIN) & (freq_szsz <= FREQ_MAX)
            freq_szsz_meV = freq_szsz[mask_szsz] * ENERGY_CONVERSION_FACTOR
            spec_szsz_filtered = spec_szsz[mask_szsz]
            line_szsz.set_data(freq_szsz_meV, spec_szsz_filtered)
        else:
            line_szsz.set_data([], [])
            freq_szsz_meV = None
            spec_szsz_filtered = None
        
        # Calculate and plot total
        if freq_smsp_meV is not None and freq_szsz_meV is not None:
            # Interpolate to common grid if needed
            if len(freq_smsp_meV) == len(freq_szsz_meV) and np.allclose(freq_smsp_meV, freq_szsz_meV):
                spec_total = spec_smsp_filtered + spec_szsz_filtered
                freq_total = freq_smsp_meV
            else:
                # Use SmSp grid as reference
                freq_smsp_raw = freq_smsp[mask_smsp]
                freq_szsz_raw = freq_szsz[mask_szsz]
                spec_szsz_interp = np.interp(freq_smsp_raw, freq_szsz_raw, spec_szsz_filtered)
                spec_total = spec_smsp_filtered + spec_szsz_interp
                freq_total = freq_smsp_meV
            
            line_total.set_data(freq_total, spec_total)
        else:
            line_total.set_data([], [])
        
        title.set_text(f'Spectral Components - {display_name} - h={h_converted:.3f} T')
        return line_smsp, line_szsz, line_total, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=2)
    anim.save(output_file, writer=writer)
    print(f"  ✓ Saved combined animation: {output_file}")
    plt.close()

def main():
    """Main function to process all data and create plots"""
    print("Finding h directories...")
    h_data = find_all_h_directories()
    
    if not h_data:
        print("No h directories found!")
        return
    
    h_values = [h for h, _ in h_data]
    h_dirs = {h: d for h, d in h_data}
    
    print(f"Found {len(h_values)} h values: {h_values}")
    
    # Find all unique species across all h directories
    print("\nFinding all species...")
    all_species = set()
    for h, h_dir in h_data:
        species = find_all_species(h_dir)
        all_species.update(species)
    
    all_species = sorted(list(all_species))
    print(f"Found {len(all_species)} species:")
    for sp in all_species:
        print(f"  - {sp}")
    
    # Identify base species (without _SF or _NSF suffix) to create DO channels
    base_species_set = set()
    for sp in all_species:
        if sp.endswith("_SF"):
            base_species_set.add(sp[:-3])  # Remove "_SF"
        elif sp.endswith("_NSF"):
            base_species_set.add(sp[:-4])  # Remove "_NSF"
    
    # Create DO channels
    do_channels = {}
    if base_species_set:
        print(f"\nCreating DO channels (SF + NSF) for {len(base_species_set)} base species...")
        for base_sp in sorted(base_species_set):
            sf_sp = base_sp + "_SF"
            nsf_sp = base_sp + "_NSF"
            if sf_sp in all_species and nsf_sp in all_species:
                print(f"  Creating DO for: {base_sp}")
                freq_data_do, spectral_data_do = create_do_channel(h_values, h_dirs, base_sp)
                if freq_data_do:
                    do_channels[base_sp + "_DO"] = (freq_data_do, spectral_data_do)
                    print(f"    ✓ Created {base_sp}_DO with {len(freq_data_do)} h-values")
                else:
                    print(f"    ✗ Failed to create {base_sp}_DO")
    
    # Process each species
    print("\n" + "="*80)
    for species in all_species:
        print(f"\nProcessing species: {species}")
        print("-"*80)
        
        # Collect data for this species across all h values
        freq_data = {}
        spectral_data = {}
        
        for h in h_values:
            h_dir = h_dirs[h]
            freq, spectral = read_spectral_data(h_dir, species)
            
            if freq is not None and spectral is not None:
                freq_data[h] = freq
                spectral_data[h] = spectral
                print(f"  ✓ h={h:.2f}: {len(freq)} data points")
            else:
                print(f"  ✗ h={h:.2f}: No data")
        
        if not spectral_data:
            print(f"  No data found for {species}, skipping...")
            continue
        
        # Create safe filename
        safe_species_name = species.replace("/", "_").replace(" ", "_")
        
        # Create plots
        print(f"\n  Creating plots for {species}...")
        
        # 1. Stacked plot
        stacked_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_stacked.png")
        create_stacked_plot(h_values, freq_data, spectral_data, species, stacked_file)
        
        # 2. Heatmap
        heatmap_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_heatmap.png")
        create_heatmap_plot(h_values, freq_data, spectral_data, species, heatmap_file)
        
        # 3. 3D surface plot
        surface_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_3d_surface.png")
        create_3d_surface_plot(h_values, freq_data, spectral_data, species, surface_file)
        
        # 4. Animation
        anim_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_animation.gif")
        create_animation(h_values, freq_data, spectral_data, species, anim_file)
        
        print(f"  ✓ Completed {species}")
    
    # Process DO channels
    if do_channels:
        print("\n" + "="*80)
        print("Processing DO channels (SF + NSF)...")
        print("="*80)
        
        for do_species, (freq_data, spectral_data) in do_channels.items():
            print(f"\nProcessing DO channel: {do_species}")
            print("-"*80)
            
            if not spectral_data:
                print(f"  No data found for {do_species}, skipping...")
                continue
            
            # Create safe filename
            safe_species_name = do_species.replace("/", "_").replace(" ", "_")
            
            # Create plots
            print(f"  Creating plots for {do_species}...")
            
            # 1. Stacked plot
            stacked_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_stacked.png")
            create_stacked_plot(h_values, freq_data, spectral_data, do_species, stacked_file)
            
            # 2. Heatmap
            heatmap_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_heatmap.png")
            create_heatmap_plot(h_values, freq_data, spectral_data, do_species, heatmap_file)
            
            # 3. 3D surface plot
            surface_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_3d_surface.png")
            create_3d_surface_plot(h_values, freq_data, spectral_data, do_species, surface_file)
            
            # 4. Animation
            anim_file = os.path.join(OUTPUT_DIR, f"{safe_species_name}_animation.gif")
            create_animation(h_values, freq_data, spectral_data, do_species, anim_file)
            
            print(f"  ✓ Completed {do_species}")
    
    # Create combined Spm + Szz plots
    print("\n" + "="*80)
    print("Creating combined Spm + Szz plots...")
    print("="*80)
    
    # Build a complete data dictionary for all species
    all_freq_data = {}
    all_spectral_data = {}
    
    # Add regular species data
    for species in all_species:
        freq_data = {}
        spectral_data = {}
        for h in h_values:
            h_dir = h_dirs[h]
            freq, spectral = read_spectral_data(h_dir, species)
            if freq is not None and spectral is not None:
                freq_data[h] = freq
                spectral_data[h] = spectral
        
        if freq_data:
            all_freq_data[species] = freq_data
            all_spectral_data[species] = spectral_data
    
    # Add DO channel data
    for do_species, (freq_data, spectral_data) in do_channels.items():
        if freq_data:
            all_freq_data[do_species] = freq_data
            all_spectral_data[do_species] = spectral_data
    
    # Identify base species and suffixes with both SmSp and SzSz
    # Pattern: SmSp_q_Qx#_Qy#_Qz#_SUFFIX or SzSz_q_Qx#_Qy#_Qz#_SUFFIX
    species_combinations = {}  # Maps (base_pattern, suffix) -> set of components
    
    for species in all_freq_data.keys():
        # Try to parse species name
        # Check if it's SmSp or SzSz type
        if species.startswith("SmSp_"):
            # Find the suffix (_SF, _NSF, or _DO)
            if "_SF" in species:
                suffix = "_SF"
                base_pattern = species.replace("SmSp_", "").replace("_SF", "")
            elif "_NSF" in species:
                suffix = "_NSF"
                base_pattern = species.replace("SmSp_", "").replace("_NSF", "")
            elif "_DO" in species:
                suffix = "_DO"
                base_pattern = species.replace("SmSp_", "").replace("_DO", "")
            else:
                # No suffix, the whole thing after SmSp_ is the pattern
                suffix = ""
                base_pattern = species.replace("SmSp_", "")
            
            key = (base_pattern, suffix)
            if key not in species_combinations:
                species_combinations[key] = set()
            species_combinations[key].add("SmSp")
            
        elif species.startswith("SzSz_"):
            # Find the suffix (_SF, _NSF, or _DO)
            if "_SF" in species:
                suffix = "_SF"
                base_pattern = species.replace("SzSz_", "").replace("_SF", "")
            elif "_NSF" in species:
                suffix = "_NSF"
                base_pattern = species.replace("SzSz_", "").replace("_NSF", "")
            elif "_DO" in species:
                suffix = "_DO"
                base_pattern = species.replace("SzSz_", "").replace("_DO", "")
            else:
                # No suffix, the whole thing after SzSz_ is the pattern
                suffix = ""
                base_pattern = species.replace("SzSz_", "")
            
            key = (base_pattern, suffix)
            if key not in species_combinations:
                species_combinations[key] = set()
            species_combinations[key].add("SzSz")
    
    # Create combined plots for species with both SmSp and SzSz
    for (base_pattern, suffix), components in species_combinations.items():
        if "SmSp" in components and "SzSz" in components:
            print(f"\nCreating combined plots for {base_pattern}{suffix}")
            print("-"*80)
            
            # Reconstruct the full species names
            smsp_species = f"SmSp_{base_pattern}{suffix}"
            szsz_species = f"SzSz_{base_pattern}{suffix}"
            
            safe_name = f"{base_pattern}{suffix}".replace("/", "_").replace(" ", "_")
            
            # 1. Combined component plot (line plots at different h values)
            combined_file = os.path.join(OUTPUT_DIR, f"{safe_name}_combined_components.png")
            create_combined_component_plot_direct(h_values, all_freq_data, all_spectral_data, 
                                                  smsp_species, szsz_species, 
                                                  f"{base_pattern}{suffix}", combined_file)
            
            # 2. Combined heatmap (3-panel: SmSp, SzSz, Total)
            heatmap_file = os.path.join(OUTPUT_DIR, f"{safe_name}_combined_heatmap.png")
            create_combined_heatmap_direct(h_values, all_freq_data, all_spectral_data, 
                                          smsp_species, szsz_species,
                                          f"{base_pattern}{suffix}", heatmap_file)
            
            # 3. Combined animation (GIF showing SmSp, SzSz, and Total evolving with h)
            anim_file = os.path.join(OUTPUT_DIR, f"{safe_name}_combined_animation.gif")
            create_combined_animation_direct(h_values, all_freq_data, all_spectral_data,
                                           smsp_species, szsz_species,
                                           f"{base_pattern}{suffix}", anim_file)
            
            print(f"  ✓ Completed combined plots for {base_pattern}{suffix}")
    
    print("\n" + "="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
