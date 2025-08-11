import os
import glob
import numpy as np
from collections import defaultdict
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from mpi4py import MPI
# Function to apply broadening in time domain
def apply_time_broadening(t_values, data, broadening_type='gaussian', sigma=None, gamma=None):
    """
    Apply broadening to time correlation data before FFT
    
    Parameters:
    t_values: Time array
    data: Complex correlation data
    broadening_type: 'gaussian' or 'lorentzian'
    sigma: Width parameter for Gaussian broadening
    gamma: Width parameter for Lorentzian broadening
    
    Returns:
    broadened_data: Data with broadening applied
    """
    if broadening_type == 'gaussian' and sigma is not None:
        # Apply Gaussian broadening: multiply by exp(-t²/(2σ²))
        broadening_factor = np.exp(-t_values**2 / (2 * sigma**2))
        broadened_data = data * broadening_factor
        print(f"Applied Gaussian broadening in time domain with σ = {sigma:.4f}")
        
    elif broadening_type == 'lorentzian' and gamma is not None:
        # Apply Lorentzian broadening: multiply by exp(-γ|t|)
        broadening_factor = np.exp(-gamma * np.abs(t_values))
        broadened_data = data * broadening_factor
        print(f"Applied Lorentzian broadening in time domain with γ = {gamma:.4f}")
        
    else:
        broadened_data = data
        print("No time domain broadening applied")
    
    return broadened_data, broadening_factor if 'broadening_factor' in locals() else np.ones_like(data)

def extract_beta(filename):
    """Extract beta value from filename."""
    match = re.search(r'beta=(\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def parse_filename(filename):
    """Extracts species and beta from filename."""
    basename = os.path.basename(filename)
    # Regex to capture the species from time_corr_rand0_(species)_beta=(value).dat
    match = re.search(r'time_corr_rand0_(.*?)_beta=([\d\.]+?)\.dat', basename)
    if match:
        species = match.group(1)
        beta = float(match.group(2))
        return species, beta
    return None, None

def parse_QFI_data(data_dir):
    # List all relevant files matching the pattern for any species
    files = glob.glob(os.path.join(data_dir, 'time_corr_rand0_*.dat'))
    print(f"Found {len(files)} files matching the pattern.")
    # Group files by species, then by beta
    species_data = defaultdict(lambda: defaultdict(list))
    species_names = set()
    for file_path in files:
        species, _ = parse_filename(file_path)
        if species:
            species_names.add(species)

    print("Species found:")
    for name in sorted(list(species_names)):
        print(name)
    for file_path in files:
        species, beta = parse_filename(file_path)
        if species is not None and beta is not None:
            species_data[species][beta].append(file_path)

    # Prepare for QFI vs beta plots for each species
    all_species_qfi_data = defaultdict(list)

    # Process each species
    for species, beta_groups in species_data.items():
        print(f"Processing species: {species}")

        # Process each beta group for the current species
        for beta, file_list in beta_groups.items():
            print(f"  Processing beta={beta} with {len(file_list)} files")

            all_data = []
            for file_path in file_list:
                try:
                    # Load the data (time, real, imag)
                    data = np.loadtxt(file_path, comments='#')
                    time = data[:, 0]
                    real_part = data[:, 1]
                    imag_part = data[:, 2]
                    val = real_part + 1j * imag_part
                    all_data.append((time, val))
                except Exception as e:
                    print(f"    Error reading {file_path}: {e}")
                    
            if not all_data:
                print(f"    No valid data for beta={beta}")
                continue

            reference_time = all_data[0][0]

            # Calculate mean correlation function
            all_complex_values = np.vstack([data[1] for data in all_data])
            mean_correlation = np.mean(all_complex_values, axis=0)

            # Determine time step
            dt = reference_time[1] - reference_time[0] if len(reference_time) > 1 else 1.0
            N = len(mean_correlation)

            # Check if data is already ordered from negative to positive time
            if reference_time[0] < 0 and reference_time[-1] > 0:
                # Data is already time-ordered from negative to positive
                t_full = reference_time
                C_full = mean_correlation
                print(f"    Using pre-ordered time data from {t_full.min():.2f} to {t_full.max():.2f}")
                
                # For FFT, we need to reorder from time-ordered to frequency-ordered
                C_fft_input = np.fft.ifftshift(C_full)
                
            else:
                # Assume data is for t >= 0, construct negative times using C(-t) = C(t)*
                t_pos = reference_time
                C_pos = mean_correlation
                
                # Construct data for t < 0 using C(-t) = C(t)*
                t_neg = -t_pos[1:][::-1]  # Exclude t=0 to avoid duplication
                C_neg = np.conj(C_pos[1:][::-1])
                
                # Combine to create full time evolution
                t_full = np.concatenate((t_neg, t_pos))
                C_full = np.concatenate((C_neg, C_pos))
                
                print(f"    Constructed negative time evolution, range: {t_full.min():.2f} to {t_full.max():.2f}")
                
                # Reorder for FFT
                C_fft_input = np.fft.ifftshift(C_full)
            
            # Apply time domain broadening (Lorentzian)
            gamma = 0.1  # Broadening parameter
            t_fft_ordered = np.fft.ifftshift(t_full)
            C_fft_input_broadened, time_broadening_factor = apply_time_broadening(
                t_fft_ordered, C_fft_input, 'lorentzian', gamma=gamma)
            
            # Take complex conjugate for FFT convention (matching the reference code)
            C_fft_input_broadened = np.conj(C_fft_input_broadened)

            # Compute FFT to get spectral function
            C_w = np.fft.fft(C_fft_input_broadened)
            S_w = dt * np.fft.fftshift(C_w) / (2 * np.pi)  / 16
            
            # Frequency axis
            omega = np.fft.fftshift(np.fft.fftfreq(len(C_fft_input_broadened), d=dt)) * 2 * np.pi

            # Take real part of spectral function
            S_omega_real = S_w.real

            # Calculate integral of S(ω) before truncation
            integral_before = np.trapz(S_omega_real, omega)
            
            # Extract positive frequencies only
            positive_freq_mask = omega > 0
            omega_pos = omega[positive_freq_mask]
            s_omega_pos = S_omega_real[positive_freq_mask]
            
            # Calculate integral of S(ω) after truncation (positive frequencies only)
            integral_after = np.trapz(s_omega_pos, omega_pos)
            
            # Calculate compensation factor
            compensation_factor = integral_before / integral_after if integral_after != 0 else 1.0
            
            print("Processing for species:", species)
            # Apply compensation to the truncated spectral function
            s_omega_pos_compensated = s_omega_pos * 1
            
            print(f"    Beta={beta}: Integral before truncation: {integral_before:.6f}")
            print(f"    Beta={beta}: Integral after truncation: {integral_after:.6f}")
            print(f"    Beta={beta}: Compensation factor: {compensation_factor:.6f}")

            # Calculate QFI using compensated positive frequencies
            integrand = s_omega_pos_compensated * np.tanh(beta * omega_pos / 2.0) * (1 - np.exp(-beta * omega_pos))
            qfi = 4*np.trapz(integrand, omega_pos)

            # Plot the spectral function
            plt.figure(figsize=(10, 6))
            plt.scatter(omega_pos, s_omega_pos, label=f'Beta={beta} QFI={qfi:.4f}')
            plt.xlabel('Frequency (rad/s)')
            plt.ylabel('Spectral Function S(ω)')
            plt.xlim(0, 3)
            plt.title(f'Spectral Function for {species} at Beta={beta}')
            plt.grid(True)
            plt.legend()
            outdir = os.path.join(data_dir, 'processed_data', species)
            os.makedirs(outdir, exist_ok=True)
            plot_filename = os.path.join(outdir, f'spectral_function_{species}_beta_{beta}.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()

            all_species_qfi_data[species].append((beta, qfi))

            # Save processed spectral data for each beta
            data_out = np.column_stack((omega, S_omega_real))
            data_filename = os.path.join(outdir, f'spectral_beta_{beta}.dat')
            np.savetxt(data_filename, data_out, header='freq spectral_function')

    # Plot QFI vs Beta for each species
    plot_outdir = os.path.join(data_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)

    for species, qfi_data in all_species_qfi_data.items():
        if not qfi_data:
            continue

        qfi_data.sort()
        qfi_beta_array = np.array(qfi_data)

        # Save QFI data
        qfi_data_filename = os.path.join(plot_outdir, f'qfi_vs_beta_{species}.dat')
        np.savetxt(qfi_data_filename, qfi_beta_array, header='beta qfi')

        # Plot QFI vs beta
        plt.figure(figsize=(10, 6))
        plt.plot(qfi_beta_array[:, 0], qfi_beta_array[:, 1], 'o-')
        plt.xlabel('Beta (β)')
        plt.ylabel('QFI')
        plt.title(f'QFI vs. Beta for {species}')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()

        qfi_plot_filename = os.path.join(plot_outdir, f'qfi_vs_beta_{species}.png')
        plt.savefig(qfi_plot_filename, dpi=300)
        plt.close()

        # Calculate and plot the derivative of QFI with respect to beta
        if len(qfi_beta_array) > 1:
            betas = qfi_beta_array[:, 0]
            qfis = qfi_beta_array[:, 1]
            
            # Use central differences for the derivative
            mid_betas = (betas[:-1] + betas[1:]) / 2
            delta_beta = np.diff(betas)
            delta_qfi = np.diff(qfis)
            qfi_derivative = delta_qfi / delta_beta

            # Plot the derivative
            plt.figure(figsize=(10, 6))
            plt.plot(mid_betas, qfi_derivative, 'o-')
            plt.xlabel('Beta (β)')
            plt.ylabel('dQFI/dβ')
            plt.title(f'Derivative of QFI vs. Beta for {species}')
            plt.xscale('log')
            plt.grid(True)
            plt.legend()

            derivative_plot_filename = os.path.join(plot_outdir, f'qfi_derivative_vs_beta_{species}.png')
            plt.savefig(derivative_plot_filename, dpi=300)
            plt.close()

            # Save derivative data
            derivative_data = np.column_stack((mid_betas, qfi_derivative))
            derivative_data_filename = os.path.join(plot_outdir, f'qfi_derivative_vs_beta_{species}.dat')
            np.savetxt(derivative_data_filename, derivative_data, header='beta dQFI/dbeta')

    print("Processing complete!")
    return all_species_qfi_data


def parse_QFI_across_Jpm(data_dir):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Find all subdirectories matching the pattern Jpm=*
    subdirs = []
    if rank == 0:
        subdirs = glob.glob(os.path.join(data_dir, 'Jpm=*'))
        subdirs.sort()  # Ensure consistent ordering
    
    # Broadcast subdirs to all processes
    subdirs = comm.bcast(subdirs, root=0)
    
    # Distribute subdirectories among processes
    local_subdirs = []
    for i, subdir in enumerate(subdirs):
        if i % size == rank:
            local_subdirs.append(subdir)
    
    # Each process handles its assigned subdirectories
    local_jpm_qfi_data = {}
    
    for subdir in local_subdirs:
        # Extract Jpm value from the directory name
        match = re.search(r'Jpm=([-]?[\d\.]+)', os.path.basename(subdir))
        if not match:
            continue
        jpm_value = float(match.group(1))
        
        print(f"[Rank {rank}] Processing directory: {subdir} for Jpm={jpm_value}")
        # Run the QFI analysis for the current Jpm value
        species_qfi_data = parse_QFI_data(os.path.join(subdir, 'output'))
        local_jpm_qfi_data[jpm_value] = species_qfi_data
    
    # Gather all results at rank 0
    all_jpm_qfi_data = comm.gather(local_jpm_qfi_data, root=0)
    
    if rank == 0:
        # Merge all results
        jpm_qfi_data = {}
        for process_data in all_jpm_qfi_data:
            jpm_qfi_data.update(process_data)
        
        # Reorganize data by species for heatmap plotting
        all_qfi_data = defaultdict(list)
        all_derivative_data = defaultdict(list)
        
        for jpm, all_species_data in jpm_qfi_data.items():
            for species, qfi_beta_list in all_species_data.items():
                for beta, qfi in qfi_beta_list:
                    all_qfi_data[species].append((jpm, beta, qfi))
                
                # Calculate derivatives for this species and jpm
                if len(qfi_beta_list) > 1:
                    qfi_beta_list.sort()
                    qfi_beta_array = np.array(qfi_beta_list)
                    betas = qfi_beta_array[:, 0]
                    qfis = qfi_beta_array[:, 1]
                    
                    # Use central differences for the derivative
                    mid_betas = (betas[:-1] + betas[1:]) / 2
                    delta_beta = np.diff(betas)
                    delta_qfi = np.diff(qfis)
                    qfi_derivative = delta_qfi / delta_beta
                    
                    for mid_beta, derivative in zip(mid_betas, qfi_derivative):
                        all_derivative_data[species].append((jpm, mid_beta, derivative))

        # Create output directory
        plot_outdir = os.path.join(data_dir, 'plots')
        os.makedirs(plot_outdir, exist_ok=True)
        
        # Plot QFI heatmaps
        for species, data_points in all_qfi_data.items():
            if not data_points:
                continue
                
            # Extract data
            jpm_vals = np.array([point[0] for point in data_points])
            beta_vals = np.array([point[1] for point in data_points])
            qfi_vals = np.array([point[2] for point in data_points])
            
            # Create interpolation grid
            jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
            beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
            jpm_grid = np.linspace(jpm_min, jpm_max, 100)
            beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)
            JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
            
            # Interpolate QFI values
            QFI = griddata((jpm_vals, beta_vals), qfi_vals, (JPM, BETA), method='cubic')
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JPM, BETA, QFI, shading='auto', cmap='viridis')
            plt.colorbar(label='QFI')
            
            # Overlay original data points
            plt.scatter(jpm_vals, beta_vals, c='red', s=20, edgecolors='black')
            
            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.yscale('log')
            plt.title(f'QFI Heatmap for {species}')
            
            heatmap_filename = os.path.join(plot_outdir, f'qfi_heatmap_{species}.png')
            plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved QFI heatmap for {species} to {heatmap_filename}")
        
        # Plot derivative heatmaps
        for species, data_points in all_derivative_data.items():
            if not data_points:
                continue
                
            # Extract data
            jpm_vals = np.array([point[0] for point in data_points])
            beta_vals = np.array([point[1] for point in data_points])
            deriv_vals = np.array([point[2] for point in data_points])
            
            # Create interpolation grid
            jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
            beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
            jpm_grid = np.linspace(jpm_min, jpm_max, 100)
            beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)
            JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
            
            # Interpolate derivative values
            DERIV = griddata((jpm_vals, beta_vals), deriv_vals, (JPM, BETA), method='cubic')
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JPM, BETA, DERIV, shading='auto', cmap='viridis')
            plt.colorbar(label='dQFI/dβ')

            # Overlay original data points
            plt.scatter(jpm_vals, beta_vals, c='red', s=20, edgecolors='black')

            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.yscale('log')
            plt.title(f'dQFI/dβ Heatmap for {species}')
            
            heatmap_filename = os.path.join(plot_outdir, f'qfi_derivative_heatmap_{species}.png')
            plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved derivative heatmap for {species} to {heatmap_filename}")
        
        print("Heatmap generation complete!")
        return jpm_qfi_data
    else:
        return None


def plot_heatmaps_from_processed_data(data_dir):
    """Plot heatmaps by reading processed QFI data from subdirectories."""
    
    # Find all subdirectories matching the pattern Jpm=*
    subdirs = glob.glob(os.path.join(data_dir, 'Jpm=*'))
    
    # Data storage for heatmaps
    all_qfi_data = defaultdict(list)
    all_derivative_data = defaultdict(list)
    
    for subdir in subdirs:
        # Extract Jpm value from the directory name
        match = re.search(r'Jpm=([-]?[\d\.]+)', os.path.basename(subdir))
        if not match:
            continue
        jpm_value = float(match.group(1))
        
        plots_dir = os.path.join(subdir, 'output', 'plots')
        if not os.path.exists(plots_dir):
            continue
            
        print(f"Reading processed data from: {plots_dir} for Jpm={jpm_value}")
        
        # Find QFI files
        qfi_files = glob.glob(os.path.join(plots_dir, 'qfi_vs_beta_*.dat'))
        for qfi_file in qfi_files:
            # Extract species name from filename
            species_match = re.search(r'qfi_vs_beta_(.+?)\.dat', os.path.basename(qfi_file))
            if not species_match:
                continue
            species = species_match.group(1)
            
            try:
                # Read beta and QFI data
                data = np.loadtxt(qfi_file)
                if data.size == 0:
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                for row in data:
                    beta, qfi = row[0], row[1]
                    all_qfi_data[species].append((jpm_value, beta, qfi))
            except Exception as e:
                print(f"Error reading {qfi_file}: {e}")
        
        # Find derivative files
        derivative_files = glob.glob(os.path.join(plots_dir, 'qfi_derivative_vs_beta_*.dat'))
        for deriv_file in derivative_files:
            # Extract species name from filename
            species_match = re.search(r'qfi_derivative_vs_beta_(.+?)\.dat', os.path.basename(deriv_file))
            if not species_match:
                continue
            species = species_match.group(1)
            
            try:
                # Read beta and derivative data
                data = np.loadtxt(deriv_file)
                if data.size == 0:
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                for row in data:
                    beta, derivative = row[0], row[1]
                    all_derivative_data[species].append((jpm_value, beta, derivative))
            except Exception as e:
                print(f"Error reading {deriv_file}: {e}")
    
    # Create output directory
    plot_outdir = os.path.join(data_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)
    
    # Plot QFI heatmaps and scatter plots
    for species, data_points in all_qfi_data.items():
        if not data_points:
            continue
            
        # Extract data and split into positive and negative Jpm
        data_array = np.array(data_points)
        jpm_vals = data_array[:, 0]
        beta_vals = data_array[:, 1]
        qfi_vals = data_array[:, 2]
        
        # Split data
        pos_mask = jpm_vals > 0
        neg_mask = jpm_vals < 0
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Scatter plot for Jpm > 0
        ax1 = plt.subplot(3, 2, 1)
        if np.any(pos_mask):
            scatter1 = ax1.scatter(jpm_vals[pos_mask], beta_vals[pos_mask], 
                                 c=qfi_vals[pos_mask], cmap='viridis', s=50, edgecolors='black')
            plt.colorbar(scatter1, ax=ax1, label='QFI')
        ax1.set_xlabel('Jpm')
        ax1.set_ylabel('Beta (β)')
        ax1.set_yscale('log')
        ax1.set_title(f'QFI Scatter (Jpm > 0) - {species}')
        
        # Scatter plot for Jpm < 0
        ax2 = plt.subplot(3, 2, 2)
        if np.any(neg_mask):
            scatter2 = ax2.scatter(jpm_vals[neg_mask], beta_vals[neg_mask], 
                                 c=qfi_vals[neg_mask], cmap='viridis', s=50, edgecolors='black')
            plt.colorbar(scatter2, ax=ax2, label='QFI')
        ax2.set_xlabel('Jpm')
        ax2.set_ylabel('Beta (β)')
        ax2.set_yscale('log')
        ax2.set_title(f'QFI Scatter (Jpm < 0) - {species}')
        
        # Interpolated heatmap for Jpm > 0
        ax3 = plt.subplot(3, 2, 3)
        if np.any(pos_mask):
            jpm_pos = jpm_vals[pos_mask]
            beta_pos = beta_vals[pos_mask]
            qfi_pos = qfi_vals[pos_mask]
            
            # Create grid
            jpm_grid_pos = np.linspace(jpm_pos.min(), jpm_pos.max(), 200)
            beta_grid_pos = np.logspace(np.log10(beta_pos.min()), np.log10(beta_pos.max()), 200)
            JPM_pos, BETA_pos = np.meshgrid(jpm_grid_pos, beta_grid_pos)
            
            # Interpolate with high smoothness
            QFI_pos = griddata((jpm_pos, beta_pos), qfi_pos, (JPM_pos, BETA_pos), method='cubic')
            
            im1 = ax3.imshow(QFI_pos, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_pos.min(), jpm_pos.max(), 
                                 np.log10(beta_pos.min()), np.log10(beta_pos.max())],
                           interpolation='bicubic')
            plt.colorbar(im1, ax=ax3, label='QFI')
        ax3.set_xlabel('Jpm')
        ax3.set_ylabel('log10(Beta)')
        ax3.set_title(f'QFI Heatmap (Jpm > 0) - {species}')
        
        # Interpolated heatmap for Jpm < 0
        ax4 = plt.subplot(3, 2, 4)
        if np.any(neg_mask):
            jpm_neg = jpm_vals[neg_mask]
            beta_neg = beta_vals[neg_mask]
            qfi_neg = qfi_vals[neg_mask]
            
            # Create grid
            jpm_grid_neg = np.linspace(jpm_neg.min(), jpm_neg.max(), 200)
            beta_grid_neg = np.logspace(np.log10(beta_neg.min()), np.log10(beta_neg.max()), 200)
            JPM_neg, BETA_neg = np.meshgrid(jpm_grid_neg, beta_grid_neg)
            
            # Interpolate with high smoothness
            QFI_neg = griddata((jpm_neg, beta_neg), qfi_neg, (JPM_neg, BETA_neg), method='cubic')
            
            im2 = ax4.imshow(QFI_neg, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_neg.min(), jpm_neg.max(), 
                                 np.log10(beta_neg.min()), np.log10(beta_neg.max())],
                           interpolation='bicubic')
            plt.colorbar(im2, ax=ax4, label='QFI')
        ax4.set_xlabel('Jpm')
        ax4.set_ylabel('log10(Beta)')
        ax4.set_title(f'QFI Heatmap (Jpm < 0) - {species}')
        
        # Combined heatmap
        ax5 = plt.subplot(3, 1, 3)
        if len(data_points) > 0:
            # Create combined grid
            jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
            beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
            jpm_grid_combined = np.linspace(jpm_min, jpm_max, 400)
            beta_grid_combined = np.logspace(np.log10(beta_min), np.log10(beta_max), 400)
            JPM_combined, BETA_combined = np.meshgrid(jpm_grid_combined, beta_grid_combined)
            
            # Interpolate combined data
            QFI_combined = griddata((jpm_vals, beta_vals), qfi_vals, 
                                  (JPM_combined, BETA_combined), method='cubic')
            
            im3 = ax5.imshow(QFI_combined, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_min, jpm_max, np.log10(beta_min), np.log10(beta_max)],
                           interpolation='bicubic')
            plt.colorbar(im3, ax=ax5, label='QFI')
            
            # Add vertical line at Jpm = 0
            ax5.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.7)
            
        ax5.set_xlabel('Jpm')
        ax5.set_ylabel('log10(Beta)')
        ax5.set_title(f'QFI Combined Heatmap - {species}')
        
        plt.tight_layout()
        combined_filename = os.path.join(plot_outdir, f'qfi_analysis_{species}.png')
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved QFI analysis for {species} to {combined_filename}")
    
    # Plot derivative heatmaps and scatter plots
    for species, data_points in all_derivative_data.items():
        if not data_points:
            continue
            
        # Extract data and split into positive and negative Jpm
        data_array = np.array(data_points)
        jpm_vals = data_array[:, 0]
        beta_vals = data_array[:, 1]
        deriv_vals = data_array[:, 2]
        
        # Split data
        pos_mask = jpm_vals > 0
        neg_mask = jpm_vals < 0
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Scatter plot for Jpm > 0
        ax1 = plt.subplot(3, 2, 1)
        if np.any(pos_mask):
            scatter1 = ax1.scatter(jpm_vals[pos_mask], beta_vals[pos_mask], 
                                 c=deriv_vals[pos_mask], cmap='viridis', s=50, edgecolors='black')
            plt.colorbar(scatter1, ax=ax1, label='dQFI/dβ')
        ax1.set_xlabel('Jpm')
        ax1.set_ylabel('Beta (β)')
        ax1.set_yscale('log')
        ax1.set_title(f'dQFI/dβ Scatter (Jpm > 0) - {species}')
        
        # Scatter plot for Jpm < 0
        ax2 = plt.subplot(3, 2, 2)
        if np.any(neg_mask):
            scatter2 = ax2.scatter(jpm_vals[neg_mask], beta_vals[neg_mask], 
                                 c=deriv_vals[neg_mask], cmap='viridis', s=50, edgecolors='black')
            plt.colorbar(scatter2, ax=ax2, label='dQFI/dβ')
        ax2.set_xlabel('Jpm')
        ax2.set_ylabel('Beta (β)')
        ax2.set_yscale('log')
        ax2.set_title(f'dQFI/dβ Scatter (Jpm < 0) - {species}')
        
        # Interpolated heatmap for Jpm > 0
        ax3 = plt.subplot(3, 2, 3)
        if np.any(pos_mask):
            jpm_pos = jpm_vals[pos_mask]
            beta_pos = beta_vals[pos_mask]
            deriv_pos = deriv_vals[pos_mask]
            
            # Create grid
            jpm_grid_pos = np.linspace(jpm_pos.min(), jpm_pos.max(), 200)
            beta_grid_pos = np.logspace(np.log10(beta_pos.min()), np.log10(beta_pos.max()), 200)
            JPM_pos, BETA_pos = np.meshgrid(jpm_grid_pos, beta_grid_pos)
            
            # Interpolate with high smoothness
            DERIV_pos = griddata((jpm_pos, beta_pos), deriv_pos, (JPM_pos, BETA_pos), method='cubic')
            
            im1 = ax3.imshow(DERIV_pos, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_pos.min(), jpm_pos.max(), 
                                 np.log10(beta_pos.min()), np.log10(beta_pos.max())],
                           interpolation='bicubic')
            plt.colorbar(im1, ax=ax3, label='dQFI/dβ')
        ax3.set_xlabel('Jpm')
        ax3.set_ylabel('log10(Beta)')
        ax3.set_title(f'dQFI/dβ Heatmap (Jpm > 0) - {species}')
        
        # Interpolated heatmap for Jpm < 0
        ax4 = plt.subplot(3, 2, 4)
        if np.any(neg_mask):
            jpm_neg = jpm_vals[neg_mask]
            beta_neg = beta_vals[neg_mask]
            deriv_neg = deriv_vals[neg_mask]
            
            # Create grid
            jpm_grid_neg = np.linspace(jpm_neg.min(), jpm_neg.max(), 200)
            beta_grid_neg = np.logspace(np.log10(beta_neg.min()), np.log10(beta_neg.max()), 200)
            JPM_neg, BETA_neg = np.meshgrid(jpm_grid_neg, beta_grid_neg)
            
            # Interpolate with high smoothness
            DERIV_neg = griddata((jpm_neg, beta_neg), deriv_neg, (JPM_neg, BETA_neg), method='cubic')
            
            im2 = ax4.imshow(DERIV_neg, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_neg.min(), jpm_neg.max(), 
                                 np.log10(beta_neg.min()), np.log10(beta_neg.max())],
                           interpolation='bicubic')
            plt.colorbar(im2, ax=ax4, label='dQFI/dβ')
        ax4.set_xlabel('Jpm')
        ax4.set_ylabel('log10(Beta)')
        ax4.set_title(f'dQFI/dβ Heatmap (Jpm < 0) - {species}')
        
        # Combined heatmap
        ax5 = plt.subplot(3, 1, 3)
        if len(data_points) > 0:
            # Create combined grid
            jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
            beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
            jpm_grid_combined = np.linspace(jpm_min, jpm_max, 400)
            beta_grid_combined = np.logspace(np.log10(beta_min), np.log10(beta_max), 400)
            JPM_combined, BETA_combined = np.meshgrid(jpm_grid_combined, beta_grid_combined)
            
            # Interpolate combined data
            DERIV_combined = griddata((jpm_vals, beta_vals), deriv_vals, 
                                    (JPM_combined, BETA_combined), method='cubic')
            
            im3 = ax5.imshow(DERIV_combined, aspect='auto', origin='lower', cmap='viridis',
                           extent=[jpm_min, jpm_max, np.log10(beta_min), np.log10(beta_max)],
                           interpolation='bicubic')
            plt.colorbar(im3, ax=ax5, label='dQFI/dβ')
            
            # Add vertical line at Jpm = 0
            ax5.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.7)
            
        ax5.set_xlabel('Jpm')
        ax5.set_ylabel('log10(Beta)')
        ax5.set_title(f'dQFI/dβ Combined Heatmap - {species}')
        
        plt.tight_layout()
        combined_filename = os.path.join(plot_outdir, f'qfi_derivative_analysis_{species}.png')
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved derivative analysis for {species} to {combined_filename}")
    
    print("Heatmap generation from processed data complete!")


if __name__ == "__main__":
    # Path to the directory containing the data files
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    across_QFI = sys.argv[2] if len(sys.argv) > 2 else 'False'
    across_QFI = across_QFI.lower() == 'true'
    if across_QFI:
        parse_QFI_across_Jpm(data_dir)
        plot_heatmaps_from_processed_data(data_dir)
    else:
        parse_QFI_data(data_dir)
    print("All processing complete.")
