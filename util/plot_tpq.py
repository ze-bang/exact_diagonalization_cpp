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
                    val = data[:, 1] + 1j * data[:, 2]
                    all_data.append((time, val))
                except Exception as e:
                    print(f"    Error reading {file_path}: {e}")

            if not all_data:
                print(f"    No valid data for beta={beta}")
                continue

            # Averaging and FFT
            reference_time = all_data[0][0]
            all_values = np.array([d[1] for d in all_data])
            mean_val = np.mean(all_values, axis=0)

            # Apply Lorentzian filter
            gamma = 0.1
            lorentzian_filter = np.exp(-gamma * np.abs(reference_time))
            filtered_mean_val = mean_val * lorentzian_filter

            # FFT
            dt = reference_time[1] - reference_time[0]
            freq = np.fft.fftfreq(len(reference_time), d=dt) * (2 * np.pi)
            spectral_func = np.abs(np.fft.fft(filtered_mean_val)) * dt / (2 * np.pi)

            # Sort by frequency
            sort_indices = np.argsort(freq)
            freq = freq[sort_indices]
            spectral_func = spectral_func[sort_indices]



            # Calculate QFI
            positive_freq_mask = freq < 0
            omega = np.flip(-freq[positive_freq_mask])
            s_omega = np.flip(spectral_func[positive_freq_mask])


            integrand = s_omega * np.tanh(beta * omega / 2.0)
            d_omega = omega[1] - omega[0]
            qfi = 4 * np.sum(integrand) * d_omega

            # Plot the spectral function
            plt.figure(figsize=(10, 6))
            plt.plot(omega, s_omega, label=f'Beta={beta} QFI={qfi:.4f}')
            plt.xlabel('Frequency (rad/s)')
            plt.ylabel('Spectral Function')
            plt.xlim(0, np.max(-freq))
            plt.title(f'Spectral Function for {species} at Beta={beta}')
            plt.grid(True)
            plt.legend()
            outdir = os.path.join(data_dir, 'processed_data', species)
            os.makedirs(outdir, exist_ok=True)
            plot_filename = os.path.join(outdir, f'spectral_function_{species}_beta_{beta}.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            

            all_species_qfi_data[species].append((beta, qfi))

            # Optional: Save processed spectral data for each beta
            data_out = np.column_stack((freq, spectral_func))
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

        qfi_plot_filename = os.path.join(plot_outdir, f'qfi_vs_beta_{species}.png')
        plt.savefig(qfi_plot_filename, dpi=300)
        plt.close()

        # Calculate and plot the derivative of QFI with respect to beta
        if len(qfi_beta_array) > 1:
            betas = qfi_beta_array[:, 0]
            qfis = qfi_beta_array[:, 1]
            
            # Use central differences for the derivative
            # We'll have one less point for the derivative
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
    # Find all subdirectories matching the pattern Jpm=*
    subdirs = glob.glob(os.path.join(data_dir, 'Jpm=*'))
    jpm_qfi_data = {}

    for subdir in subdirs:
        # Extract Jpm value from the directory name
        match = re.search(r'Jpm=([-]?[\d\.]+)', os.path.basename(subdir))
        if not match:
            continue
        jpm_value = float(match.group(1))
        
        print(f"\nProcessing directory: {subdir} for Jpm={jpm_value}")
        # Run the QFI analysis for the current Jpm value
        species_qfi_data = parse_QFI_data(os.path.join(subdir, 'output'))
        jpm_qfi_data[jpm_value] = species_qfi_data

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
            
        # Sort by beta
        data_points.sort(key=lambda x: x[1])
        
        # Extract data
        jpm_vals = [point[0] for point in data_points]
        beta_vals = [point[1] for point in data_points]
        qfi_vals = [point[2] for point in data_points]
        
        # Plot as scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(jpm_vals, beta_vals, c=qfi_vals, cmap='viridis', s=50)
        plt.colorbar(scatter, label='QFI')
        plt.xlabel('Jpm')
        plt.ylabel('Beta (β)')
        plt.yscale('log')
        plt.title(f'QFI for {species}')
        
        heatmap_filename = os.path.join(plot_outdir, f'qfi_scatter_{species}.png')
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved QFI scatter plot for {species} to {heatmap_filename}")
    
    # Plot derivative heatmaps
    for species, data_points in all_derivative_data.items():
        if not data_points:
            continue
            
        # Sort by beta
        data_points.sort(key=lambda x: x[1])
        
        # Extract data
        jpm_vals = [point[0] for point in data_points]
        beta_vals = [point[1] for point in data_points]
        deriv_vals = [point[2] for point in data_points]
        
        # Plot as scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(jpm_vals, beta_vals, c=deriv_vals, cmap='viridis', s=50)
        plt.colorbar(scatter, label='dQFI/dβ')
        plt.xlabel('Jpm')
        plt.ylabel('Beta (β)')
        plt.yscale('log')
        plt.title(f'dQFI/dβ for {species}')
        
        heatmap_filename = os.path.join(plot_outdir, f'qfi_derivative_scatter_{species}.png')
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved derivative scatter plot for {species} to {heatmap_filename}")
    
    print("Scatter plot generation complete!")
    return jpm_qfi_data


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
        
        # Create grid for interpolation (linear spacing for Jpm, log spacing for beta)
        jpm_grid = np.linspace(jpm_min, jpm_max, 50)
        beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 50)
        JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
        
        # Interpolate QFI values
        QFI = griddata((jpm_vals, beta_vals), qfi_vals, (JPM, BETA), method='cubic', fill_value=np.nan)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(QFI, extent=[jmp_min, jpm_max, beta_min, beta_max], 
                       aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, label='QFI')
        plt.xlabel('Jpm')
        plt.ylabel('Beta (β)')
        plt.yscale('log')
        plt.title(f'QFI Heatmap for {species}')
        
        # Overlay original data points
        plt.scatter(jpm_vals, beta_vals, c='white', s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
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
        
        # Create grid for interpolation
        jpm_grid = np.linspace(jpm_min, jpm_max, 50)
        beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 50)
        JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
        
        # Interpolate derivative values
        DERIV = griddata((jpm_vals, beta_vals), deriv_vals, (JPM, BETA), method='cubic', fill_value=np.nan)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(DERIV, extent=[jpm_min, jpm_max, beta_min, beta_max], 
                       aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, label='dQFI/dβ')
        plt.xlabel('Jpm')
        plt.ylabel('Beta (β)')
        plt.yscale('log')
        plt.title(f'dQFI/dβ Heatmap for {species}')
        
        # Overlay original data points
        plt.scatter(jpm_vals, beta_vals, c='white', s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        heatmap_filename = os.path.join(plot_outdir, f'qfi_derivative_heatmap_{species}.png')
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved derivative heatmap for {species} to {heatmap_filename}")
    
    print("Heatmap generation from processed data complete!")

if __name__ == "__main__":
    # Path to the directory containing the data files
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    across_QFI = sys.argv[2] if len(sys.argv) > 2 else 'False'
    if across_QFI:
        parse_QFI_across_Jpm(data_dir)
        # plot_heatmaps_from_processed_data(data_dir)
    else:
        parse_QFI_data(data_dir)
    print("All processing complete.")
