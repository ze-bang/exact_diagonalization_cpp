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
from collections import defaultdict
from scipy.interpolate import interp1d

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

def parse_filename_new(filename):
    """Extract species (including momentum), beta, and sample index from filenames like:
       time_corr_rand0_SmSp_q0_op0_beta=1.436681.dat or time_corr_sample3_...

    Returns: (species_with_momentum: str, beta: float, sample_idx: int)
    If parsing fails, returns (None, None, None).
    """
    basename = os.path.basename(filename)
    # Support both randN and sampleN prefixes
    m = re.match(r'^time_corr_(?:rand|sample)(\d+)_(.+?)_beta=([0-9.+-eE]+)\.dat$', basename)
    if m:
        sample_idx = int(m.group(1))
        species_with_momentum = m.group(2)
        beta = float(m.group(3))
        return species_with_momentum, beta, sample_idx
    return None, None, None

def parse_QFI_data_new(structure_factor_dir, beta_tol=1e-2):
    """Parse QFI data from the new directory structure."""
    # Find all beta subdirectories
    beta_dirs = glob.glob(os.path.join(structure_factor_dir, 'beta_*'))
    print(f"Found {len(beta_dirs)} beta directories.")
    
    # Group files by species (including momentum), then by beta bins (tolerance-based)
    # species_data[species][beta_bin_index] -> list of file paths for that species and beta bin
    species_data = defaultdict(lambda: defaultdict(list))
    species_names = set()
    # Keep track of beta bin centers and values assigned (for labeling)
    beta_bins = []               # list of representative beta centers
    beta_bin_values = defaultdict(list)  # beta_bin_values[bin_idx] -> list of raw betas contributing

    def assign_beta_bin(beta_val, bins, tol):
        # Absolute tolerance grouping
        for i, c in enumerate(bins):
            if abs(beta_val - c) <= tol:
                return i
        bins.append(beta_val)
        return len(bins) - 1
    
    for beta_dir in beta_dirs:
        # Extract beta value from directory name
        beta_match = re.search(r'beta_([\d\.]+)', os.path.basename(beta_dir))
        if not beta_match:
            continue
        beta_value = float(beta_match.group(1))
        # Map this directory beta to a tolerance-based bin index
        bin_idx = assign_beta_bin(beta_value, beta_bins, beta_tol)
        beta_bin_values[bin_idx].append(beta_value)

        # Find all time correlation files in this beta directory, across all samples
        taylor_dir = os.path.join(beta_dir, 'taylor')
        files = []
        files.extend(glob.glob(os.path.join(taylor_dir, 'time_corr_rand*.dat')))

        for file_path in files:
            species_with_momentum, file_beta, sample_idx = parse_filename_new(file_path)
            if species_with_momentum:
                species_names.add(species_with_momentum)
                # Group by beta bin (use directory beta, tolerate small per-file beta deviations)
                species_data[species_with_momentum][bin_idx].append(file_path)
    
    print("Species (with momentum) found:")
    for name in sorted(list(species_names)):
        print(name)
    
    # Prepare for QFI vs beta plots for each species
    all_species_qfi_data = defaultdict(list)
    
    # Process each species
    for species, beta_groups in species_data.items():
        print(f"Processing species: {species}")
        
        # Process each beta group for the current species
        for beta_bin_idx, file_list in beta_groups.items():
            # Determine representative beta for this bin (mean of contributing values)
            beta_vals = beta_bin_values.get(beta_bin_idx, [])
            beta = float(np.mean(beta_vals)) if len(beta_vals) > 0 else np.nan
            print(f"  Processing beta≈{beta} (bin {beta_bin_idx}) with {len(file_list)} files")
            
            all_data = []
            for file_path in file_list:
                try:
                    # Load the data (time, real, imag)
                    data = np.loadtxt(file_path, comments='#')
                    time = data[:, 0]
                    real_part = data[:, 1]
                    imag_part = data[:, 2] if data.shape[1] > 2 else np.zeros_like(real_part)
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
            gamma = 0.3  # Broadening parameter
            t_fft_ordered = np.fft.ifftshift(t_full)
            C_fft_input_broadened, time_broadening_factor = apply_time_broadening(
                t_fft_ordered, C_fft_input, 'lorentzian', gamma=gamma)
            
            # Take complex conjugate for FFT convention (matching the reference code)
            C_fft_input_broadened = np.conj(C_fft_input_broadened)
            
            # Compute FFT to get spectral function
            C_w = np.fft.fft(C_fft_input_broadened)
            S_w = dt * np.fft.fftshift(C_w) / (2 * np.pi) 
            
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
            s_omega_pos_compensated = s_omega_pos * compensation_factor
            
            print(f"    Beta≈{beta:.6g}: Integral before truncation: {integral_before:.6f}")
            print(f"    Beta≈{beta:.6g}: Integral after truncation: {integral_after:.6f}")
            print(f"    Beta≈{beta:.6g}: Compensation factor: {compensation_factor:.6f}")
            
            # Calculate QFI using compensated positive frequencies
            integrand = s_omega_pos_compensated * np.tanh(beta * omega_pos / 2.0) * (1 - np.exp(-beta * omega_pos))
            qfi = 4*np.trapz(integrand, omega_pos)
            
            # Plot the spectral function
            plt.figure(figsize=(10, 6))
            plt.scatter(omega_pos, s_omega_pos, label=f'Beta≈{beta:.6g} QFI={qfi:.4f}')
            plt.xlabel('Frequency (rad/s)')
            plt.ylabel('Spectral Function S(ω)')
            plt.xlim(0, 6)
            plt.title(f'Spectral Function for {species} at Beta≈{beta:.6g}')
            plt.grid(True)
            plt.legend()
            outdir = os.path.join(structure_factor_dir, 'processed_data', species)
            os.makedirs(outdir, exist_ok=True)
            plot_filename = os.path.join(outdir, f'spectral_function_{species}_beta_{beta:.6g}.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            
            all_species_qfi_data[species].append((beta, qfi))
            
            # Save processed spectral data for each beta
            data_out = np.column_stack((omega, S_omega_real))
            data_filename = os.path.join(outdir, f'spectral_beta_{beta:.6g}.dat')
            np.savetxt(data_filename, data_out, header='freq spectral_function')
    
    # Plot QFI vs Beta for each species
    plot_outdir = os.path.join(structure_factor_dir, 'plots')
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
        
        # Path to structure_factor_results directory
        structure_factor_dir = os.path.join(subdir, 'structure_factor_results')
        
        if not os.path.exists(structure_factor_dir):
            print(f"[Rank {rank}] Structure factor directory not found: {structure_factor_dir}")
            continue
        
        print(f"[Rank {rank}] Processing directory: {subdir} for Jpm={jpm_value}")
        # Run the QFI analysis for the current Jpm value
        species_qfi_data = parse_QFI_data_new(structure_factor_dir)
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
        
        # # Plot QFI heatmaps
        # for species, data_points in all_qfi_data.items():
        #     if not data_points:
        #         continue
                
        #     # Extract data
        #     jpm_vals = np.array([point[0] for point in data_points])
        #     beta_vals = np.array([point[1] for point in data_points])
        #     qfi_vals = np.array([point[2] for point in data_points])
            
        #     # Create interpolation grid
        #     jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
        #     beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
        #     jpm_grid = np.linspace(jpm_min, jpm_max, 100)
        #     beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)
        #     JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
            
        #     # Interpolate QFI values
        #     QFI = griddata((jpm_vals, beta_vals), qfi_vals, (JPM, BETA), method='cubic')
            
        #     # Plot heatmap
        #     plt.figure(figsize=(12, 8))
        #     plt.pcolormesh(JPM, BETA, QFI, shading='auto', cmap='viridis')
        #     plt.colorbar(label='QFI')
            
        #     # Overlay original data points
        #     plt.scatter(jpm_vals, beta_vals, c='red', s=20, edgecolors='black')
            
        #     plt.xlabel('Jpm')
        #     plt.ylabel('Beta (β)')
        #     plt.yscale('log')
        #     plt.title(f'QFI Heatmap for {species}')
            
        #     heatmap_filename = os.path.join(plot_outdir, f'qfi_heatmap_{species}.png')
        #     plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"Saved QFI heatmap for {species} to {heatmap_filename}")
        
        # # Plot derivative heatmaps
        # for species, data_points in all_derivative_data.items():
        #     if not data_points:
        #         continue
                
        #     # Extract data
        #     jpm_vals = np.array([point[0] for point in data_points])
        #     beta_vals = np.array([point[1] for point in data_points])
        #     deriv_vals = np.array([point[2] for point in data_points])
            
        #     # Create interpolation grid
        #     jpm_min, jpm_max = jpm_vals.min(), jpm_vals.max()
        #     beta_min, beta_max = beta_vals.min(), beta_vals.max()
            
        #     jpm_grid = np.linspace(jpm_min, jpm_max, 100)
        #     beta_grid = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)
        #     JPM, BETA = np.meshgrid(jpm_grid, beta_grid)
            
        #     # Interpolate derivative values
        #     DERIV = griddata((jpm_vals, beta_vals), deriv_vals, (JPM, BETA), method='cubic')
            
        #     # Plot heatmap
        #     plt.figure(figsize=(12, 8))
        #     plt.pcolormesh(JPM, BETA, DERIV, shading='auto', cmap='viridis')
        #     plt.colorbar(label='dQFI/dβ')

        #     # Overlay original data points
        #     plt.scatter(jpm_vals, beta_vals, c='red', s=20, edgecolors='black')

        #     plt.xlabel('Jpm')
        #     plt.ylabel('Beta (β)')
        #     plt.yscale('log')
        #     plt.title(f'dQFI/dβ Heatmap for {species}')
            
        #     heatmap_filename = os.path.join(plot_outdir, f'qfi_derivative_heatmap_{species}.png')
        #     plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"Saved derivative heatmap for {species} to {heatmap_filename}")
        
        print("Heatmap generation complete!")
        return jpm_qfi_data
    else:
        return None

def parse_QFI_across_hi(data_dir):
    """
    Scan subdirectories named 'h=i=*' under data_dir, run QFI parsing per folder,
    and build heatmaps across the parameter h=i.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Discover parameter sweep folders on rank 0
    if rank == 0:
        sweep_dirs = sorted(glob.glob(os.path.join(data_dir, 'h=*')))
    else:
        sweep_dirs = None
    sweep_dirs = comm.bcast(sweep_dirs, root=0)

    # Round-robin assignment
    my_dirs = sweep_dirs[rank::size]

    # Local compute
    local_results = {}
    param_regex = re.compile(r'h=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    for d in my_dirs:
        m = param_regex.search(os.path.basename(d))
        if not m:
            continue
        hi_val = float(m.group(1))
        sf_path = os.path.join(d, 'structure_factor_results')
        if not os.path.isdir(sf_path):
            print(f"[Rank {rank}] Missing structure_factor_results at: {sf_path}")
            continue
        print(f"[Rank {rank}] Processing {d} (h=i={hi_val})")
        local_results[hi_val] = parse_QFI_data_new(sf_path)

    # Gather and merge on root
    gathered = comm.gather(local_results, root=0)
    if rank != 0:
        return None

    merged = {}
    for part in gathered:
        merged.update(part)

    # Reformat per-species arrays and compute derivatives
    by_species = defaultdict(list)
    by_species_deriv = defaultdict(list)
    for hi, species_map in merged.items():
        for sp, beta_qfi in species_map.items():
            for b, q in beta_qfi:
                by_species[sp].append((hi, b, q))

            if len(beta_qfi) > 1:
                bq = np.array(sorted(beta_qfi, key=lambda x: x[0]), dtype=float)
                bvals, qvals = bq[:, 0], bq[:, 1]
                mid = 0.5 * (bvals[:-1] + bvals[1:])
                dq = np.diff(qvals)
                db = np.diff(bvals)
                deriv = dq / db
                for mb, dv in zip(mid, deriv):
                    by_species_deriv[sp].append((hi, mb, dv))

    # Plotting
    out_dir = os.path.join(data_dir, 'plots_hi')
    os.makedirs(out_dir, exist_ok=True)

    # Heatmaps for QFI
    for sp, triples in by_species.items():
        if not triples:
            continue
        arr = np.array(triples, dtype=float)
        X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]

        # Build grid in parameter (linear) and beta (log)
        x_min, x_max = np.nanmin(X), np.nanmax(X)
        y_min, y_max = np.nanmin(Y[Y > 0]), np.nanmax(Y)
        x_lin = np.linspace(x_min, x_max, 120)
        y_log = np.logspace(np.log10(y_min), np.log10(y_max), 120)
        XX, YY = np.meshgrid(x_lin, y_log)
        ZZ = griddata((X, Y), Z, (XX, YY), method='cubic')

        plt.figure(figsize=(11, 7))
        mesh = plt.pcolormesh(XX, YY, ZZ, shading='auto', cmap='viridis')
        plt.colorbar(mesh, label='QFI')
        plt.scatter(X, Y, s=14, c='k', alpha=0.6, label='samples')
        plt.yscale('log')
        plt.xlabel('h=i')
        plt.ylabel('Beta (β)')
        plt.title(f'QFI heatmap (h=i sweep): {sp}')
        plt.legend(loc='best')
        fout = os.path.join(out_dir, f'qfi_heatmap_hi_{sp}.png')
        plt.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fout}")

    # Heatmaps for dQFI/dβ
    for sp, triples in by_species_deriv.items():
        if not triples:
            continue
        arr = np.array(triples, dtype=float)
        X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]

        x_min, x_max = np.nanmin(X), np.nanmax(X)
        y_min, y_max = np.nanmin(Y[Y > 0]), np.nanmax(Y)
        x_lin = np.linspace(x_min, x_max, 120)
        y_log = np.logspace(np.log10(y_min), np.log10(y_max), 120)
        XX, YY = np.meshgrid(x_lin, y_log)
        ZZ = griddata((X, Y), Z, (XX, YY), method='cubic')

        plt.figure(figsize=(11, 7))
        mesh = plt.pcolormesh(XX, YY, ZZ, shading='auto', cmap='viridis')
        plt.colorbar(mesh, label='dQFI/dβ')
        plt.scatter(X, Y, s=14, c='k', alpha=0.6, label='samples')
        plt.yscale('log')
        plt.xlabel('h=i')
        plt.ylabel('Beta (β)')
        plt.title(f'dQFI/dβ heatmap (h=i sweep): {sp}')
        plt.legend(loc='best')
        fout = os.path.join(out_dir, f'qfi_derivative_heatmap_hi_{sp}.png')
        plt.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fout}")

    print("h=i sweep processing complete!")
    return merged


def plot_heatmaps_from_processed_data(data_dir):
    """Plot heatmaps and fixed-beta line plots by reading processed QFI data from subdirectories.
       Only plot rows where there is no NaN, and save all intermediate and plot data."""
    
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
        
        plots_dir = os.path.join(subdir, 'structure_factor_results', 'plots')
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
    
    plot_outdir = os.path.join(data_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)

    # Save raw assembled points for QFI and derivatives
    for species, data_points in all_qfi_data.items():
        if data_points:
            arr = np.array(data_points, dtype=float)
            np.savetxt(os.path.join(plot_outdir, f'qfi_points_{species}.dat'), arr,
                       header='Jpm beta qfi')
    for species, data_points in all_derivative_data.items():
        if data_points:
            arr = np.array(data_points, dtype=float)
            np.savetxt(os.path.join(plot_outdir, f'qfi_derivative_points_{species}.dat'), arr,
                       header='Jpm beta dQFI_dbeta')

    def save_grid_txt(fname, beta, jpm, Z, value_label):
        if Z is None or Z.size == 0:
            return
        header_cols = ['beta'] + [f'Jpm={v:g}' for v in jpm]
        header = ' '.join(header_cols)
        out = np.column_stack((beta, Z))
        np.savetxt(fname, out, header=header)

    for species, data_points in all_qfi_data.items():
        if not data_points:
            continue

        arr = np.array(data_points, dtype=float)
        jpm_vals, beta_vals, qfi_vals = arr[:, 0], arr[:, 1], arr[:, 2]

        # Choose reference beta grid from Jpm closest to 0.08
        ref_target = 0.08
        unique_jpm = np.unique(jpm_vals)
        if unique_jpm.size == 0:
            continue
        ref_jpm = unique_jpm[np.argmin(np.abs(unique_jpm - ref_target))]
        ref_mask = np.isclose(jpm_vals, ref_jpm, rtol=1e-8, atol=1e-12)

        beta_ref = beta_vals[ref_mask]
        beta_ref = beta_ref[beta_ref > 0]
        target_beta = np.unique(beta_ref)
        target_beta.sort()

        if target_beta.size < 2:
            # Fallback to all positive betas if ref grid insufficient
            target_beta = np.unique(beta_vals[beta_vals > 0])
            target_beta.sort()
        if target_beta.size < 2:
            continue  # cannot build a meaningful grid

        jpm_neg = np.unique(jpm_vals[jpm_vals < 0])
        jpm_neg.sort()
        jpm_pos = np.unique(jpm_vals[jpm_vals > 0])
        jpm_pos.sort()

        def interp_at_jpm(j):
            mask = np.isclose(jpm_vals, j, rtol=1e-8, atol=1e-12)
            b = beta_vals[mask]
            q = qfi_vals[mask]
            if b.size == 0:
                return np.full_like(target_beta, np.nan, dtype=float)
            order = np.argsort(b)
            b, q = b[order], q[order]
            bu, inv = np.unique(b, return_inverse=True)
            q_mean = np.zeros_like(bu, dtype=float)
            counts = np.zeros_like(bu, dtype=int)
            np.add.at(q_mean, inv, q)
            np.add.at(counts, inv, 1)
            q_mean = q_mean / np.maximum(counts, 1)
            if bu.size < 2:
                return np.full_like(target_beta, np.nan, dtype=float)
            f = interp1d(bu, q_mean, kind='linear', bounds_error=False, fill_value=np.nan)
            return f(target_beta)

        Z_neg = None
        Z_pos = None

        if jpm_neg.size > 0:
            Z_neg = np.column_stack([interp_at_jpm(j) for j in jpm_neg])
        if jpm_pos.size > 0:
            Z_pos = np.column_stack([interp_at_jpm(j) for j in jpm_pos])

        # Save unfiltered grids
        if Z_neg is not None:
            np.savez(os.path.join(plot_outdir, f'qfi_grid_neg_{species}.npz'),
                     beta=target_beta, jpm=jpm_neg, Z=Z_neg)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_grid_neg_{species}.dat'),
                          target_beta, jpm_neg, Z_neg, 'QFI')
        if Z_pos is not None:
            np.savez(os.path.join(plot_outdir, f'qfi_grid_pos_{species}.npz'),
                     beta=target_beta, jpm=jpm_pos, Z=Z_pos)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_grid_pos_{species}.dat'),
                          target_beta, jpm_pos, Z_pos, 'QFI')

        # Filter rows: only keep rows with no NaN
        Z_neg_f = None
        Z_pos_f = None
        mask_rows_neg = None
        mask_rows_pos = None
        if Z_neg is not None and Z_neg.size > 0:
            mask_rows_neg = np.all(np.isfinite(Z_neg), axis=1)
            if np.any(mask_rows_neg):
                Z_neg_f = Z_neg[mask_rows_neg, :]
        if Z_pos is not None and Z_pos.size > 0:
            mask_rows_pos = np.all(np.isfinite(Z_pos), axis=1)
            if np.any(mask_rows_pos):
                Z_pos_f = Z_pos[mask_rows_pos, :]

        # Save filtered grids
        if Z_neg_f is not None:
            beta_neg_f = target_beta[mask_rows_neg]
            np.savez(os.path.join(plot_outdir, f'qfi_grid_neg_filtered_{species}.npz'),
                     beta=beta_neg_f, jpm=jpm_neg, Z=Z_neg_f)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_grid_neg_filtered_{species}.dat'),
                          beta_neg_f, jpm_neg, Z_neg_f, 'QFI')
        if Z_pos_f is not None:
            beta_pos_f = target_beta[mask_rows_pos]
            np.savez(os.path.join(plot_outdir, f'qfi_grid_pos_filtered_{species}.npz'),
                     beta=beta_pos_f, jpm=jpm_pos, Z=Z_pos_f)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_grid_pos_filtered_{species}.dat'),
                          beta_pos_f, jpm_pos, Z_pos_f, 'QFI')

        # Color scale unified across all plotted panels (filtered only)
        z_list = []
        if Z_neg_f is not None: z_list.append(Z_neg_f)
        if Z_pos_f is not None: z_list.append(Z_pos_f)
        if not z_list:
            continue
        vmin = np.nanmin([np.nanmin(z) for z in z_list])
        vmax = np.nanmax([np.nanmax(z) for z in z_list])

        # Plot Jpm<0 using filtered rows only
        if Z_neg_f is not None and Z_neg_f.size > 0:
            beta_neg_f = target_beta[mask_rows_neg]
            JN, BN = np.meshgrid(jpm_neg, beta_neg_f)
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JN, BN, Z_neg_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.yscale('log')
            plt.gca().invert_yaxis()  # large beta at bottom
            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.title(f'QFI Heatmap (Jpm<0) for {species}')
            plt.colorbar(label='QFI')
            plt.savefig(os.path.join(plot_outdir, f'qfi_heatmap_neg_{species}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Plot Jpm>0 using filtered rows only
        if Z_pos_f is not None and Z_pos_f.size > 0:
            beta_pos_f = target_beta[mask_rows_pos]
            JP, BP = np.meshgrid(jpm_pos, beta_pos_f)
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JP, BP, Z_pos_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.yscale('log')
            plt.gca().invert_yaxis()  # large beta at bottom
            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.title(f'QFI Heatmap (Jpm>0) for {species}')
            plt.colorbar(label='QFI')
            plt.savefig(os.path.join(plot_outdir, f'qfi_heatmap_pos_{species}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Side-by-side (no gap) view for Jpm<0 and Jpm>0 using filtered rows
        if (Z_neg_f is not None and Z_neg_f.size > 0) and (Z_pos_f is not None and Z_pos_f.size > 0):
            beta_neg_f = target_beta[mask_rows_neg]
            beta_pos_f = target_beta[mask_rows_pos]

            JN, BN = np.meshgrid(jpm_neg, beta_neg_f)
            JP, BP = np.meshgrid(jpm_pos, beta_pos_f)

            y_min = float(min(beta_neg_f.min(), beta_pos_f.min()))
            y_max = float(max(beta_neg_f.max(), beta_pos_f.max()))

            # Make left panel wider (dynamically scaled, lower bound 1.2x)
            neg_span = float(np.ptp(jpm_neg)) if jpm_neg.size > 0 else 0.0
            pos_span = float(np.ptp(jpm_pos)) if jpm_pos.size > 0 else 0.0
            if pos_span > 0:
                wr = neg_span / pos_span
            else:
                wr = 1.5
            wr = float(np.clip(wr, 1.2, 3.0))

            fig, (axL, axR) = plt.subplots(
                1, 2, figsize=(14, 8), sharey=True,
                gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'width_ratios': [wr, 1.0]}
            )
            mL = axL.pcolormesh(JN, BN, Z_neg_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            mR = axR.pcolormesh(JP, BP, Z_pos_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)

            for ax in (axL, axR):
                ax.set_yscale('log')
                ax.set_ylim(y_max, y_min)  # large beta at bottom
                ax.set_xlabel('Jpm')
            axL.set_ylabel('Beta (β)')
            axR.tick_params(labelleft=False)

            fig.subplots_adjust(wspace=0.0)
            cbar = fig.colorbar(mL, ax=[axL, axR], location='right', pad=0.02)
            cbar.set_label('QFI')

            fig.suptitle(f'QFI Heatmap (Jpm<0 | Jpm>0) for {species}')
            fig.savefig(os.path.join(plot_outdir, f'qfi_heatmap_side_by_side_{species}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Line plot at the largest beta with no NaN per segment (QFI vs Jpm)
        # Find largest beta row index satisfying no-NaN in each segment
        idx_neg = None
        idx_pos = None
        if Z_neg is not None and Z_neg.size > 0 and mask_rows_neg is not None and np.any(mask_rows_neg):
            valid_idxs = np.where(mask_rows_neg)[0]
            if valid_idxs.size > 0:
                idx_neg = valid_idxs[-1]
        if Z_pos is not None and Z_pos.size > 0 and mask_rows_pos is not None and np.any(mask_rows_pos):
            valid_idxs = np.where(mask_rows_pos)[0]
            if valid_idxs.size > 0:
                idx_pos = valid_idxs[-1]

        if idx_neg is not None or idx_pos is not None:
            plt.figure(figsize=(10, 6))
            color = 'C0'
            plotted_any = False

            # Negative Jpm segment
            if idx_neg is not None:
                b_neg = float(target_beta[idx_neg])
                y_neg = Z_neg[idx_neg, :]
                # All finite by construction of mask
                plt.plot(jpm_neg, y_neg, '-', lw=1.8, color=color, label=f'β={b_neg:.3g} (neg)')
                np.savetxt(os.path.join(plot_outdir, f'qfi_vs_jpm_fixed_beta_neg_{species}.dat'),
                           np.column_stack((jpm_neg, np.full_like(jpm_neg, b_neg), y_neg)),
                           header='Jpm beta QFI')
                plotted_any = True

            # Positive Jpm segment
            if idx_pos is not None:
                b_pos = float(target_beta[idx_pos])
                y_pos = Z_pos[idx_pos, :]
                plt.plot(jpm_pos, y_pos, '-', lw=1.8, color=color, label=f'β={b_pos:.3g} (pos)' if not plotted_any else f'β={b_pos:.3g} (pos)')
                np.savetxt(os.path.join(plot_outdir, f'qfi_vs_jpm_fixed_beta_pos_{species}.dat'),
                           np.column_stack((jpm_pos, np.full_like(jpm_pos, b_pos), y_pos)),
                           header='Jpm beta QFI')
                plotted_any = True

            plt.xlabel('Jpm')
            plt.ylabel('QFI')
            plt.title(f'QFI vs Jpm at largest β rows (no NaN) for {species}')
            plt.grid(True, alpha=0.3)
            if plotted_any:
                plt.legend(fontsize=9)
                fname = os.path.join(plot_outdir, f'qfi_vs_jpm_fixed_beta_{species}.png')
                plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()

    # Derivative plots (same pipeline)
    for species, data_points in all_derivative_data.items():
        if not data_points:
            continue

        arr = np.array(data_points, dtype=float)
        jpm_vals, beta_vals, deriv_vals = arr[:, 0], arr[:, 1], arr[:, 2]

        # Choose reference beta grid from Jpm closest to 0.09
        ref_target = 0.09
        unique_jpm = np.unique(jpm_vals)
        if unique_jpm.size == 0:
            continue
        ref_jpm = unique_jpm[np.argmin(np.abs(unique_jpm - ref_target))]
        ref_mask = np.isclose(jpm_vals, ref_jpm, rtol=1e-8, atol=1e-12)

        beta_ref = beta_vals[ref_mask]
        beta_ref = beta_ref[beta_ref > 0]
        target_beta = np.unique(beta_ref)
        target_beta.sort()

        if target_beta.size < 2:
            # Fallback to all positive betas if ref grid insufficient
            target_beta = np.unique(beta_vals[beta_vals > 0])
            target_beta.sort()
        if target_beta.size < 2:
            continue  # cannot build a meaningful grid

        jpm_neg = np.unique(jpm_vals[jpm_vals < 0])
        jpm_neg.sort()
        jpm_pos = np.unique(jpm_vals[jpm_vals > 0])
        jpm_pos.sort()

        def interp_deriv_at_jpm(j):
            mask = np.isclose(jpm_vals, j, rtol=1e-8, atol=1e-12)
            b = beta_vals[mask]
            d = deriv_vals[mask]
            if b.size == 0:
                return np.full_like(target_beta, np.nan, dtype=float)
            order = np.argsort(b)
            b, d = b[order], d[order]
            bu, inv = np.unique(b, return_inverse=True)
            d_mean = np.zeros_like(bu, dtype=float)
            counts = np.zeros_like(bu, dtype=int)
            np.add.at(d_mean, inv, d)
            np.add.at(counts, inv, 1)
            d_mean = d_mean / np.maximum(counts, 1)
            if bu.size < 2:
                return np.full_like(target_beta, np.nan, dtype=float)
            f = interp1d(bu, d_mean, kind='linear', bounds_error=False, fill_value=np.nan)
            return f(target_beta)

        Z_neg = None
        Z_pos = None

        if jpm_neg.size > 0:
            Z_neg = np.column_stack([interp_deriv_at_jpm(j) for j in jpm_neg])
        if jpm_pos.size > 0:
            Z_pos = np.column_stack([interp_deriv_at_jpm(j) for j in jpm_pos])

        # Save unfiltered derivative grids
        if Z_neg is not None:
            np.savez(os.path.join(plot_outdir, f'qfi_derivative_grid_neg_{species}.npz'),
                     beta=target_beta, jpm=jpm_neg, Z=Z_neg)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_derivative_grid_neg_{species}.dat'),
                          target_beta, jpm_neg, Z_neg, 'dQFI_dbeta')
        if Z_pos is not None:
            np.savez(os.path.join(plot_outdir, f'qfi_derivative_grid_pos_{species}.npz'),
                     beta=target_beta, jpm=jpm_pos, Z=Z_pos)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_derivative_grid_pos_{species}.dat'),
                          target_beta, jpm_pos, Z_pos, 'dQFI_dbeta')

        # Filter rows: only keep rows with no NaN
        Z_neg_f = None
        Z_pos_f = None
        mask_rows_neg = None
        mask_rows_pos = None
        if Z_neg is not None and Z_neg.size > 0:
            mask_rows_neg = np.all(np.isfinite(Z_neg), axis=1)
            if np.any(mask_rows_neg):
                Z_neg_f = Z_neg[mask_rows_neg, :]
        if Z_pos is not None and Z_pos.size > 0:
            mask_rows_pos = np.all(np.isfinite(Z_pos), axis=1)
            if np.any(mask_rows_pos):
                Z_pos_f = Z_pos[mask_rows_pos, :]

        # Save filtered derivative grids
        if Z_neg_f is not None:
            beta_neg_f = target_beta[mask_rows_neg]
            np.savez(os.path.join(plot_outdir, f'qfi_derivative_grid_neg_filtered_{species}.npz'),
                     beta=beta_neg_f, jpm=jpm_neg, Z=Z_neg_f)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_derivative_grid_neg_filtered_{species}.dat'),
                          beta_neg_f, jpm_neg, Z_neg_f, 'dQFI_dbeta')
        if Z_pos_f is not None:
            beta_pos_f = target_beta[mask_rows_pos]
            np.savez(os.path.join(plot_outdir, f'qfi_derivative_grid_pos_filtered_{species}.npz'),
                     beta=beta_pos_f, jpm=jpm_pos, Z=Z_pos_f)
            save_grid_txt(os.path.join(plot_outdir, f'qfi_derivative_grid_pos_filtered_{species}.dat'),
                          beta_pos_f, jpm_pos, Z_pos_f, 'dQFI_dbeta')

        # Color scale unified across all plotted panels (filtered only)
        z_list = []
        if Z_neg_f is not None: z_list.append(Z_neg_f)
        if Z_pos_f is not None: z_list.append(Z_pos_f)
        if not z_list:
            continue
        vmin = np.nanmin([np.nanmin(z) for z in z_list])
        vmax = np.nanmax([np.nanmax(z) for z in z_list])

        # Plot Jpm<0 using filtered rows
        if Z_neg_f is not None and Z_neg_f.size > 0:
            beta_neg_f = target_beta[mask_rows_neg]
            JN, BN = np.meshgrid(jpm_neg, beta_neg_f)
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JN, BN, Z_neg_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.yscale('log')
            plt.gca().invert_yaxis()  # large beta at bottom
            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.title(f'dQFI/dβ Heatmap (Jpm<0) for {species}')
            plt.colorbar(label='dQFI/dβ')
            plt.savefig(os.path.join(plot_outdir, f'qfi_derivative_heatmap_neg_{species}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Plot Jpm>0 using filtered rows
        if Z_pos_f is not None and Z_pos_f.size > 0:
            beta_pos_f = target_beta[mask_rows_pos]
            JP, BP = np.meshgrid(jpm_pos, beta_pos_f)
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(JP, BP, Z_pos_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.yscale('log')
            plt.gca().invert_yaxis()  # large beta at bottom
            plt.xlabel('Jpm')
            plt.ylabel('Beta (β)')
            plt.title(f'dQFI/dβ Heatmap (Jpm>0) for {species}')
            plt.colorbar(label='dQFI/dβ')
            plt.savefig(os.path.join(plot_outdir, f'qfi_derivative_heatmap_pos_{species}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Side-by-side (no gap) view for Jpm<0 and Jpm>0 using filtered rows
        if (Z_neg_f is not None and Z_neg_f.size > 0) and (Z_pos_f is not None and Z_pos_f.size > 0):
            beta_neg_f = target_beta[mask_rows_neg]
            beta_pos_f = target_beta[mask_rows_pos]

            JN, BN = np.meshgrid(jpm_neg, beta_neg_f)
            JP, BP = np.meshgrid(jpm_pos, beta_pos_f)

            y_min = float(min(beta_neg_f.min(), beta_pos_f.min()))
            y_max = float(max(beta_neg_f.max(), beta_pos_f.max()))

            # Make left panel wider (dynamically scaled, lower bound 1.2x)
            neg_span = float(np.ptp(jpm_neg)) if jpm_neg.size > 0 else 0.0
            pos_span = float(np.ptp(jpm_pos)) if jpm_pos.size > 0 else 0.0
            if pos_span > 0:
                wr = neg_span / pos_span
            else:
                wr = 1.5
            wr = float(np.clip(wr, 1.2, 3.0))

            fig, (axL, axR) = plt.subplots(
                1, 2, figsize=(14, 8), sharey=True,
                gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'width_ratios': [wr, 1.0]}
            )
            mL = axL.pcolormesh(JN, BN, Z_neg_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            mR = axR.pcolormesh(JP, BP, Z_pos_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)

            for ax in (axL, axR):
                ax.set_yscale('log')
                ax.set_ylim(y_max, y_min)
                ax.set_xlabel('Jpm')
            axL.set_ylabel('Beta (β)')
            axR.tick_params(labelleft=False)

            fig.subplots_adjust(wspace=0.0)
            cbar = fig.colorbar(mL, ax=[axL, axR], location='right', pad=0.02)
            cbar.set_label('dQFI/dβ')

            fig.suptitle(f'dQFI/dβ Heatmap (Jpm<0 | Jpm>0) for {species}')
            fig.savefig(os.path.join(plot_outdir, f'qfi_derivative_heatmap_side_by_side_{species}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Line plot at the largest beta with no NaN per segment (dQFI/dβ vs Jpm)
        idx_neg = None
        idx_pos = None
        if Z_neg is not None and Z_neg.size > 0 and mask_rows_neg is not None and np.any(mask_rows_neg):
            valid_idxs = np.where(mask_rows_neg)[0]
            if valid_idxs.size > 0:
                idx_neg = valid_idxs[-1]
        if Z_pos is not None and Z_pos.size > 0 and mask_rows_pos is not None and np.any(mask_rows_pos):
            valid_idxs = np.where(mask_rows_pos)[0]
            if valid_idxs.size > 0:
                idx_pos = valid_idxs[-1]

        if idx_neg is not None or idx_pos is not None:
            plt.figure(figsize=(10, 6))
            color = 'C1'
            plotted_any = False

            # Negative Jpm segment
            if idx_neg is not None:
                b_neg = float(target_beta[idx_neg])
                y_neg = Z_neg[idx_neg, :]
                plt.plot(jpm_neg, y_neg, '-', lw=1.8, color=color, label=f'β={b_neg:.3g} (neg)')
                np.savetxt(os.path.join(plot_outdir, f'qfi_derivative_vs_jpm_fixed_beta_neg_{species}.dat'),
                           np.column_stack((jpm_neg, np.full_like(jpm_neg, b_neg), y_neg)),
                           header='Jpm beta dQFI_dbeta')
                plotted_any = True

            # Positive Jpm segment
            if idx_pos is not None:
                b_pos = float(target_beta[idx_pos])
                y_pos = Z_pos[idx_pos, :]
                plt.plot(jpm_pos, y_pos, '-', lw=1.8, color=color, label=f'β={b_pos:.3g} (pos)' if not plotted_any else f'β={b_pos:.3g} (pos)')
                np.savetxt(os.path.join(plot_outdir, f'qfi_derivative_vs_jpm_fixed_beta_pos_{species}.dat'),
                           np.column_stack((jpm_pos, np.full_like(jpm_pos, b_pos), y_pos)),
                           header='Jpm beta dQFI_dbeta')
                plotted_any = True

            plt.xlabel('Jpm')
            plt.ylabel('dQFI/dβ')
            plt.title(f'dQFI/dβ vs Jpm at largest β rows (no NaN) for {species}')
            plt.grid(True, alpha=0.3)
            if plotted_any:
                fname = os.path.join(plot_outdir, f'qfi_derivative_vs_jpm_fixed_beta_{species}.png')
                plt.legend(fontsize=9)
                plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Path to the directory containing the data files
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    across_QFI = sys.argv[2] if len(sys.argv) > 2 else 'False'
    across_QFI = across_QFI.lower() == 'true'
    if across_QFI:
        parse_QFI_across_Jpm(data_dir)
        parse_QFI_across_hi(data_dir)
        plot_heatmaps_from_processed_data(data_dir)
    else:
        parse_QFI_data_new(data_dir)
    print("All processing complete.")
