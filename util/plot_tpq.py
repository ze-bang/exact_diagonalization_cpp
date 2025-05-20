import os
import glob
import numpy as np
from collections import defaultdict
import re

import matplotlib.pyplot as plt

def extract_step(filename):
    """Extract step number from filename."""
    match = re.search(r'step(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def determine_beta(step):
    """Determine beta based on step number."""
    if step in [1623, 1624]:
        return 10
    elif step in [16246, 16247]:
        return 100
    else:
        return 1000

def get_file_key(filename):
    """Extract key for grouping files (ignoring rand number)."""
    match = re.search(r'time_corr_rand\d+_(.+?)\.dat', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

# Path to the directory containing the data files
data_dir = '/home/pc_linux/exact_diagonalization_cpp/ED_16_sites_QFI_test/output/'

# List all relevant files matching the pattern
files = glob.glob(os.path.join(data_dir, 'time_corr_rand*_*.dat'))

# Group files by their base name (ignoring rand number)
grouped_files = defaultdict(list)
for file_path in files:
    key = get_file_key(file_path)
    if key:
        grouped_files[key].append(file_path)

results = {}

# Process each group of files
for base_key, file_list in grouped_files.items():
    print(f"Processing {base_key} with {len(file_list)} files")
    
    # Get the step number and determine beta
    example_file = file_list[0]
    step = extract_step(example_file)
    beta = determine_beta(step)
    print(f"Step: {step}, Beta: {beta}")
    
    # Store data from all files in this group
    all_data = []
    
    for file_path in file_list:
        try:
            # Load the data (time in column 0, real value in column 1)
            data = np.loadtxt(file_path, comments='#')
            time = data[:, 0]  # Column 0: time

            real_val = data[:, 1]  # Column 1: real value
            comp_val = data[:, 2]  # Column 2: imaginary value

            val = real_val + 1j * comp_val  # Combine real and imaginary parts

            all_data.append((time, val))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_data:
        print(f"No valid data found for {base_key}")
        continue

    # Ensure all files have the same time points
    reference_time = all_data[0][0]
    if not all(np.array_equal(data[0], reference_time) for data in all_data):
        print(f"Warning: Time points differ across files for {base_key}")
        continue
    
    # Calculate mean and standard error
    all_real_values = np.vstack([data[1] for data in all_data])
    mean_real = np.mean(all_real_values, axis=0)
    std_error = np.std(all_real_values, axis=0) / np.sqrt(len(all_data))

    # Apply Lorentzian filter to the time-domain data
    gamma = 0.02  # Broadening parameter: larger values -> more smoothing
    lorentzian_filter = np.exp(-gamma * np.abs(reference_time))
    mean_real = mean_real * lorentzian_filter

    mean_fft = np.zeros_like(mean_real)

    mean_fft[0] = mean_real[len(mean_real)//2]
    mean_fft[1:len(mean_real)//2] = mean_real[0:len(mean_real)//2-1]
    mean_fft[len(mean_real)//2+1:] = mean_real[len(mean_real)//2+1:]

    mean_fft = np.abs(np.real(np.fft.fft(mean_fft, norm="ortho")))

    reference_freq = -np.fft.fftfreq(len(mean_real), d=(reference_time[1]-reference_time[0])/(2*np.pi))


    qfi = np.trapz(mean_fft * np.tanh(beta*reference_freq / 2) * 4 / np.pi, -reference_freq)

    # Store the results
    results[base_key] = {
        'freq': reference_freq,
        'mean': mean_fft,
        'error': std_error,
        'beta': beta,
        'qfi': qfi
    }

# Plotting
for base_key, data in results.items():
    plt.figure(figsize=(10, 6))

    plt.plot(data['freq'], data['mean'],
                 label=f'β={data["beta"]}, QFI={data["qfi"]:.2f}')

    plt.xlabel('Frequency')
    plt.xlim(-2,5)
    plt.ylabel('Spectral Function * (1-exp(-ω*β))/2')
    plt.title(f'Average Spectral Function for {base_key}')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_filename = os.path.join(data_dir, f'plot_{base_key}.png')
    plt.savefig(plot_filename, dpi=300)
    
    # Save the data
    data_out = np.column_stack((data['freq'], data['mean'], data['error']))
    data_filename = os.path.join(data_dir, f'processed_{base_key}.dat')
    np.savetxt(data_filename, data_out, header='freq mean error qfi')

    plt.close()

print("Processing complete!")