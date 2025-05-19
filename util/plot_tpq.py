import numpy as np
import pandas as pd
import os

#!/usr/bin/env python3
import matplotlib.pyplot as plt

def read_data(filename):
    """Read data from file."""
    try:
        # Assuming space/tab-separated data without header
        data = np.loadtxt(filename, skiprows=1, unpack=True)
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def plot_specific_heat(file1, file2):
    """Plot specific heat from data files."""
    plt.figure(figsize=(10, 6))
    
    # Read data files
    data1 = read_data(file1)
    data2 = read_data(file2)
    
    # Plot data (assuming temperature in column 0 and specific heat in column 2)
    if data1 is not None:
        plt.plot(1/data1[0], data1[2]*data1[0]*data1[0]/16, 'o-', label="Specific Heat")
    
    if data2 is not None:
        plt.plot(1/data2[0], data2[1]/16, 's-', label="Energy")
    
    # Add labels and legend
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Specific Heat', fontsize=12)
    plt.title('Specific Heat vs Temperature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log')
    
    # Save and show plot
    plt.savefig('specific_heat.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    file1 = "ED_test_16_sites/output/SS_rand0.dat"
    file2 = "ED_test_16_sites/output/SS_rand0.dat"
    plot_specific_heat(file1, file2)