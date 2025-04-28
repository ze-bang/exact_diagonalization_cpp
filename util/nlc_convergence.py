import os
import sys
import subprocess
import argparse
import time
import glob
import logging
import numpy as np
from matplotlib.cm import get_cmap
from tqdm import tqdm

#!/usr/bin/env python3
"""
NLCE Convergence Analysis - Runs NLCE calculations with increasing order

This script performs multiple NLCE calculations with increasing maximum orders
and plots thermodynamic properties to visualize convergence.

It builds on the NLCE workflow in nlce.py to:
1. Run NLCE calculations for orders 1 to max_order
2. Collect thermodynamic data from each calculation
3. Plot properties from all orders together to visualize convergence
"""

import matplotlib.pyplot as plt


def setup_logging(log_file):
    """Set up logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_nlce_for_order(order, args):
    """Run NLCE calculation for a specific maximum order"""
    logging.info(f"="*80)
    logging.info(f"Running NLCE with maximum order {order}")
    logging.info(f"="*80)
    
    # Build command for running nlce.py with current order
    cmd = [
        'python3', 
        'util/nlce.py',
        f'--max_order={order}',
        f'--base_dir={args.base_dir}/order_{order}',
        f'--ed_executable={args.ed_executable}',
        f'--Jxx={args.Jxx}',
        f'--Jyy={args.Jyy}',
        f'--Jzz={args.Jzz}',
        f'--h={args.h}',
        f'--method={args.method}',
        f'--temp_min={args.temp_min}',
        f'--temp_max={args.temp_max}',
        f'--temp_bins={args.temp_bins}',
    ]
    
    # Add optional arguments
    if args.thermo:
        cmd.append('--thermo')
    if args.euler_resum:
        cmd.append('--euler_resum')
    if args.parallel:
        cmd.append('--parallel')
    if args.num_cores:
        cmd.append(f'--num_cores={args.num_cores}')
    if args.SI_units:
        cmd.append('--SI_units')
    if args.skip_cluster_gen:
        cmd.append('--skip_cluster_gen')
        
    # Add field direction if specified
    if args.field_dir:
        cmd.extend(['--field_dir', str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2])])
    
    # Run NLCE workflow for this order
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE for order {order}: {e}")
        return False

def collect_and_plot_results(max_order, args):
    """Collect results from all orders and create comparative plots"""
    logging.info("="*80)
    logging.info("Creating convergence plots for all orders")
    logging.info("="*80)
    
    # Create directory for convergence plots
    plot_dir = os.path.join(args.base_dir, 'convergence_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define thermodynamic properties to plot
    properties = ['energy', 'specific_heat', 'entropy']
    property_files = {
        'energy': 'nlc_energy.txt',
        'specific_heat': 'nlc_specific_heat.txt',
        'entropy': 'nlc_entropy.txt'
    }
    property_labels = {
        'energy': 'Energy per site',
        'specific_heat': 'Specific Heat per site',
        'entropy': 'Entropy per site'
    }
    
    # Get y-axis limits
    y_limits = {}
    for prop in properties:
        y_min = getattr(args, f"{prop}_ymin", None)
        y_max = getattr(args, f"{prop}_ymax", None)
        y_limits[prop] = (y_min, y_max)
    
    # Initialize plot for each property
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get colormap for different orders
    cmap = plt.get_cmap('viridis')
    
    # Load data for each order and property
    data_by_order_prop = {}
    
    for order in range(1, max_order + 1):
        data_by_order_prop[order] = {}
        nlc_dir = os.path.join(args.base_dir, f'order_{order}/nlc_results_order_{order}')
        
        for prop, filename in property_files.items():
            file_path = os.path.join(nlc_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    # Load data - skip the first line (header)
                    data = np.loadtxt(file_path, comments='#')
                    data_by_order_prop[order][prop] = data
                    logging.info(f"Loaded {prop} data for order {order} from {filename}")
                except Exception as e:
                    logging.error(f"Error loading {prop} data for order {order}: {e}")
    
    # Plot each property
    for i, prop in enumerate(properties):
        if i < len(axes):  # Ensure we have enough axes
            ax = axes[i]
            ax.set_title(property_labels[prop])
            ax.set_xlabel('Temperature')
            ax.set_ylabel(property_labels[prop])
            ax.set_xscale('log')
            ax.grid(True)
            
            # Set y-axis limits if provided
            y_min, y_max = y_limits[prop]
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            for order in range(1, max_order + 1):
                if order in data_by_order_prop and prop in data_by_order_prop[order]:
                    data = data_by_order_prop[order][prop]
                    
                    # First column is temperature, second is the property value
                    temp = data[:, 0]
                    prop_data = data[:, 1]
                    
                    # Apply temperature cutoff based on y domain if limits are provided
                    if y_min is not None and y_max is not None:
                        # Find indices where data exceeds y limits
                        valid_indices = np.where((prop_data < y_min) | (prop_data > y_max))[0]
                        
                        if len(valid_indices) > 0:
                            # Find the minimum valid temperature
                            # Sort temperatures in ascending order
                            sorted_temp_indices = np.argsort(temp)
                            found_valid_temp = False
                            
                            # Sort temperatures in descending order instead of ascending
                            sorted_temp_indices = np.argsort(temp)[::-1]
                            found_valid_temp = False
                            
                            for idx in sorted_temp_indices:
                                if idx in valid_indices:
                                    min_valid_temp = temp[idx]
                                    found_valid_temp = True
                                    break
                            
                            if not found_valid_temp:
                                # No valid temperatures found, skip this data
                                continue
                                
                            # Filter data to only include temperatures >= min_valid_temp
                            mask = temp >= min_valid_temp
                            temp = temp[mask]
                            prop_data = prop_data[mask]
                    
                    # Plot this order
                    color = cmap(order / max_order)
                    ax.plot(temp, prop_data, '-', color=color, 
                            label=f'Order {order}', linewidth=2)
    
    # Add legend to each plot and adjust layout
    for ax in axes:
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'convergence_all_properties.png'), dpi=300)
    plt.close()
    
    # Also create individual plots for each property for better detail
    for prop in properties:
        plt.figure(figsize=(10, 8))
        plt.title(property_labels[prop])
        plt.xlabel('Temperature')
        plt.ylabel(property_labels[prop])
        plt.xscale('log')
        plt.grid(True)
        
        # Set y-axis limits if provided
        y_min, y_max = y_limits[prop]
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        for order in range(1, max_order + 1):
            if order in data_by_order_prop and prop in data_by_order_prop[order]:
                data = data_by_order_prop[order][prop]
                
                # First column is temperature, second is the property value
                temp = data[:, 0]
                prop_data = data[:, 1]
                
                # Apply temperature cutoff based on y domain if limits are provided
                if y_min is not None and y_max is not None:
                    # Find indices where data exceeds y limits
                    valid_indices = np.where((prop_data >= y_min) & (prop_data <= y_max))[0]
                    
                    if len(valid_indices) > 0:
                        # Find the minimum valid temperature
                        # Sort temperatures in ascending order
                        sorted_temp_indices = np.argsort(temp)
                        found_valid_temp = False
                        
                        for idx in sorted_temp_indices:
                            if idx in valid_indices:
                                min_valid_temp = temp[idx]
                                found_valid_temp = True
                                break
                        
                        if not found_valid_temp:
                            # No valid temperatures found, skip this data
                            continue
                            
                        # Filter data to only include temperatures >= min_valid_temp
                        mask = temp >= min_valid_temp
                        temp = temp[mask]
                        prop_data = prop_data[mask]
                
                # Plot this order
                color = cmap(order / max_order)
                plt.plot(temp, prop_data, '-', color=color, 
                            label=f'Order {order}', linewidth=2)
        
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'convergence_{prop}.png'), dpi=300)
        plt.close()
    
    logging.info(f"Convergence plots saved to {plot_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NLCE convergence analysis with multiple orders')
    
    # Parameters for the convergence study
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order to calculate (will run from 1 to this value)')
    parser.add_argument('--base_dir', type=str, default='./nlce_convergence', 
                        help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', 
                        help='Path to the ED executable')
    
    # Model parameters (same as nlce.py)
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], help='Field direction (x,y,z)')
    
    # ED parameters (same as nlce.py)
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method (FULL, LANCZOS, etc.)')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.001, help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    
    # NLCE parameters (same as nlce.py)
    parser.add_argument('--euler_resum', action='store_true', help='Use Euler resummation for NLCE')
    
    # Control flow
    parser.add_argument('--skip_calculations', action='store_true', 
                        help='Skip calculations and only plot existing results')
    parser.add_argument('--start_order', type=int, default=1,
                        help='Starting order (default is 1)')
    parser.add_argument('--skip_cluster_gen', action='store_true',
                        help='Skip cluster generation and only run NLCE calculations')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', help='Run ED in parallel')
    parser.add_argument('--num_cores', type=int, default=None, help='Number of cores to use for parallel processing')
    
    # SI units
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')

    # Parse arguments
    parser.add_argument('--energy_ymin', type=float, default=None, help='Y-axis minimum for energy plot')
    parser.add_argument('--energy_ymax', type=float, default=None, help='Y-axis maximum for energy plot')
    parser.add_argument('--specific_heat_ymin', type=float, default=0, help='Y-axis minimum for specific heat plot')
    parser.add_argument('--specific_heat_ymax', type=float, default=None, help='Y-axis maximum for specific heat plot')
    parser.add_argument('--entropy_ymin', type=float, default=0, help='Y-axis minimum for entropy plot')
    parser.add_argument('--entropy_ymax', type=float, default=None, help='Y-axis maximum for entropy plot')

    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_convergence.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info(f"Starting NLCE convergence analysis up to order {args.max_order}")
    logging.info("="*80)
    
    # Run NLCE for each order unless skipped
    if not args.skip_calculations:
        for order in range(args.start_order, args.max_order + 1):
            success = run_nlce_for_order(order, args)
            if not success:
                logging.warning(f"Failed to complete order {order}. Continuing with next order.")
    
    # Collect and plot results
    collect_and_plot_results(args.max_order, args)
    
    logging.info("="*80)
    logging.info("NLCE convergence analysis completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")