import os
import sys
import subprocess
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import logging
import tempfile
import shutil

#!/usr/bin/env python3
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

def load_experimental_data(file_path):
    """Load experimental data from the specified file"""
    data = np.loadtxt(file_path)
    temp = data[:, 0]  # Temperature (K)
    spec_heat = data[:, 1]  # Specific heat (J*mol/K)
    return temp, spec_heat

def run_nlce(params, fixed_params, exp_temp, work_dir):
    """Run NLCE with the given parameters and return the calculated specific heat"""
    Jxx, Jyy, Jzz = params
    
    # Create command for nlce.py
    cmd = [
        'python3', 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlce.py'),
        '--max_order', str(fixed_params["max_order"]),
        '--Jxx', str(Jxx),
        '--Jyy', str(Jyy),
        '--Jzz', str(Jzz),
        '--h', str(fixed_params["h"]),
        '--field_dir', str(fixed_params["field_dir"][0]), str(fixed_params["field_dir"][1]), str(fixed_params["field_dir"][2]),
        '--base_dir', work_dir,
        '--temp_min', str(fixed_params["temp_min"]),
        '--temp_max', str(fixed_params["temp_max"]),
        '--temp_bins', str(fixed_params["temp_bins"]),
        '--thermo',
        '--SI_units',
        '--euler_resum'
    ]
    
    if fixed_params.get("skip_cluster_gen", False):
        cmd.append('--skip_cluster_gen')
    if fixed_params.get("skip_ham_prep", False):
        cmd.append('--skip_ham_prep')
    
    try:
        logging.info(f"Running NLCE with Jxx={Jxx}, Jyy={Jyy}, Jzz={Jzz}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Find the specific heat output file
        nlc_dir = os.path.join(work_dir, f'nlc_results_order_{fixed_params["max_order"]}')
        spec_heat_file = os.path.join(nlc_dir, 'nlc_specific_heat.txt')
        
        if not os.path.exists(spec_heat_file):
            logging.error(f"Specific heat file not found: {spec_heat_file}")
            return np.array([]), np.array([])
        
        calc_data = np.loadtxt(spec_heat_file)
        calc_temp = calc_data[:, 0]  # Temperature
        calc_spec_heat = calc_data[:, 1]  # Specific heat
        
        return calc_temp, calc_spec_heat
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE: {e}")
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return np.array([]), np.array([])

def interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp):
    """Interpolate calculated data to match experimental temperature points"""
    if len(calc_temp) == 0 or len(calc_spec_heat) == 0:
        return np.zeros_like(exp_temp)
    
    # Sort temperature values
    sort_idx = np.argsort(calc_temp)
    calc_temp = calc_temp[sort_idx]
    calc_spec_heat = calc_spec_heat[sort_idx]
    
    # Create interpolation function
    interp_func = interp1d(calc_temp, calc_spec_heat, kind='cubic', 
                          bounds_error=False, fill_value=np.nan)
    
    # Interpolate at experimental temperatures
    interp_spec_heat = interp_func(exp_temp)
    
    # Replace NaN values with zeros
    interp_spec_heat = np.nan_to_num(interp_spec_heat)
    
    return interp_spec_heat

def calc_chi_squared(params, fixed_params, exp_temp, exp_spec_heat, work_dir):
    """Calculate chi-squared between experimental and calculated specific heat"""
    calc_temp, calc_spec_heat = run_nlce(params, fixed_params, exp_temp, work_dir)
    
    # Interpolate calculated data to match experimental temperature points
    calc_interp = interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp)
    
    # Calculate chi-squared
    chi_squared = np.sum(((exp_spec_heat - calc_interp) / (exp_spec_heat + 1e-10)) ** 2)
    
    logging.info(f"Parameters: Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}, Chi-squared={chi_squared:.4f}")
    
    return chi_squared

def plot_results(exp_temp, exp_spec_heat, calc_temp, calc_spec_heat, best_params, output_dir):
    """Plot experimental data and best fit"""
    plt.figure(figsize=(10, 6))
    
    # Plot experimental data
    plt.scatter(exp_temp, exp_spec_heat, color='red', label='Experimental Data', zorder=5)
    
    # Sort calculated data by temperature
    sort_idx = np.argsort(calc_temp)
    calc_temp = calc_temp[sort_idx]
    calc_spec_heat = calc_spec_heat[sort_idx]
    
    # Plot calculated data
    plt.plot(calc_temp, calc_spec_heat, 'b-', 
             label=f'NLCE Fit (Jxx={best_params[0]:.3f}, Jyy={best_params[1]:.3f}, Jzz={best_params[2]:.3f})')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Specific Heat (J/mol·K)')
    plt.title('Specific Heat of Pr2Zr2O7: Experimental vs. NLCE Fit')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'specific_heat_fit.png')
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Plot saved to {plot_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Fit NLCE calculations to experimental specific heat data')
    
    # Input/output parameters
    parser.add_argument('--exp_data', type=str, default='specific_heat_Pr2Zr2O7.txt',
                        help='Path to experimental specific heat data')
    parser.add_argument('--output_dir', type=str, default='./fit_results',
                        help='Directory for output files')
    parser.add_argument('--work_dir', type=str, default='nlc_fit_work_dir',
                        help='Working directory for temporary files (default: create temp dir)')
    
    # Initial guess and bounds for parameters
    parser.add_argument('--initial_Jxx', type=float, default=0.0, help='Initial guess for Jxx coupling')
    parser.add_argument('--initial_Jyy', type=float, default=0.0, help='Initial guess for Jyy coupling')
    parser.add_argument('--initial_Jzz', type=float, default=1.0, help='Initial guess for Jzz coupling')
    parser.add_argument('--bound_min', type=float, default=-2.0, help='Lower bound for J parameters')
    parser.add_argument('--bound_max', type=float, default=2.0, help='Upper bound for J parameters')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=3, help='Maximum order for NLCE calculation')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], help='Field direction (x,y,z)')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')


    parser.add_argument('--skip_cluster_gen', action='store_false', help='Skip cluster generation step')
    parser.add_argument('--skip_ham_prep', action='store_true', help='Skip Hamiltonian preparation step')
    
    # Optimization parameters
    parser.add_argument('--method', type=str, default='L-BFGS-B', help='Optimization method')
    parser.add_argument('--max_iter', type=int, default=200 , help='Maximum number of iterations')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for convergence')
    
    # Temperature range for NLCE
    parser.add_argument('--temp_min', type=float, default=1.0, help='Minimum temperature for NLCE')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature for NLCE')


    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'nlc_fit.log')
    setup_logging(log_file)
    
    # Determine working directory
    temp_dir = None
    if args.work_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='nlc_fit_')
        work_dir = temp_dir
    else:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)


    # Generate clusters once before optimization starts to avoid redundant generation
    logging.info(f"Generating pyrochlore clusters up to order {args.max_order}")
    cluster_gen_cmd = [
        'python3',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_pyrochlore_clusters.py'),
        '--max_order', str(args.max_order),
        '--output_dir', work_dir+'/clusters_order_'+str(args.max_order)+'/',
    ]
    try:
        subprocess.run(cluster_gen_cmd, check=True)
        logging.info(f"Clusters successfully generated in {work_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error generating clusters: {e}")
        logging.error("Continuing without pre-generating clusters")
    
    try:
        # Load experimental data
        logging.info(f"Loading experimental data from {args.exp_data}")
        exp_temp, exp_spec_heat = load_experimental_data(args.exp_data)
        logging.info(f"Loaded {len(exp_temp)} data points")
        
        # Define fixed parameters for NLCE
        fixed_params = {
            "max_order": args.max_order,
            "h": args.h,
            "field_dir": args.field_dir,
            "temp_bins": args.temp_bins,
            "skip_cluster_gen": args.skip_cluster_gen,
            "skip_ham_prep": args.skip_ham_prep,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max
        }
        
        # Filter experimental data based on temperature range
        valid_indices = (exp_temp >= args.temp_min) & (exp_temp <= args.temp_max)
        exp_temp = exp_temp[valid_indices]
        exp_spec_heat = exp_spec_heat[valid_indices]
        logging.info(f"Filtered to {len(exp_temp)} data points within temperature range [{args.temp_min}, {args.temp_max}]")

        # Check if any data points remain
        if len(exp_temp) == 0:
            logging.error(f"No experimental data points within temperature range [{args.temp_min}, {args.temp_max}]")
            sys.exit(1)

        # Initial guess for parameters
        initial_params = [args.initial_Jxx, args.initial_Jyy, args.initial_Jzz]
        
        # Parameter bounds
        bounds = [(args.bound_min, args.bound_max) for _ in range(3)]
        
        # Perform optimization
        logging.info(f"Starting optimization with method {args.method}")
        logging.info(f"Initial parameters: Jxx={initial_params[0]}, Jyy={initial_params[1]}, Jzz={initial_params[2]}")
        
        result = minimize(
            calc_chi_squared,
            initial_params,
            args=(fixed_params, exp_temp, exp_spec_heat, work_dir),
            method=args.method,
            bounds=bounds if args.method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
            options={'maxiter': args.max_iter, 'disp': True, 'ftol': args.tolerance}
        )
        
        best_params = result.x
        logging.info(f"Optimization finished: {result.message}")
        logging.info(f"Best parameters: Jxx={best_params[0]:.4f}, Jyy={best_params[1]:.4f}, Jzz={best_params[2]:.4f}")
        logging.info(f"Final chi-squared: {result.fun:.4f}")
        
        # Run NLCE with best parameters to get final fit
        calc_temp, calc_spec_heat = run_nlce(best_params, fixed_params, exp_temp, work_dir)
        
        # Plot results
        plot_results(exp_temp, exp_spec_heat, calc_temp, calc_spec_heat, best_params, args.output_dir)
        
        # Save best parameters
        params_file = os.path.join(args.output_dir, 'best_parameters.txt')
        with open(params_file, 'w') as f:
            f.write(f"# Best-fit parameters for Pr2Zr2O7 specific heat\n")
            f.write(f"Jxx = {best_params[0]}\n")
            f.write(f"Jyy = {best_params[1]}\n")
            f.write(f"Jzz = {best_params[2]}\n")
        
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None and os.path.exists(temp_dir):
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    logging.info("Fitting completed successfully!")

if __name__ == "__main__":
    main()