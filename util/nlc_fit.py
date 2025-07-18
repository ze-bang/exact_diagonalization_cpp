import os
import sys
import subprocess
import argparse
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import interp1d
import logging
import tempfile
import shutil
import json

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

def load_multiple_experimental_data(exp_data_configs):
    """Load experimental data from multiple files with their respective parameters"""
    exp_datasets = []
    for config in exp_data_configs:
        temp, spec_heat = load_experimental_data(config['file'])
        exp_datasets.append({
            'temp': temp,
            'spec_heat': spec_heat,
            'h': config['h'],
            'field_dir': config.get('field_dir', [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
            'weight': config.get('weight', 1.0)
        })
    return exp_datasets

def run_nlce(params, fixed_params, exp_temp, work_dir, h_field=None):
    """Run NLCE with the given parameters and return the calculated specific heat"""
    Jxx, Jyy, Jzz = params
    
    # Use provided h_field if given, otherwise use the one from fixed_params
    h_value = h_field if h_field is not None else fixed_params["h"]
    
    # Create command for nlce.py
    if fixed_params["ED_method"] == 'FULL' or fixed_params["ED_method"] == 'OSS':
        cmd = [
            'python3', 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlce.py'),
            '--max_order', str(fixed_params["max_order"]),
            '--Jxx', str(Jxx),
            '--Jyy', str(Jyy),
            '--Jzz', str(Jzz),
            '--h', str(h_value),
            '--ed_executable', str(fixed_params["ED_path"]),
            '--field_dir', str(fixed_params["field_dir"][0]), str(fixed_params["field_dir"][1]), str(fixed_params["field_dir"][2]),
            '--base_dir', work_dir,
            '--temp_min', str(fixed_params["temp_min"]),
            '--temp_max', str(fixed_params["temp_max"]),
            '--temp_bins', str(fixed_params["temp_bins"]),
            '--thermo',
            '--SI_units'
        ]
    elif fixed_params["ED_method"] == 'mTPQ':
        cmd = [
            'python3', 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlce.py'),
            '--max_order', str(fixed_params["max_order"]),
            '--Jxx', str(Jxx),
            '--Jyy', str(Jyy),
            '--Jzz', str(Jzz),
            '--h', str(h_value),
            '--ed_executable', str(fixed_params["ED_path"]),
            '--field_dir', str(fixed_params["field_dir"][0]), str(fixed_params["field_dir"][1]), str(fixed_params["field_dir"][2]),
            '--base_dir', work_dir,
            '--temp_min', str(fixed_params["temp_min"]),
            '--temp_max', str(fixed_params["temp_max"]),
            '--temp_bins', str(fixed_params["temp_bins"]),
            '--thermo',
            '--SI_units',
            '--euler_resum',
            '--method=mTPQ'
        ]
    
    cmd.append('--skip_cluster_gen')
    if fixed_params.get("skip_ham_prep", False):
        cmd.append('--skip_ham_prep')
    
    if fixed_params.get("measure_spin", False):
        cmd.append('--measure_spin')
    
    

    
    try:
        logging.info(f"Running NLCE with Jxx={Jxx}, Jyy={Jyy}, Jzz={Jzz}, h={h_value}")
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
        logging.info("NLCE calculation failed, trying with lanczos")

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

def calc_chi_squared(params, fixed_params, exp_datasets, work_dir):
    """Calculate combined chi-squared between experimental and calculated specific heat for multiple datasets"""
    total_chi_squared = 0.0
    
    for i, dataset in enumerate(exp_datasets):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        weight = dataset['weight']
        
        # Run NLCE with dataset-specific h field
        calc_temp, calc_spec_heat = run_nlce(params, fixed_params, exp_temp, work_dir, h_field=h_field)
        
        # Interpolate calculated data to match experimental temperature points
        calc_interp = interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp)
        
        # Calculate chi-squared for this dataset
        dataset_chi_squared = np.sum(((exp_spec_heat - calc_interp) / (exp_spec_heat + 1e-10)) ** 2)
        
        # Apply weight and add to total
        weighted_chi_squared = weight * dataset_chi_squared
        total_chi_squared += weighted_chi_squared
        
        logging.info(f"Dataset {i+1} (h={h_field}): Chi-squared={dataset_chi_squared:.4f}, Weighted={weighted_chi_squared:.4f}")
    
    logging.info(f"Parameters: Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}, Total Chi-squared={total_chi_squared:.4f}")
    
    return total_chi_squared

def plot_results(exp_datasets, fixed_params, best_params, work_dir, output_dir):
    """Plot experimental data and best fit for all datasets"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_datasets)))
    
    for i, (dataset, color) in enumerate(zip(exp_datasets, colors)):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        
        # Plot experimental data
        plt.scatter(exp_temp, exp_spec_heat, color=color, alpha=0.7, 
                   label=f'Exp Data (h={h_field})', zorder=5)
        
        # Calculate and plot fitted data
        calc_temp, calc_spec_heat = run_nlce(best_params, fixed_params, exp_temp, work_dir, h_field=h_field)
        
        if len(calc_temp) > 0:
            # Sort calculated data by temperature
            sort_idx = np.argsort(calc_temp)
            calc_temp = calc_temp[sort_idx]
            calc_spec_heat = calc_spec_heat[sort_idx]
            
            # Plot calculated data
            plt.plot(calc_temp, calc_spec_heat, color=color, linestyle='-', linewidth=2,
                    label=f'NLCE Fit (h={h_field})')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Specific Heat (J/mol·K)')
    plt.title(f'Specific Heat Fit: Jxx={best_params[0]:.3f}, Jyy={best_params[1]:.3f}, Jzz={best_params[2]:.3f}')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'specific_heat_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Fit NLCE calculations to experimental specific heat data')
    
    # Input/output parameters
    parser.add_argument('--exp_data', type=str, default='specific_heat_Pr2Zr2O7.txt',
                        help='Path to experimental specific heat data (for single file mode)')
    parser.add_argument('--exp_config', type=str, default=None,
                        help='Path to JSON config file for multiple experimental datasets')
    parser.add_argument('--output_dir', type=str, default='./fit_results',
                        help='Directory for output files')
    parser.add_argument('--work_dir', type=str, default='nlc_fit_work_dir',
                        help='Working directory for temporary files (default: create temp dir)')
    parser.add_argument('--ed_executable', type=str, default='./build/ED',
                        help='Path to the ED executable')


    # Initial guess and bounds for parameters
    parser.add_argument('--initial_Jxx', type=float, default=0.0, help='Initial guess for Jxx coupling')
    parser.add_argument('--initial_Jyy', type=float, default=0.0, help='Initial guess for Jyy coupling')
    parser.add_argument('--initial_Jzz', type=float, default=1.0, help='Initial guess for Jzz coupling')
    parser.add_argument('--bound_min', type=float, default=-100.0, help='Lower bound for J parameters')
    parser.add_argument('--bound_max', type=float, default=100.0, help='Upper bound for J parameters')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=3, help='Maximum order for NLCE calculation')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[1, 1, 1]/np.sqrt(3), help='Field direction (x,y,z)')
    parser.add_argument('--temp_bins', type=int, default=1000, help='Number of temperature bins')


    parser.add_argument('--skip_cluster_gen', action='store_true', help='Skip cluster generation step')
    parser.add_argument('--skip_ham_prep', action='store_true', help='Skip Hamiltonian preparation step')
    
    # Optimization parameters
    parser.add_argument('--method', type=str, default='Nelder_Mead', help='Optimization method')
    parser.add_argument('--ED_method', type=str, default='FULL', help='ED method for NLCE')
    parser.add_argument('--max_iter', type=int, default=5000 , help='Maximum number of iterations')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for convergence')
    
    # Temperature range for NLCE
    parser.add_argument('--temp_min', type=float, default=1.0, help='Minimum temperature for NLCE')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature for NLCE')


    parser.add_argument('--measure_spin', action='store_true', help='Measure spin instead of specific heat')

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
    if not args.skip_cluster_gen:
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
    
    # Define fixed parameters for NLCE
    fixed_params = {
        "max_order": args.max_order,
        "h": args.h,
        "field_dir": args.field_dir,
        "temp_bins": args.temp_bins,
        "skip_ham_prep": args.skip_ham_prep,
        "temp_min": args.temp_min,
        "temp_max": args.temp_max,
        "measure_spin": args.measure_spin,
        "ED_method": args.ED_method,
        "ED_path": args.ed_executable
    }

    try:
        # Load experimental data
        if args.exp_config:
            # Load multiple datasets from config file
            logging.info(f"Loading experimental data configuration from {args.exp_config}")
            with open(args.exp_config, 'r') as f:
                config = json.load(f)
            
            exp_datasets = load_multiple_experimental_data(config['experimental_data'])
            logging.info(f"Loaded {len(exp_datasets)} experimental datasets")
            
            # Update fixed_params with any global config values
            if 'global_params' in config:
                fixed_params.update(config['global_params'])
        else:
            # Single dataset mode (backward compatibility)
            logging.info(f"Loading experimental data from {args.exp_data}")
            exp_temp, exp_spec_heat = load_experimental_data(args.exp_data)
            
            # Create single dataset
            exp_datasets = [{
                'temp': exp_temp,
                'spec_heat': exp_spec_heat,
                'h': args.h,
                'field_dir': args.field_dir,
                'weight': 1.0
            }]
            logging.info(f"Loaded {len(exp_temp)} data points")
        

        
        # Filter experimental data based on temperature range for all datasets
        filtered_datasets = []
        for dataset in exp_datasets:
            exp_temp = dataset['temp']
            exp_spec_heat = dataset['spec_heat']
            
            valid_indices = (exp_temp >= args.temp_min) & (exp_temp <= args.temp_max)
            filtered_temp = exp_temp[valid_indices]
            filtered_spec_heat = exp_spec_heat[valid_indices]
            
            if len(filtered_temp) > 0:
                filtered_datasets.append({
                    'temp': filtered_temp,
                    'spec_heat': filtered_spec_heat,
                    'h': dataset['h'],
                    'field_dir': dataset['field_dir'],
                    'weight': dataset['weight']
                })
                logging.info(f"Dataset h={dataset['h']}: Filtered to {len(filtered_temp)} data points within temperature range [{args.temp_min}, {args.temp_max}]")
            else:
                logging.warning(f"Dataset h={dataset['h']}: No data points within temperature range [{args.temp_min}, {args.temp_max}], skipping")
        
        if not filtered_datasets:
            logging.error("No experimental data points within temperature range for any dataset")
            sys.exit(1)
        
        exp_datasets = filtered_datasets

        # Initial guess for parameters
        initial_params = [args.initial_Jxx, args.initial_Jyy, args.initial_Jzz]
        
        # Parameter bounds
        bounds = [(args.bound_min, args.bound_max),
                  (args.bound_min, args.bound_max),
                  (args.bound_min, args.bound_max)]
        
        def constraint_func(params):
            Jxx, Jyy, Jzz = params
            # Return array of constraint values (should be >= 0)
            return np.array([
                0.125*Jzz - Jxx,
                0.2*Jzz - Jyy + 0.4 * Jxx,
                Jyy + 0.4 * Jxx + 0.2*Jzz,
            ])
        
        constraints = NonlinearConstraint(constraint_func, 0, np.inf)

        # Perform optimization
        logging.info(f"Starting optimization with method {args.method}")
        logging.info(f"Initial parameters: Jxx={initial_params[0]}, Jyy={initial_params[1]}, Jzz={initial_params[2]}")
        
        result = minimize(
            calc_chi_squared,
            initial_params,
            args=(fixed_params, exp_datasets, work_dir),
            method=args.method,
            bounds=bounds if args.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'COBYLA'] else None,
            constraints=constraints if args.method in ['SLSQP', 'COBYLA', 'trust-constr'] else None,
            options={'maxiter': args.max_iter, 'disp': True, 'ftol': args.tolerance}
        )
        
        best_params = result.x
        logging.info(f"Optimization finished: {result.message}")
        logging.info(f"Best parameters: Jxx={best_params[0]:.4f}, Jyy={best_params[1]:.4f}, Jzz={best_params[2]:.4f}")
        logging.info(f"Final chi-squared: {result.fun:.4f}")
        
        # Plot results
        plot_results(exp_datasets, fixed_params, best_params, work_dir, args.output_dir)
        
        # Save best parameters
        params_file = os.path.join(args.output_dir, 'best_parameters.txt')
        with open(params_file, 'w') as f:
            f.write(f"# Best-fit parameters for specific heat\n")
            f.write(f"Jxx = {best_params[0]}\n")
            f.write(f"Jyy = {best_params[1]}\n")
            f.write(f"Jzz = {best_params[2]}\n")
            f.write(f"# Final chi-squared: {result.fun:.4f}\n")
        
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None and os.path.exists(temp_dir):
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    logging.info("Fitting completed successfully!")

if __name__ == "__main__":
    main()