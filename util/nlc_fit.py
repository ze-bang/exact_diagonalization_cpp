import os
import sys
import subprocess
import argparse
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, basinhopping, dual_annealing
from scipy.stats import qmc
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Bayesian optimization imports
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Install with: pip install scikit-optimize")
    print("Bayesian optimization methods will not be available.")
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

def apply_gaussian_broadening(temp, spec_heat, sigma):
    """Apply Gaussian broadening to specific heat data"""
    if sigma <= 0:
        return spec_heat
    
    # Convert temperature to log space for more uniform broadening
    log_temp = np.log(temp)
    dt = np.mean(np.diff(log_temp))
    
    # Apply Gaussian filter in log-temperature space
    # sigma_pixels converts sigma from log-temperature units to array indices
    sigma_pixels = sigma / dt
    
    # Apply broadening
    broadened_spec_heat = gaussian_filter1d(spec_heat, sigma=sigma_pixels, mode='nearest')
    
    return broadened_spec_heat

def run_nlce(params, fixed_params, exp_temp, work_dir, h_field=None):
    """Run NLCE with the given parameters and return the calculated specific heat"""
    # Extract J parameters (first 3) - other parameters are handled in calc_chi_squared
    Jxx, Jyy, Jzz = params[:3]
    
    # Use provided h_field if given, otherwise use the one from fixed_params
    h_value = h_field if h_field is not None else fixed_params["h"]
    
    # Apply g_renorm scaling if it's being fitted (4th parameter if not fitting broadening, or after broadening params)
    if fixed_params.get("fit_g_renorm", False):
        n_datasets = fixed_params.get("n_datasets", 1)
        fit_broadening = fixed_params.get("fit_broadening", False)
        
        if fit_broadening:
            # g_renorm is after J params and sigma params
            g_renorm_idx = 3 + n_datasets
        else:
            # g_renorm is the 4th parameter (index 3)
            g_renorm_idx = 3
        
        if len(params) > g_renorm_idx:
            g_renorm = params[g_renorm_idx]
            h_value *= g_renorm
    
    # Create command for nlce.py
    if fixed_params["ED_method"] == 'FULL' or fixed_params["ED_method"] == 'OSS':
        cmd = [
            'python3', 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlce.py'),
            '--max_order', str(fixed_params["max_order"]),
            '--Jxx', f'{Jxx:.12f}',
            '--Jyy', f'{Jyy:.12f}',
            '--Jzz', f'{Jzz:.12f}',
            '--h', f'{h_value:.12f}',
            '--ed_executable', str(fixed_params["ED_path"]),
            '--field_dir', f'{fixed_params["field_dir"][0]:.12f}', f'{fixed_params["field_dir"][1]:.12f}', f'{fixed_params["field_dir"][2]:.12f}',
            '--base_dir', work_dir,
            '--temp_min', f'{fixed_params["temp_min"]:.8f}',
            '--temp_max', f'{fixed_params["temp_max"]:.8f}',
            '--temp_bins', str(fixed_params["temp_bins"]),
            '--thermo',
            '--SI_units',
            '--symmetrized'
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
    
    # Extract J parameters and sigma parameters
    n_datasets = len(exp_datasets)
    Jxx, Jyy, Jzz = params[:3]
    
    # Extract sigma parameters if fitting broadening
    fit_broadening = fixed_params.get("fit_broadening", False)
    if fit_broadening:
        sigmas = params[3:3+n_datasets] if len(params) > 3 else [0.0] * n_datasets
    else:
        sigmas = [2, 1, 1, 1]
        # sigmas = [0, 0, 0, 0]
    
    # Extract g_renorm parameter if fitting
    fit_g_renorm = fixed_params.get("fit_g_renorm", False)
    if fit_g_renorm:
        if fit_broadening:
            g_renorm_idx = 3 + n_datasets
        else:
            g_renorm_idx = 3
        g_renorm = params[g_renorm_idx] if len(params) > g_renorm_idx else 1.0
    else:
        g_renorm = 1.0
    
    for i, dataset in enumerate(exp_datasets):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        weight = dataset['weight']
        sigma = sigmas[i] if i < len(sigmas) else 0.0
        
        # Run NLCE with dataset-specific h field (g_renorm scaling handled in run_nlce)
        calc_temp, calc_spec_heat = run_nlce(params, fixed_params, exp_temp, work_dir, h_field=h_field)
        
        # Apply Gaussian broadening if sigma > 0
        if len(calc_spec_heat) > 0 and sigma > 0:
            calc_spec_heat = apply_gaussian_broadening(calc_temp, calc_spec_heat, sigma)
        
        # Interpolate calculated data to match experimental temperature points
        calc_interp = interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp)
        
        # Calculate chi-squared for this dataset
        dataset_chi_squared = np.sum(((exp_spec_heat - calc_interp) / (exp_spec_heat + 1e-10)) ** 2)
        
        # Apply weight and add to total
        weighted_chi_squared = weight * dataset_chi_squared
        total_chi_squared += weighted_chi_squared
        
        logging.info(f"Dataset {i+1} (h={h_field}, σ={sigma:.4f}): Chi-squared={dataset_chi_squared:.4f}, Weighted={weighted_chi_squared:.4f}")
    
    param_str = f"Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}"
    if fit_broadening and len(params) > 3:
        sigma_str = ", ".join([f"σ{i+1}={s:.4f}" for i, s in enumerate(sigmas)])
        param_str += f", {sigma_str}"
    if fit_g_renorm:
        param_str += f", g_renorm={g_renorm:.4f}"
    logging.info(f"Parameters: {param_str}, Total Chi-squared={total_chi_squared:.4f}")
    
    return total_chi_squared

def plot_results(exp_datasets, fixed_params, best_params, work_dir, output_dir):
    """Plot experimental data and best fit for all datasets"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_datasets)))
    n_datasets = len(exp_datasets)
    fit_broadening = fixed_params.get("fit_broadening", False)
    
    if fit_broadening:
        sigmas = best_params[3:3+n_datasets] if len(best_params) > 3 else [0.0] * n_datasets
    else:
        sigmas = [2, 1, 1, 1]
    
    # Extract g_renorm if fitted
    fit_g_renorm = fixed_params.get("fit_g_renorm", False)
    if fit_g_renorm:
        if fit_broadening:
            g_renorm_idx = 3 + n_datasets
        else:
            g_renorm_idx = 3
        g_renorm = best_params[g_renorm_idx] if len(best_params) > g_renorm_idx else 1.0
    else:
        g_renorm = 1.0
    
    for i, (dataset, color) in enumerate(zip(exp_datasets, colors)):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        sigma = sigmas[i] if i < len(sigmas) else 0.0
        
        # Plot experimental data
        plt.scatter(exp_temp, exp_spec_heat, color=color, alpha=0.7, 
                   label=f'Exp Data (h={h_field})', zorder=5)
        
        # Calculate and plot fitted data
        calc_temp, calc_spec_heat = run_nlce(best_params, fixed_params, exp_temp, work_dir, h_field=h_field)
        
        if len(calc_temp) > 0:
            # Apply Gaussian broadening if sigma > 0
            if sigma > 0:
                calc_spec_heat = apply_gaussian_broadening(calc_temp, calc_spec_heat, sigma)
            
            # Sort calculated data by temperature
            sort_idx = np.argsort(calc_temp)
            calc_temp = calc_temp[sort_idx]
            calc_spec_heat = calc_spec_heat[sort_idx]
            
            # Plot calculated data
            label_str = f'NLCE Fit (h={h_field}'
            if sigma > 0:
                label_str += f', σ={sigma:.3f}'
            if fit_g_renorm:
                label_str += f', g={g_renorm:.3f}'
            label_str += ')'
            
            plt.plot(calc_temp, calc_spec_heat, color=color, linestyle='-', linewidth=2,
                    label=label_str)
    
    title_str = f'Specific Heat Fit: Jxx={best_params[0]:.3f}, Jyy={best_params[1]:.3f}, Jzz={best_params[2]:.3f}'
    if fit_g_renorm:
        title_str += f', g_renorm={g_renorm:.3f}'
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Specific Heat (J/mol·K)')
    plt.title(title_str)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'specific_heat_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_file}")
    
    plt.close()

def multi_start_optimization(obj_func, initial_params, bounds, n_starts=10, method='SLSQP', 
                            constraints=None, args=(), **kwargs):
    """
    Perform multi-start optimization with different initial conditions
    """
    best_result = None
    best_score = np.inf
    all_results = []
    
    # Generate initial points using Latin Hypercube Sampling for better coverage
    sampler = qmc.LatinHypercube(d=len(initial_params), seed=42)
    sample = sampler.random(n=n_starts)
    
    # Scale to bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
    
    logging.info(f"Starting multi-start optimization with {n_starts} different initial conditions using {method}")
    
    for i, start_point in enumerate(scaled_samples):
        logging.info(f"Multi-start iteration {i+1}/{n_starts}: Starting from {start_point}")
        
        try:
            result = minimize(
                obj_func,
                start_point,
                args=args,
                method=method,
                bounds=bounds,
                constraints=constraints,
                **kwargs
            )
            
            all_results.append(result)
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                logging.info(f"New best result found: chi-squared = {best_score:.4f}")
                
        except Exception as e:
            logging.warning(f"Multi-start iteration {i+1} failed: {e}")
            continue
    
    logging.info(f"Multi-start optimization completed. Best chi-squared: {best_score:.4f}")
    return best_result, all_results

def basin_hopping_optimization(obj_func, initial_params, bounds, args=(), **kwargs):
    """
    Basin hopping optimization to escape local minima
    """
    logging.info("Starting basin hopping optimization")
    
    # Define bounds as a callable for basin hopping
    def bounds_func(x):
        for i, (low, high) in enumerate(bounds):
            if x[i] < low or x[i] > high:
                return False
        return True
    
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "args": args,
        "options": kwargs.get('options', {})
    }
    
    result = basinhopping(
        obj_func,
        initial_params,
        minimizer_kwargs=minimizer_kwargs,
        niter=kwargs.get('niter', 100),
        accept_test=bounds_func,
        seed=42
    )
    
    logging.info(f"Basin hopping completed. Best chi-squared: {result.fun:.4f}")
    return result

def differential_evolution_optimization(obj_func, bounds, args=(), **kwargs):
    """
    Differential evolution global optimization
    """
    logging.info("Starting differential evolution optimization")
    
    result = differential_evolution(
        obj_func,
        bounds,
        args=args,
        maxiter=kwargs.get('maxiter', 1000),
        popsize=kwargs.get('popsize', 15),
        atol=kwargs.get('atol', 0.01),
        seed=42,
        disp=True
    )
    
    logging.info(f"Differential evolution completed. Best chi-squared: {result.fun:.4f}")
    return result

def dual_annealing_optimization(obj_func, bounds, args=(), **kwargs):
    """
    Dual annealing (simulated annealing variant) optimization
    """
    logging.info("Starting dual annealing optimization")
    
    result = dual_annealing(
        obj_func,
        bounds,
        args=args,
        maxiter=kwargs.get('maxiter', 1000),
        seed=42
    )
    
    logging.info(f"Dual annealing completed. Best chi-squared: {result.fun:.4f}")
    return result

def bayesian_optimization(obj_func, bounds, args=(), method='gp', **kwargs):
    """
    Bayesian optimization using Gaussian Processes, Random Forest, or Gradient Boosting
    """
    if not BAYESIAN_OPT_AVAILABLE:
        raise ImportError("scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
    
    logging.info(f"Starting Bayesian optimization using {method}")
    
    # Convert bounds to skopt format
    dimensions = [Real(low, high, name=f'param_{i}') for i, (low, high) in enumerate(bounds)]
    
    # Define objective function for skopt (which expects named parameters)
    @use_named_args(dimensions)
    def objective(**params):
        # Convert named parameters back to array
        param_array = np.array([params[f'param_{i}'] for i in range(len(bounds))])
        return obj_func(param_array, *args)
    
    # Set optimization parameters
    n_calls = kwargs.get('n_calls', 100)
    n_initial_points = kwargs.get('n_initial_points', 10)
    acq_func = kwargs.get('acq_func', 'EI')  # Expected Improvement
    
    # Choose the optimization method
    if method == 'gp':
        # Gaussian Process
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=42,
            verbose=True
        )
    elif method == 'forest':
        # Random Forest
        result = forest_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=42,
            verbose=True
        )
    elif method == 'gbrt':
        # Gradient Boosting Regression Trees
        result = gbrt_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=42,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown Bayesian optimization method: {method}")
    
    logging.info(f"Bayesian optimization ({method}) completed. Best chi-squared: {result.fun:.4f}")
    logging.info(f"Best parameters: {result.x}")
    logging.info(f"Total function evaluations: {len(result.func_vals)}")
    
    return result

def adaptive_bayesian_optimization(obj_func, bounds, args=(), **kwargs):
    """
    Adaptive Bayesian optimization that switches acquisition functions based on progress
    """
    if not BAYESIAN_OPT_AVAILABLE:
        raise ImportError("scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
    
    logging.info("Starting adaptive Bayesian optimization")
    
    # Convert bounds to skopt format
    dimensions = [Real(low, high, name=f'param_{i}') for i, (low, high) in enumerate(bounds)]
    
    @use_named_args(dimensions)
    def objective(**params):
        param_array = np.array([params[f'param_{i}'] for i in range(len(bounds))])
        return obj_func(param_array, *args)
    
    n_calls = kwargs.get('n_calls', 100)
    n_initial_points = kwargs.get('n_initial_points', 15)
    
    # Phase 1: Exploration with Expected Improvement
    logging.info("Phase 1: Exploration phase using Expected Improvement")
    result1 = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls // 2,
        n_initial_points=n_initial_points,
        acq_func='EI',
        random_state=42,
        verbose=True
    )
    
    # Phase 2: Exploitation with Lower Confidence Bound
    logging.info("Phase 2: Exploitation phase using Lower Confidence Bound")
    result2 = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls // 2,
        x0=result1.x_iters,
        y0=result1.func_vals,
        acq_func='LCB',
        random_state=42,
        verbose=True
    )
    
    # Return the best overall result
    if result1.fun < result2.fun:
        best_result = result1
        logging.info("Best result found in exploration phase")
    else:
        best_result = result2
        logging.info("Best result found in exploitation phase")
    
    logging.info(f"Adaptive Bayesian optimization completed. Best chi-squared: {best_result.fun:.4f}")
    return best_result

def multi_objective_bayesian_optimization(obj_func, bounds, args=(), **kwargs):
    """
    Multi-fidelity Bayesian optimization using different surrogate models
    """
    if not BAYESIAN_OPT_AVAILABLE:
        raise ImportError("scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
    
    logging.info("Starting multi-fidelity Bayesian optimization")
    
    results = {}
    methods = ['gp', 'forest', 'gbrt']
    n_calls_per_method = kwargs.get('n_calls', 100) // len(methods)
    
    for method in methods:
        try:
            result = bayesian_optimization(
                obj_func, bounds, args=args, method=method,
                n_calls=n_calls_per_method,
                n_initial_points=kwargs.get('n_initial_points', 10),
                **kwargs
            )
            results[method] = result
        except Exception as e:
            logging.warning(f"Bayesian optimization with {method} failed: {e}")
    
    if not results:
        logging.error("All Bayesian optimization methods failed!")
        return None
    
    # Find the best result
    best_method = min(results.keys(), key=lambda k: results[k].fun)
    best_result = results[best_method]
    
    logging.info(f"Best Bayesian method: {best_method} with chi-squared: {best_result.fun:.4f}")
    
    # Log all results for comparison
    for method_name, result in results.items():
        logging.info(f"{method_name}: chi-squared = {result.fun:.4f}, params = {result.x}")
    
    return best_result

def robust_optimization(obj_func, initial_params, bounds, method='auto', constraints=None, args=(), **kwargs):
    """
    Robust optimization combining multiple algorithms
    """
    results = {}
    
    if method == 'auto' or method == 'multi_start':
        try:
            result, all_results = multi_start_optimization(
                obj_func, initial_params, bounds, 
                n_starts=kwargs.get('n_starts', 10),
                constraints=constraints, args=args
            )
            results['multi_start'] = result
        except Exception as e:
            logging.warning(f"Multi-start optimization failed: {e}")
    
    if method == 'auto' or method == 'differential_evolution':
        try:
            result = differential_evolution_optimization(obj_func, bounds, args=args, **kwargs)
            results['differential_evolution'] = result
        except Exception as e:
            logging.warning(f"Differential evolution failed: {e}")
    
    if method == 'auto' or method == 'basin_hopping':
        try:
            result = basin_hopping_optimization(obj_func, initial_params, bounds, args=args, **kwargs)
            results['basin_hopping'] = result
        except Exception as e:
            logging.warning(f"Basin hopping failed: {e}")
    
    if method == 'auto' or method == 'dual_annealing':
        try:
            result = dual_annealing_optimization(obj_func, bounds, args=args, **kwargs)
            results['dual_annealing'] = result
        except Exception as e:
            logging.warning(f"Dual annealing failed: {e}")
    
    # Bayesian optimization methods
    if BAYESIAN_OPT_AVAILABLE:
        if method == 'auto' or method == 'bayesian_gp':
            try:
                result = bayesian_optimization(obj_func, bounds, args=args, method='gp', **kwargs)
                results['bayesian_gp'] = result
            except Exception as e:
                logging.warning(f"Bayesian optimization (GP) failed: {e}")
        
        if method == 'auto' or method == 'bayesian_forest':
            try:
                result = bayesian_optimization(obj_func, bounds, args=args, method='forest', **kwargs)
                results['bayesian_forest'] = result
            except Exception as e:
                logging.warning(f"Bayesian optimization (Forest) failed: {e}")
        
        if method == 'auto' or method == 'adaptive_bayesian':
            try:
                result = adaptive_bayesian_optimization(obj_func, bounds, args=args, **kwargs)
                results['adaptive_bayesian'] = result
            except Exception as e:
                logging.warning(f"Adaptive Bayesian optimization failed: {e}")
        
        if method == 'auto' or method == 'multi_fidelity_bayesian':
            try:
                result = multi_objective_bayesian_optimization(obj_func, bounds, args=args, **kwargs)
                results['multi_fidelity_bayesian'] = result
            except Exception as e:
                logging.warning(f"Multi-fidelity Bayesian optimization failed: {e}")
    else:
        if method in ['bayesian_gp', 'bayesian_forest', 'adaptive_bayesian', 'multi_fidelity_bayesian']:
            logging.error("Bayesian optimization methods require scikit-optimize. Install with: pip install scikit-optimize")
    
    # Return the best result
    if not results:
        logging.error("All optimization methods failed!")
        return None
    
    best_method = min(results.keys(), key=lambda k: results[k].fun)
    best_result = results[best_method]
    
    logging.info(f"Best optimization method: {best_method} with chi-squared: {best_result.fun:.4f}")
    
    # Log all results for comparison
    for method_name, result in results.items():
        logging.info(f"{method_name}: chi-squared = {result.fun:.4f}, params = {result.x}")
    
    return best_result

def analyze_optimization_landscape(obj_func, best_params, bounds, args=(), n_samples=1000):
    """
    Analyze the optimization landscape around the best solution
    """
    logging.info("Analyzing optimization landscape around best solution")
    
    # Generate random samples around the best solution
    param_names = ['Jxx', 'Jyy', 'Jzz']
    landscape_data = []
    
    for i in range(n_samples):
        # Generate random perturbations
        perturbation = np.random.normal(0, 0.1, len(best_params))
        test_params = best_params + perturbation
        
        # Ensure parameters stay within bounds
        for j, (low, high) in enumerate(bounds):
            test_params[j] = np.clip(test_params[j], low, high)
        
        try:
            chi_squared = obj_func(test_params, *args)
            landscape_data.append({
                'params': test_params.copy(),
                'chi_squared': chi_squared
            })
        except:
            continue
    
    if landscape_data:
        # Sort by chi-squared
        landscape_data.sort(key=lambda x: x['chi_squared'])
        
        logging.info("Top 10 parameter combinations:")
        for i, data in enumerate(landscape_data[:10]):
            params = data['params']
            chi_sq = data['chi_squared']
            logging.info(f"  {i+1}: Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}, χ²={chi_sq:.4f}")
    
    return landscape_data

def plot_bayesian_convergence(result, output_dir):
    """
    Plot the convergence of Bayesian optimization
    """
    if not BAYESIAN_OPT_AVAILABLE or not hasattr(result, 'func_vals'):
        logging.warning("Cannot plot Bayesian convergence: either skopt not available or result doesn't contain function values")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot convergence
        plt.subplot(2, 2, 1)
        iterations = range(1, len(result.func_vals) + 1)
        plt.plot(iterations, result.func_vals, 'b-', alpha=0.7)
        plt.scatter(iterations, result.func_vals, c=range(len(result.func_vals)), 
                   cmap='viridis', s=20, alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Chi-squared')
        plt.title('Bayesian Optimization Convergence')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative minimum
        plt.subplot(2, 2, 2)
        cumulative_min = np.minimum.accumulate(result.func_vals)
        plt.plot(iterations, cumulative_min, 'r-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Chi-squared So Far')
        plt.title('Cumulative Best Result')
        plt.grid(True, alpha=0.3)
        
        # Plot parameter evolution (first 3 parameters only)
        param_names = ['Jxx', 'Jyy', 'Jzz']
        colors = ['red', 'green', 'blue']
        
        plt.subplot(2, 2, 3)
        for i, (name, color) in enumerate(zip(param_names[:3], colors)):
            if i < len(result.x_iters[0]):
                param_values = [x[i] for x in result.x_iters]
                plt.plot(iterations, param_values, color=color, label=name, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot acquisition function behavior (if available)
        plt.subplot(2, 2, 4)
        if hasattr(result, 'func_vals') and len(result.func_vals) > 10:
            # Show the improvement over iterations
            improvements = []
            current_best = result.func_vals[0]
            for val in result.func_vals:
                if val < current_best:
                    improvements.append(current_best - val)
                    current_best = val
                else:
                    improvements.append(0)
            
            plt.bar(range(len(improvements)), improvements, alpha=0.7, color='green')
            plt.xlabel('Iteration')
            plt.ylabel('Improvement')
            plt.title('Per-Iteration Improvement')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, 'bayesian_optimization_convergence.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Bayesian convergence plot saved to {plot_file}")
        plt.close()
        
    except ImportError:
        logging.warning("Matplotlib not available for plotting Bayesian convergence")
    except Exception as e:
        logging.warning(f"Error creating Bayesian convergence plot: {e}")

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
    parser.add_argument('--initial_sigma', type=float, default=0.1, help='Initial guess for Gaussian broadening width')
    parser.add_argument('--initial_g_renorm', type=float, default=1.0, help='Initial guess for g-factor renormalization')
    parser.add_argument('--bound_min', type=float, default=-10.0, help='Lower bound for J parameters')
    parser.add_argument('--bound_max', type=float, default=10.0, help='Upper bound for J parameters')
    parser.add_argument('--sigma_bound_min', type=float, default=0.0, help='Lower bound for sigma parameters')
    parser.add_argument('--sigma_bound_max', type=float, default=10.0, help='Upper bound for sigma parameters')
    parser.add_argument('--g_renorm_bound_min', type=float, default=0.8, help='Lower bound for g_renorm parameter')
    parser.add_argument('--g_renorm_bound_max', type=float, default=1.2, help='Upper bound for g_renorm parameter')
    parser.add_argument('--fit_broadening', action='store_true', help='Include Gaussian broadening as fitting parameters')
    parser.add_argument('--fit_g_renorm', action='store_true', help='Include g-factor renormalization as fitting parameter')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=3, help='Maximum order for NLCE calculation')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[1, 1, 1]/np.sqrt(3), help='Field direction (x,y,z)')
    parser.add_argument('--temp_bins', type=int, default=1000, help='Number of temperature bins')


    parser.add_argument('--skip_cluster_gen', action='store_true', help='Skip cluster generation step')
    parser.add_argument('--skip_ham_prep', action='store_true', help='Skip Hamiltonian preparation step')
    
    # Optimization parameters
    parser.add_argument('--method', type=str, default='auto', 
                        choices=['auto', 'multi_start', 'differential_evolution', 'basin_hopping', 
                                'dual_annealing', 'bayesian_gp', 'bayesian_forest', 'adaptive_bayesian',
                                'multi_fidelity_bayesian', 'Nelder-Mead', 'L-BFGS-B', 'SLSQP', 'COBYLA'],
                        help='Optimization method (auto tries multiple global methods including Bayesian)')
    parser.add_argument('--n_starts', type=int, default=10, help='Number of random starts for multi-start optimization')
    parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
    parser.add_argument('--n_calls', type=int, default=100, help='Number of function calls for Bayesian optimization')
    parser.add_argument('--n_initial_points', type=int, default=10, help='Number of initial points for Bayesian optimization')
    parser.add_argument('--acq_func', type=str, default='EI', choices=['EI', 'LCB', 'PI'],
                        help='Acquisition function for Bayesian optimization (EI=Expected Improvement, LCB=Lower Confidence Bound, PI=Probability of Improvement)')
    parser.add_argument('--ED_method', type=str, default='FULL', help='ED method for NLCE')
    parser.add_argument('--max_iter', type=int, default=5000 , help='Maximum number of iterations')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for convergence')
    
    # Temperature range for NLCE
    parser.add_argument('--temp_min', type=float, default=1.0, help='Minimum temperature for NLCE')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature for NLCE')

    # Plotting options
    parser.add_argument('--plot_only', action='store_true', help='Only plot initial parameters without optimization')

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
        "ED_path": args.ed_executable,
        "fit_broadening": args.fit_broadening,
        "fit_g_renorm": args.fit_g_renorm
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
        
        # Store number of datasets in fixed_params for parameter indexing
        fixed_params["n_datasets"] = len(exp_datasets)

        # Initial guess for parameters
        initial_params = [args.initial_Jxx, args.initial_Jyy, args.initial_Jzz]
        
        # Add sigma parameters if fitting broadening
        if args.fit_broadening:
            initial_params.extend([args.initial_sigma] * len(exp_datasets))
        
        # Add g_renorm parameter if fitting
        if args.fit_g_renorm:
            initial_params.append(args.initial_g_renorm)
        
        # Parameter bounds
        bounds = [(args.bound_min, args.bound_max),
                  (args.bound_min, args.bound_max),
                  (args.bound_min, args.bound_max)]
        
        # Add sigma bounds if fitting broadening
        if args.fit_broadening:
            bounds.extend([(args.sigma_bound_min, args.sigma_bound_max)])
            bounds.extend([(args.sigma_bound_min/2, args.sigma_bound_max/2)] * (len(exp_datasets)-1))

        # Add g_renorm bounds if fitting
        if args.fit_g_renorm:
            bounds.append((args.g_renorm_bound_min, args.g_renorm_bound_max))
        
        def constraint_func(params):
            Jxx, Jyy, Jzz = params[:3]
            # Return array of constraint values (should be >= 0)
            return np.array([
                0.125*Jzz - Jxx,
                0.2*Jzz - Jyy + 0.4 * Jxx,
                Jyy + 0.4 * Jxx + 0.2*Jzz,
            ])
        
        constraints = NonlinearConstraint(constraint_func, 0, np.inf)

        if args.plot_only:
            # Skip optimization and just plot initial parameters
            logging.info("Plot-only mode: Using initial parameters without optimization")
            best_params = initial_params
            
            # Calculate chi-squared for initial parameters
            initial_chi_squared = calc_chi_squared(initial_params, fixed_params, exp_datasets, work_dir)
            logging.info(f"Initial parameters: Jxx={initial_params[0]:.4f}, Jyy={initial_params[1]:.4f}, Jzz={initial_params[2]:.4f}")
            if args.fit_broadening and len(initial_params) > 3:
                sigma_str = ", ".join([f"σ{i+1}={initial_params[3+i]:.4f}" for i in range(len(exp_datasets))])
                logging.info(f"Initial broadening parameters: {sigma_str}")
            if args.fit_g_renorm:
                if args.fit_broadening:
                    g_renorm_idx = 3 + len(exp_datasets)
                else:
                    g_renorm_idx = 3
                if len(initial_params) > g_renorm_idx:
                    logging.info(f"Initial g_renorm parameter: {initial_params[g_renorm_idx]:.4f}")
            logging.info(f"Initial chi-squared: {initial_chi_squared:.4f}")
            
            # Create a mock result object for consistency with plotting function
            class MockResult:
                def __init__(self, x, fun):
                    self.x = x
                    self.fun = fun
                    self.success = True
                    self.message = "Plot-only mode (no optimization performed)"
            
            result = MockResult(initial_params, initial_chi_squared)
        else:
            # Perform optimization
            logging.info(f"Starting optimization with method {args.method}")
            logging.info(f"Initial parameters: Jxx={initial_params[0]}, Jyy={initial_params[1]}, Jzz={initial_params[2]}")
            
            # Choose optimization strategy
            if args.method in ['auto', 'multi_start', 'differential_evolution', 'basin_hopping', 'dual_annealing',
                              'bayesian_gp', 'bayesian_forest', 'adaptive_bayesian', 'multi_fidelity_bayesian']:
                # Use robust global optimization
                result = robust_optimization(
                    calc_chi_squared,
                    initial_params,
                    bounds,
                    method=args.method,
                    constraints=constraints if args.method not in ['differential_evolution', 'dual_annealing'] and not args.method.startswith('bayesian') and args.method != 'adaptive_bayesian' and args.method != 'multi_fidelity_bayesian' else None,
                    args=(fixed_params, exp_datasets, work_dir),
                    maxiter=args.max_iter,
                    n_starts=args.n_starts,
                    popsize=args.popsize,
                    atol=args.tolerance,
                    n_calls=args.n_calls,
                    n_initial_points=args.n_initial_points,
                    acq_func=args.acq_func,
                    options={'disp': True, 'ftol': args.tolerance}
                )
            else:
                # Use traditional scipy.optimize.minimize
                result = minimize(
                    calc_chi_squared,
                    initial_params,
                    args=(fixed_params, exp_datasets, work_dir),
                    method=args.method,
                    bounds=bounds if args.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'COBYLA'] else None,
                    constraints=constraints if args.method in ['SLSQP', 'COBYLA', 'trust-constr'] else None,
                    options={'maxiter': args.max_iter, 'disp': True, 'ftol': args.tolerance}
                )
            
            if result is None:
                logging.error("Optimization failed!")
                sys.exit(1)
            
            # Handle different result formats (scipy vs skopt)
            if hasattr(result, 'message'):
                logging.info(f"Optimization finished: {result.message}")
            else:
                logging.info("Optimization finished successfully")

        best_params = result.x
        logging.info(f"Best parameters: Jxx={best_params[0]:.4f}, Jyy={best_params[1]:.4f}, Jzz={best_params[2]:.4f}")
        if args.fit_broadening and len(best_params) > 3:
            sigma_str = ", ".join([f"σ{i+1}={best_params[3+i]:.4f}" for i in range(len(exp_datasets))])
            logging.info(f"Best broadening parameters: {sigma_str}")
        if args.fit_g_renorm:
            if args.fit_broadening:
                g_renorm_idx = 3 + len(exp_datasets)
            else:
                g_renorm_idx = 3
            if len(best_params) > g_renorm_idx:
                logging.info(f"Best g_renorm parameter: {best_params[g_renorm_idx]:.4f}")
        logging.info(f"Final chi-squared: {result.fun:.4f}")
        
        # Analyze optimization landscape (only if optimization was performed)
        if not args.plot_only and args.method in ['auto', 'multi_start', 'differential_evolution', 'basin_hopping', 'dual_annealing',
                          'bayesian_gp', 'bayesian_forest', 'adaptive_bayesian', 'multi_fidelity_bayesian']:
            landscape_data = analyze_optimization_landscape(
                calc_chi_squared, 
                best_params, 
                bounds, 
                args=(fixed_params, exp_datasets, work_dir),
                n_samples=100  # Reduced for computational efficiency
            )
        
        # Plot results
        plot_results(exp_datasets, fixed_params, best_params, work_dir, args.output_dir)
        
        # Plot Bayesian optimization convergence if applicable
        if not args.plot_only and (args.method in ['bayesian_gp', 'bayesian_forest', 'adaptive_bayesian', 'multi_fidelity_bayesian'] or (args.method == 'auto' and BAYESIAN_OPT_AVAILABLE)):
            plot_bayesian_convergence(result, args.output_dir)
        
        # Save comprehensive results
        params_file = os.path.join(args.output_dir, 'best_parameters.txt')
        with open(params_file, 'w') as f:
            f.write(f"# Best-fit parameters for specific heat\n")
            f.write(f"# Optimization method: {args.method}\n")
            f.write(f"# Final chi-squared: {result.fun:.6f}\n")
            f.write(f"# Success: {result.success}\n")
            f.write(f"# Message: {result.message}\n")
            f.write(f"Jxx = {best_params[0]:.8f}\n")
            f.write(f"Jyy = {best_params[1]:.8f}\n")
            f.write(f"Jzz = {best_params[2]:.8f}\n")
            if args.fit_broadening and len(best_params) > 3:
                for i in range(len(exp_datasets)):
                    f.write(f"sigma_{i+1} = {best_params[3+i]:.8f}\n")
            if args.fit_g_renorm:
                if args.fit_broadening:
                    g_renorm_idx = 3 + len(exp_datasets)
                else:
                    g_renorm_idx = 3
                if len(best_params) > g_renorm_idx:
                    f.write(f"g_renorm = {best_params[g_renorm_idx]:.8f}\n")
            
        # Save detailed results
        results_file = os.path.join(args.output_dir, 'optimization_results.json')
        results_dict = {
            'method': args.method,
            'best_parameters': {
                'Jxx': float(best_params[0]),
                'Jyy': float(best_params[1]),
                'Jzz': float(best_params[2])
            },
            'chi_squared': float(result.fun),
            'success': bool(result.success),
            'message': str(result.message),
            'nfev': getattr(result, 'nfev', None),
            'nit': getattr(result, 'nit', None)
        }
        
        # Add Bayesian optimization specific results
        if hasattr(result, 'func_vals') and hasattr(result, 'x_iters'):
            results_dict['bayesian_details'] = {
                'total_evaluations': len(result.func_vals),
                'convergence_history': [float(x) for x in result.func_vals],
                'parameter_history': [[float(x) for x in params] for params in result.x_iters],
                'acquisition_function': args.acq_func if args.method.startswith('bayesian') or args.method == 'adaptive_bayesian' else None
            }
        
        # Add broadening parameters if they were fitted
        if args.fit_broadening and len(best_params) > 3:
            broadening_params = {}
            for i in range(len(exp_datasets)):
                broadening_params[f'sigma_{i+1}'] = float(best_params[3+i])
            results_dict['broadening_parameters'] = broadening_params
        
        # Add g_renorm parameter if it was fitted
        if args.fit_g_renorm:
            if args.fit_broadening:
                g_renorm_idx = 3 + len(exp_datasets)
            else:
                g_renorm_idx = 3
            if len(best_params) > g_renorm_idx:
                results_dict['g_renorm'] = float(best_params[g_renorm_idx])
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None and os.path.exists(temp_dir):
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    logging.info("Fitting completed successfully!")

if __name__ == "__main__":
    main()