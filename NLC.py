import numpy as np
from scipy.interpolate import interp1d
import os
import subprocess
import shutil
import matplotlib.pyplot as plt

def average_energy(eigenvalues, temperatures, kb=1.0):
    """
    Compute average energy and specific heat as a function of temperature.
    
    Parameters:
    -----------
    eigenvalues : array_like
        Array containing the full energy spectrum.
    temperatures : array_like
        Array of temperatures.
    kb : float, optional
        Boltzmann constant (default: 1.0).
    
    Returns:
    --------
    avg_energy : ndarray
        Average energy at each temperature.
    specific_heat : ndarray
        Specific heat at each temperature.
    """
    # Convert inputs to numpy arrays if they aren't already
    eigenvalues = np.asarray(eigenvalues)
    temperatures = np.asarray(temperatures)
    
    # Initialize arrays to store results
    avg_energy = np.zeros_like(temperatures, dtype=float)
    
    # Ground state energy
    e_min = np.min(eigenvalues)
    
    for i, T in enumerate(temperatures):
        if T == 0 or np.isclose(T, 0):  # Handle T=0 or very small T case
            avg_energy[i] = e_min  # Ground state energy
            specific_heat[i] = 0.0  # Specific heat is zero at T=0
        else:
            beta = 1.0 / (kb * T)
            
            # For numerical stability, shift eigenvalues by the minimum value
            shifted_eigenvalues = eigenvalues - e_min
            
            # Calculate Boltzmann factors and partition function
            boltzmann_factors = np.exp(-beta * shifted_eigenvalues)
            Z = np.sum(boltzmann_factors)
            
            # Calculate average energy (add back the minimum energy)
            avg_shifted_energy = np.sum(shifted_eigenvalues * boltzmann_factors) / Z
            avg_energy[i] = avg_shifted_energy + e_min
    return avg_energy


def specific_heat(eigenvalues, temperatures, kb=1.0):
    """
    Compute average energy and specific heat as a function of temperature.
    
    Parameters:
    -----------
    eigenvalues : array_like
        Array containing the full energy spectrum.
    temperatures : array_like
        Array of temperatures.
    kb : float, optional
        Boltzmann constant (default: 1.0).
    
    Returns:
    --------
    avg_energy : ndarray
        Average energy at each temperature.
    specific_heat : ndarray
        Specific heat at each temperature.
    """
    # Convert inputs to numpy arrays if they aren't already
    eigenvalues = np.asarray(eigenvalues)
    temperatures = np.asarray(temperatures)
    
    # Initialize arrays to store results
    avg_energy = np.zeros_like(temperatures, dtype=float)
    specific_heat = np.zeros_like(temperatures, dtype=float)
    
    # Ground state energy
    e_min = np.min(eigenvalues)
    
    kb = 1.380649e-23  # Boltzmann constant in J/K
    n_avogadro = 6.02214076e23  # Avogadro's number in mol^-1
    kb = kb * n_avogadro  # Convert to J/mol/K

    kb_meV = 0.00861733  # Boltzmann constant in meV/K

    for i, T in enumerate(temperatures):
        if T == 0 or np.isclose(T, 0):  # Handle T=0 or very small T case
            avg_energy[i] = e_min  # Ground state energy
            specific_heat[i] = 0.0  # Specific heat is zero at T=0
        else:
            beta = 1.0 / (kb_meV * T)
            
            # For numerical stability, shift eigenvalues by the minimum value
            shifted_eigenvalues = eigenvalues - e_min
            
            # Calculate Boltzmann factors and partition function
            boltzmann_factors = np.exp(-beta * shifted_eigenvalues)
            Z = np.sum(boltzmann_factors)
            
            # Calculate average energy (add back the minimum energy)
            avg_shifted_energy = np.sum(shifted_eigenvalues * boltzmann_factors) / Z
            
            # Calculate average energy squared (for the specific heat)
            avg_shifted_energy_squared = np.sum(shifted_eigenvalues**2 * boltzmann_factors) / Z
            
            # Calculate specific heat (this formula is invariant to energy shifts)
            specific_heat[i] =  (avg_shifted_energy_squared - avg_shifted_energy**2)/(kb*T)**2
    
    return specific_heat


# Cluster name, Subcluster name, Multiplicity
ClusterInfo = [["2", "1", 2], ["3", "1", 3], ["3", "2", 2], ["4a", "1", 4], ["4a", "2", 3], ["4a", "3", 3], ["4b", "1", 4], ["4b", "2", 3], ["4b", "3", 2], ["5a", "1", 5], ["5a", "2", 4], ["5a", "3", 6], ["5a", "4a", 4]]

def NLC_weight(observable, cluster_name, dir, temperatures):
    """
    Compute the NLC weight for a given observable.
    
    Parameters:
    -----------
    observable : str
        The observable for which to compute the NLC weight.
    cluster_size : int
        Size of the cluster.
    subcluster_size : int
        Size of the subcluster.
    multiplicity : int
        Multiplicity of the cluster.
    N : int
        Number of sites in the system.
    
    Returns:
    --------
    nlc_weight : float
        The NLC weight for the given observable.
    """
    
    if cluster_name == "1":
        eigenvalues = np.genfromtxt(dir + "/1/output/spectrum.dat", dtype=float)
        return observable(eigenvalues, temperatures)
    
    else:
        # Compute the NLC weight
        eigenvalues = np.genfromtxt(dir + "/" + cluster_name + "/output/spectrum.dat", dtype=float)

        nlc_weight = observable(eigenvalues, temperatures)
        for i in ClusterInfo:
            if i[0] == cluster_name:
                nlc_weight = nlc_weight - int(i[2])*NLC_weight(observable, i[1], dir, temperatures)
        
        return nlc_weight

    
def run_lanczos(dir):
    """
    Call the C++ lanczos program on specified subdirectories.
    
    Parameters:
    -----------
    dir : str
        Base directory path.
    
    Returns:
    --------
    bool
        True if all executions were successful, False otherwise.
    """
    
    # List of subdirectories to process
    subdirs = ['1', '2', '3', '4a', '4b']
    
    # Process each subdirectory
    for subdir in subdirs:
        full_path = os.path.join(dir, subdir)
        print(f"Running lanczos on {full_path}")
        
        try:
            # Run the lanczos program and wait for it to complete
            result = subprocess.run(
                ["./build/lanczos", full_path, "0"],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Lanczos successfully completed on {full_path}")
            print(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing lanczos on {full_path}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True
    


def NLC_compute(params, temperatures, dir):
    """
    Call helper_non_kramer script with specified parameters.
    
    Parameters:
    -----------
    Jxx : float
        X-axis coupling constant.
    Jyy : float
        Y-axis coupling constant.
    Jzz : float
        Z-axis coupling constant.
    dir : str
        Directory path.
    """
    
    # Clear directory if it exists
    if os.path.exists(dir):
        shutil.rmtree(dir)

    # Create the directory structure
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, '1/lanczos_eigenvectors'), exist_ok=True)
    os.makedirs(os.path.join(dir, '2/lanczos_eigenvectors'), exist_ok=True)
    os.makedirs(os.path.join(dir, '3/lanczos_eigenvectors'), exist_ok=True)
    os.makedirs(os.path.join(dir, '4a/lanczos_eigenvectors'), exist_ok=True)
    os.makedirs(os.path.join(dir, '4b/lanczos_eigenvectors'), exist_ok=True)

    # Convert parameters to strings
    jxx_str = str(params[0])
    jyy_str = str(params[1])
    jzz_str = str(params[2])
    
    # Call the helper script with arguments
    try:
        result = subprocess.run(
            ["python", "helper_non_kramers.py", jxx_str, jyy_str, jzz_str, "0", "1", "1", "1", dir+'/1/', "1"],
            check=True,
            capture_output=True,
            text=True
        )
        result = subprocess.run(
            ["python", "helper_non_kramers.py", jxx_str, jyy_str, jzz_str, "0", "1", "1", "1", dir+'/2/', "2"],
            check=True,
            capture_output=True,
            text=True
        )
        result = subprocess.run(
            ["python", "helper_non_kramers.py", jxx_str, jyy_str, jzz_str, "0", "1", "1", "1", dir+'/3/', "3"],
            check=True,
            capture_output=True,
            text=True
        )        
        result = subprocess.run(
            ["python", "helper_non_kramers.py", jxx_str, jyy_str, jzz_str, "0", "1", "1", "1", dir+'/4a/', "4a"],
            check=True,
            capture_output=True,
            text=True
        )
        result = subprocess.run(
            ["python", "helper_non_kramers.py", jxx_str, jyy_str, jzz_str, "0", "1", "1", "1", dir+'/4b/', "4b"],
            check=True,
            capture_output=True,
            text=True
        )
        print("helper_non_kramers.py executed successfully")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing helper_non_kramer.py: {e}")
        print("Error output:", e.stderr)
    
    run_lanczos(dir)
    nlc_sum = np.zeros_like(temperatures, dtype=float)
    for i in ClusterInfo:
        if i[0] == "5a":
            nlc_sum += NLC_weight(specific_heat, i[1], dir, temperatures)*int(i[2])
    return nlc_sum/16


import scipy.optimize as optimize

def load_reference_data(filename):
    """
    Load reference data from a file. This function assumes the data file
    contains temperature and specific heat values in two columns.
    
    Parameters:
    -----------
    filename : str
        Path to the data file
    
    Returns:
    --------
    temp_data : ndarray
        Temperature values
    heat_data : ndarray
        Specific heat values
    """
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # Assuming first column is temperature, second is specific heat

def fit_nlc_model(initial_params, reference_data_file, dir_path, 
                    temperatures, cluster_names=None, bounds=None):
    """
    Fit a model to specific heat data by minimizing the error.
    
    Parameters:
    -----------
    model_func : callable
        Function that takes parameters and returns a function to calculate specific heat
    initial_params : array_like
        Initial guess for parameters to optimize
    reference_data_file : str
        Path to file containing reference data
    dir_path : str
        Directory containing eigenvalue data
    temperatures : array_like
        Temperatures at which to calculate specific heat
    cluster_names : list, optional
        List of cluster names to include in NLC sum
    bounds : tuple, optional
        Bounds for parameters (min, max)
    
    Returns:
    --------
    opt_params : ndarray
        Optimized parameters
    min_error : float
        Minimum error achieved
    """
    # Load reference data
    temp_data, heat_data = load_reference_data(reference_data_file)
        
    def error_function(params):
        """Calculate error between model and reference data"""
        # Get specific heat function with current parameters
        
        nlc_sum = NLC_compute(params, temp_data, dir_path)

        # Calculate error using mean squared error
        # First interpolate model at reference temperatures

        # Calculate MSE
        mse = np.mean((nlc_sum - heat_data) ** 2)
        return mse
    
    # Perform optimization
    if bounds:
        result = optimize.minimize(error_function, initial_params, bounds=bounds)
    else:
        result = optimize.minimize(error_function, initial_params)
    
    opt_params = result.x
    min_error = result.fun
    
    print(f"Optimization complete. Minimum error: {min_error}")
    print(f"Optimal parameters: {opt_params}")
    
    return opt_params, min_error

def plot_fit_results(opt_params, reference_data_file, dir_path):
    """
    Plot the fitted results against reference data.
    
    Parameters:
    -----------
    opt_params : array_like
        Optimized parameters
    model_func : callable
        Function that takes parameters and returns a function to calculate specific heat
    reference_data_file : str
        Path to file containing reference data
    dir_path : str
        Directory containing eigenvalue data
    temperatures : array_like
        Temperatures at which to calculate specific heat
    cluster_names : list, optional
        List of cluster names to include in NLC sum
    """
    # Load reference data
    temp_data, heat_data = load_reference_data(reference_data_file)
    
    # Calculate fitted model    
    nlc_sum = NLC_compute(opt_params, temp_data, dir_path)

    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(temp_data, heat_data, 'o', label='Reference Data')
    plt.plot(temp_data, nlc_sum, '-', label='Fitted Model')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.xscale('log')
    plt.legend()
    plt.title('Fitted Specific Heat vs Reference Data')
    plt.grid(True)
    plt.savefig('fit_results.png')


def plot_spec_heat_from_file(dir, temperatures):
    A = np.genfromtxt(dir + "/output/spectrum.dat", dtype=float)
    print(A)
    E = specific_heat(A, temperatures)
    plt.plot(temperatures, E, label='Specific Heat')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.xscale('log')
    plt.legend()
    plt.title('Specific Heat vs Temperature')
    plt.grid(True)
    plt.savefig(dir+'specific_heat_plot.png')
    plt.show()


opt_params, min_error = fit_nlc_model(
    initial_params=[0.2, 0.2, 1.0],
    reference_data_file='specific_heat_Pr2Zr2O7.txt',
    dir_path='./data',
    temperatures=np.linspace(0.01, 10, 100),
    cluster_names=['1', '2', '3', '4a', '4b'],
    bounds=((0, None), (0, None), (0, None))
)

plot_fit_results(opt_params,'specific_heat_Pr2Zr2O7.txt', './data')

# plot_spec_heat_from_file("ED_XXZ_test/", np.logspace(-2, 1, 100))