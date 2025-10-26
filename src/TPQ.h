// TPQ.h - Thermal Pure Quantum state implementation

#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include "blas_lapack_wrapper.h"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include "observables.h"
#include "construct_ham.h"
#include <memory>


#define GET_VARIABLE_NAME(Variable) (#Variable)

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, unsigned int seed);
/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists(const std::string& path);
/**
 * Calculate energy and variance for a TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    int N
);
std::vector<SingleSiteOperator> createSzOperators(int num_sites, float spin_length);
std::vector<SingleSiteOperator> createSxOperators(int num_sites, float spin_length);
std::vector<SingleSiteOperator> createSyOperators(int num_sites, float spin_length);

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    int sublattice_size
);

Complex calculateSpm_onsite(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Spm_ops,
    int sublattice_size
);

std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> createDoubleSiteOperators(int num_sites, float spin_length);

std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> createSingleOperators_pair(int num_sites, float spin_length);


std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    int sublattice_size
);

std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    int sublattice_size
);

/**
 * Write TPQ data to file
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, int step);

/**
 * Read TPQ data from file
 */
bool readTPQData(const std::string& filename, int step, double& energy, 
                double& temp, double& specificHeat);

/**
 * Time evolve TPQ state using Taylor expansion of exp(-iH*delta_t)
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_state(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int n_max = 100,
    bool normalize = true
);

/**
 * Create a time evolution operator using Taylor expansion of exp(-iH*delta_t)
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 * @return Function that applies time evolution to a complex vector
 */
std::function<void(const Complex*, Complex*, int)> create_time_evolution_operator(
    std::function<void(const Complex*, Complex*, int)> H,
    double delta_t,
    int n_max = 10,
    bool normalize = true
);

/**
 * Save the current TPQ state to a file
 * 
 * @param tpq_state TPQ state vector to save
 * @param filename Name of the file to save to
 * @return True if successful
 */
bool save_tpq_state(const ComplexVector& tpq_state, const std::string& filename);

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename);

/**
 * Load eigenvector data from a raw binary file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @param N Expected size of the vector
 * @return True if successful
 */
bool load_raw_data(ComplexVector& tpq_state, const std::string& filename, int N);

/**
 * Calculate spectral function from a TPQ state using real-time evolution
 * 
 * @param H Hamiltonian operator function
 * @param O Observable operator function
 * @param tpq_state Current TPQ state
 * @param N Dimension of the Hilbert space
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_points Number of frequency points
 * @param tmax Maximum evolution time
 * @param dt Time step
 * @param eta Broadening parameter
 * @param use_lorentzian Use Lorentzian (true) or Gaussian (false) broadening
 * @return Structure containing frequencies and spectral function values
 */
SpectralFunctionData calculate_spectral_function_from_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& tpq_state,
    int N,
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double tmax = 10.0,
    double dt = 0.01,
    double eta = 0.1,
    bool use_lorentzian = false,
    int n_max = 100 // Order of Taylor expansion
);

std::vector<std::vector<Complex>> calculate_spectral_function_from_tpq_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,   
    const ComplexVector& tpq_state,
    int N,
    const int num_steps
);

/**
 * Incremental version that writes time correlation data as computation happens
 */
void calculate_spectral_function_from_tpq_U_t_incremental(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,   
    const ComplexVector& tpq_state,
    int N,
    const int num_steps,
    double dt,
    std::vector<std::ofstream>& output_files
);

/**
 * Compute spin expectations (S^+, S^-, S^z) at each site using a TPQ state
 * 
 * @param tpq_state The TPQ state vector
 * @param num_sites Number of lattice sites
 * @param spin_l Spin value (e.g., 0.5 for spin-1/2)
 * @param output_file Output file path
 * @param print_output Whether to print results to console
 * @return Vector of spin expectation values organized as [site][S+,S-,Sz]
 */
std::vector<std::vector<Complex>> compute_spin_expectations_from_tpq(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_l,
    const std::string& output_file = "",
    bool print_output = true
);




void writeFluctuationData(
    const std::string& flct_file,
    const std::vector<std::string>& spin_corr,
    double inv_temp,
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sx_ops,
    const std::vector<SingleSiteOperator>& Sy_ops,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    int sublattice_size,
    int step
);

/**
 * Get a TPQ state at a specific inverse temperature by loading the closest available state
 * 
 * @param tpq_dir Directory containing TPQ data
 * @param sample TPQ sample index
 * @param target_beta Target inverse temperature
 * @param N Dimension of Hilbert space
 * @return TPQ state vector at the specified temperature
 */
ComplexVector get_tpq_state_at_temperature(
    const std::string& tpq_dir,
    int sample,
    double target_beta,
    int N
);

/**
 * Initialize TPQ output files with appropriate headers
 * 
 * @param dir Directory for output files
 * @param sample Current sample index
 * @param sublattice_size Size of sublattice for measurements
 * @return Tuple of filenames (ss_file, norm_file, flct_file, spin_corr)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeTPQFiles(
    const std::string& dir,
    int sample,
    int sublattice_size
);

/**
 * Compute and save dynamics for observables in TPQ evolution
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector
 * @param observables List of observables to compute
 * @param observable_names Names of the observables
 * @param N Dimension of the Hilbert space
 * @param dir Output directory
 * @param sample Current sample index
 * @param step Current TPQ step
 * @param omega_min Minimum frequency 
 * @param omega_max Maximum frequency
 * @param num_points Number of frequency points
 * @param t_end Maximum evolution time
 * @param dt Time step
 */
void computeObservableDynamics(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables,
    const std::vector<std::string>& observable_names,
    int N, 
    const std::string& dir,
    int sample,
    int step,
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.1
);

// Forward-time only evolution of observable correlations C_O(t)=<psi(t)|O^\u2020 O|psi(t)>, leveraging hermiticity.
// Removed negative time evolution to halve computational cost.
// Modified to write time correlation data incrementally during computation.
void computeObservableDynamics_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables_1,
    const std::vector<Operator>& observables_2,
    const std::vector<std::string>& observable_names,
    int N,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end = 100.0,
    double dt = 0.01
);

/**
 * Calculate spectrum function from TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param tpq_sample Sample index to use from TPQ calculation
 * @param tpq_step TPQ step to use
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param omega_step Step size in frequency domain
 * @param eta Broadening factor
 * @param tpq_dir Directory containing TPQ data
 * @param out_file Output file for spectrum
 */
void calculate_spectrum_from_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int tpq_sample,
    int tpq_step,
    double omega_min,
    double omega_max,
    double omega_step,
    double eta,
    const std::string& tpq_dir,
    const std::string& out_file
);

/**
 * Krylov-based time evolution using Lanczos method
 * This is much more accurate and stable than Taylor expansion
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param krylov_dim Dimension of Krylov subspace (typically 20-50)
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int krylov_dim = 30,
    bool normalize = true
);

/**
 * Chebyshev polynomial-based time evolution
 * Excellent for systems with bounded spectra
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param E_min Minimum eigenvalue of H (estimate)
 * @param E_max Maximum eigenvalue of H (estimate)
 * @param num_terms Number of Chebyshev polynomials to use
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_chebyshev(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    double E_min,
    double E_max,
    int num_terms = 100,
    bool normalize = true
);

/**
 * 4th-order Runge-Kutta time evolution
 * Best for time-dependent Hamiltonians or when high accuracy is needed
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_rk4(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    bool normalize = true
);

/**
 * Adaptive time evolution with automatic method selection
 * Chooses the best method based on system size and accuracy requirements
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector (will be modified)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param accuracy_level Accuracy level: 1=fast, 2=balanced, 3=high accuracy
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_tpq_adaptive(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int accuracy_level = 2,
    bool normalize = true
);

/**
 * Compute dynamical correlations using Krylov method with pre-constructed operators
 * 
 * This is the more general version that takes Operator objects directly for maximum versatility.
 * Computes the time correlation function C(t) = ⟨ψ|O_1†(0)O_2(t)|ψ⟩
 * using the Krylov-based time evolution method for high accuracy.
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector
 * @param operators_1 Vector of first Operator objects in correlation (applied at t=0)
 * @param operators_2 Vector of second Operator objects in correlation (applied at time t)
 * @param operator_names Names corresponding to each operator pair
 * @param N Dimension of the Hilbert space
 * @param dir Output directory
 * @param sample Current sample index
 * @param inv_temp Inverse temperature
 * @param t_end Maximum evolution time
 * @param dt Time step
 * @param krylov_dim Dimension of Krylov subspace for time evolution
 */
void computeDynamicCorrelationsKrylov(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& operators_1,
    const std::vector<Operator>& operators_2,
    const std::vector<std::string>& operator_names,
    int N,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end = 100.0,
    double dt = 0.01,
    int krylov_dim = 30
);

/**
 * Standard TPQ (microcanonical) implementation
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_spectrum = false,
    double LargeValue = 1e5,
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    int num_points = 10000,
    double t_end = 50.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    int sublattice_size = 1,
    int num_sites = 16
);

// Canonical TPQ using imaginary-time propagation e^{-βH} |r>
inline void imaginary_time_evolve_tpq_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    int N,
    double delta_beta,
    int n_max = 50,
    bool normalize = true
);

void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double beta_max,
    int num_samples,
    int temp_interval,
    std::vector<double>& energies,
    std::string dir = "",
    double delta_beta = 0.1,
    int taylor_order = 50,
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    int num_points = 10000,
    double t_end = 50.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    int sublattice_size = 1,
    int num_sites = 16
);


#endif // TPQ_H