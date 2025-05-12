// TPQ.h - Thermal Pure Quantum state implementation

#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include "observables.h"
#include "construct_ham.h"

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v[i] = Complex(real, imag);
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        std::string cmd = "mkdir -p " + path;
        return system(cmd.c_str()) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Path exists and is a directory
        return true;
    } else {
        // Path exists but is not a directory
        return false;
    }
}

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
) {
    // Calculate H|v⟩
    ComplexVector Hv(N);
    H(v.data(), Hv.data(), N);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        energy_complex += std::conj(v[i]) * Hv[i];
    }
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩
    ComplexVector H2v(N);
    H(Hv.data(), H2v.data(), N);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        h2_complex += std::conj(v[i]) * H2v[i];
    }
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

std::pair<Complex, Complex> calculateSzandSz2(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length
){
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    Complex Sz_exps = Complex(0.0, 0.0);
    Complex Sz2_exps = Complex(0.0, 0.0);

    // Create S operators for each site
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.emplace_back(num_sites, spin_length, 2, site);
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators
        std::vector<Complex> Sz_psi = Sz_ops[site].apply(std::vector<Complex>(tpq_state.begin(), tpq_state.end()));
        
        // Calculate expectation values
        Complex Sz_exp = Complex(0.0, 0.0);
        
        for (int i = 0; i < N; i++) {
            Sz_exp += std::conj(tpq_state[i]) * Sz_psi[i];
        }
        
        // Store expectation values
        Sz_exps += Sz_exp;

        std::vector<Complex> Sz2_psi = Sz_ops[site].apply(std::vector<Complex>(Sz_psi.begin(), Sz_psi.end()));

        Complex Sz2_exp = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            Sz2_exp += std::conj(tpq_state[i]) * Sz2_psi[i];
        }
        Sz2_exps += Sz2_exp;
    }
    
    return {Sz_exps/double(num_sites), Sz2_exps/double(num_sites)};
}

/**
 * Write TPQ data to file
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, int step) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(16) << inv_temp << " " << energy << " " 
             << variance << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        file.close();
    }
}

/**
 * Read TPQ data from file
 */
bool readTPQData(const std::string& filename, int step, double& energy, 
                double& temp, double& specificHeat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    double inv_temp, e, var, n, doublon;
    int s;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> inv_temp >> e >> var >> n >> doublon >> s)) {
            continue;
        }
        
        if (s == step) {
            energy = e;
            temp = 1.0/inv_temp;
            specificHeat = (var-e*e)*(inv_temp*inv_temp);
            return true;
        }
    }
    
    return false;
}


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
    int n_max = 10,
    bool normalize = true
) {
    // Temporary vectors for calculation
    ComplexVector result(N);
    ComplexVector term(N);
    ComplexVector Hterm(N);
    
    // Copy initial state to term
    std::copy(tpq_state.begin(), tpq_state.end(), term.begin());
    
    // Copy initial state to result for the first term in Taylor series
    std::copy(tpq_state.begin(), tpq_state.end(), result.begin());
    
    // Precompute coefficients for each term in the Taylor series
    std::vector<Complex> coefficients(n_max + 1);
    coefficients[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        coefficients[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    // Apply Taylor expansion terms
    for (int order = 1; order <= n_max; order++) {
        // Apply H to the previous term
        H(term.data(), Hterm.data(), N);
        std::swap(term, Hterm);
        
        // Add this term to the result
        for (int i = 0; i < N; i++) {
            result[i] += coefficients[order] * term[i];
        }
    }
    
    // Replace tpq_state with the evolved state
    std::swap(tpq_state, result);
    
    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, tpq_state.data(), 1);
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, tpq_state.data(), 1);
    }
}

/**
 * Save the current TPQ state to a file
 * 
 * @param tpq_state TPQ state vector to save
 * @param filename Name of the file to save to
 * @return True if successful
 */
bool save_tpq_state(const ComplexVector& tpq_state, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    size_t size = tpq_state.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tpq_state.data()), size * sizeof(Complex));
    
    out.close();
    return true;
}

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    tpq_state.resize(size);
    in.read(reinterpret_cast<char*>(tpq_state.data()), size * sizeof(Complex));
    
    in.close();
    return true;
}

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
    double tmax = 100.0,
    double dt = 0.1,
    double eta = 0.1,
    bool use_lorentzian = false,
    int n_max = 10 // Order of Taylor expansion
) {
    SpectralFunctionData result;
    
    // Generate frequency points
    result.frequencies.resize(num_points);
    double omega_step = (omega_max - omega_min) / (num_points - 1);
    for (int i = 0; i < num_points; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.spectral_function.resize(num_points, 0.0);
    
    // Goal is to calculate C(t) = <ψ|e^{iHt}O†e^{-iHt}O|ψ>
    // where |ψ> is the TPQ state and O is the operator
    // As such we need to calculate  <ψ(t)|O†|Oψ(t)> 


    // Calculate O|ψ>
    ComplexVector O_psi(N);
    O(tpq_state.data(), O_psi.data(), N);
    
    // Prepare for time evolution
    int num_steps = static_cast<int>(tmax / dt) + 1;
    std::vector<Complex> time_correlation(num_steps);
    
    // Create a copy of the state for time evolution
    ComplexVector evolved_state = tpq_state;
    ComplexVector temp_state(N);
    
    // <ψ|O†
    O(evolved_state.data(), temp_state.data(), N);
    
    // Calculate initial correlation C(0) = <ψ|O†O|ψ>
    Complex initial_corr = Complex(0.0, 0.0);
    for (int i = 0; i < N; i++) {
        initial_corr += std::conj(temp_state[i]) * O_psi[i];
    }
    time_correlation[0] = initial_corr;
    
    std::cout << "Starting real-time evolution for correlation function..." << std::endl;
    
    // Time evolve and calculate correlation function C(t) = <ψ|O†e^{iHt}Oe^{-iHt}|ψ>
    for (int step = 1; step < num_steps; step++) {
        double t = step * dt;
        
        // Evolve temp_state = <ψ|e^{iHt}
        time_evolve_tpq_state(H, evolved_state, N, t, n_max, true);
        // <ψ|e^{iHt} O†
        O(evolved_state.data(), temp_state.data(), N);
        // Evolve temp_state =  e^{-iHt}O|ψ> 
        time_evolve_tpq_state(H, O_psi, N, t, n_max, true);

        
        // Calculate correlation C(t)
        Complex corr_t = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            initial_corr += std::conj(temp_state[i]) * O_psi[i];
        }
        time_correlation[step] = corr_t;
        
        if (step % 100 == 0) {
            std::cout << "  Completed time step " << step << " of " << num_steps << std::endl;
        }
    }
    
    std::cout << "Calculating spectral function via Fourier transform..." << std::endl;
    
    // Perform Fourier transform to get spectral function
    for (int i = 0; i < num_points; i++) {
        double omega = result.frequencies[i];
        Complex spectral_value = Complex(0.0, 0.0);
        
        for (int step = 0; step < num_steps; step++) {
            double t = step * dt;
            Complex phase = std::exp(Complex(0.0, -omega * t));
            
            // Add damping factor
            double damping;
            if (use_lorentzian) {
                damping = std::exp(-eta * t);
            } else {
                damping = std::exp(-eta * t * t / 2.0);
            }
            
            spectral_value += time_correlation[step] * phase * damping * dt;
        }
        
        // The spectral function is the real part of the Fourier transform
        result.spectral_function[i] = spectral_value.real();
    }
    
    return result;
}


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
) {
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    // Create S operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators
        std::vector<Complex> Sp_psi = Sp_ops[site].apply(std::vector<Complex>(tpq_state.begin(), tpq_state.end()));
        std::vector<Complex> Sm_psi = Sm_ops[site].apply(std::vector<Complex>(tpq_state.begin(), tpq_state.end()));
        std::vector<Complex> Sz_psi = Sz_ops[site].apply(std::vector<Complex>(tpq_state.begin(), tpq_state.end()));
        
        // Calculate expectation values
        Complex Sp_exp = Complex(0.0, 0.0);
        Complex Sm_exp = Complex(0.0, 0.0);
        Complex Sz_exp = Complex(0.0, 0.0);
        
        for (int i = 0; i < N; i++) {
            Sp_exp += std::conj(tpq_state[i]) * Sp_psi[i];
            Sm_exp += std::conj(tpq_state[i]) * Sm_psi[i];
            Sz_exp += std::conj(tpq_state[i]) * Sz_psi[i];
        }
        
        // Store expectation values
        expectations[0][site] = Sp_exp;
        expectations[1][site] = Sm_exp;
        expectations[2][site] = Sz_exp;
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values from TPQ state:" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^+ (real)" 
                << std::setw(20) << "S^+ (imag)" 
                << std::setw(20) << "S^- (real)"
                << std::setw(20) << "S^- (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file if output_file is specified
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "# Site S+_real S+_imag S-_real S-_imag Sz_real Sz_imag" << std::endl;
            for (int site = 0; site < num_sites; site++) {
                out << site << " "
                    << std::setprecision(10) << expectations[0][site].real() << " "
                    << std::setprecision(10) << expectations[0][site].imag() << " "
                    << std::setprecision(10) << expectations[1][site].real() << " "
                    << std::setprecision(10) << expectations[1][site].imag() << " "
                    << std::setprecision(10) << expectations[2][site].real() << " "
                    << std::setprecision(10) << expectations[2][site].imag() << std::endl;
            }
            out.close();
            std::cout << "Spin expectations saved to " << output_file << std::endl;
        }
    }
    
    return expectations;
}

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
) {
    std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::ifstream file(ss_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open TPQ data file " << ss_file << std::endl;
        return ComplexVector(N);
    }
    
    // Skip header
    std::string line;
    std::getline(file, line);
    
    double best_beta = 0.0;
    int best_step = 0;
    double min_diff = std::numeric_limits<double>::max();
    
    // Find the step with the closest inverse temperature
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double beta, energy, variance, norm, doublon;
        int step;
        
        if (!(iss >> beta >> energy >> variance >> norm >> doublon >> step)) {
            continue;
        }
        
        double diff = std::abs(beta - target_beta);
        if (diff < min_diff) {
            min_diff = diff;
            best_beta = beta;
            best_step = step;
        }
    }
    file.close();
    
    if (best_step == 0) {
        std::cerr << "Error: Could not find appropriate TPQ state" << std::endl;
        return ComplexVector(N);
    }
    
    std::cout << "Loading TPQ state at step " << best_step 
              << ", beta = " << best_beta 
              << " (target beta = " << target_beta << ")" << std::endl;
    
    // Load the state from file
    std::string state_file = tpq_dir + "/tpq_state_" + std::to_string(sample) 
                             + "_step" + std::to_string(best_step) + ".dat";
    
    ComplexVector tpq_state(N);
    if (!load_tpq_state(tpq_state, state_file)) {
        std::cerr << "Error: Could not load TPQ state from " << state_file << std::endl;
        return ComplexVector(N);
    }
    
    return tpq_state;
}



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
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.1,
    float spin_length = 0.5,
    bool measure_sz = false
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    int num_sites = static_cast<int>(std::log2(N));

    eigenvalues.clear();

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
        std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
        
        // Initialize output files
        {
            std::ofstream ss_out(ss_file);
            ss_out << "# inv_temp energy variance num doublon step" << std::endl;
            
            std::ofstream norm_out(norm_file);
            norm_out << "# inv_temp norm first_norm step" << std::endl;
            
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag) step" << std::endl;
        }
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        ComplexVector v0(N);
        H(v1.data(), v0.data(), N);

        // For each element, compute v0 = LargeValue*v1 - v0
        for (int i = 0; i < N; i++) {
            v0[i] = (LargeValue * v1[i]) - v0[i];
        }

        
        // Calculate initial energy and norm
        auto [energy, variance] = calculateEnergyAndVariance(H, v1, N);
        double first_norm = cblas_dznrm2(N, v1.data(), 1);
        double current_norm = first_norm;
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        writeTPQData(ss_file, inv_temp, energy, variance, current_norm, 0);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << 0 << std::endl;
        }
        
        // Step 1: Calculate v0 = H|v1⟩
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance(H, v1, N);
        double nsite = N; // This should be the actual number of sites, approximating as N for now
        inv_temp = (2.0) / (LargeValue - energy1);
        
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
            
            std::ofstream flct_out(flct_file, std::ios::app);
            flct_out << std::setprecision(16) << inv_temp << " " 
                     << 0.0 << " " << 0.0 << " " << 0.0 << " " 
                     << 0.0 << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Store previous v0 as temp
            ComplexVector temp = v0;
            
            // Update v0 = H|v1⟩ - v0
            H(v1.data(), v0.data(), N);
            
            // For each element, compute v0 = LargeValue*v1 - v0
            for (int i = 0; i < N; i++) {
                v0[i] = (LargeValue * v1[i]) - v0[i];
            }

            // Update v1 = v1 / ||v1||
            std::swap(v1, temp);
            current_norm = cblas_dznrm2(N, v1.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v1.data(), 1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v1, N);
            
            // Update inverse temperature
            inv_temp = (2.0*step) / (LargeValue - energy_step);
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                if (measure_sz){
                    std::ofstream flct_out(flct_file, std::ios::app);
                    auto [Sz, Sz2] = calculateSzandSz2(v1, num_sites, spin_length);
                    flct_out << std::setprecision(16) << inv_temp << " " << Sz.real() << " " << Sz.imag() << " " << Sz2.real() << " " << Sz2.imag() << " " << step << std::endl;
                }
                // Optionally compute dynamical susceptibiltity
                if (compute_observables) {
                    // Compute dynamical susceptibilities
                    std::string dyn_file = dir + "/dyn_rand" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat";
                                        
                    // Save the current TPQ state for later analysis
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat";
                    save_tpq_state(v1, state_file);
                    for (auto observable : observables) {
                        // Create a lambda to adapt Operator to the required function signature
                        auto operatorFunc = [&observable](const Complex* in, Complex* out, int N) {
                            // Convert input to vector
                            std::vector<Complex> input(in, in + N);
                            // Apply operator
                            std::vector<Complex> result = observable.apply(input);
                            // Copy result to output
                            std::copy(result.begin(), result.end(), out);
                        };
                        
                        // Calculate spectral function with current parameters
                        SpectralFunctionData spectrum = calculate_spectral_function_from_tpq(
                            H, operatorFunc, v1, N, omega_min, omega_max, num_points, t_end, dt, 0.1, false);
                                            
                        // Write spectral function to file
                        std::ofstream dyn_out(dyn_file);
                        if (dyn_out.is_open()) {
                            dyn_out << "# omega spectral_function" << std::endl;
                            for (size_t i = 0; i < spectrum.frequencies.size(); i++) {
                                dyn_out << std::setprecision(16) 
                                        << spectrum.frequencies[i] << " " 
                                        << spectrum.spectral_function[i] << std::endl;
                            }
                            dyn_out.close();
                            std::cout << "Dynamical susceptibility saved to " << dyn_file << std::endl;
                        }
                    }

                }
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy);
    }
}

/**
 * Canonical TPQ implementation (using imaginary time evolution)
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param delta_tau Time step for imaginary time evolution
 * @param compute_spectrum Whether to compute spectrum
 */
void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    double delta_tau = 0.0, // Default 0 means use 1/LargeValue
    bool compute_spectrum = false,
    int n_max = 10, // Order of Taylor expansion
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.1,
    float spin_length = 0.5,
    bool measure_sz = false
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    // Set default delta_tau if not specified
    if (delta_tau <= 0.0) {
        delta_tau = 1.0/1e10;
    }

    int num_sites = static_cast<int>(std::log2(N));
    
    eigenvalues.clear();

    // Define the exponential imaginary time evolution operator function outside the loops
    auto expMinusHalfDeltaTauH = [&H, delta_tau, n_max, N](const Complex* v_in, Complex* v_out) {
        // Copy v_in to v_out for the first term in Taylor series
        std::copy(v_in, v_in + N, v_out);
        
        // Allocate memory for temporary vectors
        ComplexVector term(N);
        ComplexVector Hterm(N);
        std::copy(v_in, v_in + N, term.data());
        
        // Precompute coefficients for each term in the Taylor series
        std::vector<double> coefficients(n_max + 1);
        coefficients[0] = 1.0;  // 0th order term
        double factorial = 1.0;
        
        for (int order = 1; order <= n_max; order++) {
            factorial *= order;
            double coef = (order % 2 == 1) ? -1.0 : 1.0;  // Alternating sign
            coefficients[order] = coef * std::pow(delta_tau/2.0, order) / factorial;
        }
        
        // Apply Taylor expansion terms
        for (int order = 1; order <= n_max; order++) {
            // Apply H to the previous term
            H(term.data(), Hterm.data(), N);
            std::swap(term, Hterm);
            
            // Add this term to the result
            for (int i = 0; i < N; i++) {
                v_out[i] += coefficients[order] * term[i];
            }
        }
    };
    
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "Canonical TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
        std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
        std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
        
        // Initialize output files
        {
            std::ofstream ss_out(ss_file);
            ss_out << "# inv_temp energy variance num doublon step" << std::endl;
            
            std::ofstream norm_out(norm_file);
            norm_out << "# inv_temp norm first_norm step" << std::endl;
            
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag) step" << std::endl;
        }
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        ComplexVector v0 = v1; // In canonical TPQ, v0 starts as v1
        
        // Calculate initial energy and norm
        auto [energy, variance] = calculateEnergyAndVariance(H, v1, N);
        double first_norm = cblas_dznrm2(N, v1.data(), 1);
        double current_norm = first_norm;
        
        // Initial inverse temperature is 0
        double inv_temp = 0.0;
        
        // Write initial state (infinite temperature)
        writeTPQData(ss_file, inv_temp, energy, variance, current_norm, 0);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << 0 << std::endl;
        }
        
        // Main canonical TPQ loop
        for (int step = 1; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Canonical TPQ: Apply exp(-delta_tau*H/2) to v1 using 4th order approximation
            expMinusHalfDeltaTauH(v1.data(), v0.data());

            // Normalize v0
            current_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Swap v0 and v1 for next iteration
            std::swap(v0, v1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v1, N);
            
            // Update inverse temperature
            inv_temp += delta_tau;
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                if (measure_sz){
                    std::ofstream flct_out(flct_file, std::ios::app);
                    auto [Sz, Sz2] = calculateSzandSz2(v1, num_sites, spin_length);
                    flct_out << std::setprecision(16) << inv_temp << " " << Sz.real() << " " << Sz.imag() << " " << Sz2.real() << " " << Sz2.imag() << " " << step << std::endl;
                }
                // Optionally compute dynamical susceptibiltity
                if (compute_observables) {
                    // Compute dynamical susceptibilities
                    std::string dyn_file = dir + "/dyn_rand" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat";
                                        
                    // Save the current TPQ state for later analysis
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat";
                    save_tpq_state(v1, state_file);
                    for (auto observable : observables) {
                        // Create a lambda to adapt Operator to the required function signature
                        auto operatorFunc = [&observable](const Complex* in, Complex* out, int N) {
                            // Convert input to vector
                            std::vector<Complex> input(in, in + N);
                            // Apply operator
                            std::vector<Complex> result = observable.apply(input);
                            // Copy result to output
                            std::copy(result.begin(), result.end(), out);
                        };
                        
                        // Calculate spectral function with current parameters
                        SpectralFunctionData spectrum = calculate_spectral_function_from_tpq(
                            H, operatorFunc, v1, N, omega_min, omega_max, num_points, t_end, dt, 0.1, false);
                                                   
                        // Write spectral function to file
                        std::ofstream dyn_out(dyn_file);
                        if (dyn_out.is_open()) {
                            dyn_out << "# omega spectral_function" << std::endl;
                            for (size_t i = 0; i < spectrum.frequencies.size(); i++) {
                                dyn_out << std::setprecision(16) 
                                        << spectrum.frequencies[i] << " " 
                                        << spectrum.spectral_function[i] << std::endl;
                            }
                            dyn_out.close();
                            std::cout << "Dynamical susceptibility saved to " << dyn_file << std::endl;
                        }
                    }

                }
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy);
    }
}

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
) {
    std::cout << "Calculating spectrum from TPQ state..." << std::endl;
    
    // Read TPQ data
    std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(tpq_sample) + ".dat";
    double energy, temp, specificHeat;
    
    if (!readTPQData(ss_file, tpq_step, energy, temp, specificHeat)) {
        std::cerr << "Error: Could not read TPQ data from " << ss_file << std::endl;
        return;
    }
    
    std::cout << "Using TPQ state at step " << tpq_step 
              << ", temperature: " << temp 
              << ", energy: " << energy << std::endl;
    
    // Open output file
    std::ofstream spectrum_file(out_file);
    if (!spectrum_file.is_open()) {
        std::cerr << "Error: Could not open output file " << out_file << std::endl;
        return;
    }
    spectrum_file << "# omega re(spectrum) im(spectrum)" << std::endl;
    
    // Calculate number of frequency points
    int n_omega = static_cast<int>((omega_max - omega_min) / omega_step) + 1;
    
    // Pre-factor for Gaussian broadening
    double pre_factor = 2.0 * temp * temp * specificHeat;
    double factor = 1.0 / sqrt(M_PI * pre_factor);
    
    // Calculate spectrum for each frequency
    for (int i = 0; i < n_omega; i++) {
        double omega = omega_min + i * omega_step;
        Complex z(omega, eta); // Complex frequency with broadening
        
        // This is a simplified version - the full algorithm would perform
        // continued fraction expansion using Lanczos tridiagonalization
        
        // Calculate the spectrum using Gaussian broadening approximation
        double spectrum_val = factor * exp(-pow((omega - energy), 2) / pre_factor);
        
        spectrum_file << std::setprecision(16) 
                     << omega << " " 
                     << spectrum_val << " " 
                     << 0.0 << std::endl;
    }
    
    spectrum_file.close();
    std::cout << "Spectrum calculation complete. Written to " << out_file << std::endl;
}



#endif // TPQ_H