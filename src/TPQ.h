// TPQ.h - Thermal Pure Quantum state implementation

#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <mkl.h>
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

// Forward declaration for CUDA wrapper
#if defined(WITH_CUDA)
#include "TPQ_cuda.cuh"
#endif

#define GET_VARIABLE_NAME(Variable) (#Variable)

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

    return std::move(v); // Explicitly signal move semantics for the return value
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

std::vector<SingleSiteOperator> createSzOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sz_ops;
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.emplace_back(num_sites, spin_length, 2, site);
    }
    return Sz_ops;
}

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    ComplexVector Sz_exps(sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Sz2_exps(sublattice_size+1, Complex(0.0, 0.0));
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        int i = site % sublattice_size;

        // Apply operators - use direct vector construction to avoid copy
        std::vector<Complex> Sz_psi = Sz_ops[i].apply({tpq_state.begin(), tpq_state.end()});
        
        // Calculate expectation values
        Complex Sz_exp = Complex(0.0, 0.0);
        
        for (int j = 0; j < N; j++) {
            Sz_exp += std::conj(tpq_state[j]) * Sz_psi[j];
        }
        
        // Store expectation values
        Sz_exps[i] += Sz_exp;

        // Apply operator directly to avoid temporary vector copy
        std::vector<Complex> Sz2_psi = Sz_ops[i].apply(std::move(Sz_psi));

        Complex Sz2_exp = Complex(0.0, 0.0);
        for (int j = 0; j < N; j++) {
            Sz2_exp += std::conj(tpq_state[j]) * Sz2_psi[j];
        }
        Sz2_exps[i] += Sz2_exp;
    }

    for (int i = 0; i < sublattice_size; i++) {
        Sz_exps[i] /= double(num_sites);
        Sz2_exps[i] /= double(num_sites);
        Sz_exps[sublattice_size] += Sz_exps[i];
        Sz2_exps[sublattice_size] += Sz2_exps[i];
    }


    return {Sz_exps, Sz2_exps};
}


std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> createDoubleSiteOperators(int num_sites, float spin_length) {
    std::vector<DoubleSiteOperator> Szz_ops;
    std::vector<DoubleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            Szz_ops.emplace_back(num_sites, spin_length, 2, site, 2, site2);
            Spm_ops.emplace_back(num_sites, spin_length, 0, site, 1, site2);
        }
    }
    return {Szz_ops, Spm_ops};
}


std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> createSingleOperators_pair(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Szz_ops;
    std::vector<SingleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        Szz_ops.emplace_back(num_sites, spin_length, 2, site);
        Spm_ops.emplace_back(num_sites, spin_length, 0, site);
    }
    return {Szz_ops, Spm_ops};
}



std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));

    // Create S operators for each site
    std::vector<DoubleSiteOperator> Szz_ops;
    std::vector<DoubleSiteOperator> Spm_ops;

    Szz_ops = double_site_ops.first;
    Spm_ops = double_site_ops.second;            // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            int n1 = site % sublattice_size;
            int n2 = site2 % sublattice_size;

            // Apply operators
            std::vector<Complex> Szz_psi = Szz_ops[site*num_sites+site2].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Spm_psi = Spm_ops[site*num_sites+site2].apply({tpq_state.begin(), tpq_state.end()});

            // Calculate expectation values
            Complex Szz_exp = Complex(0.0, 0.0);
            Complex Spm_exp = Complex(0.0, 0.0);


            for (int i = 0; i < N; i++) {
                Szz_exp += std::conj(tpq_state[i]) * Szz_psi[i];
            }
            for (int i = 0; i < N; i++) {
                Spm_exp += std::conj(tpq_state[i]) * Spm_psi[i];
            }
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
    }
    
    return {Szz_exps, Spm_exps};

}

std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    int N = 1 << num_sites;  // 2^num_sites
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spp_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    // Create S operators for each site
    std::vector<SingleSiteOperator> Szz_ops;
    std::vector<SingleSiteOperator> Spm_ops;

    Szz_ops = double_site_ops.first;
    Spm_ops = double_site_ops.second;            // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            int n1 = site % sublattice_size;
            int n2 = site2 % sublattice_size;

            // Apply operators
            // SzSz
            std::vector<Complex> Szz_psi = Szz_ops[site].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Szz_psi2 = Szz_ops[site2].apply({tpq_state.begin(), tpq_state.end()});

            // S+S-
            std::vector<Complex> Spm_psi = Spm_ops[site].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Spm_psi2 = Spm_ops[site2].apply({tpq_state.begin(), tpq_state.end()});

            // S+S+
            std::vector<Complex> Spp_psi = Spm_ops[site2].apply({Spm_psi.begin(), Spm_psi.end()});

            // Calculate expectation values
            Complex Szz_exp = Complex(0.0, 0.0);
            Complex Spm_exp = Complex(0.0, 0.0);
            Complex Spp_exp = Complex(0.0, 0.0);
            Complex Spz_exp = Complex(0.0, 0.0);

            for (int i = 0; i < N; i++) {
                Szz_exp += std::conj(Szz_psi[i]) * Szz_psi2[i];
            }
            for (int i = 0; i < N; i++) {
                Spm_exp += std::conj(Spm_psi[i]) * Spm_psi2[i];
            }
            for (int i = 0; i < N; i++) {
                Spp_exp += std::conj(tpq_state[i]) * Spp_psi[i];
            }
            for (int i = 0; i < N; i++) {
                Spz_exp += std::conj(Spm_psi[i]) * Szz_psi2[i];
            }
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
            Spp_exps[n1*sublattice_size+n2] += Spp_exp;
            Spz_exps[n1*sublattice_size+n2] += Spz_exp;

        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spp_exps[i] /= double(num_sites);
        Spz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
        Spp_exps[sublattice_size*sublattice_size] += Spp_exps[i];
        Spz_exps[sublattice_size*sublattice_size] += Spz_exps[i];
    }
    
    return {Szz_exps, Spm_exps, Spp_exps, Spz_exps};

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
    int n_max = 100,
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
) {
    // Precompute coefficients for each term in the Taylor series
    auto coefficients = std::make_shared<std::vector<Complex>>(n_max + 1);
    (*coefficients)[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        (*coefficients)[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    // Return a function that applies the time evolution
    return [H, coefficients, n_max, normalize](const Complex* input, Complex* output, int size) -> void {
        // Temporary vectors for calculation
        std::vector<Complex> term(size);
        std::vector<Complex> Hterm(size);
        std::vector<Complex> result(size);
        
        // Copy input to term and result
        std::copy(input, input + size, term.begin());
        std::copy(input, input + size, result.begin());
        
        // Apply Taylor expansion terms
        for (int order = 1; order <= n_max; order++) {
            // Apply H to the previous term
            H(term.data(), Hterm.data(), size);
            std::swap(term, Hterm);
            
            // Add this term to the result
            for (int i = 0; i < size; i++) {
                result[i] += (*coefficients)[order] * term[i];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            double norm = cblas_dznrm2(size, result.data(), 1);
            if (norm > 0.0) {
                Complex scale_factor = Complex(1.0/norm, 0.0);
                cblas_zscal(size, &scale_factor, result.data(), 1);
            }
        }
        
        // Copy result to output
        std::copy(result.begin(), result.end(), output);
    };
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
    double tmax = 10.0,
    double dt = 0.01,
    double eta = 0.1,
    bool use_lorentzian = false,
    int n_max = 100 // Order of Taylor expansion
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
        
        // Evolve temp_state = <ψ|e^{iHt}
        time_evolve_tpq_state(H, evolved_state, N, dt, n_max, true);
        // <ψ|e^{iHt} O†
        O(evolved_state.data(), temp_state.data(), N);
        // Evolve temp_state =  e^{-iHt}O|ψ> 
        time_evolve_tpq_state(H, O_psi, N, dt, n_max, true);

        
        // Calculate correlation C(t)
        Complex corr_t = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            corr_t += std::conj(temp_state[i]) * O_psi[i];
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


std::vector<std::vector<Complex>> calculate_spectral_function_from_tpq_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators,
    const ComplexVector& tpq_state,
    int N,
    const int num_steps
) {
    int num_operators = operators.size();
    
    // Pre-allocate all buffers needed for calculation
    std::vector<ComplexVector> O_psi_vec(num_operators, ComplexVector(N));        // O|ψ> for each operator
    std::vector<ComplexVector> O_psi_next_vec(num_operators, ComplexVector(N));   // For time evolution of O|ψ>
    ComplexVector state(N);        // |ψ(t)>
    ComplexVector state_next(N);   // For time evolution of |ψ>
    std::vector<ComplexVector> O_dag_state_vec(num_operators, ComplexVector(N));  // O†|ψ(t)> for each operator
    
    // Initialize state to tpq_state
    std::copy(tpq_state.begin(), tpq_state.end(), state.begin());
    
    // Calculate O|ψ> once for each operator
    for (int op = 0; op < num_operators; op++) {
        operators[op](state.data(), O_psi_vec[op].data(), N);
    }
    
    // Prepare time evolution
    std::vector<std::vector<Complex>> time_correlations(num_operators, std::vector<Complex>(num_steps));
    
    // Calculate initial O†|ψ> for each operator
    for (int op = 0; op < num_operators; op++) {
        operators[op](state.data(), O_dag_state_vec[op].data(), N);
        
        // Calculate initial correlation C(0) = <ψ|O†O|ψ>
        time_correlations[op][0] = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            time_correlations[op][0] += std::conj(O_dag_state_vec[op][i]) * O_psi_vec[op][i];
        }
    }
    
    std::cout << "Starting real-time evolution for correlation function..." << std::endl;
        
    // Time evolution loop
    for (int step = 1; step < num_steps; step++) {
        // Evolve state: |ψ(t)> = U_t|ψ(t-dt)>
        U_t(state.data(), state_next.data(), N);
        
        // For each operator
        for (int op = 0; op < num_operators; op++) {
            // Evolve O_psi: O|ψ(t)> = U_t(O|ψ(t-dt)>)0
            U_t(O_psi_vec[op].data(), O_psi_next_vec[op].data(), N);
            
            // Calculate O†|ψ(t)>
            operators[op](state_next.data(), O_dag_state_vec[op].data(), N);
            
            // Calculate correlation C(t) = <ψ(t)|O†O|ψ(t)>
            time_correlations[op][step] = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                time_correlations[op][step] += std::conj(O_dag_state_vec[op][i]) * O_psi_next_vec[op][i];
            }
            
            // Update O_psi for next iteration
            std::swap(O_psi_vec[op], O_psi_next_vec[op]);
        }
        
        // Update state for next iteration
        std::swap(state, state_next);
        
        if (step % 100 == 0) {
            std::cout << "  Completed time step " << step << " of " << num_steps << std::endl;
        }
    }
    
    std::cout << "Calculating spectral function via Fourier transform..." << std::endl;
    
    return time_correlations;
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
        std::vector<Complex> Sp_psi = Sp_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        std::vector<Complex> Sm_psi = Sm_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        std::vector<Complex> Sz_psi = Sz_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        
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



void writeFluctuationData(
    const std::string& flct_file,
    const std::vector<std::string>& spin_corr,
    double inv_temp,
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    int sublattice_size,
    int step
) {
    std::ofstream flct_out(flct_file, std::ios::app);
    auto [Sz, Sz2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sz_ops, sublattice_size);
    
    flct_out << std::setprecision(16) << inv_temp 
             << " " << Sz[sublattice_size].real() << " " << Sz[sublattice_size].imag() 
             << " " << Sz2[sublattice_size].real() << " " << Sz2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out << " " << Sz[i].real() << " " << Sz[i].imag() 
                 << " " << Sz2[i].real() << " " << Sz2[i].imag();
    }
    flct_out << " " << step << std::endl;

    auto [Szz, Spm, Spp, Spz] = calculateSzzSpm(tpq_state, num_sites, spin_length, double_site_ops, sublattice_size);
    for (size_t idx = 0; idx < spin_corr.size(); idx++) {
        std::ofstream corr_out(spin_corr[idx], std::ios::app);
        
        corr_out << std::setprecision(16) << inv_temp;
        
        // Write total (last element)
        std::vector<Complex>* data_ptr = nullptr;
        if (idx == 0) data_ptr = &Szz;
        else if (idx == 1) data_ptr = &Spm;
        else if (idx == 2) data_ptr = &Spp;
        else if (idx == 3) data_ptr = &Spz;
        
        corr_out << " " << (*data_ptr)[sublattice_size*sublattice_size].real() 
                 << " " << (*data_ptr)[sublattice_size*sublattice_size].imag();
        
        // Write individual correlations
        for (int i = 0; i < sublattice_size*sublattice_size; i++) {
            corr_out << " " << (*data_ptr)[i].real() 
                     << " " << (*data_ptr)[i].imag();
        }
        
        corr_out << " " << step << std::endl;
        corr_out.close();
    }
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
) {
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
    std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
    
    // Create vector of spin correlation files
    std::vector<std::string> spin_corr_files;
    std::vector<std::string> suffixes = {"SzSz", "SpSm", "SmSm", "SpSz"};
    
    for (const auto& suffix : suffixes) {
        std::string filename = dir + "/spin_corr_" + suffix + "_rand" + std::to_string(sample) + ".dat";
        spin_corr_files.push_back(filename);
    }
    
    // Initialize output files
    {
        std::ofstream ss_out(ss_file);
        ss_out << "# inv_temp energy variance num doublon step" << std::endl;
        
        std::ofstream norm_out(norm_file);
        norm_out << "# inv_temp norm first_norm step" << std::endl;
        
        std::ofstream flct_out(flct_file);
        flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag)";

        for (int i = 0; i < sublattice_size; i++) {
            flct_out << " sz" << i << "(real) sz" << i << "(imag)"  << " sz2" << i << "(real) sz2" << i << "(imag)";
        }
        flct_out << " step" << std::endl;

        // Initialize each spin correlation file
        for (const auto& file : spin_corr_files) {
            std::ofstream spin_out(file);
            spin_out << "# inv_temp total(real) total(imag)";
            
            for (int i = 0; i < sublattice_size*sublattice_size; i++) {
                spin_out << " site" << i << "(real) site" << i << "(imag)";
            }
            spin_out << " step" << std::endl;
        }
    }
    
    return {ss_file, norm_file, flct_file, spin_corr_files};
}

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
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_step" + std::to_string(step) + ".dat";
    save_tpq_state(tpq_state, state_file);

    for (size_t i = 0; i < observables.size(); i++) {
        std::cout << "Computing dynamical susceptibility for sample " << sample 
                  << ", step " << step << ", observable: " << observable_names[i] << std::endl;
                  
        std::string dyn_file = dir + "/dyn_rand" + std::to_string(sample) + "_" 
                             + observable_names[i] + "_step" + std::to_string(step) + ".dat";
                             
        // Create a lambda to adapt Operator to the required function signature
        auto operatorFunc = [&observables, i](const Complex* in, Complex* out, int size) {
            // Convert input to vector
            std::vector<Complex> input(in, in + size);
            // Apply operator
            std::vector<Complex> result = observables[i].apply(input);
            // Copy result to output
            std::copy(result.begin(), result.end(), out);
        };
        
        // Calculate spectral function with current parameters
        SpectralFunctionData spectrum = calculate_spectral_function_from_tpq(
            H, operatorFunc, tpq_state, N, omega_min, omega_max, num_points, t_end, dt, 0.1, false);
            
        // Write spectral function to file
        std::ofstream dyn_out(dyn_file);
        if (dyn_out.is_open()) {
            dyn_out << "# omega spectral_function" << std::endl;
            for (size_t j = 0; j < spectrum.frequencies.size(); j++) {
                dyn_out << std::setprecision(16) 
                      << spectrum.frequencies[j] << " " 
                      << spectrum.spectral_function[j].real() << " "
                      << spectrum.spectral_function[j].imag() << std::endl;
            }
            dyn_out.close();
            std::cout << "Dynamical susceptibility saved to " << dyn_file << std::endl;
        }
    }
}


void computeObservableDynamics_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    std::function<void(const Complex*, Complex*, int)> U_nt,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables,
    const std::vector<std::string>& observable_names,
    int N, 
    const std::string& dir,
    int sample,
    double inv_temp,
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.01
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
    save_tpq_state(tpq_state, state_file);

    std::cout << "Computing dynamical susceptibility for sample " << sample 
              << ", beta = " << inv_temp << ", for " << observables.size() << " observables" << std::endl;
    
    // Create array of operator functions
    std::vector<std::function<void(const Complex*, Complex*, int)>> operatorFuncs;
    operatorFuncs.reserve(observables.size());
    
    for (size_t i = 0; i < observables.size(); i++) {
        operatorFuncs.emplace_back([&observables, i](const Complex* in, Complex* out, int size) {
            // Convert input to vector
            std::vector<Complex> input(in, in + size);
            // Apply operator
            std::vector<Complex> result = observables[i].apply(input);
            // Copy result to output
            std::copy(result.begin(), result.end(), out);
        });
    }
    
    // Calculate spectral function for all operators at once
    auto time_correlations = calculate_spectral_function_from_tpq_U_t(
        U_t, operatorFuncs, tpq_state, N, int(t_end/dt+1));
    
    auto negative_time_correlations = calculate_spectral_function_from_tpq_U_t(
        U_nt, operatorFuncs, tpq_state, N, int(t_end/dt+1));

    // Process and save results for each observable
    for (size_t i = 0; i < observables.size(); i++) {
        std::string time_corr_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                             + observable_names[i] + "_beta=" + std::to_string(inv_temp) + ".dat";
        
        std::vector<double> time_points(time_correlations[i].size());
        for (size_t j = 0; j < time_correlations[i].size(); j++) {
            time_points[j] = j * dt;
        }

        // Combine negative and positive time correlations into one vector
        std::vector<Complex> combined_time_correlation;
        std::vector<double> combined_time_points;
        combined_time_correlation.reserve(time_correlations[i].size() + negative_time_correlations[i].size() - 1);
        combined_time_points.reserve(time_correlations[i].size() + negative_time_correlations[i].size() - 1);
        
        // Add negative time correlations first (in reverse order, skipping t=0)
        for (int j = negative_time_correlations[i].size() - 1; j > 0; j--) {
            combined_time_correlation.push_back(negative_time_correlations[i][j]);
            combined_time_points.push_back(-j * dt);
        }

        // Add positive time correlations
        combined_time_correlation.insert(combined_time_correlation.end(), 
                                        time_correlations[i].begin(), 
                                        time_correlations[i].end());
                                        
        combined_time_points.insert(combined_time_points.end(), 
                                    time_points.begin(), 
                                    time_points.end());

        // Write time correlation to file
        std::ofstream time_corr_out(time_corr_file);
        if (time_corr_out.is_open()) {
            time_corr_out << "# t time_correlation" << std::endl;
            for (size_t j = 0; j < combined_time_correlation.size(); j++) {
                time_corr_out << std::setprecision(16) 
                      << combined_time_points[j] << " " 
                      << combined_time_correlation[j].real() << " "
                      << combined_time_correlation[j].imag() << std::endl;
            }
            time_corr_out.close();
            std::cout << "Time correlation saved to " << time_corr_file << std::endl;
        }
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
) {
    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::min(krylov_dim, N);
    
    // Pre-allocate all memory to avoid repeated allocations
    static thread_local std::vector<ComplexVector> krylov_vectors;
    static thread_local std::vector<double> alpha;
    static thread_local std::vector<double> beta;
    static thread_local ComplexVector w;
    static thread_local int last_N = 0;
    static thread_local int last_krylov_dim = 0;
    
    // Resize only if dimensions changed
    if (last_N != N || last_krylov_dim < krylov_dim) {
        krylov_vectors.resize(krylov_dim);
        for (auto& vec : krylov_vectors) {
            vec.resize(N);
        }
        alpha.resize(krylov_dim);
        beta.resize(krylov_dim - 1);
        w.resize(N);
        last_N = N;
        last_krylov_dim = krylov_dim;
    }
    
    // Initialize first Krylov vector as normalized input state
    double norm = cblas_dznrm2(N, tpq_state.data(), 1);
    if (norm < 1e-14) {
        // Handle zero state
        return;
    }
    
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zcopy(N, tpq_state.data(), 1, krylov_vectors[0].data(), 1);
    cblas_zscal(N, &scale_factor, krylov_vectors[0].data(), 1);
    
    // Lanczos iteration with improved numerical stability
    int effective_dim = krylov_dim;
    constexpr double ortho_threshold = 1e-10;
    constexpr double breakdown_threshold = 1e-14;
    
    for (int j = 0; j < krylov_dim - 1; j++) {
        // Apply Hamiltonian: w = H * v_j
        H(krylov_vectors[j].data(), w.data(), N);
        
        // Compute alpha_j = Re(<v_j | H | v_j>) - use BLAS for efficiency
        Complex alpha_complex;
        cblas_zdotc_sub(N, krylov_vectors[j].data(), 1, w.data(), 1, &alpha_complex);
        alpha[j] = alpha_complex.real();
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, krylov_vectors[j].data(), 1, w.data(), 1);
        
        // Full reorthogonalization for better stability
        // This is more expensive but necessary for numerical accuracy
        for (int i = 0; i <= j; i++) {
            Complex overlap;
            cblas_zdotc_sub(N, krylov_vectors[i].data(), 1, w.data(), 1, &overlap);
            if (std::abs(overlap) > ortho_threshold) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, krylov_vectors[i].data(), 1, w.data(), 1);
            }
        }
        
        // Compute beta_{j} = ||w||
        beta[j] = cblas_dznrm2(N, w.data(), 1);
        
        // Check for breakdown
        if (beta[j] < breakdown_threshold) {
            effective_dim = j + 1;
            break;
        }
        
        // v_{j+1} = w / beta_{j}
        Complex inv_beta = Complex(1.0/beta[j], 0.0);
        cblas_zcopy(N, w.data(), 1, krylov_vectors[j+1].data(), 1);
        cblas_zscal(N, &inv_beta, krylov_vectors[j+1].data(), 1);
    }
    
    // Handle the last alpha
    if (effective_dim == krylov_dim) {
        H(krylov_vectors[effective_dim-1].data(), w.data(), N);
        Complex alpha_complex;
        cblas_zdotc_sub(N, krylov_vectors[effective_dim-1].data(), 1, w.data(), 1, &alpha_complex);
        alpha[effective_dim-1] = alpha_complex.real();
    }
    
    // Use Eigen's optimized routines for the small tridiagonal problem
    Eigen::MatrixXd H_krylov(effective_dim, effective_dim);
    
    // Build tridiagonal matrix efficiently
    H_krylov.setZero();
    for (int i = 0; i < effective_dim; i++) {
        H_krylov(i, i) = alpha[i];
    }
    for (int i = 0; i < effective_dim - 1; i++) {
        H_krylov(i, i+1) = beta[i];
        H_krylov(i+1, i) = beta[i];
    }
    
    // Compute eigendecomposition (more stable than matrix exponential)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_krylov);
    const auto& eigenvalues = solver.eigenvalues();
    const auto& eigenvectors = solver.eigenvectors();
    // Apply time evolution using eigendecomposition
    // exp(-i*H*t)|ψ⟩ = V * exp(-i*D*t) * V^† * |e_0⟩
    Eigen::VectorXcd coeffs(effective_dim);
    for (int i = 0; i < effective_dim; i++) {
        coeffs(i) = std::exp(Complex(0, -eigenvalues(i) * delta_t)) * eigenvectors(i, 0) * norm;
    }
    
    // Reconstruct state in original basis
    std::fill(tpq_state.begin(), tpq_state.end(), Complex(0, 0));
    
    // Use BLAS-3 operation for better cache efficiency
    for (int i = 0; i < effective_dim; i++) {
        Complex coeff(0, 0);
        for (int j = 0; j < effective_dim; j++) {
            coeff += eigenvectors(j, i) * coeffs(j);
        }
        cblas_zaxpy(N, &coeff, krylov_vectors[i].data(), 1, tpq_state.data(), 1);
    }
    
    // Normalize if requested
    if (normalize) {
        double final_norm = cblas_dznrm2(N, tpq_state.data(), 1);
        if (final_norm > 1e-14) {
            Complex final_scale = Complex(1.0/final_norm, 0.0);
            cblas_zscal(N, &final_scale, tpq_state.data(), 1);
        }
    }
}

#if defined(WITH_CUDA)
// GPU overload: evolve state on GPU using a device-side matvec callback
inline void time_evolve_tpq_krylov(
    GpuMatvec H_dev,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int krylov_dim,
    bool normalize,
    CudaContext& ctx
){
    DeviceVector d_state(N);
    copyHostToDevice(d_state, tpq_state.data(), N);
    time_evolve_tpq_krylov_cuda(H_dev, d_state.ptr, N, delta_t, krylov_dim, normalize, ctx);
    copyDeviceToHost(tpq_state.data(), d_state, N);
}
#endif

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
) {
    // Scale Hamiltonian to [-1, 1] range for Chebyshev expansion
    double E_center = (E_max + E_min) / 2.0;
    double E_scale = (E_max - E_min) / 2.0;
    
    // Scaled time parameter
    double tau = delta_t * E_scale;
    
    // Create scaled Hamiltonian operator: H_scaled = (H - E_center * I) / E_scale
    auto H_scaled = [&](const Complex* input, Complex* output, int size) {
        H(input, output, size);
        // output = (H - E_center * I) * input / E_scale
        Complex center_factor = Complex(-E_center / E_scale, 0.0);
        Complex scale_factor = Complex(1.0 / E_scale, 0.0);
        
        cblas_zscal(size, &scale_factor, output, 1);
        cblas_zaxpy(size, &center_factor, input, 1, output, 1);
    };
    
    // Initialize result and Chebyshev polynomials
    ComplexVector result(N, Complex(0, 0));
    ComplexVector T_prev(N), T_curr(N), T_next(N);
    
    // T_0(H_scaled) |ψ⟩ = |ψ⟩
    std::copy(tpq_state.begin(), tpq_state.end(), T_prev.begin());
    
    // First term: J_0(-iτ) * T_0
    double J_0 = std::cyl_bessel_j(0, tau);
    Complex coeff_0 = Complex(J_0, 0.0);
    cblas_zaxpy(N, &coeff_0, T_prev.data(), 1, result.data(), 1);
    
    if (num_terms > 1) {
        // T_1(H_scaled) |ψ⟩ = H_scaled |ψ⟩
        H_scaled(tpq_state.data(), T_curr.data(), N);
        
        // Second term: 2 * J_1(-iτ) * (-i) * T_1
        double J_1 = std::cyl_bessel_j(1, tau);
        Complex coeff_1 = Complex(0, -2.0 * J_1);
        cblas_zaxpy(N, &coeff_1, T_curr.data(), 1, result.data(), 1);
        
        // Higher order terms using Chebyshev recurrence
        for (int n = 2; n < num_terms; n++) {
            // T_n = 2 * H_scaled * T_{n-1} - T_{n-2}
            H_scaled(T_curr.data(), T_next.data(), N);
            Complex two = Complex(2.0, 0.0);
            Complex neg_one = Complex(-1.0, 0.0);
            
            cblas_zscal(N, &two, T_next.data(), 1);
            cblas_zaxpy(N, &neg_one, T_prev.data(), 1, T_next.data(), 1);
            
            // Add contribution: 2 * J_n(-iτ) * (-i)^n * T_n
            double J_n = std::cyl_bessel_j(n, tau);
            Complex i_power = std::pow(Complex(0, -1), n);
            Complex coeff_n = 2.0 * J_n * i_power;
            
            cblas_zaxpy(N, &coeff_n, T_next.data(), 1, result.data(), 1);
            
            // Cycle for next iteration
            std::swap(T_prev, T_curr);
            std::swap(T_curr, T_next);
        }
    }
    
    // Apply phase factor from center shift: exp(-i * E_center * delta_t)
    Complex phase = std::exp(Complex(0, -E_center * delta_t));
    cblas_zscal(N, &phase, result.data(), 1);
    
    // Replace state with evolved result
    std::swap(tpq_state, result);
    
    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, tpq_state.data(), 1);
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, tpq_state.data(), 1);
    }
}

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
) {
    ComplexVector k1(N), k2(N), k3(N), k4(N);
    ComplexVector temp_state(N);
    Complex neg_i = Complex(0, -1);
    Complex half = Complex(0.5, 0);
    
    // k1 = -i * H * ψ(t)
    H(tpq_state.data(), k1.data(), N);
    cblas_zscal(N, &neg_i, k1.data(), 1);
    
    // temp_state = ψ(t) + (Δt/2) * k1
    std::copy(tpq_state.begin(), tpq_state.end(), temp_state.begin());
    Complex dt_half = Complex(delta_t / 2.0, 0);
    cblas_zaxpy(N, &dt_half, k1.data(), 1, temp_state.data(), 1);
    
    // k2 = -i * H * (ψ(t) + (Δt/2) * k1)
    H(temp_state.data(), k2.data(), N);
    cblas_zscal(N, &neg_i, k2.data(), 1);
    
    // temp_state = ψ(t) + (Δt/2) * k2
    std::copy(tpq_state.begin(), tpq_state.end(), temp_state.begin());
    cblas_zaxpy(N, &dt_half, k2.data(), 1, temp_state.data(), 1);
    
    // k3 = -i * H * (ψ(t) + (Δt/2) * k2)
    H(temp_state.data(), k3.data(), N);
    cblas_zscal(N, &neg_i, k3.data(), 1);
    
    // temp_state = ψ(t) + Δt * k3
    std::copy(tpq_state.begin(), tpq_state.end(), temp_state.begin());
    Complex dt = Complex(delta_t, 0);
    cblas_zaxpy(N, &dt, k3.data(), 1, temp_state.data(), 1);
    
    // k4 = -i * H * (ψ(t) + Δt * k3)
    H(temp_state.data(), k4.data(), N);
    cblas_zscal(N, &neg_i, k4.data(), 1);
    
    // ψ(t + Δt) = ψ(t) + (Δt/6) * (k1 + 2*k2 + 2*k3 + k4)
    Complex dt_sixth = Complex(delta_t / 6.0, 0);
    Complex two_dt_sixth = Complex(delta_t / 3.0, 0);
    
    cblas_zaxpy(N, &dt_sixth, k1.data(), 1, tpq_state.data(), 1);
    cblas_zaxpy(N, &two_dt_sixth, k2.data(), 1, tpq_state.data(), 1);
    cblas_zaxpy(N, &two_dt_sixth, k3.data(), 1, tpq_state.data(), 1);
    cblas_zaxpy(N, &dt_sixth, k4.data(), 1, tpq_state.data(), 1);
    
    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, tpq_state.data(), 1);
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, tpq_state.data(), 1);
    }
}

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
) {
    if (accuracy_level == 1) {
        // Fast: Use improved Taylor with higher order
        time_evolve_tpq_state(H, tpq_state, N, delta_t, 50, normalize);
    } else if (accuracy_level == 2) {
        // Balanced: Use Krylov method
        int krylov_dim = std::min(30, N/2);
        time_evolve_tpq_krylov(H, tpq_state, N, delta_t, krylov_dim, normalize);
    } else {
        // High accuracy: Use RK4 for small systems, Krylov for large
        if (N < 1000) {
            time_evolve_tpq_rk4(H, tpq_state, N, delta_t, normalize);
        } else {
            int krylov_dim = std::min(50, N/2);
            time_evolve_tpq_krylov(H, tpq_state, N, delta_t, krylov_dim, normalize);
        }
    }
}

/**
 * Compute S(q)S(-q) time evolution using Krylov method with momentum-dependent operators
 * 
 * This function creates SumOperators for each momentum vector and spin component,
 * then computes the time correlation function C^α(q,t) = ⟨ψ|S^α(-q,0)S^α(q,t)|ψ⟩
 * using the Krylov-based time evolution method for high accuracy.
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state vector
 * @param momentum_positions List of momentum vectors [Qx, Qy, Qz]
 * @param positions_file Path to file containing site positions
 * @param N Dimension of the Hilbert space
 * @param num_sites Number of lattice sites
 * @param spin_length Spin quantum number
 * @param dir Output directory
 * @param sample Current sample index
 * @param inv_temp Inverse temperature
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_points Number of frequency points
 * @param t_end Maximum evolution time
 * @param dt Time step
 * @param krylov_dim Dimension of Krylov subspace for time evolution
 */
void computeSpinStructureFactorKrylov(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::vector<std::vector<double>>& momentum_positions,
    const std::string& positions_file,
    int N,
    int num_sites,
    float spin_length,
    const std::string& dir,
    int sample,
    double inv_temp,
    double omega_min = -10.0,
    double omega_max = 10.0,
    int num_points = 1000,
    double t_end = 100.0,
    double dt = 0.01,
    int krylov_dim = 30,
    std::vector<std::pair<int, int>> spin_combinations = {{ {0, 0}, {2, 2}, {0, 1}, {0, 2}, {2, 1} }},
    std::vector<const char*> combination_names = { "SpSm", "SzSz", "SpSp", "SpSz", "SzSp" }
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
    save_tpq_state(tpq_state, state_file);

    std::cout << "Computing spin structure factor S(q)S(-q) using Krylov method for sample " << sample 
              << ", beta = " << inv_temp << ", for " << momentum_positions.size() << " momentum points" << std::endl;
    
    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::min(krylov_dim, N/2);
    
    int num_steps = static_cast<int>(t_end / dt) + 1;
    
    // Pre-allocate reusable buffers to avoid repeated allocations
    ComplexVector evolved_psi(N);
    ComplexVector evolved_S_minus_q_psi(N);
    ComplexVector S_q_psi(N);
    ComplexVector S_minus_q_psi(N);
    ComplexVector S_q_evolved_psi(N);
    std::vector<Complex> time_correlation(num_steps);
    
    // Process each momentum point
    for (size_t q_idx = 0; q_idx < momentum_positions.size(); q_idx++) {
        const auto& Q_vector = momentum_positions[q_idx];
        
        // Create momentum label for filenames
        std::stringstream q_label_ss;
        q_label_ss << "q_" << q_idx 
                   << "_Qx" << Q_vector[0] 
                   << "_Qy" << Q_vector[1] 
                   << "_Qz" << Q_vector[2];
        std::string q_label = q_label_ss.str();
        
        std::cout << "Processing momentum point " << (q_idx + 1) << "/" << momentum_positions.size() 
                  << ": Q = [" << Q_vector[0] << ", " << Q_vector[1] << ", " << Q_vector[2] << "]" << std::endl;
        
        // Pre-compute negative Q vector
        const std::vector<double> neg_Q_vector = {-Q_vector[0], -Q_vector[1], -Q_vector[2]};
        
        
        for (size_t combo_idx = 0; combo_idx < spin_combinations.size(); combo_idx++) {
            int op_type_1 = spin_combinations[combo_idx].first;   // First operator type
            int op_type_2 = spin_combinations[combo_idx].second;  // Second operator type
            std::string op_name = std::string(combination_names[combo_idx]) + "_" + q_label;
            
            std::cout << "  Computing " << op_name << " correlations..." << std::endl;

            SumOperator S_q_op(num_sites, spin_length, op_type_1, Q_vector, positions_file);
            SumOperator S_minus_q_op(num_sites, spin_length, op_type_2, neg_Q_vector, positions_file);

            // Apply S^α(-q) to initial state: |φ⟩ = S^α(-q)|ψ⟩
            S_minus_q_op.apply(tpq_state.data(), S_minus_q_psi.data(), N);
            
            // Calculate initial correlation: C(0) = ⟨ψ|S^β(q)S^α(-q)|ψ⟩
            S_q_op.apply(tpq_state.data(), S_q_psi.data(), N);
            
            // Use BLAS for dot product
            Complex initial_corr;
            cblas_zdotc_sub(N, S_q_psi.data(), 1, S_minus_q_psi.data(), 1, &initial_corr);
            time_correlation[0] = initial_corr;

            std::cout << "    Initial correlation C(0) = " 
                      << initial_corr.real() << " + i*" 
                      << initial_corr.imag() << std::endl;
            
            // Initialize evolution states
            std::copy(tpq_state.begin(), tpq_state.end(), evolved_psi.begin());
            std::copy(S_minus_q_psi.begin(), S_minus_q_psi.end(), evolved_S_minus_q_psi.begin());
            
            // Open file for writing time correlation data
            std::string base_name = dir + "/" + op_name + "_sample" + std::to_string(sample) + 
                       "_beta" + std::to_string(inv_temp);
            std::string time_corr_file = base_name + "_time_correlation.dat";
            
            std::ofstream time_corr_out(time_corr_file);
            if (time_corr_out.is_open()) {
                time_corr_out << "# time real(C(t)) imag(C(t))" << std::endl;
                time_corr_out << std::setprecision(16);
                
                // Write initial time point
                time_corr_out << 0.0 << " " 
                    << time_correlation[0].real() << " " 
                    << time_correlation[0].imag() << std::endl;
                time_corr_out.flush(); // Ensure data is written to disk
            }
            
            // Time evolution loop using Krylov method
            for (int step = 1; step < num_steps; step++) {
                // Evolve states using Krylov method
                time_evolve_tpq_krylov(H, evolved_psi, N, dt, krylov_dim, true);
                time_evolve_tpq_krylov(H, evolved_S_minus_q_psi, N, dt, krylov_dim, true);
                
                // Apply S^β(q) to evolved state
                S_q_op.apply(evolved_psi.data(), S_q_evolved_psi.data(), N);
                
                // Calculate correlation using BLAS
                Complex corr_t;
                cblas_zdotc_sub(N, S_q_evolved_psi.data(), 1, evolved_S_minus_q_psi.data(), 1, &corr_t);
                time_correlation[step] = corr_t;
                
                // Write current time point to file
                if (time_corr_out.is_open()) {
                    double t = step * dt;
                    time_corr_out << t << " " 
                        << time_correlation[step].real() << " " 
                        << time_correlation[step].imag() << std::endl;
                    
                    // Flush every 100 steps to ensure data is written
                    if (step % 100 == 0) {
                        time_corr_out.flush();
                    }
                }
                
                if (step % 100 == 0) {
                    std::cout << "    Completed time step " << step << " / " << num_steps << std::endl;
                    std::cout << "    Current correlation: " 
                              << time_correlation[step].real() << " + i*" 
                              << time_correlation[step].imag() << " at t = " << step * dt << std::endl;
                }
            }
            
            // Close the file
            if (time_corr_out.is_open()) {
                time_corr_out.close();
                std::cout << "    Time correlation saved to " << time_corr_file << std::endl;
            }
        }
        std::cout << "Completed momentum point " << (q_idx + 1) << " of " << momentum_positions.size() << std::endl;
    }
    
    std::cout << "Spin structure factor calculation using Krylov method complete for sample " << sample << std::endl;
}

/**
 * Wrapper function to compute spin structure factor using Krylov method during TPQ calculation
 * 
 * This function provides an easy interface to compute S(q)S(-q) correlations with the
 * high-accuracy Krylov time evolution method.
 * 
 * @param H Hamiltonian operator function
 * @param tpq_state Current TPQ state
 * @param positions_file Path to lattice positions file
 * @param lattice_type Type of lattice for momentum point generation
 * @param N Dimension of Hilbert space
 * @param num_sites Number of lattice sites
 * @param spin_length Spin quantum number
 * @param dir Output directory
 * @param sample Sample index
 * @param inv_temp Current inverse temperature
 * @param custom_momentum_points Optional custom momentum vectors
 * @param krylov_dim Dimension of Krylov subspace (default: 30)
 */
void computeTPQSpinStructureFactorKrylov(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::string& positions_file,
    int N,
    int num_sites, 
    float spin_length,
    const std::string& dir,
    int sample,
    double inv_temp,
    const std::vector<std::vector<double>>& custom_momentum_points = {},
    int krylov_dim = 30,
    std::vector<std::pair<int, int>> spin_combinations = {{ {0, 0}, {2, 2}, {0, 1}, {0, 2}, {2, 1} }},
    std::vector<const char*> combination_names = { "SpSm", "SzSz", "SpSp", "SpSz", "SzSp" }
) {
    // Use custom momentum points if provided, otherwise generate standard ones
    std::vector<std::vector<double>> momentum_points;
    if (!custom_momentum_points.empty()) {
        momentum_points = custom_momentum_points;
        std::cout << "Using " << momentum_points.size() << " custom momentum points" << std::endl;
    } else {
        momentum_points = {{0.0, 0.0, 0.0}}; // Default to zero momentum if none provided
    }
    
    // Compute the spin structure factor using Krylov method
    computeSpinStructureFactorKrylov(
        H,
        tpq_state,
        momentum_points,
        positions_file,
        N, num_sites, spin_length,
        dir, sample, inv_temp,
        -10.0,      // omega_min
        10.0,       // omega_max
        2000,       // num_points  
        50.0,       // t_end
        0.01,       // dt
        krylov_dim,  // Krylov dimension
        spin_combinations, // spin combinations
        combination_names // combination names
    );
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
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    int num_points = 10000,
    double t_end = 50.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    int sublattice_size = 1
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    
    int num_sites = static_cast<int>(std::log2(N));

    eigenvalues.clear();

    // Create Sz operators
    std::vector<SingleSiteOperator> Sz_ops = createSzOperators(num_sites, spin_length);
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops = createSingleOperators_pair(num_sites, spin_length);

    std::function<void(const Complex*, Complex*, int)> U_t;
    std::function<void(const Complex*, Complex*, int)> U_nt;
    std::vector<std::vector<double>> momentum_positions;
    if (compute_observables) {
        momentum_positions = {{0,0,0},
                             {0,0,4*M_PI},
                             {0,0,2*M_PI}};
    }

    std::string position_file;
    if (!dir.empty()) {
        size_t last_slash_pos = dir.find_last_of('/');
        if (last_slash_pos != std::string::npos) {
            // Check if the last character is a slash, if so, find the previous one
            if (last_slash_pos == dir.length() - 1) {
                last_slash_pos = dir.find_last_of('/', last_slash_pos - 1);
            }
            if (last_slash_pos != std::string::npos) {
                position_file = dir.substr(0, last_slash_pos) + "/positions.dat";
            } else {
                position_file = "positions.dat"; // In case dir is just a name without slashes
            }
        } else {
            position_file = "positions.dat"; // Relative path
        }
    }



    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::vector<bool> temp_measured(num_temp_points, false);
        std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        ComplexVector v0(N);
        H(v1.data(), v0.data(), N);

        // For each element, compute v0 = (L-H)|v1⟩ = Lv1 - v0
        for (int i = 0; i < N; i++) {
            v0[i] = (LargeValue * num_sites * v1[i]) - v0[i];
        }

        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance(H, v0, N);
        double nsite = N; // This should be the actual number of sites, approximating as N for now
        inv_temp = (2.0) / (LargeValue* num_sites - energy1);

        double first_norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale_factor = Complex(1.0/first_norm, 0.0);

        cblas_zscal(N, &scale_factor, v0.data(), 1);

        double current_norm = first_norm;
        
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Compute v1 = H|v0⟩
            H(v0.data(), v1.data(), N);
            
            // For each element, compute v0 = (L-H)|v0⟩ = L*v0 - v1
            for (int i = 0; i < N; i++) {
                v0[i] = (LargeValue * num_sites * v0[i]) - v1[i];
            }

            current_norm = cblas_dznrm2(N, v0.data(), 1);
            scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v0, N);
            
            // Update inverse temperature
            inv_temp = (2.0*step) / (LargeValue * num_sites - energy_step);
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            energy1 = energy_step;

            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                if (measure_sz){
                    writeFluctuationData(flct_file, spin_corr, inv_temp, v0, num_sites, spin_length, Sz_ops, double_site_ops, sublattice_size, step);
                }
            }
            // If inv_temp is at one of the specified inverse temperature points, compute observables
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(inv_temp - measure_inv_temp[i]) < 4e-3) {
                    std::cout << "Computing observables at inv_temp = " << inv_temp << std::endl;
                    if (compute_observables) {
                        // computeSpinStructureFactorKrylov(H, v0, momentum_positions, position_file, N, num_sites, spin_length, dir, sample, inv_temp);
                        // Just save the state for now
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
                        save_tpq_state(v0, state_file);
                    }
                    temp_measured[i] = true; // Mark this temperature as measured
                }
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy1);
    }
}


#if defined(WITH_CUDA)
// Device-side helper: energy and variance using GPU matvec and BLAS wrappers
inline std::pair<double,double> calculateEnergyAndVariance_cuda(
    GpuMatvec H_dev,
    const DeviceVector& d_v,
    int N,
    CudaContext& ctx
){
    DeviceVector d_Hv(N);
    DeviceVector d_H2v(N);
    // H|v>
    H_dev(d_v.ptr, d_Hv.ptr, N);
    // energy = <v|H|v> (real for Hermitian H)
    Complex e = dz_dotc_device(N, d_v.ptr, d_Hv.ptr, ctx);
    double energy = e.real();
    // H^2|v>
    H_dev(d_Hv.ptr, d_H2v.ptr, N);
    Complex h2 = dz_dotc_device(N, d_v.ptr, d_H2v.ptr, ctx);
    double variance = h2.real() - energy * energy;
    return {energy, variance};
}

// CUDA-optimized microcanonical TPQ main loop operating in device memory.
// Uses GPU BLAS for norms/axpy/scal and a device matvec callback for H.
inline void microcanonical_tpq_cuda(
    GpuMatvec H_dev,
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
    CudaContext* pCtx = nullptr
){
    // Create output directory if needed
    if (!dir.empty()) { ensureDirectoryExists(dir); }

    int num_sites = static_cast<int>(std::log2(N));
    eigenvalues.clear();

    // Pre-create host-side operator structures needed for fluctuation I/O
    std::vector<SingleSiteOperator> Sz_ops = createSzOperators(num_sites, spin_length);
    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);

    // Temperature checkpoints (same as CPU path)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);
    double log_max = std::log10(1000);
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    CudaContext localCtx;
    CudaContext& ctx = pCtx ? *pCtx : localCtx;

    for (int sample = 0; sample < num_samples; ++sample) {
        std::vector<bool> temp_measured(num_temp_points, false);
        std::cout << "TPQ (CUDA) sample " << sample+1 << " of " << num_samples << std::endl;

        // Setup filenames
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);

        // Generate initial random state on host, then copy to device
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector h_v1 = generateTPQVector(N, seed);

        DeviceVector d_v0(N), d_v1(N), d_tmp(N);
        copyHostToDevice(d_v1, h_v1.data(), N);

        // d_tmp = H|v1>
        H_dev(d_v1.ptr, d_tmp.ptr, N);
        // d_v0 = (L*ns)*v1 - H|v1>
        dz_copy_device(N, d_v1.ptr, d_v0.ptr, ctx);
        dz_scal_device(N, Complex(LargeValue * num_sites, 0.0), d_v0.ptr, ctx);
        dz_axpy_device(N, Complex(-1.0, 0.0), d_tmp.ptr, d_v0.ptr, ctx);

        // Initial energy/variance and normalization
        auto [energy1, variance1] = calculateEnergyAndVariance_cuda(H_dev, d_v0, N, ctx);
        double inv_temp = (2.0) / (LargeValue * num_sites - energy1);
        double first_norm = dz_nrm2_device(d_v0.ptr, N, ctx);
        if (first_norm > 0.0) { dz_scal_device(N, Complex(1.0/first_norm, 0.0), d_v0.ptr, ctx); }

        double current_norm = first_norm;
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, /*step=*/1);
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " "
                     << current_norm << " " << first_norm << " " << 1 << std::endl;
        }

        // Main TPQ loop
        int step = 2;
        for (; step <= max_iter; ++step) {
            if (step % (std::max(1, max_iter/10)) == 0 || step == max_iter) {
                std::cout << "  [CUDA] Step " << step << " of " << max_iter << std::endl;
            }

            // d_v1 = H|v0>
            H_dev(d_v0.ptr, d_v1.ptr, N);

            // d_v0 = (L*ns)*v0 - v1
            dz_scal_device(N, Complex(LargeValue * num_sites, 0.0), d_v0.ptr, ctx);
            dz_axpy_device(N, Complex(-1.0, 0.0), d_v1.ptr, d_v0.ptr, ctx);

            // Normalize
            current_norm = dz_nrm2_device(d_v0.ptr, N, ctx);
            if (current_norm > 0.0) {
                dz_scal_device(N, Complex(1.0/current_norm, 0.0), d_v0.ptr, ctx);
            }

            // Energy/variance and inverse temperature
            auto [energy_step, variance_step] = calculateEnergyAndVariance_cuda(H_dev, d_v0, N, ctx);
            inv_temp = (2.0 * step) / (LargeValue * num_sites - energy_step);

            // Write outputs
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " "
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }

            energy1 = energy_step;

            // Optional fluctuation measurements: copy to host only when needed
            if ((measure_sz && (step % temp_interval == 0 || step == max_iter))) {
                ComplexVector h_v0(N);
                copyDeviceToHost(h_v0.data(), d_v0, N);
                writeFluctuationData(flct_file, spin_corr, inv_temp, h_v0,
                                     num_sites, spin_length, Sz_ops, double_site_ops, sublattice_size, step);
            }

            // Save states near target inverse temperatures (for later observables)
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(inv_temp - measure_inv_temp[i]) < 4e-3) {
                    if (compute_observables) {
                        ComplexVector h_v0(N);
                        copyDeviceToHost(h_v0.data(), d_v0, N);
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
                        save_tpq_state(h_v0, state_file);
                    }
                    temp_measured[i] = true;
                }
            }
        }

        // Store final energy
        eigenvalues.push_back(energy1);
    }
}

// Unified entry: choose CPU or CUDA path at runtime. When CUDA is requested but not available,
// it will fall back to CPU.
inline void microcanonical_tpq_unified(
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
    bool use_cuda = false
){
    if (!use_cuda) {
        // CPU path
        microcanonical_tpq(H, N, max_iter, num_samples, temp_interval, eigenvalues, dir,
                           compute_spectrum, LargeValue, compute_observables, observables, observable_names,
                           omega_min, omega_max, num_points, t_end, dt, spin_length, measure_sz, sublattice_size);
        return;
    }

    // CUDA path: by default wrap host matvec (for convenience). For best performance,
    // supply a native device-side matvec to microcanonical_tpq_cuda.
    try {
        CudaContext ctx;
        GpuMatvec H_dev = wrap_host_matvec(H);
        microcanonical_tpq_cuda(H_dev, N, max_iter, num_samples, temp_interval, eigenvalues, dir,
                                compute_spectrum, LargeValue, compute_observables, observables, observable_names,
                                omega_min, omega_max, num_points, t_end, dt, spin_length, measure_sz, sublattice_size, &ctx);
    } catch (const std::exception& e) {
        std::cerr << "CUDA path failed (" << e.what() << "), falling back to CPU." << std::endl;
        microcanonical_tpq(H, N, max_iter, num_samples, temp_interval, eigenvalues, dir,
                           compute_spectrum, LargeValue, compute_observables, observables, observable_names,
                           omega_min, omega_max, num_points, t_end, dt, spin_length, measure_sz, sublattice_size);
    }
}
#endif // WITH_CUDA


#endif // TPQ_H