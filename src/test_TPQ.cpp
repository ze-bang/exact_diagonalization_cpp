#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <string>
#include "TPQ.h"

// Simple test Hamiltonian: Two-level system (2x2) with energy gap
void testHamiltonian(const Complex* v_in, Complex* v_out, int N) {
    // Hamiltonian matrix elements: [1.0, 0.2; 0.2, -1.0]
    // This represents a simple two-level system with gap and coupling
    if (N != 2) {
        std::cerr << "This test Hamiltonian is implemented for N=2 only" << std::endl;
        return;
    }
    
    v_out[0] = 1.0 * v_in[0] + 0.2 * v_in[1];
    v_out[1] = 0.2 * v_in[0] - 1.0 * v_in[1];
}

// Helper function to compute specific heat from energy and variance
double computeSpecificHeat(double energy, double variance, double beta) {
    return beta * beta * variance;
}

// Helper function to print thermodynamic data
void printThermodynamics(const std::string& type, double beta, double energy, double variance) {
    double temp = 1.0 / beta;
    double specific_heat = computeSpecificHeat(energy, variance, beta);
    
    std::cout << std::fixed << std::setprecision(6)
              << type << " - Temperature: " << std::setw(10) << temp
              << " | Energy: " << std::setw(10) << energy
              << " | Variance: " << std::setw(10) << variance
              << " | Specific Heat: " << std::setw(10) << specific_heat
              << std::endl;
}

// Helper function to plot thermodynamic data
void plotThermodynamicData(const std::string& dir, int sample) {
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::string plot_script = dir + "/plot_thermodynamics.gp";
    
    // Create gnuplot script
    std::ofstream plot(plot_script);
    plot << "set terminal png enhanced\n";
    plot << "set output '" << dir << "/thermodynamics.png'\n";
    plot << "set xlabel 'Inverse Temperature (β)'\n";
    plot << "set ylabel 'Energy / Specific Heat'\n";
    plot << "set grid\n";
    plot << "set key top right\n";
    plot << "plot '" << ss_file << "' using 1:2 with lines title 'Energy', \\\n";
    plot << "     '" << ss_file << "' using 1:(($3)*($1)*($1)) with lines title 'Specific Heat'\n";
    plot.close();
    
    // Execute gnuplot script
    std::string cmd = "gnuplot " + plot_script;
    system(cmd.c_str());
    std::cout << "Thermodynamic plot created at: " << dir << "/thermodynamics.png" << std::endl;
}

// Helper function to analyze cooling behavior
void analyzeThermodynamicCooling(const std::string& dir, int sample) {
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::ifstream file(ss_file);
    std::string line;
    std::getline(file, line); // Skip header
    
    std::vector<double> inverse_temps;
    std::vector<double> energies;
    std::vector<double> variances;
    std::vector<double> specific_heats;
    
    double inv_temp, energy, variance, num, doublon;
    int step;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> inv_temp >> energy >> variance >> num >> doublon >> step)) {
            continue;
        }
        
        inverse_temps.push_back(inv_temp);
        energies.push_back(energy);
        variances.push_back(variance);
        specific_heats.push_back(computeSpecificHeat(energy, variance, inv_temp));
    }
    
    std::cout << "\nThermodynamic Analysis during Cooling:\n";
    std::cout << "===================================\n";
    std::cout << "- Initial energy: " << energies.front() << std::endl;
    std::cout << "- Final energy: " << energies.back() << std::endl;
    
    if (energies.size() > 1) {
        std::cout << "- Energy decrease: " << (energies.front() - energies.back()) << std::endl;
        
        // Find maximum specific heat and its temperature
        auto max_specific_heat_it = std::max_element(specific_heats.begin(), specific_heats.end());
        int max_idx = std::distance(specific_heats.begin(), max_specific_heat_it);
        
        std::cout << "- Maximum specific heat: " << *max_specific_heat_it 
                  << " at β = " << inverse_temps[max_idx] 
                  << " (T = " << 1.0/inverse_temps[max_idx] << ")" << std::endl;
    }
}

int main() {
    std::cout << "Testing TPQ Thermodynamics as the System Cools Down" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Test parameters
    const int N = 2;  // Hilbert space dimension
    const int max_iter = 1000000;
    const int num_samples = 10;
    const int temp_interval = 100;
    std::vector<double> eigenvalues;
    
    // Create output directories
    std::string mcdir = "tpq_microcanonical_test";
    std::string cdir = "tpq_canonical_test";
    ensureDirectoryExists(mcdir);
    ensureDirectoryExists(cdir);

    // 1. Run microcanonical TPQ
    std::cout << "\nRunning microcanonical TPQ..." << std::endl;
    microcanonical_tpq(testHamiltonian, N, max_iter, num_samples, 
                       temp_interval, eigenvalues, mcdir, true);
    
    // 2. Run canonical TPQ with small delta_tau for better resolution
    std::cout << "\nRunning canonical TPQ..." << std::endl;
    canonical_tpq(testHamiltonian, N, max_iter, num_samples, 
                  temp_interval, eigenvalues, cdir, 1e-4, true);

    // 3. Analyze thermodynamic data from both methods
    std::cout << "\nAnalyzing thermodynamic data..." << std::endl;
    
    // Print a few data points from microcanonical TPQ
    std::string mc_file = mcdir + "/SS_rand0.dat";
    std::ifstream mc_data(mc_file);
    std::string line;
    std::getline(mc_data, line); // Skip header
    
    std::cout << "\nMicrocanonical TPQ results (selective points):" << std::endl;
    
    for (int i = 0; i < 10000; i++) {
        if (std::getline(mc_data, line)) {
            double inv_temp, energy, variance, num, doublon;
            int step;
            std::istringstream iss(line);
            if (iss >> inv_temp >> energy >> variance >> num >> doublon >> step) {
                printThermodynamics("MC-TPQ", inv_temp, energy, variance);
            }
        }
    }
    
    // Print final results from microcanonical TPQ
    mc_data.clear();
    mc_data.seekg(0, std::ios::beg);
    std::getline(mc_data, line); // Skip header
    std::string last_line;
    while (std::getline(mc_data, line)) {
        last_line = line;
    }
    
    if (!last_line.empty()) {
        double inv_temp, energy, variance, num, doublon;
        int step;
        std::istringstream iss(last_line);
        if (iss >> inv_temp >> energy >> variance >> num >> doublon >> step) {
            std::cout << "Final state: ";
            printThermodynamics("MC-TPQ", inv_temp, energy, variance);
        }
    }
    
    // Print a few data points from canonical TPQ
    std::string c_file = cdir + "/SS_rand0.dat";
    std::ifstream c_data(c_file);
    std::getline(c_data, line); // Skip header
    
    std::cout << "\nCanonical TPQ results (selective points):" << std::endl;
    
    for (int i = 0; i < 5; i++) {
        if (std::getline(c_data, line)) {
            double inv_temp, energy, variance, num, doublon;
            int step;
            std::istringstream iss(line);
            if (iss >> inv_temp >> energy >> variance >> num >> doublon >> step) {
                printThermodynamics("C-TPQ", inv_temp, energy, variance);
            }
        }
    }
    
    // Print final results from canonical TPQ
    c_data.clear();
    c_data.seekg(0, std::ios::beg);
    std::getline(c_data, line); // Skip header
    last_line = "";
    while (std::getline(c_data, line)) {
        last_line = line;
    }
    
    if (!last_line.empty()) {
        double inv_temp, energy, variance, num, doublon;
        int step;
        std::istringstream iss(last_line);
        if (iss >> inv_temp >> energy >> variance >> num >> doublon >> step) {
            std::cout << "Final state: ";
            printThermodynamics("C-TPQ", inv_temp, energy, variance);
        }
    }
    
    // 4. Create plots of the thermodynamic data
    plotThermodynamicData(mcdir, 0);
    plotThermodynamicData(cdir, 0);
    
    // 5. Analyze thermodynamic behavior as system cools
    analyzeThermodynamicCooling(mcdir, 0);
    analyzeThermodynamicCooling(cdir, 0);

    return 0;
}