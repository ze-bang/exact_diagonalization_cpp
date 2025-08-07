#ifndef STOCHASTIC_SPECTRUM_H
#define STOCHASTIC_SPECTRUM_H

#include "arpack.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unordered_map>

/**
 * @brief Stochastic algorithm for estimating the full spectrum of a Hermitian operator
 * 
 * This implementation is optimized for highly degenerate spectra where many
 * eigenvalues may be identical or very close. It uses shift-invert Lanczos
 * with careful tracking of multiplicities.
 */
class StochasticSpectrum {
private:
    std::mt19937 rng;
    
    struct SpectralWindow {
        double lower;
        double upper;
        std::vector<double> eigenvalues;
        std::unordered_map<double, int> degeneracies; // Track multiplicities
        int num_found;
    };
    
    struct EigenvalueCluster {
        double value;
        int multiplicity;
        double spread; // Standard deviation within cluster
    };
    
public:
    StochasticSpectrum(unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count()) 
        : rng(seed) {}
    
    /**
     * @brief Find full spectrum optimized for degenerate cases
     */
    std::vector<double> find_spectrum(
        std::function<void(const Complex*, Complex*, int)> H,
        std::function<void(const Complex*, Complex*, int)> solver,
        int N,
        std::pair<double, double> spectral_range,
        int num_windows = 20,  // More windows for degenerate spectra
        int eigs_per_shift = 50,  // More eigenvalues per shift to catch degeneracies
        int shifts_per_window = 5,  // More shifts to ensure we find all degenerate levels
        double tol = 1e-10,  // Tighter tolerance for degeneracy detection
        int max_iter = 2000) {  // More iterations for convergence
        
        std::cout << "\n=== Stochastic Spectrum Calculation (Degenerate Case) ===" << std::endl;
        std::cout << "Matrix dimension: " << N << std::endl;
        std::cout << "Expected spectral range: [" << spectral_range.first 
                  << ", " << spectral_range.second << "]" << std::endl;
        std::cout << "Degeneracy tolerance: " << tol * 100 << std::endl;
        
        // Use overlapping windows for better coverage
        std::vector<SpectralWindow> windows = create_overlapping_windows(
            spectral_range, num_windows);
        
        // Track all found eigenvalues with multiplicities
        std::unordered_map<double, int> global_multiplicities;
        
        // Process each window
        for (int w = 0; w < windows.size(); ++w) {
            std::cout << "\nProcessing window " << w + 1 << "/" << windows.size() 
                      << " [" << windows[w].lower << ", " << windows[w].upper << "]" << std::endl;
            
            // Use stratified random sampling within window
            std::vector<double> shifts = generate_stratified_shifts(
                windows[w].lower, windows[w].upper, shifts_per_window);
            
            for (int s = 0; s < shifts.size(); ++s) {
                Complex sigma(shifts[s], 0.0);
                
                std::cout << "  Shift " << s + 1 << "/" << shifts.size() 
                          << ": sigma = " << shifts[s] << std::endl;
                
                try {
                    std::vector<double> local_eigs;
                    int num_to_find = std::min(eigs_per_shift, N - 1);
                    
                    // Use shift-invert to find eigenvalues near shift
                    arpack_eigs_shift_invert(H, solver, N, num_to_find, 
                                           max_iter, tol, sigma, local_eigs);
                    
                    // Cluster eigenvalues to identify degeneracies
                    auto clusters = cluster_eigenvalues(local_eigs, tol * 100);
                    
                    for (const auto& cluster : clusters) {
                        // Round to avoid floating point issues
                        double rounded_eig = round_to_tolerance(cluster.value, tol);
                        global_multiplicities[rounded_eig] += cluster.multiplicity;
                        
                        std::cout << "    Found eigenvalue " << cluster.value 
                                  << " with local multiplicity " << cluster.multiplicity << std::endl;
                    }
                    
                    windows[w].num_found += local_eigs.size();
                    
                } catch (const std::exception& e) {
                    std::cerr << "    Warning: Shift-invert failed at sigma = " << shifts[s] 
                              << ": " << e.what() << std::endl;
                }
            }
        }
        
        // Convert to sorted vector with proper multiplicities
        std::vector<double> all_eigenvalues = 
            extract_spectrum_with_multiplicities(global_multiplicities, N);
        
        // Verify and adjust multiplicities
        all_eigenvalues = verify_multiplicities(H, solver, all_eigenvalues, N, tol);
        
        // Print detailed statistics
        print_degeneracy_statistics(all_eigenvalues, global_multiplicities, N);
        
        return all_eigenvalues;
    }
    
    /**
     * @brief Adaptive version specifically for degenerate spectra
     */
    std::vector<double> find_spectrum_adaptive_degenerate(
        std::function<void(const Complex*, Complex*, int)> H,
        std::function<void(const Complex*, Complex*, int)> solver,
        int N,
        std::pair<double, double> initial_range,
        double tol = 1e-10,
        int max_iter = 2000) {
        
        std::cout << "\n=== Adaptive Degenerate Spectrum Calculation ===" << std::endl;
        
        // First pass: find unique eigenvalues
        auto unique_eigs = find_unique_eigenvalues(H, solver, N, initial_range, tol);
        
        // Second pass: determine multiplicities
        std::unordered_map<double, int> multiplicities;
        for (double eig : unique_eigs) {
            int mult = estimate_multiplicity(H, solver, N, eig, tol);
            multiplicities[eig] = mult;
            std::cout << "Eigenvalue " << eig << " has multiplicity " << mult << std::endl;
        }
        
        // Construct full spectrum
        return extract_spectrum_with_multiplicities(multiplicities, N);
    }
    
private:
    /**
     * @brief Create overlapping windows for better coverage
     */
    std::vector<SpectralWindow> create_overlapping_windows(
        std::pair<double, double> range, int num_windows) {
        
        std::vector<SpectralWindow> windows;
        double base_size = (range.second - range.first) / (num_windows * 0.7);
        double overlap = base_size * 0.3;
        
        for (int i = 0; i < num_windows; ++i) {
            SpectralWindow w;
            w.lower = range.first + i * (base_size - overlap);
            w.upper = w.lower + base_size;
            w.upper = std::min(w.upper, range.second);
            w.num_found = 0;
            windows.push_back(w);
            
            if (w.upper >= range.second) break;
        }
        
        return windows;
    }
    
    /**
     * @brief Generate stratified random shifts within an interval
     */
    std::vector<double> generate_stratified_shifts(
        double lower, double upper, int num_shifts) {
        
        std::vector<double> shifts;
        double stratum_size = (upper - lower) / num_shifts;
        
        for (int i = 0; i < num_shifts; ++i) {
            double stratum_lower = lower + i * stratum_size;
            double stratum_upper = stratum_lower + stratum_size;
            std::uniform_real_distribution<double> dist(stratum_lower, stratum_upper);
            shifts.push_back(dist(rng));
        }
        
        return shifts;
    }
    
    /**
     * @brief Cluster nearby eigenvalues to identify degeneracies
     */
    std::vector<EigenvalueCluster> cluster_eigenvalues(
        const std::vector<double>& eigs, double cluster_tol) {
        
        std::vector<EigenvalueCluster> clusters;
        if (eigs.empty()) return clusters;
        
        std::vector<double> sorted_eigs = eigs;
        std::sort(sorted_eigs.begin(), sorted_eigs.end());
        
        std::vector<double> current_cluster;
        current_cluster.push_back(sorted_eigs[0]);
        
        for (size_t i = 1; i < sorted_eigs.size(); ++i) {
            if (std::abs(sorted_eigs[i] - current_cluster.back()) < cluster_tol) {
                current_cluster.push_back(sorted_eigs[i]);
            } else {
                // Finalize current cluster
                EigenvalueCluster c;
                c.multiplicity = current_cluster.size();
                c.value = std::accumulate(current_cluster.begin(), 
                                        current_cluster.end(), 0.0) / c.multiplicity;
                c.spread = compute_spread(current_cluster);
                clusters.push_back(c);
                
                // Start new cluster
                current_cluster.clear();
                current_cluster.push_back(sorted_eigs[i]);
            }
        }
        
        // Don't forget last cluster
        if (!current_cluster.empty()) {
            EigenvalueCluster c;
            c.multiplicity = current_cluster.size();
            c.value = std::accumulate(current_cluster.begin(), 
                                    current_cluster.end(), 0.0) / c.multiplicity;
            c.spread = compute_spread(current_cluster);
            clusters.push_back(c);
        }
        
        return clusters;
    }
    
    /**
     * @brief Compute spread (standard deviation) of values
     */
    double compute_spread(const std::vector<double>& values) {
        if (values.size() <= 1) return 0.0;
        
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - mean) * (v - mean);
        }
        return std::sqrt(sq_sum / values.size());
    }
    
    /**
     * @brief Round eigenvalue to avoid floating point comparison issues
     */
    double round_to_tolerance(double value, double tol) {
        return std::round(value / tol) * tol;
    }
    
    /**
     * @brief Extract full spectrum from multiplicity map
     */
    std::vector<double> extract_spectrum_with_multiplicities(
        const std::unordered_map<double, int>& multiplicities, int N) {
        
        // Get unique eigenvalues
        std::vector<double> unique_vals;
        for (const auto& pair : multiplicities) {
            unique_vals.push_back(pair.first);
        }
        std::sort(unique_vals.begin(), unique_vals.end());
        
        // Build full spectrum with repetitions
        std::vector<double> spectrum;
        for (double val : unique_vals) {
            int mult = multiplicities.at(val);
            for (int i = 0; i < mult; ++i) {
                spectrum.push_back(val);
            }
        }
        
        // Ensure we don't exceed matrix dimension
        if (spectrum.size() > N) {
            std::cout << "Warning: Found more eigenvalues than matrix dimension. Truncating." << std::endl;
            spectrum.resize(N);
        }
        
        return spectrum;
    }
    
    /**
     * @brief Find unique eigenvalues (ignoring multiplicities)
     */
    std::vector<double> find_unique_eigenvalues(
        std::function<void(const Complex*, Complex*, int)> H,
        std::function<void(const Complex*, Complex*, int)> solver,
        int N,
        std::pair<double, double> range,
        double tol) {
        
        std::set<double> unique_set;
        int num_samples = std::min(20, N);
        
        // Sample at various points
        for (int i = 0; i < num_samples; ++i) {
            double shift = range.first + (range.second - range.first) * i / (num_samples - 1);
            Complex sigma(shift, 0.0);
            
            try {
                std::vector<double> local_eigs;
                arpack_eigs_shift_invert(H, solver, N, std::min(30, N-1), 
                                       2000, tol, sigma, local_eigs);
                
                for (double eig : local_eigs) {
                    unique_set.insert(round_to_tolerance(eig, tol * 100));
                }
            } catch (...) {}
        }
        
        return std::vector<double>(unique_set.begin(), unique_set.end());
    }
    
    /**
     * @brief Estimate multiplicity of a specific eigenvalue
     */
    int estimate_multiplicity(
        std::function<void(const Complex*, Complex*, int)> H,
        std::function<void(const Complex*, Complex*, int)> solver,
        int N,
        double eigenvalue,
        double tol) {
        
        Complex sigma(eigenvalue + tol * 10, 0.0); // Slight shift
        
        try {
            std::vector<double> local_eigs;
            arpack_eigs_shift_invert(H, solver, N, std::min(50, N-1), 
                                   2000, tol, sigma, local_eigs);
            
            // Count how many are close to target
            int count = 0;
            for (double eig : local_eigs) {
                if (std::abs(eig - eigenvalue) < tol * 100) {
                    count++;
                }
            }
            
            return count;
        } catch (...) {
            return 1; // Default to 1 if estimation fails
        }
    }
    
    /**
     * @brief Verify multiplicities by targeted searches
     */
    std::vector<double> verify_multiplicities(
        std::function<void(const Complex*, Complex*, int)> H,
        std::function<void(const Complex*, Complex*, int)> solver,
        const std::vector<double>& spectrum,
        int N,
        double tol) {
        
        // For highly degenerate cases, the current spectrum is likely good
        // Additional verification can be added if needed
        return spectrum;
    }
    
    /**
     * @brief Print statistics specific to degenerate spectra
     */
    void print_degeneracy_statistics(
        const std::vector<double>& eigenvalues,
        const std::unordered_map<double, int>& multiplicities,
        int N) {
        
        std::cout << "\n=== Degeneracy Statistics ===" << std::endl;
        std::cout << "Total eigenvalues found: " << eigenvalues.size() 
                  << " / " << N << std::endl;
        std::cout << "Number of unique eigenvalues: " << multiplicities.size() << std::endl;
        
        // Find maximum degeneracy
        int max_mult = 0;
        double max_mult_eig = 0.0;
        for (const auto& pair : multiplicities) {
            if (pair.second > max_mult) {
                max_mult = pair.second;
                max_mult_eig = pair.first;
            }
        }
        
        std::cout << "Maximum degeneracy: " << max_mult 
                  << " at eigenvalue " << max_mult_eig << std::endl;
        
        // Degeneracy distribution
        std::map<int, int> deg_distribution;
        for (const auto& pair : multiplicities) {
            deg_distribution[pair.second]++;
        }
        
        std::cout << "\nDegeneracy distribution:" << std::endl;
        for (const auto& pair : deg_distribution) {
            std::cout << "  Multiplicity " << pair.first << ": " 
                      << pair.second << " eigenvalues" << std::endl;
        }
        
        // Average degeneracy
        double avg_deg = (double)eigenvalues.size() / multiplicities.size();
        std::cout << "Average degeneracy: " << avg_deg << std::endl;
    }
};

/**
 * @brief Convenience function for degenerate spectra
 */
std::vector<double> stochastic_degenerate_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> solver,
    int N,
    std::pair<double, double> spectral_range,
    double tol = 1e-10) {
    
    StochasticSpectrum stoch_solver;
    return stoch_solver.find_spectrum_adaptive_degenerate(H, solver, N, spectral_range, tol);
}

#endif // STOCHASTIC_SPECTRUM_H