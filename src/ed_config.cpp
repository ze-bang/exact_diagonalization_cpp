#include "ed_config.h"
// NOTE: We include ed_wrapper.h ONLY in the implementation of conversion functions
// This is at the end of the file to avoid including it globally
#include <algorithm>
#include <cctype>

// ============================================================================
// Temporary forward declaration to avoid including ed_wrapper.h here
// The actual enum is defined in ed_wrapper.h
// ============================================================================
enum class DiagonalizationMethod {
    LANCZOS,
    LANCZOS_SELECTIVE,
    LANCZOS_NO_ORTHO,
    BLOCK_LANCZOS,
    SHIFT_INVERT,
    SHIFT_INVERT_ROBUST,
    CG,
    BLOCK_CG,
    DAVIDSON,
    BICG,
    LOBPCG,
    KRYLOV_SCHUR,
    FULL,
    OSS,
    mTPQ,
    mTPQ_MPI,
    cTPQ,
    mTPQ_CUDA,
    FTLM,
    LTLM,
    ARPACK_SM,
    ARPACK_LM,
    ARPACK_SHIFT_INVERT,
    ARPACK_ADVANCED,
    LANCZOS_GPU,
    LANCZOS_GPU_FIXED_SZ
};

// ============================================================================
// EDConfig Implementation
// ============================================================================

// Default constructor with LANCZOS as default method
EDConfig::EDConfig() : method(DiagonalizationMethod::LANCZOS) {}

EDConfig EDConfig::fromFile(const std::string& filename) {
    EDConfig config;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << std::endl;
        return config;
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse based on key
        try {
            if (key == "method") {
                auto m = ed_config::parseMethod(value);
                if (m) config.method = *m;
            }
            else if (key == "num_eigenvalues") config.diag.num_eigenvalues = std::stoi(value);
            else if (key == "max_iterations") config.diag.max_iterations = std::stoi(value);
            else if (key == "tolerance") config.diag.tolerance = std::stod(value);
            else if (key == "compute_eigenvectors") config.diag.compute_eigenvectors = (value == "true" || value == "1");
            else if (key == "shift") config.diag.shift = std::stod(value);
            else if (key == "block_size") config.diag.block_size = std::stoi(value);
            else if (key == "num_sites") config.system.num_sites = std::stoi(value);
            else if (key == "spin_length") config.system.spin_length = std::stof(value);
            else if (key == "hamiltonian_dir") config.system.hamiltonian_dir = value;
            else if (key == "use_fixed_sz") config.system.use_fixed_sz = (value == "true" || value == "1");
            else if (key == "n_up") config.system.n_up = std::stoi(value);
            else if (key == "output_dir") config.workflow.output_dir = value;
            else if (key == "num_samples") config.thermal.num_samples = std::stoi(value);
            else if (key == "temp_min") config.thermal.temp_min = std::stod(value);
            else if (key == "temp_max") config.thermal.temp_max = std::stod(value);
            else if (key == "temp_bins") config.thermal.num_temp_bins = std::stoi(value);
            else if (key == "calc_observables") config.observable.calculate = (value == "true" || value == "1");
            else if (key == "measure_spin") config.observable.measure_spin = (value == "true" || value == "1");
            else if (key == "run_standard") config.workflow.run_standard = (value == "true" || value == "1");
            else if (key == "run_symmetrized") config.workflow.run_symmetrized = (value == "true" || value == "1");
            else if (key == "compute_thermo") config.workflow.compute_thermo = (value == "true" || value == "1");
            else {
                std::cerr << "Warning: Unknown config key '" << key << "' at line " << line_num << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_num << ": " << e.what() << std::endl;
        }
    }
    
    return config;
}

EDConfig EDConfig::fromCommandLine(int argc, char* argv[]) {
    EDConfig config;
    
    if (argc < 2) {
        return config;  // Return default config
    }
    
    // First argument is directory
    config.system.hamiltonian_dir = argv[1];
    config.workflow.output_dir = config.system.hamiltonian_dir + "/output";
    
    // Parse remaining arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        auto parse_value = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.length());
        };
        
        try {
            if (arg.find("--method=") == 0) {
                auto m = ed_config::parseMethod(parse_value("--method="));
                if (m) config.method = *m;
            }
            else if (arg.find("--eigenvalues=") == 0) {
                auto val = parse_value("--eigenvalues=");
                if (val == "FULL") {
                    // Will be set after num_sites is known
                    config.diag.num_eigenvalues = -1; // Special marker
                } else {
                    config.diag.num_eigenvalues = std::stoi(val);
                }
            }
            else if (arg.find("--iterations=") == 0) config.diag.max_iterations = std::stoi(parse_value("--iterations="));
            else if (arg.find("--tolerance=") == 0) config.diag.tolerance = std::stod(parse_value("--tolerance="));
            else if (arg == "--eigenvectors") config.diag.compute_eigenvectors = true;
            else if (arg.find("--shift=") == 0) config.diag.shift = std::stod(parse_value("--shift="));
            else if (arg.find("--block-size=") == 0) config.diag.block_size = std::stoi(parse_value("--block-size="));
            else if (arg.find("--num_sites=") == 0) config.system.num_sites = std::stoi(parse_value("--num_sites="));
            else if (arg.find("--spin_length=") == 0) config.system.spin_length = std::stof(parse_value("--spin_length="));
            else if (arg == "--fixed-sz") config.system.use_fixed_sz = true;
            else if (arg.find("--n-up=") == 0) config.system.n_up = std::stoi(parse_value("--n-up="));
            else if (arg.find("--output=") == 0) config.workflow.output_dir = parse_value("--output=");
            else if (arg.find("--samples=") == 0) config.thermal.num_samples = std::stoi(parse_value("--samples="));
            else if (arg.find("--temp_min=") == 0) config.thermal.temp_min = std::stod(parse_value("--temp_min="));
            else if (arg.find("--temp_max=") == 0) config.thermal.temp_max = std::stod(parse_value("--temp_max="));
            else if (arg.find("--temp_bins=") == 0) config.thermal.num_temp_bins = std::stoi(parse_value("--temp_bins="));
            else if (arg.find("--num_order=") == 0) config.thermal.num_order = std::stoi(parse_value("--num_order="));
            else if (arg.find("--num_measure_freq=") == 0) config.thermal.num_measure_freq = std::stoi(parse_value("--num_measure_freq="));
            else if (arg.find("--delta_tau=") == 0) config.thermal.delta_tau = std::stod(parse_value("--delta_tau="));
            else if (arg.find("--large_value=") == 0) config.thermal.large_value = std::stod(parse_value("--large_value="));
            else if (arg == "--calc_observables") config.observable.calculate = true;
            else if (arg == "--measure_spin") config.observable.measure_spin = true;
            else if (arg == "--standard") config.workflow.run_standard = true;
            else if (arg == "--symmetrized") config.workflow.run_symmetrized = true;
            else if (arg == "--thermo") config.workflow.compute_thermo = true;
            else if (arg == "--skip_ED") config.workflow.skip_ed = true;
            else if (arg.find("--sublattice_size=") == 0) config.system.sublattice_size = std::stoi(parse_value("--sublattice_size="));
            else if (arg.find("--omega_min=") == 0) config.observable.omega_min = std::stod(parse_value("--omega_min="));
            else if (arg.find("--omega_max=") == 0) config.observable.omega_max = std::stod(parse_value("--omega_max="));
            else if (arg.find("--num_points=") == 0) config.observable.num_points = std::stoi(parse_value("--num_points="));
            else if (arg.find("--t_end=") == 0) config.observable.t_end = std::stod(parse_value("--t_end="));
            else if (arg.find("--dt=") == 0) config.observable.dt = std::stod(parse_value("--dt="));
            else if (arg.find("--max_subspace=") == 0) config.diag.max_subspace = std::stoi(parse_value("--max_subspace="));
            // ARPACK options
            else if (arg.find("--arpack-which=") == 0) config.arpack.which = parse_value("--arpack-which=");
            else if (arg.find("--arpack-ncv=") == 0) config.arpack.ncv = std::stoi(parse_value("--arpack-ncv="));
            else if (arg.find("--arpack-max-restarts=") == 0) config.arpack.max_restarts = std::stoi(parse_value("--arpack-max-restarts="));
            else if (arg == "--arpack-shift-invert") config.arpack.shift_invert = true;
            else if (arg.find("--arpack-sigma=") == 0) config.arpack.sigma = std::stod(parse_value("--arpack-sigma="));
            else if (arg == "--arpack-verbose") config.arpack.verbose = true;
            else if (arg.find("--config=") == 0) {
                // Load from config file and merge
                auto file_config = EDConfig::fromFile(parse_value("--config="));
                config = file_config.merge(config); // Command line takes precedence
            }
            else if (arg != "--help") {
                std::cerr << "Warning: Unknown option: " << arg << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing argument '" << arg << "': " << e.what() << std::endl;
        }
    }
    
    // Auto-detect num_sites if not specified
    if (config.system.num_sites == 0) {
        config.autoDetectNumSites();
    }
    
    // Handle FULL spectrum case
    if (config.diag.num_eigenvalues == -1 && config.system.num_sites > 0) {
        config.diag.num_eigenvalues = (1ULL << config.system.num_sites);
    }
    
    // Default to standard workflow if nothing specified
    if (!config.workflow.run_standard && !config.workflow.run_symmetrized) {
        config.workflow.run_standard = true;
    }
    
    return config;
}

EDConfig& EDConfig::merge(const EDConfig& other) {
    // Simple merge: other overrides this where values differ from defaults
    // This is a simplified version - could be more sophisticated
    method = other.method;
    
    // Merge diag
    if (other.diag.num_eigenvalues != 1) diag.num_eigenvalues = other.diag.num_eigenvalues;
    if (other.diag.max_iterations != 100000) diag.max_iterations = other.diag.max_iterations;
    if (other.diag.tolerance != 1e-10) diag.tolerance = other.diag.tolerance;
    if (other.diag.compute_eigenvectors) diag.compute_eigenvectors = true;
    
    // Merge system
    if (other.system.num_sites != 0) system.num_sites = other.system.num_sites;
    if (other.system.spin_length != 0.5f) system.spin_length = other.system.spin_length;
    if (!other.system.hamiltonian_dir.empty()) system.hamiltonian_dir = other.system.hamiltonian_dir;
    
    // Merge workflow
    if (other.workflow.run_standard) workflow.run_standard = true;
    if (other.workflow.run_symmetrized) workflow.run_symmetrized = true;
    if (other.workflow.compute_thermo) workflow.compute_thermo = true;
    if (!other.workflow.output_dir.empty()) workflow.output_dir = other.workflow.output_dir;
    
    return *this;
}

bool EDConfig::validate(std::ostream& err) const {
    bool valid = true;
    
    if (system.num_sites == 0) {
        err << "Error: num_sites must be specified or auto-detected\n";
        valid = false;
    }
    
    if (system.hamiltonian_dir.empty()) {
        err << "Error: hamiltonian_dir must be specified\n";
        valid = false;
    }
    
    if (diag.num_eigenvalues < 1) {
        err << "Error: num_eigenvalues must be >= 1\n";
        valid = false;
    }
    
    if (diag.tolerance <= 0) {
        err << "Error: tolerance must be positive\n";
        valid = false;
    }
    
    return valid;
}

void EDConfig::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write config to " << filename << std::endl;
        return;
    }
    
    file << "# ED Configuration\n";
    file << "# Generated configuration file\n\n";
    
    file << "[Diagonalization]\n";
    file << "method = " << ed_config::methodToString(method) << "\n";
    file << "num_eigenvalues = " << diag.num_eigenvalues << "\n";
    file << "max_iterations = " << diag.max_iterations << "\n";
    file << "tolerance = " << diag.tolerance << "\n";
    file << "compute_eigenvectors = " << (diag.compute_eigenvectors ? "true" : "false") << "\n\n";
    
    file << "[System]\n";
    file << "num_sites = " << system.num_sites << "\n";
    file << "spin_length = " << system.spin_length << "\n";
    file << "hamiltonian_dir = " << system.hamiltonian_dir << "\n\n";
    
    file << "[Workflow]\n";
    file << "output_dir = " << workflow.output_dir << "\n";
    file << "run_standard = " << (workflow.run_standard ? "true" : "false") << "\n";
    file << "run_symmetrized = " << (workflow.run_symmetrized ? "true" : "false") << "\n";
    file << "compute_thermo = " << (workflow.compute_thermo ? "true" : "false") << "\n";
}

void EDConfig::print(std::ostream& out) const {
    out << "========================================\n";
    out << "ED Configuration Summary\n";
    out << "========================================\n\n";
    
    out << "Method: " << ed_config::methodToString(method) << "\n";
    out << "System: " << system.num_sites << " sites, spin = " << system.spin_length << "\n";
    
    if (system.use_fixed_sz) {
        int n_up_actual = (system.n_up >= 0) ? system.n_up : system.num_sites / 2;
        double sz = n_up_actual - system.num_sites / 2.0;
        out << "Fixed Sz: n_up = " << n_up_actual << " (Sz = " << sz << ")\n";
        
        // Calculate dimension reduction
        auto binomial = [](int n, int k) {
            if (k > n || k < 0) return 0;
            if (k == 0 || k == n) return 1;
            long long result = 1;
            for (int i = 1; i <= k; ++i) {
                result = result * (n - k + i) / i;
            }
            return (int)result;
        };
        int full_dim = 1 << system.num_sites;
        int fixed_dim = binomial(system.num_sites, n_up_actual);
        out << "Hilbert space: " << fixed_dim << " (reduced from " << full_dim 
            << ", factor: " << (double)full_dim / fixed_dim << "x)\n";
    }
    
    out << "Eigenvalues: " << diag.num_eigenvalues << " (tol=" << diag.tolerance << ")\n";
    out << "Output: " << workflow.output_dir << "\n";
    
    if (workflow.run_standard) out << "  - Running standard diagonalization\n";
    if (workflow.run_symmetrized) out << "  - Running symmetrized diagonalization\n";
    if (workflow.compute_thermo) out << "  - Computing thermodynamics\n";
    if (observable.calculate) out << "  - Calculating observables\n";
    if (observable.measure_spin) out << "  - Measuring spin expectations\n";
    
    out << "========================================\n";
}

bool EDConfig::autoDetectNumSites() {
    std::string positions_file = system.hamiltonian_dir + "/positions.dat";
    std::ifstream file(positions_file);
    
    if (!file.is_open()) {
        return false;
    }
    
    int max_site_id = -1;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        int site_id;
        
        if (iss >> site_id) {
            max_site_id = std::max(max_site_id, site_id);
        }
    }
    
    if (max_site_id >= 0) {
        system.num_sites = max_site_id + 1;
        std::cout << "Auto-detected num_sites = " << system.num_sites << " from positions.dat\n";
        return true;
    }
    
    return false;
}

// ============================================================================
// Conversion Utilities
// ============================================================================

namespace ed_config {

std::optional<DiagonalizationMethod> parseMethod(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "lanczos") return DiagonalizationMethod::LANCZOS;
    if (lower == "full") return DiagonalizationMethod::FULL;
    if (lower == "mtpq") return DiagonalizationMethod::mTPQ;
    if (lower == "ctpq") return DiagonalizationMethod::cTPQ;
    if (lower == "cg") return DiagonalizationMethod::CG;
    if (lower == "davidson") return DiagonalizationMethod::DAVIDSON;
    if (lower == "arpack" || lower == "arpack_sm") return DiagonalizationMethod::ARPACK_SM;
    if (lower == "arpack_lm") return DiagonalizationMethod::ARPACK_LM;
    if (lower == "arpack_advanced") return DiagonalizationMethod::ARPACK_ADVANCED;
    if (lower == "oss") return DiagonalizationMethod::OSS;

    std::cerr << "Warning: Unknown method '" << str << "', using LANCZOS\n";
    return std::nullopt;
}

std::string methodToString(DiagonalizationMethod method) {
    switch (method) {
        case DiagonalizationMethod::LANCZOS: return "LANCZOS";
        case DiagonalizationMethod::FULL: return "FULL";
        case DiagonalizationMethod::mTPQ: return "mTPQ";
        case DiagonalizationMethod::cTPQ: return "cTPQ";
        case DiagonalizationMethod::CG: return "CG";
        case DiagonalizationMethod::DAVIDSON: return "DAVIDSON";
        case DiagonalizationMethod::ARPACK_SM: return "ARPACK_SM";
        case DiagonalizationMethod::ARPACK_LM: return "ARPACK_LM";
        case DiagonalizationMethod::ARPACK_ADVANCED: return "ARPACK_ADVANCED";
        case DiagonalizationMethod::OSS: return "OSS";
        default: return "UNKNOWN";
    }
}

EDConfig defaultConfigFor(DiagonalizationMethod method) {
    EDConfig config(method);
    
    switch (method) {
        case DiagonalizationMethod::mTPQ:
        case DiagonalizationMethod::cTPQ:
            config.thermal.num_samples = 10;
            config.workflow.compute_thermo = true;
            break;
            
        case DiagonalizationMethod::FULL:
            config.diag.num_eigenvalues = -1; // Will be set based on system size
            config.workflow.compute_thermo = true;
            break;
            
        case DiagonalizationMethod::ARPACK_ADVANCED:
            config.arpack.max_restarts = 3;
            config.arpack.two_phase_refine = true;
            break;
            
        default:
            break;
    }
    
    return config;
}

} // namespace ed_config

// ============================================================================
// CommandLineParser Implementation
// ============================================================================

CommandLineParser& CommandLineParser::addOption(
    const std::string& long_name,
    const std::string& short_name,
    const std::string& description,
    bool has_value,
    bool required,
    const std::string& category
) {
    Option opt;
    opt.long_name = long_name;
    opt.short_name = short_name;
    opt.description = description;
    opt.has_value = has_value;
    opt.required = required;
    opt.category = category;
    options_.push_back(opt);
    return *this;
}

bool CommandLineParser::parse(int argc, char* argv[]) {
    program_name_ = argv[0];
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Handle --key=value
        if (arg.find("--") == 0) {
            size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                std::string key = arg.substr(2, eq - 2);
                std::string value = arg.substr(eq + 1);
                values_[key] = value;
            } else {
                // Boolean flag
                std::string key = arg.substr(2);
                values_[key] = "true";
            }
        }
    }
    
    // Check required options
    for (const auto& opt : options_) {
        if (opt.required && values_.find(opt.long_name) == values_.end()) {
            std::cerr << "Error: Required option --" << opt.long_name << " not provided\n";
            return false;
        }
    }
    
    return true;
}

std::optional<std::string> CommandLineParser::get(const std::string& name) const {
    auto it = values_.find(name);
    if (it != values_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool CommandLineParser::has(const std::string& name) const {
    return values_.find(name) != values_.end();
}

void CommandLineParser::printHelp(std::ostream& out) const {
    out << "Usage: " << program_name_ << " <directory> [options]\n\n";
    
    // Group by category
    std::map<std::string, std::vector<Option>> grouped;
    for (const auto& opt : options_) {
        grouped[opt.category].push_back(opt);
    }
    
    for (const auto& [category, opts] : grouped) {
        out << category << ":\n";
        for (const auto& opt : opts) {
            out << "  --" << opt.long_name;
            if (opt.has_value) out << "=<value>";
            if (opt.required) out << " (required)";
            out << "\n      " << opt.description << "\n";
        }
        out << "\n";
    }
}
