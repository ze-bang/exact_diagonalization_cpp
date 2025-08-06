#ifndef ARPACK_H
#define ARPACK_H

#include <iostream>
#include <vector>
#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <map>
#include <iomanip>
#include <fstream>
#include <numeric>

// ARPACK FORTRAN functions are typically postfixed with an underscore
extern "C" {
    void znaupd_(int *ido, char *bmat, int *n, char *which, int *nev,
                 double *tol, std::complex<double> *resid, int *ncv,
                 std::complex<double> *v, int *ldv, int *iparam,
                 int *ipntr, std::complex<double> *workd,
                 std::complex<double> *workl, int *lworkl,
                 double *rwork, int *info);

    void zneupd_(int *rvec, char *howmny, int *select,
                 std::complex<double> *d, std::complex<double> *z,
                 int *ldz, std::complex<double> *sigma,
                 std::complex<double> *workev, char *bmat, int *n,
                 char *which, int *nev, double *tol,
                 std::complex<double> *resid, int *ncv,
                 std::complex<double> *v, int *ldv, int *iparam,
                 int *ipntr, std::complex<double> *workd,
                 std::complex<double> *workl, int *lworkl,
                 double *rwork, int *info);
}

// Type definition for complex vector
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Helper to save eigenvectors
void save_eigenvector(const std::string& dir, int index, const ComplexVector& evec) {
    if (dir.empty()) return;
    std::string filename = dir + "/eigenvector_" + std::to_string(index) + ".dat";
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(evec.data()), evec.size() * sizeof(Complex));
}

// Helper to save eigenvalues
void save_eigenvalues(const std::string& dir, const std::vector<double>& evals) {
    if (dir.empty()) return;
    std::string filename = dir + "/eigenvalues.dat";
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    size_t n_evals = evals.size();
    outfile.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(evals.data()), n_evals * sizeof(double));
}

/**
 * @brief Solves a standard eigenvalue problem (A*x = lambda*x) using the ARPACK library.
 *
 * This function is a C++ wrapper for the ARPACK znaupd/zneupd routines, designed for
 * finding a few eigenvalues and (optionally) eigenvectors of a large, sparse,
 * complex non-Hermitian matrix. For Hermitian matrices, it is more efficient but still correct.
 *
 * @param H A function that computes the matrix-vector product y = A*x.
 * @param N The dimension of the matrix A.
 * @param num_eigs The number of eigenvalues to compute (NEV).
 * @param max_iter The maximum number of Arnoldi update iterations allowed.
 * @param tol The desired relative accuracy of the computed Ritz values.
 * @param eigenvalues Output vector to store the real parts of the computed eigenvalues.
 * @param dir Directory to save eigenvectors and eigenvalues. If empty, nothing is saved.
 * @param compute_eigenvectors If true, eigenvectors are computed and saved.
 * @param which A string specifying which eigenvalues to compute (e.g., "LM", "SM", "LR", "SR").
 *              "LM": Largest Magnitude (default)
 *              "SM": Smallest Magnitude
 *              "LR": Largest Real part
 *              "SR": Smallest Real part
 *              "LI": Largest Imaginary part
 *              "SI": Smallest Imaginary part
 * @param ncv The number of Arnoldi vectors to generate at each iteration. Must be > num_eigs.
 *            A good rule of thumb is ncv = 2*num_eigs + 1. If 0, a default is used.
 */
void arpack_eigs(std::function<void(const Complex*, Complex*, int)> H, int N, int num_eigs,
                 int max_iter, double tol, std::vector<double>& eigenvalues,
                 std::string dir = "", bool compute_eigenvectors = false,
                 const std::string& which_str = "LM", int ncv_in = 0) {

    std::cout << "Starting ARPACK diagonalization..." << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Number of eigenvalues to find: " << num_eigs << std::endl;
    std::cout << "Selection criteria: " << which_str << std::endl;

    // ARPACK parameters
    int ido = 0;
    char bmat = 'I'; // Standard eigenvalue problem
    char which[3];
    strncpy(which, which_str.c_str(), 2);
    which[2] = '\0';

    int nev = num_eigs;
    ComplexVector resid(N);
    // Set NCV
    int ncv = (ncv_in > nev) ? ncv_in : std::min(N, std::max(2 * nev + 1, 20));
    if (ncv <= nev || ncv > N) {
        std::cerr << "ARPACK Error: NCV (" << ncv << ") must be > NEV (" << nev << ") and <= N (" << N << ")." << std::endl;
        std::cerr << "NCV_in was: " << ncv_in << std::endl;
        throw std::runtime_error("ARPACK failed due to NCV parameter error.");
    }
    std::cout << "Using NCV (Krylov subspace dim): " << ncv << std::endl;

    ComplexVector v(N * ncv);
    int ldv = N;
    int iparam[11];
    int ipntr[14];
    int lworkl = 3 * ncv * ncv + 5 * ncv;
    ComplexVector workl(lworkl);
    ComplexVector workd(3 * N);
    std::vector<double> rwork(ncv);
    int info = 0; // Use a zero starting vector (ARPACK will generate a random one)

    // Initialize iparam
    iparam[0] = 1;       // Exact shifts
    iparam[2] = max_iter;
    iparam[3] = 1;       // Block size
    iparam[6] = 1;       // Mode 1: A*x = lambda*x

    // Main ARPACK loop (reverse communication)
    int iter_count = 0;
    do {
        znaupd_(&ido, &bmat, &N, which, &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
                &lworkl, rwork.data(), &info);

        if (ido == -1 || ido == 1) {
            // Perform matrix-vector product y = A*x
            // x is in workd(ipntr[0]-1), y must be placed in workd(ipntr[1]-1)
            H(&workd[ipntr[0] - 1], &workd[ipntr[1] - 1], N);
        } else if (ido == 99) {
            break; // Normal exit
        } else {
            std::cerr << "ARPACK znaupd error: ido = " << ido << ", info = " << info << std::endl;
            throw std::runtime_error("ARPACK znaupd failed.");
        }
        iter_count++;
    } while (iter_count <= max_iter);

    if (info < 0) {
        std::cerr << "ARPACK znaupd error: info = " << info << std::endl;
        throw std::runtime_error("Error with ARPACK znaupd parameters.");
    }
    if (info == 1) {
        std::cout << "Warning: ARPACK znaupd reached maximum number of iterations." << std::endl;
    }
    if (info == 3) {
        std::cerr << "ARPACK znaupd error: No shifts could be applied." << std::endl;
        throw std::runtime_error("ARPACK znaupd failed due to shift error.");
    }

    std::cout << "ARPACK znaupd finished in " << iparam[2] << " iterations." << std::endl;
    std::cout << "Number of converged Ritz values: " << iparam[4] << std::endl;

    // Post-processing with zneupd
    int rvec = compute_eigenvectors ? 1 : 0;
    char howmny = 'A'; // Compute all NEV Ritz vectors
    std::vector<int> select(ncv);
    ComplexVector d(nev + 1);
    ComplexVector z;
    int ldz = N;
    if (compute_eigenvectors) {
        z.resize(ldz * (nev + 1));
    }
    Complex sigma = {0.0, 0.0};
    ComplexVector workev(2 * ncv);

    zneupd_(&rvec, &howmny, select.data(), d.data(), z.data(), &ldz, &sigma,
            workev.data(), &bmat, &N, which, &nev, &tol, resid.data(),
            &ncv, v.data(), &ldv, iparam, ipntr, workd.data(),
            workl.data(), &lworkl, rwork.data(), &info);

    if (info != 0) {
        std::cerr << "ARPACK zneupd error: info = " << info << std::endl;
        throw std::runtime_error("ARPACK zneupd failed.");
    }

    // Extract results
    int n_conv = iparam[4];
    eigenvalues.resize(n_conv);
    std::vector<Complex> complex_eigenvalues(n_conv);
    for (int i = 0; i < n_conv; ++i) {
        eigenvalues[i] = d[i].real();
        complex_eigenvalues[i] = d[i];
    }

    // Compute residuals for convergence assessment
    std::vector<double> residuals(n_conv);
    if (compute_eigenvectors) {
        ComplexVector temp(N);
        for (int i = 0; i < n_conv; ++i) {
            // Get eigenvector
            ComplexVector evec(N);
            std::copy(&z[i * N], &z[i * N + N], evec.begin());
            
            // Compute A*v
            H(evec.data(), temp.data(), N);
            
            // Compute A*v - lambda*v
            for (int j = 0; j < N; ++j) {
                temp[j] -= d[i] * evec[j];
            }
            
            // Compute ||A*v - lambda*v||
            double residual = 0.0;
            for (int j = 0; j < N; ++j) {
                residual += std::norm(temp[j]);
            }
            residuals[i] = std::sqrt(residual);
        }
    }

    // Sort eigenvalues by real part (as is common in physics)
    std::vector<size_t> p(n_conv);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](size_t i, size_t j) {
        return eigenvalues[i] < eigenvalues[j];
    });

    std::vector<double> sorted_eigenvalues(n_conv);
    std::vector<double> sorted_residuals(n_conv);
    for(int i=0; i<n_conv; ++i) {
        sorted_eigenvalues[i] = eigenvalues[p[i]];
        if (compute_eigenvectors) {
            sorted_residuals[i] = residuals[p[i]];
        }
    }
    eigenvalues = sorted_eigenvalues;

    // Create output directory
    if (!dir.empty() && (compute_eigenvectors || !eigenvalues.empty())) {
        system(("mkdir -p " + dir).c_str());
    }

    // Save eigenvectors if requested
    if (compute_eigenvectors) {
        std::cout << "Saving " << n_conv << " eigenvectors..." << std::endl;
        std::vector<ComplexVector> sorted_eigenvectors(n_conv, ComplexVector(N));
        for (int i = 0; i < n_conv; ++i) {
            std::copy(&z[p[i] * N], &z[p[i] * N + N], sorted_eigenvectors[i].begin());
        }

        for (int i = 0; i < n_conv; ++i) {
            save_eigenvector(dir, i, sorted_eigenvectors[i]);
        }
    }

    // Save eigenvalues
    save_eigenvalues(dir, eigenvalues);

    std::cout << "ARPACK diagonalization completed successfully." << std::endl;
    std::cout << "Found " << n_conv << " eigenvalues." << std::endl;
    std::cout << "Eigenvalues: ";
    for(int i=0; i<std::min((int)eigenvalues.size(), 10); ++i) {
        std::cout << std::fixed << std::setprecision(8) << eigenvalues[i] << " ";
    }
    if(eigenvalues.size() > 10) std::cout << "...";
    std::cout << std::endl;

    // Output convergence information
    if (compute_eigenvectors) {
        std::cout << "Residuals (||A*v - lambda*v||): ";
        for(int i=0; i<std::min((int)sorted_residuals.size(), 10); ++i) {
            std::cout << std::scientific << std::setprecision(2) << sorted_residuals[i] << " ";
        }
        if(sorted_residuals.size() > 10) std::cout << "...";
        std::cout << std::endl;
        
        // Check convergence
        double max_residual = *std::max_element(sorted_residuals.begin(), sorted_residuals.end());
        std::cout << "Maximum residual: " << std::scientific << std::setprecision(6) << max_residual << std::endl;
        std::cout << "Tolerance requested: " << std::scientific << std::setprecision(6) << tol << std::endl;
        if (max_residual < tol) {
            std::cout << "Convergence: SATISFIED" << std::endl;
        } else {
            std::cout << "Convergence: NOT FULLY SATISFIED" << std::endl;
        }
    } else {
        std::cout << "Note: Residuals computed only when eigenvectors are requested." << std::endl;
    }
}

/**
 * @brief Solves a standard eigenvalue problem in shift-invert mode.
 *
 * This function finds eigenvalues of A closest to a specified shift `sigma`.
 * It solves the problem (A - sigma*I)^-1 * x = nu * x, where the eigenvalues
 * lambda of A are related to nu by lambda = sigma + 1/nu.
 *
 * @param H A function that computes the matrix-vector product y = A*x.
 * @param solver A function that solves the linear system (A - sigma*I)*y = x for y.
 * @param N The dimension of the matrix A.
 * @param num_eigs The number of eigenvalues to compute (NEV).
 * @param max_iter The maximum number of Arnoldi update iterations allowed.
 * @param tol The desired relative accuracy of the computed Ritz values.
 * @param sigma The shift value. Eigenvalues closest to this value will be computed.
 * @param eigenvalues Output vector to store the real parts of the computed eigenvalues.
 * @param dir Directory to save eigenvectors and eigenvalues.
 * @param compute_eigenvectors If true, eigenvectors are computed and saved.
 * @param ncv The number of Arnoldi vectors to generate.
 */
void arpack_eigs_shift_invert(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> solver,
    int N, int num_eigs, int max_iter, double tol, Complex sigma,
    std::vector<double>& eigenvalues, std::string dir = "",
    bool compute_eigenvectors = false, int ncv_in = 0) {

    std::cout << "Starting ARPACK in shift-invert mode..." << std::endl;
    std::cout << "Shift sigma = (" << sigma.real() << ", " << sigma.imag() << ")" << std::endl;

    // ARPACK parameters
    int ido = 0;
    char bmat = 'I';
    char which[] = "LM"; // For shift-invert, we want largest magnitude eigenvalues of the operator

    int nev = num_eigs;
    ComplexVector resid(N);
    int ncv = (ncv_in > nev) ? ncv_in : std::min(N, std::max(2 * nev + 1, 20));
    if (ncv <= nev || ncv > N) {
        throw std::runtime_error("ARPACK Error: NCV must be > NEV and <= N.");
    }
    std::cout << "Using NCV (Krylov subspace dim): " << ncv << std::endl;

    ComplexVector v(N * ncv);
    int ldv = N;
    int iparam[11];
    int ipntr[14];
    int lworkl = 3 * ncv * ncv + 5 * ncv;
    ComplexVector workl(lworkl);
    ComplexVector workd(3 * N);
    std::vector<double> rwork(ncv);
    int info = 1;

    // Initialize iparam for shift-invert mode
    iparam[0] = 1;
    iparam[2] = max_iter;
    iparam[3] = 1;
    iparam[6] = 3; // Shift-invert mode

    // Main ARPACK loop
    int iter_count = 0;
    do {
        znaupd_(&ido, &bmat, &N, which, &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
                &lworkl, rwork.data(), &info);

        if (ido == -1 || ido == 1) {
            // Apply operator (A - sigma*I)^-1
            // x is in workd(ipntr[0]-1), y must be placed in workd(ipntr[1]-1)
            solver(&workd[ipntr[0] - 1], &workd[ipntr[1] - 1], N);
        } else if (ido == 99) {
            break;
        } else {
            std::cerr << "ARPACK znaupd error: ido = " << ido << ", info = " << info << std::endl;
            throw std::runtime_error("ARPACK znaupd failed.");
        }
        iter_count++;
    } while (iter_count <= max_iter);

    if (info < 0) {
        std::cerr << "ARPACK znaupd error: info = " << info << std::endl;
        throw std::runtime_error("Error with ARPACK znaupd parameters.");
    }

    // Post-processing with zneupd
    int rvec = compute_eigenvectors ? 1 : 0;
    char howmny = 'A';
    std::vector<int> select(ncv);
    ComplexVector d(nev + 1);
    ComplexVector z;
    int ldz = N;
    if (compute_eigenvectors) {
        z.resize(ldz * (nev + 1));
    }
    ComplexVector workev(2 * ncv);

    // Note: zneupd needs the shift sigma
    zneupd_(&rvec, &howmny, select.data(), d.data(), z.data(), &ldz, &sigma,
            workev.data(), &bmat, &N, which, &nev, &tol, resid.data(),
            &ncv, v.data(), &ldv, iparam, ipntr, workd.data(),
            workl.data(), &lworkl, rwork.data(), &info);

    if (info != 0) {
        std::cerr << "ARPACK zneupd error: info = " << info << std::endl;
        throw std::runtime_error("ARPACK zneupd failed.");
    }

    // Extract results
    int n_conv = iparam[4];
    eigenvalues.resize(n_conv);
    for (int i = 0; i < n_conv; ++i) {
        eigenvalues[i] = d[i].real();
    }

    // Sort eigenvalues
    std::sort(eigenvalues.begin(), eigenvalues.end());

    // Create output directory
    if (!dir.empty() && (compute_eigenvectors || !eigenvalues.empty())) {
        system(("mkdir -p " + dir).c_str());
    }

    // Save eigenvectors if requested
    if (compute_eigenvectors) {
        std::cout << "Saving " << n_conv << " eigenvectors..." << std::endl;
        for (int i = 0; i < n_conv; ++i) {
            ComplexVector evec(N);
            std::copy(&z[i * N], &z[i * N + N], evec.begin());
            save_eigenvector(dir, i, evec);
        }
    }

    // Save eigenvalues
    save_eigenvalues(dir, eigenvalues);

    std::cout << "ARPACK shift-invert completed successfully." << std::endl;
    std::cout << "Found " << n_conv << " eigenvalues near sigma." << std::endl;
}

#endif // ARPACK_H