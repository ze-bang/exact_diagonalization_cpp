#include <vector>
#include <complex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Define complex number type and matrix type for convenience
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

/**
 * Operator class that can represent arbitrary quantum operators
 * through bit flip operations and scalar multiplications
 * Optimized with cuBLAS and cuSPARSE for GPU acceleration
 */
class Operator {
private:
    std::vector<TransformFunction> transforms_;
    int n_bits_; // Number of bits in the basis representation
    mutable Eigen::SparseMatrix<Complex> sparseMatrix_;
    mutable bool matrixBuilt_ = false;
    
    // CUDA resources
    mutable cublasHandle_t cublasHandle_ = nullptr;
    mutable cusparseHandle_t cusparseHandle_ = nullptr;
    mutable cusparseSpMatDescr_t matDescr_ = nullptr;
    mutable int* d_csrRowPtr_ = nullptr;
    mutable int* d_csrColInd_ = nullptr;
    mutable cuDoubleComplex* d_csrVal_ = nullptr;
    mutable bool cudaInitialized_ = false;
    mutable int nnz_ = 0;
    
    // Initialize CUDA resources
    void initializeCuda() const {
        if (!cudaInitialized_) {
            checkCudaErrors(cublasCreate(&cublasHandle_));
            checkCudaErrors(cusparseCreate(&cusparseHandle_));
            cudaInitialized_ = true;
        }
    }
    
    // Clean up CUDA resources
    void cleanupCuda() const {
        if (matDescr_) {
            cusparseDestroySpMat(matDescr_);
            matDescr_ = nullptr;
        }
        if (d_csrRowPtr_) {
            cudaFree(d_csrRowPtr_);
            d_csrRowPtr_ = nullptr;
        }
        if (d_csrColInd_) {
            cudaFree(d_csrColInd_);
            d_csrColInd_ = nullptr;
        }
        if (d_csrVal_) {
            cudaFree(d_csrVal_);
            d_csrVal_ = nullptr;
        }
    }
    
    // Build the sparse matrix from transforms if needed
    void buildSparseMatrix() const {
        if (matrixBuilt_) return;
        
        int dim = 1 << n_bits_;
        sparseMatrix_.resize(dim, dim);
        
        // Use triplets to efficiently build the sparse matrix
        std::vector<Eigen::Triplet<Complex>> triplets;
        
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {
                    triplets.emplace_back(j, i, scalar);
                }
            }
        }
        
        sparseMatrix_.setFromTriplets(triplets.begin(), triplets.end());
        matrixBuilt_ = true;
        
        // Upload to GPU
        initializeCuda();
        uploadMatrixToGpu();
    }
    
    // Upload matrix to GPU in CSR format
    void uploadMatrixToGpu() const {
        // Clean up previous data
        cleanupCuda();
        
        // Convert to CSR format
        Eigen::SparseMatrix<Complex, Eigen::RowMajor> csrMatrix = sparseMatrix_;
        nnz_ = csrMatrix.nonZeros();
        int dim = 1 << n_bits_;
        
        // Allocate GPU memory
        checkCudaErrors(cudaMalloc((void**)&d_csrRowPtr_, (dim + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_csrColInd_, nnz_ * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_csrVal_, nnz_ * sizeof(cuDoubleComplex)));
        
        // Copy data
        std::vector<cuDoubleComplex> hostValues(nnz_);
        for (int i = 0; i < nnz_; i++) {
            hostValues[i].x = csrMatrix.valuePtr()[i].real();
            hostValues[i].y = csrMatrix.valuePtr()[i].imag();
        }
        
        checkCudaErrors(cudaMemcpy(d_csrRowPtr_, csrMatrix.outerIndexPtr(), 
                                   (dim + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_csrColInd_, csrMatrix.innerIndexPtr(), 
                                   nnz_ * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_csrVal_, hostValues.data(), 
                                   nnz_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // Create sparse matrix descriptor
        cusparseCreateCsr(&matDescr_, dim, dim, nnz_,
                          d_csrRowPtr_, d_csrColInd_, d_csrVal_,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    }
    
public:
    // Function type for transforming basis states
    using TransformFunction = std::function<std::pair<int, Complex>(int)>;

    // Constructor
    Operator(int n_bits) : n_bits_(n_bits) {}
    
    // Destructor
    ~Operator() {
        if (cudaInitialized_) {
            cleanupCuda();
            cusparseDestroy(cusparseHandle_);
            cublasDestroy(cublasHandle_);
        }
    }
    
    // Prevent copying to avoid CUDA resource issues
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    
    // Allow moving
    Operator(Operator&& other) noexcept 
        : n_bits_(other.n_bits_), transforms_(std::move(other.transforms_)),
          sparseMatrix_(std::move(other.sparseMatrix_)), matrixBuilt_(other.matrixBuilt_) {
        // Move CUDA resources
        cublasHandle_ = other.cublasHandle_;
        cusparseHandle_ = other.cusparseHandle_;
        matDescr_ = other.matDescr_;
        d_csrRowPtr_ = other.d_csrRowPtr_;
        d_csrColInd_ = other.d_csrColInd_;
        d_csrVal_ = other.d_csrVal_;
        cudaInitialized_ = other.cudaInitialized_;
        nnz_ = other.nnz_;
        
        // Reset source
        other.cublasHandle_ = nullptr;
        other.cusparseHandle_ = nullptr;
        other.matDescr_ = nullptr;
        other.d_csrRowPtr_ = nullptr;
        other.d_csrColInd_ = nullptr;
        other.d_csrVal_ = nullptr;
        other.cudaInitialized_ = false;
    }

    // Mark matrix as needing rebuild when new transform is added
    void addTransform(TransformFunction transform) {
        transforms_.push_back(transform);
        matrixBuilt_ = false;  // Matrix needs to be rebuilt
    }

    // Apply the operator to a complex vector using GPU acceleration
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        int dim = 1 << n_bits_;
        
        if (vec.size() != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input vector dimension does not match operator dimension");
        }
        
        // Build the sparse matrix if not already built
        buildSparseMatrix();
        
        // Allocate device memory for vectors
        cuDoubleComplex* d_x = nullptr;
        cuDoubleComplex* d_y = nullptr;
        checkCudaErrors(cudaMalloc((void**)&d_x, dim * sizeof(cuDoubleComplex)));
        checkCudaErrors(cudaMalloc((void**)&d_y, dim * sizeof(cuDoubleComplex)));
        
        // Convert and copy input vector to device
        std::vector<cuDoubleComplex> hostX(dim);
        for (int i = 0; i < dim; i++) {
            hostX[i].x = vec[i].real();
            hostX[i].y = vec[i].imag();
        }
        checkCudaErrors(cudaMemcpy(d_x, hostX.data(), dim * sizeof(cuDoubleComplex), 
                                   cudaMemcpyHostToDevice));
        
        // Create vector descriptors
        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateDnVec(&vecX, dim, d_x, CUDA_C_64F);
        cusparseCreateDnVec(&vecY, dim, d_y, CUDA_C_64F);
        
        // Perform sparse matrix-vector multiplication
        const cuDoubleComplex alpha{1.0, 0.0};
        const cuDoubleComplex beta{0.0, 0.0};
        void* buffer = nullptr;
        size_t bufferSize = 0;
        
        cusparseSpMV_bufferSize(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matDescr_, vecX, &beta, vecY, 
                                CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        
        checkCudaErrors(cudaMalloc(&buffer, bufferSize));
        
        cusparseSpMV(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matDescr_, vecX, &beta, vecY, 
                     CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        
        // Copy result back to host
        std::vector<cuDoubleComplex> hostY(dim);
        checkCudaErrors(cudaMemcpy(hostY.data(), d_y, dim * sizeof(cuDoubleComplex), 
                                   cudaMemcpyDeviceToHost));
        
        // Convert to std::vector<Complex>
        std::vector<Complex> resultVec(dim);
        for (int i = 0; i < dim; i++) {
            resultVec[i] = Complex(hostY[i].x, hostY[i].y);
        }
        
        // Clean up
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cudaFree(buffer);
        cudaFree(d_x);
        cudaFree(d_y);
        
        return resultVec;
    }

    // Print the operator as a matrix
    Matrix returnMatrix() {
        int dim = 1 << n_bits_;
        Matrix matrix(dim, std::vector<Complex>(dim, 0.0));
        for (int i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {
                    matrix[j][i] += scalar;
                }
            }
        }
        return matrix;
    }
    
    // Load operator definition from a file
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        // Skip the first line (header)
        std::getline(file, line);
        
        // Read the number of lines
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Skip the next 3 lines (separators/headers)
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        // Process transform data
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            if (!(lineStream >> Op >> indx >> E >> F)) {
                continue; // Skip invalid lines
            }
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check if all bits match their expected values
                if (Op == 2){
                    return {basis, Complex(E,F)*0.5*pow(-1,(basis >> indx) & 1)};
                }
                else{
                    if (((basis >> indx) & 1) != Op) {
                        // Flip the A bit
                        int flipped_basis = basis ^ (1 << indx);
                        return {flipped_basis, Complex(E, F)};                    
                    }
                }
                // Default case: no transformation applies
                return {-1, Complex(0.0, 0.0)};
            });
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;
    }
    
    void loadFromInterAllFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "Reading file: " << filename << std::endl;
        std::string line;
        
        // Skip the first line (header)
        std::getline(file, line);
        
        // Read the number of lines
        std::getline(file, line);
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Skip the next 3 lines
        for (int i = 0; i < 3; ++i) {
            std::getline(file, line);
        }
        
        // Process transform data
        int lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op1, indx1, Op2, indx2;
            double E, F;
            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                continue; // Skip invalid lines
            }
            addTransform([=](int basis) -> std::pair<int, Complex> {
                // Check what type of operators we're dealing with
                if (Op1 == 2 && Op2 == 2) {
                    // Both are identity operators with phase factors
                    int bit1 = (basis >> indx1) & 1;
                    int bit2 = (basis >> indx2) & 1;
                    return {basis, Complex(E, F)* 0.25 * pow(-1, bit1) * pow(-1, bit2)};
                } 
                else if (Op1 == 2) {
                    // Op1 is identity with phase, Op2 is bit flip
                    int bit1 = (basis >> indx1) & 1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit2_matches) {
                        int flipped_basis = basis ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F) * 0.5 * pow(-1, bit1)};
                    }
                } 
                else if (Op2 == 2) {
                    // Op2 is identity with phase, Op1 is bit flip
                    int bit2 = (basis >> indx2) & 1;
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    
                    if (bit1_matches) {
                        // Flip the first bit
                        int flipped_basis = basis ^ (1 << indx1);
                        return {flipped_basis, Complex(E, F)* 0.5 * pow(-1, bit2)};
                    }
                } 
                else {
                    // Both are bit flip operators
                    bool bit1_matches = ((basis >> indx1) & 1) != Op1;
                    bool bit2_matches = ((basis >> indx2) & 1) != Op2;
                    
                    if (bit1_matches && bit2_matches) {
                        // Flip both bits
                        int flipped_basis = basis ^ (1 << indx1) ^ (1 << indx2);
                        return {flipped_basis, Complex(E, F)};
                    }
                }
                // Default case: no transformation applies
                return {-1, Complex(0.0, 0.0)};
            });
            lineCount++;
        }
        std::cout << "File read complete." << std::endl;    
    }
};
