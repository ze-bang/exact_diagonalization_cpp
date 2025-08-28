// chunked_complex_vector.h - Chunked Complex Vector implementation for large arrays
// This implementation uses vector of vectors to handle arrays larger than 2^32 elements

#ifndef CHUNKED_COMPLEX_VECTOR_H
#define CHUNKED_COMPLEX_VECTOR_H

#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <functional>

// Type definition for complex numbers
using Complex = std::complex<double>;

/**
 * ChunkedComplexVector: A structure to handle very large complex vectors
 * that exceed the size limitations of std::vector.
 * 
 * This class splits the large vector into chunks of manageable size,
 * storing them as a vector of vectors. Each chunk can hold up to CHUNK_SIZE elements.
 */
class ChunkedComplexVector {
private:
    static constexpr size_t CHUNK_SIZE = 1ULL << 30; // 2^30 elements per chunk (~8GB per chunk)
    std::vector<std::vector<Complex>> chunks;
    size_t total_size_;

public:
    // Constructors
    ChunkedComplexVector() : total_size_(0) {}
    
    explicit ChunkedComplexVector(size_t size) : total_size_(size) {
        size_t num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        chunks.reserve(num_chunks);
        
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t chunk_size = std::min(CHUNK_SIZE, size - i * CHUNK_SIZE);
            chunks.emplace_back(chunk_size, Complex(0.0, 0.0));
        }
    }
    
    ChunkedComplexVector(size_t size, const Complex& value) : total_size_(size) {
        size_t num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        chunks.reserve(num_chunks);
        
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t chunk_size = std::min(CHUNK_SIZE, size - i * CHUNK_SIZE);
            chunks.emplace_back(chunk_size, value);
        }
    }
    
    // Copy constructor
    ChunkedComplexVector(const ChunkedComplexVector& other) 
        : chunks(other.chunks), total_size_(other.total_size_) {}
    
    // Move constructor
    ChunkedComplexVector(ChunkedComplexVector&& other) noexcept
        : chunks(std::move(other.chunks)), total_size_(other.total_size_) {
        other.total_size_ = 0;
    }
    
    // Assignment operators
    ChunkedComplexVector& operator=(const ChunkedComplexVector& other) {
        if (this != &other) {
            chunks = other.chunks;
            total_size_ = other.total_size_;
        }
        return *this;
    }
    
    ChunkedComplexVector& operator=(ChunkedComplexVector&& other) noexcept {
        if (this != &other) {
            chunks = std::move(other.chunks);
            total_size_ = other.total_size_;
            other.total_size_ = 0;
        }
        return *this;
    }
    
    // Element access
    Complex& operator[](size_t index) {
        size_t chunk_idx = index / CHUNK_SIZE;
        size_t local_idx = index % CHUNK_SIZE;
        return chunks[chunk_idx][local_idx];
    }
    
    const Complex& operator[](size_t index) const {
        size_t chunk_idx = index / CHUNK_SIZE;
        size_t local_idx = index % CHUNK_SIZE;
        return chunks[chunk_idx][local_idx];
    }
    
    Complex& at(size_t index) {
        if (index >= total_size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    const Complex& at(size_t index) const {
        if (index >= total_size_) {
            throw std::out_of_range("Index out of range");
        }
        return (*this)[index];
    }
    
    // Size and capacity
    size_t size() const { return total_size_; }
    bool empty() const { return total_size_ == 0; }
    size_t num_chunks() const { return chunks.size(); }
    size_t chunk_size() const { return CHUNK_SIZE; }
    
    // Resize
    void resize(size_t new_size, const Complex& value = Complex(0.0, 0.0)) {
        if (new_size == total_size_) return;
        
        size_t new_num_chunks = (new_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t old_num_chunks = chunks.size();
        
        if (new_num_chunks > old_num_chunks) {
            // Add new chunks
            chunks.reserve(new_num_chunks);
            for (size_t i = old_num_chunks; i < new_num_chunks; ++i) {
                size_t chunk_size = std::min(CHUNK_SIZE, new_size - i * CHUNK_SIZE);
                chunks.emplace_back(chunk_size, value);
            }
        } else if (new_num_chunks < old_num_chunks) {
            // Remove chunks
            chunks.resize(new_num_chunks);
        }
        
        // Resize the last chunk if necessary
        if (!chunks.empty()) {
            size_t last_chunk_size = new_size - (new_num_chunks - 1) * CHUNK_SIZE;
            chunks.back().resize(last_chunk_size, value);
        }
        
        total_size_ = new_size;
    }
    
    // Clear
    void clear() {
        chunks.clear();
        total_size_ = 0;
    }
    
    // Fill with value
    void fill(const Complex& value) {
        for (auto& chunk : chunks) {
            std::fill(chunk.begin(), chunk.end(), value);
        }
    }
    
    // Get raw pointer to chunk data (for BLAS operations)
    Complex* chunk_data(size_t chunk_idx) {
        if (chunk_idx >= chunks.size()) {
            throw std::out_of_range("Chunk index out of range");
        }
        return chunks[chunk_idx].data();
    }
    
    const Complex* chunk_data(size_t chunk_idx) const {
        if (chunk_idx >= chunks.size()) {
            throw std::out_of_range("Chunk index out of range");
        }
        return chunks[chunk_idx].data();
    }
    
    // Get size of specific chunk
    size_t get_chunk_size(size_t chunk_idx) const {
        if (chunk_idx >= chunks.size()) {
            throw std::out_of_range("Chunk index out of range");
        }
        return chunks[chunk_idx].size();
    }
    
    // Arithmetic operators
    ChunkedComplexVector& operator+=(const ChunkedComplexVector& other) {
        if (total_size_ != other.total_size_) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            for (size_t i = 0; i < chunks[chunk_idx].size(); ++i) {
                chunks[chunk_idx][i] += other.chunks[chunk_idx][i];
            }
        }
        return *this;
    }
    
    ChunkedComplexVector& operator-=(const ChunkedComplexVector& other) {
        if (total_size_ != other.total_size_) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            for (size_t i = 0; i < chunks[chunk_idx].size(); ++i) {
                chunks[chunk_idx][i] -= other.chunks[chunk_idx][i];
            }
        }
        return *this;
    }
    
    ChunkedComplexVector& operator*=(const Complex& scalar) {
        for (auto& chunk : chunks) {
            for (auto& element : chunk) {
                element *= scalar;
            }
        }
        return *this;
    }
    
    ChunkedComplexVector& operator/=(const Complex& scalar) {
        if (scalar == Complex(0.0, 0.0)) {
            throw std::invalid_argument("Division by zero");
        }
        
        for (auto& chunk : chunks) {
            for (auto& element : chunk) {
                element /= scalar;
            }
        }
        return *this;
    }
    
    // Iterator support
    class iterator {
    private:
        ChunkedComplexVector* vec_;
        size_t index_;
        
    public:
        iterator(ChunkedComplexVector* vec, size_t index) : vec_(vec), index_(index) {}
        
        Complex& operator*() { return (*vec_)[index_]; }
        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++index_; return tmp; }
        bool operator!=(const iterator& other) const { return index_ != other.index_; }
        bool operator==(const iterator& other) const { return index_ == other.index_; }
    };
    
    class const_iterator {
    private:
        const ChunkedComplexVector* vec_;
        size_t index_;
        
    public:
        const_iterator(const ChunkedComplexVector* vec, size_t index) : vec_(vec), index_(index) {}
        
        const Complex& operator*() const { return (*vec_)[index_]; }
        const_iterator& operator++() { ++index_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++index_; return tmp; }
        bool operator!=(const const_iterator& other) const { return index_ != other.index_; }
        bool operator==(const const_iterator& other) const { return index_ == other.index_; }
    };
    
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, total_size_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, total_size_); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, total_size_); }
};

// Binary arithmetic operators
inline ChunkedComplexVector operator+(const ChunkedComplexVector& lhs, const ChunkedComplexVector& rhs) {
    ChunkedComplexVector result = lhs;
    result += rhs;
    return result;
}

inline ChunkedComplexVector operator-(const ChunkedComplexVector& lhs, const ChunkedComplexVector& rhs) {
    ChunkedComplexVector result = lhs;
    result -= rhs;
    return result;
}

inline ChunkedComplexVector operator*(const ChunkedComplexVector& vec, const Complex& scalar) {
    ChunkedComplexVector result = vec;
    result *= scalar;
    return result;
}

inline ChunkedComplexVector operator*(const Complex& scalar, const ChunkedComplexVector& vec) {
    ChunkedComplexVector result = vec;
    result *= scalar;
    return result;
}

inline ChunkedComplexVector operator/(const ChunkedComplexVector& vec, const Complex& scalar) {
    ChunkedComplexVector result = vec;
    result /= scalar;
    return result;
}

// Utility functions for compatibility with existing code
namespace ChunkedVectorUtils {
    // Apply a function to each chunk (useful for BLAS operations)
    template<typename Func>
    void apply_to_chunks(ChunkedComplexVector& vec, Func func) {
        for (size_t i = 0; i < vec.num_chunks(); ++i) {
            func(vec.chunk_data(i), vec.get_chunk_size(i), i);
        }
    }
    
    template<typename Func>
    void apply_to_chunks(const ChunkedComplexVector& vec, Func func) {
        for (size_t i = 0; i < vec.num_chunks(); ++i) {
            func(vec.chunk_data(i), vec.get_chunk_size(i), i);
        }
    }
    
    // Apply a function to corresponding chunks of two vectors
    template<typename Func>
    void apply_to_chunk_pairs(ChunkedComplexVector& vec1, const ChunkedComplexVector& vec2, Func func) {
        if (vec1.size() != vec2.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        
        for (size_t i = 0; i < vec1.num_chunks(); ++i) {
            func(vec1.chunk_data(i), vec2.chunk_data(i), vec1.get_chunk_size(i), i);
        }
    }
    
    // Compute dot product
    Complex dot_product(const ChunkedComplexVector& a, const ChunkedComplexVector& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        
        Complex result(0.0, 0.0);
        for (size_t chunk_idx = 0; chunk_idx < a.num_chunks(); ++chunk_idx) {
            const Complex* a_data = a.chunk_data(chunk_idx);
            const Complex* b_data = b.chunk_data(chunk_idx);
            size_t chunk_size = a.get_chunk_size(chunk_idx);
            
            for (size_t i = 0; i < chunk_size; ++i) {
                result += std::conj(a_data[i]) * b_data[i];
            }
        }
        return result;
    }
    
    // Compute norm
    double norm(const ChunkedComplexVector& vec) {
        double norm_squared = 0.0;
        for (size_t chunk_idx = 0; chunk_idx < vec.num_chunks(); ++chunk_idx) {
            const Complex* data = vec.chunk_data(chunk_idx);
            size_t chunk_size = vec.get_chunk_size(chunk_idx);
            
            for (size_t i = 0; i < chunk_size; ++i) {
                double real_part = data[i].real();
                double imag_part = data[i].imag();
                norm_squared += real_part * real_part + imag_part * imag_part;
            }
        }
        return std::sqrt(norm_squared);
    }
    
    // Normalize vector
    void normalize(ChunkedComplexVector& vec) {
        double vec_norm = norm(vec);
        if (vec_norm > 0.0) {
            vec *= Complex(1.0 / vec_norm, 0.0);
        }
    }
    
    // Copy from regular std::vector<Complex> to ChunkedComplexVector
    void copy_from_vector(ChunkedComplexVector& chunked_vec, const std::vector<Complex>& regular_vec) {
        chunked_vec.resize(regular_vec.size());
        for (size_t i = 0; i < regular_vec.size(); ++i) {
            chunked_vec[i] = regular_vec[i];
        }
    }
    
    // Copy from ChunkedComplexVector to regular std::vector<Complex> (if size allows)
    std::vector<Complex> to_vector(const ChunkedComplexVector& chunked_vec) {
        if (chunked_vec.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("ChunkedComplexVector too large to convert to std::vector");
        }
        
        std::vector<Complex> result;
        result.reserve(chunked_vec.size());
        
        for (size_t chunk_idx = 0; chunk_idx < chunked_vec.num_chunks(); ++chunk_idx) {
            const Complex* data = chunked_vec.chunk_data(chunk_idx);
            size_t chunk_size = chunked_vec.get_chunk_size(chunk_idx);
            
            for (size_t i = 0; i < chunk_size; ++i) {
                result.push_back(data[i]);
            }
        }
        
        return result;
    }
}

// Operator application support for ChunkedComplexVector
namespace ChunkedOperators {
    // Type definition for operators that work on chunks
    using ChunkedOperatorFunc = std::function<void(const Complex*, Complex*, size_t)>;
    
    /**
     * Apply an operator to a ChunkedComplexVector
     * This function applies the operator chunk by chunk to handle large vectors
     * 
     * @param op The operator function to apply
     * @param input Input chunked vector
     * @param output Output chunked vector (will be resized if necessary)
     */
    void apply_operator(ChunkedOperatorFunc op, const ChunkedComplexVector& input, ChunkedComplexVector& output) {
        if (output.size() != input.size()) {
            output.resize(input.size());
        }
        
        for (size_t chunk_idx = 0; chunk_idx < input.num_chunks(); ++chunk_idx) {
            const Complex* input_data = input.chunk_data(chunk_idx);
            Complex* output_data = output.chunk_data(chunk_idx);
            size_t chunk_size = input.get_chunk_size(chunk_idx);
            
            op(input_data, output_data, chunk_size);
        }
    }
    
    /**
     * Wrapper to convert standard operator functions to work with ChunkedComplexVector
     * This adapts the typical std::function<void(const Complex*, Complex*, int)> operators
     * to work with our chunked structure
     */
    class OperatorWrapper {
    private:
        std::function<void(const Complex*, Complex*, int)> op_;
        
    public:
        OperatorWrapper(std::function<void(const Complex*, Complex*, int)> op) : op_(op) {}
        
        void operator()(const ChunkedComplexVector& input, ChunkedComplexVector& output) {
            if (output.size() != input.size()) {
                output.resize(input.size());
            }
            
            // For now, we need to be careful about the int parameter in the original operator
            // Most operators expect the full vector size, not chunk size
            size_t total_size = input.size();
            if (total_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Vector too large for standard operator (size > int max)");
            }
            
            // For operators that work on the whole vector, we may need to temporarily
            // create contiguous memory or modify the operator to work chunk-wise
            // For now, let's provide a chunk-wise application
            for (size_t chunk_idx = 0; chunk_idx < input.num_chunks(); ++chunk_idx) {
                const Complex* input_data = input.chunk_data(chunk_idx);
                Complex* output_data = output.chunk_data(chunk_idx);
                int chunk_size = static_cast<int>(input.get_chunk_size(chunk_idx));
                
                op_(input_data, output_data, chunk_size);
            }
        }
        
        /**
         * Alternative method for operators that need to work on the full vector
         * This creates temporary contiguous memory (expensive but sometimes necessary)
         */
        void apply_contiguous(const ChunkedComplexVector& input, ChunkedComplexVector& output) {
            if (input.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Vector too large for contiguous operator application");
            }
            
            // Create temporary contiguous vectors
            std::vector<Complex> temp_input = ChunkedVectorUtils::to_vector(input);
            std::vector<Complex> temp_output(temp_input.size());
            
            // Apply operator
            op_(temp_input.data(), temp_output.data(), static_cast<int>(temp_input.size()));
            
            // Copy back to chunked output
            ChunkedVectorUtils::copy_from_vector(output, temp_output);
        }
    };
    
    /**
     * Create a Hamiltonian operator that works efficiently with ChunkedComplexVector
     * This is a more sophisticated version that can handle large vectors by
     * processing them in manageable chunks
     */
    class ChunkedHamiltonianOperator {
    private:
        std::function<void(const Complex*, Complex*, int)> hamiltonian_;
        size_t max_contiguous_size_;
        
    public:
        ChunkedHamiltonianOperator(
            std::function<void(const Complex*, Complex*, int)> hamiltonian,
            size_t max_contiguous_size = 1ULL << 28  // 256M elements = ~2GB
        ) : hamiltonian_(hamiltonian), max_contiguous_size_(max_contiguous_size) {}
        
        void operator()(const ChunkedComplexVector& input, ChunkedComplexVector& output) {
            if (output.size() != input.size()) {
                output.resize(input.size());
            }
            
            if (input.size() <= max_contiguous_size_) {
                // Small enough to handle contiguously
                OperatorWrapper wrapper(hamiltonian_);
                wrapper.apply_contiguous(input, output);
            } else {
                // Too large - need to apply chunk-wise
                // Note: This assumes the Hamiltonian can be applied locally to chunks
                // For non-local Hamiltonians, more sophisticated handling is needed
                OperatorWrapper wrapper(hamiltonian_);
                wrapper(input, output);
            }
        }
    };
    
    /**
     * Utility function to create a chunked operator from a standard operator
     */
    ChunkedOperatorFunc make_chunked_operator(std::function<void(const Complex*, Complex*, int)> op) {
        return [op](const Complex* input, Complex* output, size_t size) {
            if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Chunk size too large for operator");
            }
            op(input, output, static_cast<int>(size));
        };
    }
    
    /**
     * Apply multiple operators in sequence (useful for time evolution)
     */
    void apply_operator_sequence(
        const std::vector<ChunkedOperatorFunc>& operators,
        const ChunkedComplexVector& input,
        ChunkedComplexVector& output,
        std::vector<ChunkedComplexVector>& temp_vectors
    ) {
        if (operators.empty()) {
            output = input;
            return;
        }
        
        // Ensure we have enough temporary vectors
        if (temp_vectors.size() < operators.size() - 1) {
            temp_vectors.resize(operators.size() - 1);
            for (auto& temp : temp_vectors) {
                temp.resize(input.size());
            }
        }
        
        // Apply first operator
        apply_operator(operators[0], input, 
                      operators.size() == 1 ? output : temp_vectors[0]);
        
        // Apply intermediate operators
        for (size_t i = 1; i < operators.size() - 1; ++i) {
            apply_operator(operators[i], temp_vectors[i-1], temp_vectors[i]);
        }
        
        // Apply final operator
        if (operators.size() > 1) {
            apply_operator(operators.back(), temp_vectors[operators.size()-2], output);
        }
    }
}

// BLAS-like operations for ChunkedComplexVector
namespace ChunkedBLAS {
    // AXPY operation: y = a*x + y
    void axpy(const Complex& a, const ChunkedComplexVector& x, ChunkedComplexVector& y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        
        for (size_t chunk_idx = 0; chunk_idx < x.num_chunks(); ++chunk_idx) {
            const Complex* x_data = x.chunk_data(chunk_idx);
            Complex* y_data = y.chunk_data(chunk_idx);
            size_t chunk_size = x.get_chunk_size(chunk_idx);
            
            for (size_t i = 0; i < chunk_size; ++i) {
                y_data[i] += a * x_data[i];
            }
        }
    }
    
    // SCAL operation: x = a*x
    void scal(const Complex& a, ChunkedComplexVector& x) {
        for (size_t chunk_idx = 0; chunk_idx < x.num_chunks(); ++chunk_idx) {
            Complex* x_data = x.chunk_data(chunk_idx);
            size_t chunk_size = x.get_chunk_size(chunk_idx);
            
            for (size_t i = 0; i < chunk_size; ++i) {
                x_data[i] *= a;
            }
        }
    }
    
    // COPY operation: y = x
    void copy(const ChunkedComplexVector& x, ChunkedComplexVector& y) {
        if (y.size() != x.size()) {
            y.resize(x.size());
        }
        
        for (size_t chunk_idx = 0; chunk_idx < x.num_chunks(); ++chunk_idx) {
            const Complex* x_data = x.chunk_data(chunk_idx);
            Complex* y_data = y.chunk_data(chunk_idx);
            size_t chunk_size = x.get_chunk_size(chunk_idx);
            
            std::memcpy(y_data, x_data, chunk_size * sizeof(Complex));
        }
    }
}

// Type alias for backward compatibility
using LargeComplexVector = ChunkedComplexVector;

#endif // CHUNKED_COMPLEX_VECTOR_H
