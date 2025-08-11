cmake_minimum_required(VERSION 3.18)

# Options (define before project)
option(WITH_CUDA "Build with CUDA support" ON)
option(WITH_MPI "Build with MPI support" ON)

# Set languages based on options
if(WITH_CUDA)
    project(ExactDiagonalization LANGUAGES CXX CUDA)
    enable_language(CUDA)
else()
    project(ExactDiagonalization LANGUAGES CXX)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if CUDA is enabled
if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${SRC_DIR})

# Find required packages
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
option(WITH_MKL "Build with Intel MKL support" ON)
option(USE_ONEMKL "Use Intel oneMKL instead of traditional MKL" ON)

if(WITH_MKL)
    if(USE_ONEMKL)
        # Use oneMKL from oneAPI
        find_package(MKL CONFIG QUIET)
        if(MKL_FOUND)
            message(STATUS "Using Intel oneMKL from: ${MKLROOT}")
        else()
            message(STATUS "Intel oneMKL not found, falling back to traditional MKL detection")
            set(USE_ONEMKL OFF)
        endif()
    endif()
    
    if(NOT USE_ONEMKL)
        # Use traditional MKL - add include directories
        find_path(MKL_INCLUDE_DIR mkl.h
            HINTS
                /usr/include/mkl
                /opt/intel/mkl/include
                /opt/intel/oneapi/mkl/latest/include
                /opt/intel/compilers_and_libraries/linux/mkl/include
            PATH_SUFFIXES mkl
        )
        
        if(MKL_INCLUDE_DIR)
            include_directories(SYSTEM ${MKL_INCLUDE_DIR})
            message(STATUS "Found MKL headers at: ${MKL_INCLUDE_DIR}")
        else()
            message(WARNING "MKL headers not found, disabling MKL support")
            set(WITH_MKL OFF)
        endif()
    endif()
    
    if(WITH_MKL)
        add_definitions(-DWITH_MKL)
    endif()
endif()

# Print MKL vendor information
# message(STATUS "MKL library: ${MKL_LIBRARIES}")

# Print BLAS and LAPACK vendor information
message(STATUS "BLAS library: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK library: ${LAPACK_LIBRARIES}")

# Try to detect vendors
if(BLAS_LIBRARIES MATCHES "mkl")
    message(STATUS "BLAS/LAPACK vendor: Intel MKL")
elseif(BLAS_LIBRARIES MATCHES "openblas")
    message(STATUS "BLAS vendor: OpenBLAS")
elseif(BLAS_LIBRARIES MATCHES "atlas")
    message(STATUS "BLAS vendor: ATLAS")
elseif(BLAS_LIBRARIES MATCHES "Accelerate")
    message(STATUS "BLAS/LAPACK vendor: Apple Accelerate Framework")
elseif(BLAS_LIBRARIES MATCHES "FlexiBLAS")
    message(STATUS "BLAS vendor: FlexiBLAS")
else()
    message(STATUS "BLAS/LAPACK vendor: Unknown/Generic")
endif()
# Try to find LAPACKE
find_package(LAPACKE QUIET)
if(NOT LAPACKE_FOUND)
    # Try to find it manually
    find_library(LAPACKE_LIBRARY NAMES lapacke)
    if(LAPACKE_LIBRARY)
        set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
    else()
        message(FATAL_ERROR "LAPACKE library not found")
    endif()
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# CUDA setup
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DTPQ_HAVE_CUDA)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# Find ARPACK
find_library(ARPACK_LIBRARY NAMES arpack arpack-ng)
if(NOT ARPACK_LIBRARY)
    message(FATAL_ERROR "ARPACK library not found")
endif()

# MPI setup
if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
endif()

# Source files
set(SOURCES
    ${SRC_DIR}/ed_run.cpp
)

# Header-only sources that should be included in compilation
set(HEADER_SOURCES
    ${SRC_DIR}/*.h
)

# Add CUDA sources if available and CUDA is enabled
if(WITH_CUDA)
    # Explicitly add our CUDA files
    set(CUDA_SOURCES
        ${SRC_DIR}/TPQ_cuda.cu
        ${SRC_DIR}/TPQ_cuda_utils.cu
    )
    
    # Check if the files exist and add them
    foreach(cuda_file ${CUDA_SOURCES})
        if(EXISTS ${cuda_file})
            list(APPEND SOURCES ${cuda_file})
            message(STATUS "Adding CUDA source: ${cuda_file}")
        else()
            message(WARNING "CUDA source file not found: ${cuda_file}")
        endif()
    endforeach()
    
    # Also search for any other .cu files
    file(GLOB ADDITIONAL_CUDA_SOURCES ${SRC_DIR}/*.cu)
    foreach(cuda_file ${ADDITIONAL_CUDA_SOURCES})
        if(NOT ${cuda_file} IN_LIST SOURCES)
            list(APPEND SOURCES ${cuda_file})
            message(STATUS "Adding additional CUDA source: ${cuda_file}")
        endif()
    endforeach()
    
    # Create CUDA-only sources (without ed_run.cpp for tests)
    set(CUDA_ONLY_SOURCES ${CUDA_SOURCES})
else()
    message(STATUS "CUDA disabled, skipping CUDA sources")
endif()

# Define executable
add_executable(ED ${SOURCES})

# Add test executable
add_executable(TEST ${SRC_DIR}/test_observables.cpp)

# Add TPQ_DSSF executable (only if MPI is enabled since it uses MPI)
add_executable(TPQ_DSSF ${SRC_DIR}/TPQ_DSSF.cpp)


# Add CUDA example executable if CUDA is enabled and example exists
if(WITH_CUDA)
    set(CUDA_EXAMPLE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/examples/tpq_cuda_example.cpp)
    set(CUDA_TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/examples/cuda_test.cpp)
    
    if(EXISTS ${CUDA_EXAMPLE_FILE})
        add_executable(CUDA_EXAMPLE ${CUDA_EXAMPLE_FILE} ${SOURCES})
        message(STATUS "Adding CUDA example executable")
    else()
        message(STATUS "CUDA example file not found at: ${CUDA_EXAMPLE_FILE}")
    endif()
    
    if(EXISTS ${CUDA_TEST_FILE})
        add_executable(CUDA_TEST ${CUDA_TEST_FILE} ${CUDA_ONLY_SOURCES})
        message(STATUS "Adding CUDA test executable")
    endif()
endif()

if(WITH_MKL)
    if(USE_ONEMKL AND MKL_FOUND)
        # Link oneMKL using its modern CMake targets
        target_link_libraries(ED MKL::MKL)
    else()
        # Link traditional MKL (already linked via BLAS_LIBRARIES)
        # No additional linking needed since MKL is detected as BLAS/LAPACK
    endif()
endif()

# Link libraries
target_link_libraries(ED
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${ARPACK_LIBRARY}
)

# Link libraries for TEST executable
target_link_libraries(TEST
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${ARPACK_LIBRARY}
)

# Link libraries for TPQ_DSSF executable (only if it exists)
if(TARGET TPQ_DSSF)
    target_link_libraries(TPQ_DSSF
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()

# Link libraries for CUDA example if it exists
if(TARGET CUDA_EXAMPLE)
    target_link_libraries(CUDA_EXAMPLE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()

# Link libraries for CUDA test if it exists
if(TARGET CUDA_TEST)
    target_link_libraries(CUDA_TEST
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()

if(WITH_MKL)
    if(USE_ONEMKL AND MKL_FOUND)
        # Link oneMKL using its modern CMake targets
        target_link_libraries(ED MKL::MKL)
        target_link_libraries(TEST MKL::MKL)
        
        if(TARGET TPQ_DSSF)
            target_link_libraries(TPQ_DSSF MKL::MKL)
        endif()
        
        # Link MKL to CUDA example if it exists
        if(TARGET CUDA_EXAMPLE)
            target_link_libraries(CUDA_EXAMPLE MKL::MKL)
        endif()
        
        # Link MKL to CUDA test if it exists
        if(TARGET CUDA_TEST)
            target_link_libraries(CUDA_TEST MKL::MKL)
        endif()
    else()
        # Link traditional MKL (already linked via BLAS_LIBRARIES)
        # No additional linking needed since MKL is detected as BLAS/LAPACK
    endif()
endif()

# Add CUDA libraries if enabled
if(WITH_CUDA)
    target_link_libraries(ED
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
    )
    
    # Add CUDA libraries to other executables as well
    target_link_libraries(TEST
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
    )
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
        )
    endif()
    
    # Link CUDA libraries to example if it exists
    if(TARGET CUDA_EXAMPLE)
        target_link_libraries(CUDA_EXAMPLE
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
        )
    endif()
    
    # Link CUDA libraries to test if it exists
    if(TARGET CUDA_TEST)
        target_link_libraries(CUDA_TEST
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
        )
    endif()
endif()

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
    target_link_libraries(TEST ${MPI_CXX_LIBRARIES})
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF ${MPI_CXX_LIBRARIES})
    endif()
    
    # Link MPI to CUDA example if it exists
    if(TARGET CUDA_EXAMPLE)
        target_link_libraries(CUDA_EXAMPLE ${MPI_CXX_LIBRARIES})
    endif()
    
    # Link MPI to CUDA test if it exists
    if(TARGET CUDA_TEST)
        target_link_libraries(CUDA_TEST ${MPI_CXX_LIBRARIES})
    endif()
endif()

# Install targets
install(TARGETS ED DESTINATION bin)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(WITH_CUDA)
    message(STATUS "  CUDA Support: YES")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    message(STATUS "  CUDA Support: NO")
endif()
message(STATUS "  MPI Support: ${WITH_MPI}")

# Build configuration summary and performance options
message(STATUS "")
message(STATUS "=== Performance Configuration Summary ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Aggressive Optimizations: ${ENABLE_AGGRESSIVE_OPTS}")
message(STATUS "  Profile-Guided Optimization: ${ENABLE_PROFILE_GUIDED_OPTS}")
message(STATUS "  Native Architecture: ${ENABLE_NATIVE_ARCH}")
message(STATUS "  CUDA Support: ${WITH_CUDA}")
message(STATUS "  MPI Support: ${WITH_MPI}")
message(STATUS "  OpenMP Support: ${OpenMP_CXX_FOUND}")

# CPU information
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    execute_process(COMMAND grep -c "^processor" /proc/cpuinfo OUTPUT_VARIABLE CPU_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND grep "model name" /proc/cpuinfo COMMAND head -1 COMMAND sed "s/.*: //" OUTPUT_VARIABLE CPU_MODEL OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "  CPU: ${CPU_MODEL} (${CPU_COUNT} cores)")
    
    # Check for specific CPU features
    execute_process(COMMAND grep -q "avx2" /proc/cpuinfo RESULT_VARIABLE AVX2_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(AVX2_RESULT EQUAL 0)
        message(STATUS "  AVX2: Available")
    endif()
    
    execute_process(COMMAND grep -q "fma" /proc/cpuinfo RESULT_VARIABLE FMA_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(FMA_RESULT EQUAL 0)
        message(STATUS "  FMA: Available")
    endif()
endif()

# CUDA-specific compile options
if(WITH_CUDA)
    set_property(TARGET ED PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(ED PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    
    # Set CUDA compile options for other targets if they have CUDA sources
    if(TARGET TEST)
        set_property(TARGET TEST PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(TEST PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    
    if(TARGET TPQ_DSSF)
        set_property(TARGET TPQ_DSSF PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(TPQ_DSSF PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    
    # Set CUDA compile options for CUDA example if it exists
    if(TARGET CUDA_EXAMPLE)
        set_property(TARGET CUDA_EXAMPLE PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(CUDA_EXAMPLE PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    
    # Set CUDA compile options for CUDA test if it exists
    if(TARGET CUDA_TEST)
        set_property(TARGET CUDA_TEST PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(CUDA_TEST PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
endif()

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()
