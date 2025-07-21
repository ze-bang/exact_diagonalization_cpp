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

# Add CUDA sources if available and CUDA is enabled
if(WITH_CUDA)
    file(GLOB CUDA_SOURCES ${SRC_DIR}/*.cu)
    if(CUDA_SOURCES)
        list(APPEND SOURCES ${CUDA_SOURCES})
        message(STATUS "Found CUDA source files: ${CUDA_SOURCES}")
    else()
        message(STATUS "No CUDA source files found in ${SRC_DIR}")
    endif()
endif()

# Define executable
add_executable(ED ${SOURCES})

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

if(WITH_MKL)
    if(USE_ONEMKL AND MKL_FOUND)
        # Link oneMKL using its modern CMake targets
        target_link_libraries(ED MKL::MKL)
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
endif()

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
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

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ED PRIVATE -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(ED PRIVATE -g)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(ED PRIVATE -O3 -xHost)
endif()

# CUDA-specific compile options
if(WITH_CUDA)
    set_property(TARGET ED PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(ED PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()
