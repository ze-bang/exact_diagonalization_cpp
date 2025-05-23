cmake_minimum_required(VERSION 3.18)
project(ExactDiagonalization LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(WITH_CUDA "Build with CUDA support" OFF)
option(WITH_MPI "Build with MPI support" ON)
option(ComputeCanada "Build on ComputeCanada" ON)
option(WITH_MKL "Build with Intel MKL support" ON)
option(USE_ONEMKL "Use Intel oneMKL instead of traditional MKL" ON)

if (WITH_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Source directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${SRC_DIR})

# Source files
set(SOURCES
    ${SRC_DIR}/ed_run.cpp
)

# Define executable
add_executable(ED ${SOURCES})

# MKL Configuration for ComputeCanada
if (ComputeCanada AND WITH_MKL)
    # Set MKL root
    set(MKLROOT $ENV{MKLROOT})
    
    if(USE_ONEMKL)
        # Try modern MKL first
        find_package(MKL QUIET)
        if(MKL_FOUND)
            target_link_libraries(ED MKL::MKL)
            message(STATUS "Using Intel oneMKL from: ${MKLROOT}")
        else()
            # Fallback to manual linking
            set(USE_ONEMKL OFF)
        endif()
    endif()
    
    if(NOT USE_ONEMKL OR NOT MKL_FOUND)
        # Manual MKL linking for ComputeCanada
        find_library(MKL_INTEL_ILP64 mkl_intel_ilp64 PATHS ${MKLROOT}/lib)
        find_library(MKL_INTEL_THREAD mkl_intel_thread PATHS ${MKLROOT}/lib)
        find_library(MKL_CORE mkl_core PATHS ${MKLROOT}/lib)
        find_library(IOMP5 iomp5 PATHS ${MKLROOT}/lib)
        
        if(MKL_INTEL_ILP64 AND MKL_INTEL_THREAD AND MKL_CORE)
            target_compile_definitions(ED PRIVATE MKL_ILP64)
            target_include_directories(ED PRIVATE ${MKLROOT}/include)
            
            # Proper MKL link order
            target_link_libraries(ED 
                ${MKL_INTEL_ILP64}
                ${MKL_INTEL_THREAD}
                ${MKL_CORE}
                ${IOMP5}
                pthread
                m
                dl
            )
            message(STATUS "Using manual MKL linking with ILP64 interface")
        else()
            message(FATAL_ERROR "Required MKL libraries not found")
        endif()
    endif()
    
    add_definitions(-DWITH_MKL)
    
else()
    # Non-ComputeCanada MKL setup
    if(WITH_MKL)
        if(USE_ONEMKL)
            find_package(MKL REQUIRED)
            target_link_libraries(ED MKL::MKL)
            message(STATUS "Using Intel oneMKL")
        endif()
        add_definitions(-DWITH_MKL)
    else()
        # Use system BLAS/LAPACK
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        target_link_libraries(ED ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
endif()

# Find LAPACKE
find_package(LAPACKE QUIET)
if(NOT LAPACKE_FOUND)
    find_library(LAPACKE_LIBRARY NAMES lapacke PATHS ${MKLROOT}/lib)
    if(LAPACKE_LIBRARY)
        set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
    else()
        message(FATAL_ERROR "LAPACKE library not found")
    endif()
endif()
target_link_libraries(ED ${LAPACKE_LIBRARIES})

# Find Eigen3
find_package(Eigen3 REQUIRED)
target_include_directories(ED SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})

# Find ARPACK
find_library(ARPACK_LIBRARY NAMES arpack arpack-ng)
if(NOT ARPACK_LIBRARY)
    message(FATAL_ERROR "ARPACK library not found")
endif()
target_link_libraries(ED ${ARPACK_LIBRARY})

# MPI setup
if(WITH_MPI)
    find_package(MPI REQUIRED)
    target_include_directories(ED SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
    target_compile_definitions(ED PRIVATE WITH_MPI)
endif()

# CUDA setup
if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # Create CUDA wrapper
    file(WRITE ${CMAKE_BINARY_DIR}/lanczos_cuda_wrapper.cu 
         "#include \"${SRC_DIR}/lanczos_cuda.h\"\n")
    
    add_library(lanczos_cuda STATIC ${CMAKE_BINARY_DIR}/lanczos_cuda_wrapper.cu)
    target_link_libraries(lanczos_cuda CUDA::cublas CUDA::cusolver)
    
    target_link_libraries(ED lanczos_cuda)
    target_compile_definitions(ED PRIVATE WITH_CUDA)
endif()

# Add OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP Support: YES")
else()
    message(STATUS "OpenMP Support: NO")
endif()

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ED PRIVATE -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(ED PRIVATE -g)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(ED PRIVATE -O3 -xHost)
endif()

# Install targets
install(TARGETS ED DESTINATION bin)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${WITH_CUDA}")
message(STATUS "  MPI Support: ${WITH_MPI}")
message(STATUS "  MKL Support: ${WITH_MKL}")