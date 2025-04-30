cmake_minimum_required(VERSION 3.15)
project(Exact_Diagonalization_Cpp LANGUAGES CXX)

# Options
option(USE_CUDA "Enable CUDA acceleration" ON)
option(USE_MPI "Enable MPI parallelization" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add CUDA if enabled
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Detect CUDA architecture if not specified
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "Auto")
    endif()
    
    # Add CUDA includes and libraries
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
endif()

# Find OpenMP if enabled
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
    else()
        message(WARNING "OpenMP not found. Disabling OpenMP support.")
        set(USE_OPENMP OFF)
    endif()
endif()

# Find MPI if enabled
if(USE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        include_directories(${MPI_CXX_INCLUDE_PATH})
        add_definitions(-DUSE_MPI)
    else()
        message(WARNING "MPI not found. Disabling MPI support.")
        set(USE_MPI OFF)
    endif()
endif()

# Find required libraries
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Find LAPACKE (part of LAPACK)
find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)
find_library(LAPACKE_LIBRARY NAMES lapacke)
if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
    include_directories(${LAPACKE_INCLUDE_DIR})
else()
    message(FATAL_ERROR "LAPACKE library not found")
endif()

# Find Eigen3
find_package(Eigen3 3.3 QUIET)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_definitions(-DUSE_EIGEN)
else()
    message(WARNING "Eigen3 not found. Some features might not be available.")
endif()

# Find ezARPACK (optional)
find_package(ezARPACK 1.0 CONFIG QUIET)
if(ezARPACK_FOUND)
    add_definitions(-DUSE_EZARPACK)
else()
    message(WARNING "ezARPACK not found. Some features might not be available.")
endif()

# Set compiler flags based on compiler and build type
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
endif()

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add main executable
add_executable(ED src/ed_run.cpp)

# Add CUDA-specific sources if CUDA is enabled
if(USE_CUDA)
    # Convert lanczos_cuda.h to lanczos_cuda.cu for compilation
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lanczos_cuda.h
                  ${CMAKE_CURRENT_BINARY_DIR}/src/lanczos_cuda.cu COPYONLY)
    
    # Explicitly add the CUDA implementation
    target_sources(ED PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src/lanczos_cuda.cu)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/src/lanczos_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

# Set up target libraries based on what's available
target_link_libraries(ED PRIVATE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARY}
)

# Add OpenMP libraries if enabled and found
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(ED PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add MPI libraries if enabled and found
if(USE_MPI AND MPI_FOUND)
    target_link_libraries(ED PRIVATE MPI::MPI_CXX)
endif()

# Add CUDA libraries if enabled
if(USE_CUDA)
    target_link_libraries(ED PRIVATE CUDA::cublas CUDA::cusolver CUDA::cudart)
endif()

# Add Eigen if found
if(Eigen3_FOUND)
    target_link_libraries(ED PRIVATE Eigen3::Eigen)
endif()

# Add ezARPACK if found
if(ezARPACK_FOUND)
    target_link_libraries(ED PRIVATE ezarpack::ezarpack)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "=====================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA enabled: ${USE_CUDA}")
message(STATUS "MPI enabled: ${USE_MPI}")
message(STATUS "OpenMP enabled: ${USE_OPENMP}")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "ezARPACK found: ${ezARPACK_FOUND}")
message(STATUS "")
