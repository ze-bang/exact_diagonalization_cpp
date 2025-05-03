cmake_minimum_required(VERSION 3.18)
project(ExactDiagonalization LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(WITH_CUDA "Build with CUDA support" ON)
option(WITH_MPI "Build with MPI support" ON)



# Source directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${SRC_DIR})

# Find required packages
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

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

# CUDA setup
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    
    # CUDA compilation flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O3")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
    endif()
    
    # Find cuBLAS and cuSOLVER
    find_library(CUBLAS_LIBRARY cublas)
    find_library(CUSOLVER_LIBRARY cusolver)
    
    if(NOT CUBLAS_LIBRARY OR NOT CUSOLVER_LIBRARY)
        message(FATAL_ERROR "cuBLAS or cuSOLVER library not found")
    endif()
    
    # Create CUDA wrapper file for lanczos_cuda.h
    file(WRITE ${CMAKE_BINARY_DIR}/lanczos_cuda_wrapper.cu 
         "#include \"${SRC_DIR}/lanczos_cuda.h\"\n")
    
    # Compile CUDA wrapper
    cuda_add_library(lanczos_cuda STATIC ${CMAKE_BINARY_DIR}/lanczos_cuda_wrapper.cu)
    target_link_libraries(lanczos_cuda ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
    
    # Enable CUDA in code
    add_definitions(-DWITH_CUDA)
endif()

# Source files
set(SOURCES
    ${SRC_DIR}/ed_run.cpp
)

# Define executable
add_executable(ED ${SOURCES})

# Link libraries
target_link_libraries(ED
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${ARPACK_LIBRARY}
)

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
endif()

# Add CUDA if enabled
if(WITH_CUDA)
    target_link_libraries(ED 
        lanczos_cuda
        ${CUDA_LIBRARIES}
        ${CUBLAS_LIBRARY}
        ${CUSOLVER_LIBRARY}
    )
endif()

# Install targets
install(TARGETS ED DESTINATION bin)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${WITH_CUDA}")
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

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()

# # Add test_lanczos executable
add_executable(test_lanczos ${SRC_DIR}/test_lanczos.cpp)

# Link test_lanczos with the same libraries as ED
target_link_libraries(test_lanczos
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${ARPACK_LIBRARY}
)

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(test_lanczos ${MPI_CXX_LIBRARIES})
endif()

# Add CUDA if enabled
if(WITH_CUDA)
    target_link_libraries(test_lanczos 
        lanczos_cuda
        ${CUDA_LIBRARIES}
        ${CUBLAS_LIBRARY}
        ${CUSOLVER_LIBRARY}
    )
endif()

# Apply the same optimization flags as the main executable
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_lanczos PRIVATE -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(test_lanczos PRIVATE -g)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(test_lanczos PRIVATE -O3 -xHost)
endif()

# Add OpenMP to test_lanczos if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_lanczos OpenMP::OpenMP_CXX)
endif()

# Add test_lanczos_cuda executable (only if CUDA is enabled)
if(WITH_CUDA)
    cuda_add_executable(test_lanczos_cuda ${SRC_DIR}/test_lanczos_cuda.cu)

    # Link with libraries
    target_link_libraries(test_lanczos_cuda
        lanczos_cuda
        ${CUDA_LIBRARIES}
        ${CUBLAS_LIBRARY}
        ${CUSOLVER_LIBRARY}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
    
    # Add MPI if enabled
    if(WITH_MPI)
        target_link_libraries(test_lanczos_cuda ${MPI_CXX_LIBRARIES})
    endif()
    
    # Add OpenMP to test_lanczos_cuda if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_lanczos_cuda OpenMP::OpenMP_CXX)
    endif()
endif()