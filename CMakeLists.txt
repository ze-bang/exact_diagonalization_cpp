cmake_minimum_required(VERSION 3.10)
project(exact_diagonalization_cpp VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option for OpenMP
option(USE_OpenMP "Use OpenMP for parallelization" ON)

# Platform-specific settings
if(APPLE)
    # macOS-specific settings
    # Using Homebrew LLVM for better OpenMP support
    if(EXISTS "/opt/homebrew/opt/llvm/bin/clang++")
        set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
        set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
        set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
        set(OPENMP_INCLUDES "/opt/homebrew/opt/llvm/include")
    endif()
    
    # Homebrew include paths
    set(BREW_PREFIX "/opt/homebrew")
    if(NOT EXISTS ${BREW_PREFIX})
        # Try to find Homebrew prefix
        execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE BREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    
    if(EXISTS ${BREW_PREFIX})
        include_directories("${BREW_PREFIX}/opt/openblas/include")
        include_directories("${BREW_PREFIX}/opt/lapack/include")
        include_directories("${BREW_PREFIX}/opt/eigen/include")
        include_directories("${BREW_PREFIX}/opt/mpich/include")
        
        link_directories("${BREW_PREFIX}/opt/openblas/lib")
        link_directories("${BREW_PREFIX}/opt/lapack/lib")
    endif()
elseif(LINUX)
    # Linux-specific settings
    # Standard paths for libraries
    include_directories("/usr/include")
    include_directories("/usr/local/include")
    link_directories("/usr/lib")
    link_directories("/usr/local/lib")
endif()

# Find dependencies
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)

# OpenMP setup
if(USE_OpenMP)
    if(APPLE)
        # macOS-specific OpenMP setup (using Homebrew LLVM)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
            set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif()
    endif()
    
    # Find OpenMP
    find_package(OpenMP REQUIRED)
    
    if(OpenMP_FOUND)
        if(APPLE AND OPENMP_INCLUDES)
            include_directories("${OPENMP_INCLUDES}")
        endif()
        
        if(APPLE AND OPENMP_LIBRARIES)
            link_directories("${OPENMP_LIBRARIES}")
        endif()
        
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Add source directory
include_directories("${PROJECT_SOURCE_DIR}/src")

# Add executables
add_executable(ED "${PROJECT_SOURCE_DIR}/src/ed_run.cpp")

# Linking libraries
target_link_libraries(ED PRIVATE
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

# Platform-specific linking
if(APPLE)
    target_link_libraries(ED PRIVATE
        openblas
        lapacke
    )
else()
    target_link_libraries(ED PRIVATE
        lapacke
        cblas
    )
endif()

# Common libraries
target_link_libraries(ED PRIVATE
    Eigen3::Eigen
)

# Optional ARPACK support
option(USE_ARPACK "Use ARPACK for eigenvalue calculations" OFF)
if(USE_ARPACK)
    # Find ezARPACK
    find_package(ezARPACK CONFIG)
    if(ezARPACK_FOUND)
        target_link_libraries(ED PRIVATE ezarpack)
    else()
        message(WARNING "ezARPACK not found, ARPACK methods will not be available")
    endif()
endif()

# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  OpenMP: ${USE_OpenMP}")
message(STATUS "  ARPACK: ${USE_ARPACK}")