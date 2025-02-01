cmake_minimum_required(VERSION 3.22)  # Minimum CMake version

project(solverPoisson)                   # Project name

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)

# Default to Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Optimization flags for each build type
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")                 # Debug: No optimization, debug info
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -DNDEBUG")         # Release: High optimization

set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "Enable CUDA backend for Alpaka")

# Set the alpaka_USE_MDSPAN flag
# Options: OFF, SYSTEM, FETCH
set(mdspan_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/alpaka/_deps/mdspan-build)
list(APPEND CMAKE_PREFIX_PATH ${mdspan_DIR})
set(alpaka_USE_MDSPAN SYSTEM)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/alpaka/_deps/mdspan-src/include/)
#set(alpaka_USE_MDSPAN FETCH)

# Add the executable
add_subdirectory(thirdParty/alpaka)
alpaka_add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)



# Manually set MPI
set(MPI_openmpi_LIBRARY1 /gpfs/apps/MN5/ACC/OPENMPI/4.1.5/GCC/lib/libmpi.so)
set(MPI_openmpi_LIBRARY2 /gpfs/apps/MN5/ACC/OPENMPI/4.1.5/GCC/lib/libmpicxx.so)
#set(MPI_impi_LIBRARY1 /gpfs/apps/MN5/GPP/ONEAPI/2024.2/mpi/2021.13/lib/libmpi.so)
#set(MPI_impi_LIBRARY2 /gpfs/apps/MN5/GPP/ONEAPI/2024.2/mpi/2021.13/lib/libmpicxx.so)

include_directories(/gpfs/apps/MN5/ACC/OPENMPI/4.1.5/GCC/include)
target_link_libraries(solverPoisson PUBLIC ${MPI_openmpi_LIBRARY1} )

# Include directories
target_include_directories(solverPoisson PUBLIC include)