cmake_minimum_required(VERSION 3.22)  # Minimum CMake version

# Check if compilers are not already set and provide defaults
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER cc)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER CC)
endif()

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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")         # Release: High optimization

set(alpaka_ACC_GPU_HIP_ENABLE ON CACHE BOOL "Enable HIP backend for Alpaka")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx90a")



# Add the executable
#add_executable(solverPoisson src/main.cpp)
add_subdirectory(thirdParty/alpaka)
alpaka_add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)


# MPI compiler for CXX
# set(MPI_CXX_COMPILER /opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/bin/mpicxx)
# MPI CXX compiler wrapper include directories
# set(MPI_CXX_COMPILER_INCLUDE_DIRS /opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include)
# MPI CXX header directory
# set(MPI_CXX_HEADER_DIR /opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include)
# MPI compiler for C
# set(MPI_C_COMPILER /opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/bin/mpicc)
# Location of the mpi_cray library for MPI
# 
set(MPI_mpi_cray_LIBRARY /opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/lib/libmpi_cray.so)
set(Cray_xpmem_LIBRARY /opt/cray/xpmem/2.8.2-1.0_3.9__g84a27a5.shasta/lib64/libxpmem.so)
set(MPI_GTL_amd_gfx90a_LIBRARY /opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_hsa.so)

# Include directories
include_directories(/opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include)

target_link_libraries(solverPoisson PUBLIC ${MPI_mpi_cray_LIBRARY} ${Cray_xpmem_LIBRARY} ${MPI_GTL_amd_gfx90a_LIBRARY})
# Add HIP compiler flags
target_compile_options(solverPoisson PUBLIC --offload-arch=gfx90a)

# Include directories
target_include_directories(solverPoisson PUBLIC include)



