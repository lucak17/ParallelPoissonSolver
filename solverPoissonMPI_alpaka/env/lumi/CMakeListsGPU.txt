cmake_minimum_required(VERSION 3.22)  # Minimum CMake version

# Check if compilers are not already set and provide defaults
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
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
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -DNDEBUG")         # Release: High optimization

set(alpaka_ACC_GPU_HIP_ENABLE ON CACHE BOOL "Enable HIP backend for Alpaka")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx90a")

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


# Add HIP compiler flags
target_compile_options(solverPoisson PUBLIC --offload-arch=gfx90a)

# Include directories
target_include_directories(solverPoisson PUBLIC include)



