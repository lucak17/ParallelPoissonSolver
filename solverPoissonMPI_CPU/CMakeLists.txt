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

# Add the executable
add_executable(solverPoisson src/main.cpp)

# Include directories
target_include_directories(solverPoisson PUBLIC include)

# Correct library paths
#target_link_libraries(solverPoisson PRIVATE cubew cubelib)



