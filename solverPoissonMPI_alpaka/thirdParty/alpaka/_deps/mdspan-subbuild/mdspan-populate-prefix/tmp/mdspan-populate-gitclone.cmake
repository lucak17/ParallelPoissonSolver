# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitclone-lastrun.txt" AND EXISTS "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitinfo.txt" AND
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/kokkos/mdspan.git" "mdspan-src"
    WORKING_DIRECTORY "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/kokkos/mdspan.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "973ef6415a6396e5f0a55cb4c99afd1d1d541681" --
  WORKING_DIRECTORY "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '973ef6415a6396e5f0a55cb4c99afd1d1d541681'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitinfo.txt" "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/mdspan-populate-gitclone-lastrun.txt'")
endif()
