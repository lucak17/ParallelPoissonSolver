# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-build"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/tmp"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src"
  "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
