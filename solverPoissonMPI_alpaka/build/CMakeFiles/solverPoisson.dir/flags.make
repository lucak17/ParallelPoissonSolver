# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# compile HIP with /opt/rocm-5.7.0/llvm/bin/clang++
HIP_DEFINES = -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=47 -DALPAKA_DEBUG=0 -DALPAKA_USE_MDSPAN -D__HIP_ROCclr__=1

HIP_INCLUDES = -I/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/thirdParty/alpaka/_deps/mdspan-src/include -I/opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include -I/cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/include -isystem /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/thirdParty/alpaka/include -isystem /pdc/software/23.12/spack/boost-1.83.0-f3qnoa2/include -isystem /opt/rocm-5.7.0/include/rocrand -isystem /opt/rocm-5.7.0/include/hiprand -isystem /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/_deps/mdspan-src/include

HIP_FLAGS = -O3 -DNDEBUG -std=c++17 --offload-arch=gfx906 --offload-arch=gfx90a -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false

