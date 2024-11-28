#module --force purge
module load LUMI/24.03  partition/G
module load PrgEnv-cray/8.5.0
module load craype-accel-amd-gfx90a
module load rocm/6.0.3 
export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.29 
module load Boost/1.83.0-cpeCray-24.03

module load buildtools/24.03

## set environment variables required for compiling and linking
##   see (https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#compiling-with-hipcc)

export MPICH_GPU_SUPPORT_ENABLED=1

export PATH=$ROCM_PATH/llvm/bin:$PATH
export CC=hipcc
export CXX=hipcc

export CXXFLAGS="$CXXFLAGS -I${MPICH_DIR}/include"
export HIPFLAGS=$CXXFLAGS
export LDFLAGS="$LDFLAGS -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
