module load PDC/23.12
module load PrgEnv-cray/8.5.0
module load craype-accel-amd-gfx90a
module load rocm/5.7.0
export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.28 
module load boost/1.83.0-gcc-f3q
module load cmake/3.27.7


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo ${SCRIPT_DIR}
cp "$SCRIPT_DIR/CMakeLists.txt" "$SCRIPT_DIR/../../."
cp "$SCRIPT_DIR/run.slurm" "$SCRIPT_DIR/../../run/"