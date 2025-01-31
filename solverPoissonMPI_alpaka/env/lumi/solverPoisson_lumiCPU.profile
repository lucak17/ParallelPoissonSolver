module --force purge
module load LUMI/24.03  partition/C 
#module load PrgEnv-cray/8.5.0
module load PrgEnv-gnu/8.5.0
module load cray-mpich/8.1.29 
#module load Boost/1.83.0-cpeCray-24.03
module load Boost/1.83.0-cpeGNU-24.03
module load buildtools/24.03
## set environment variables required for compiling and linking
##   see (https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#compiling-with-hipcc)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo ${SCRIPT_DIR}
cp "$SCRIPT_DIR/CMakeListsCPU.txt" "$SCRIPT_DIR/../../CMakeLists.txt"
cp "$SCRIPT_DIR/runCPU.slurm" "$SCRIPT_DIR/../../run/runCPU.slurm"