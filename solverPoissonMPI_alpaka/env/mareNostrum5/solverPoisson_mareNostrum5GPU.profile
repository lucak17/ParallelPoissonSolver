module --force purge
module load cmake/3.25.1
module load gcc/11.4.0
module load openmpi/4.1.5-gcc
module load boost/1.84.0-gcc-ompi
module load cuda/11.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo ${SCRIPT_DIR}
cp "$SCRIPT_DIR/CMakeLists.txt" "$SCRIPT_DIR/../../."
cp "$SCRIPT_DIR/run.slurm" "$SCRIPT_DIR/../../run"