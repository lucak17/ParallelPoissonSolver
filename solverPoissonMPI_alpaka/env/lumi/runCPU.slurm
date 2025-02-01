#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:06:00
#SBATCH --account=project_465001586
#SBATCH --partition=standard
#SBATCH --nodes=2              # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --cpus-per-task=128
#SBATCH -o solver.o
#SBATCH -e solver.e
###SBATCH --array=1


# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

##srun  ../build/solverPoisson 1 1 1 > "output_${SLURM_ARRAY_TASK_ID}.txt"
srun  ../build/solverPoisson 1 1 2