#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:01:00
#SBATCH --account=project_465001586
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks=16
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=400G
#SBATCH --ntasks-per-gpu=1
##SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -e solver.e
#SBATCH -o solver.o

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1

# CC main.cpp -Wall -O3 -o solver.out
srun ../build/solverPoisson 2 4 2