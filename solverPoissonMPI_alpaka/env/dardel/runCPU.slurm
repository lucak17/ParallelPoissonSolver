#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:01:00
#SBATCH -A naiss2024-5-582
#SBATCH -p main
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH -e solver.e
#SBATCH -o solver.o


export OMP_NUM_THREADS=64

srun  ../build/solverPoisson 1 1 1
