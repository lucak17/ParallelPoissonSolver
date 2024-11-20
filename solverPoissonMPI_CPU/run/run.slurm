#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:01:00
#SBATCH -A naiss2024-22-531
#SBATCH --nodes=1
#SBATCH -p main
#SBATCH --ntasks-per-node=16
#SBATCH -e solver.e
#SBATCH -o solver.o


# CC main.cpp -Wall -O3 -o solver.out
srun ../build/solverPoisson 4 4 4