#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:01:00
#SBATCH -A naiss2024-22-531
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=400G
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -e solver.e
#SBATCH -o solver.o

export OMP_NUM_THREADS=1

# CC main.cpp -Wall -O3 -o solver.out
srun ../build/solverPoisson 2 2 2