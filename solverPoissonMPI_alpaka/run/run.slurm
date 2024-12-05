#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:02:00
#SBATCH --account=project_465001586
#SBATCH --partition=standard-g
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
####SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
####SBATCH --ntasks-per-gpu=1
#SBATCH -o solver.o
#SBATCH -e solver.e


cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
#CPU_BIND="map_cpu:49,57,17,25"
 

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1

#--cpu-bind=${CPU_BIND}
#srun --cpu-bind=${CPU_BIND} ./select_gpu ../build/solverPoisson 1 1 1
srun ./select_gpu ../build/solverPoisson 1 1 1
#srun ./select_gpu omnitrace-instrument -i 64 -- ../build/solverPoisson 2 2 2
rm -rf ./select_gpu
