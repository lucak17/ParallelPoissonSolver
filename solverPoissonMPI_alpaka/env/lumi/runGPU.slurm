#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:01:00
#SBATCH --account=project_465001586
#SBATCH --partition=standard-g
#SBATCH --nodes=2              # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
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
 

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1


srun --cpu-bind=${CPU_BIND} ./select_gpu ../build/solverPoisson 2 2 4
rm -rf ./select_gpu
