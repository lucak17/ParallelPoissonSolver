#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:00:20
#SBATCH -A naiss2024-22-531
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
###SBATCH --mem=400G
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -e solver.e
#SBATCH -o solver.o



cat << EOF > select_gpu
#!/bin/bash

if [ "\$SLURM_LOCALID" -gt 23 ]; then
    export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID - 24))
elif [ "\$SLURM_LOCALID" -gt 15 ]; then
    export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID - 16))
elif [ "\$SLURM_LOCALID" -gt 7 ]; then
    export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID - 8))
else
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
fi

exec \$*
EOF

chmod +x ./select_gpu




CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
#CPU_BIND="map_cpu:49,50,56,57,16,17,24,25,1,2,9,10,33,34,41,42"
#CPU_BIND="map_cpu:48,49,50,51,52,55,56,57,14,15,16,17,18,20,21,22,24,25,1,2,3,4,5,6,7,8,9,10,33,34,41,42"


export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} ./select_gpu ../build/solverPoisson 2 2 2
#srun  ../build/solverPoisson 1 1 1
# rm -rf ./select_gpu

# CC main.cpp -Wall -O3 -o solver.out
# srun ../build/solverPoisson 2 4 2