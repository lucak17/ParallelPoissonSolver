#!/bin/bash -l
#SBATCH -A ehpc71
#SBATCH -q acc_ehpc
###SBATCH -q acc_training
#SBATCH --time=00:02:00
#SBATCH --job-name=poisson

#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=20
#SBATCH --exclusive
#SBATCH -o solver.o
#SBATCH -e solver.e
###SBATCH --array=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

cat << EOF > select_gpu
#!/bin/bash

####export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
if [ "\$SLURM_LOCALID" = 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
    export UCX_NET_DEVICES=mlx5_0:1
elif [ "\$SLURM_LOCALID" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
    export UCX_NET_DEVICES=mlx5_1:1
elif [ "\$SLURM_LOCALID" = 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
    export UCX_NET_DEVICES=mlx5_4:1
elif [ "\$SLURM_LOCALID" = 3 ]; then
    export CUDA_VISIBLE_DEVICES=3
    export UCX_NET_DEVICES=mlx5_5:1
fi
exec \$*
EOF

chmod +x ./select_gpu


CPU_BIND="map_cpu:1,21,41,61"
# "output_${SLURM_ARRAY_TASK_ID}.txt"
srun --cpu-bind=${CPU_BIND} ./select_gpu ../build/solverPoisson 2 2 2

#srun nsys profile  --trace=cuda,mpi --gpu-metrics-device=all -o profile_out ../build/solverPoisson 1 1 1
#srun ncu --set full --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared --metrics group:memory__chart --metrics group:memory__shared_table -o profile_ncu_shared ../build/solverPoisson 1 1 1
#srun ncu --set full -o profile_ncu_shared_jump ../build/solverPoisson 1 1 1
