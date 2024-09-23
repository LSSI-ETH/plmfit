#!/bin/bash
#SBATCH --cpus-per-task=1

export HF_HOME='/cluster/scratch/estamkopoulo/'
export HF_HUB_CACHE='/cluster/scratch/estamkopoulo/'

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONFAULTHANDLER=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Export the global rank using SLURM_PROCID
export RANK=$SLURM_PROCID
echo "JOB ID: $SLURM_JOBID"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1

nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > ${11}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
NVIDIA_SMI_PID=$!

while true; do
  myjobs -j $SLURM_JOBID >> ${11}/task_monitor.log 2>&1
  sleep 1
done &
CPU_FREE_PID=$!

srun python3 plmfit --function $1 --ft_method $2 --target_layers $3 --head_config $4 \
        --data_type $5 --split $6 --plm $7 --layer $8 --reduction $9 \
        --output_dir ${10} --experiment_dir ${11} --experiment_name ${12} --gpus ${13} --nodes ${14} --beta True --experimenting ${15}

kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID