#!/bin/bash
#SBATCH --cpus-per-task=1

export DATA_DIR='/cluster/home/estamkopoulo/plmfit_workspace/plmfit/plmfit'
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
module load gcc/8.2.0  python_gpu/3.11.2
module load cuda/12.1.1

nvcc --version
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 100 > ${9}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
NVIDIA_SMI_PID=$!

# Start logging CPU RAM usage
watch -n 100 free -m > ${9}/cpu_usage.log 2>&1 &
CPU_FREE_PID=$!

srun python3 plmfit --function $1 --ft_method $2 --target_layers $3 --head_config $4 \
        --data_type $5 --plm $6 --layer $7 \
        --output_dir $8 --experiment_dir $9 --experiment_name ${10} --gpus ${11} --nodes ${12} --beta True --experimenting ${13}
kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID