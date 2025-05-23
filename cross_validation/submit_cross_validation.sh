#!/bin/bash
#SBATCH --job-name=cross_eval
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -a && source .env && set +a

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONFAULTHANDLER=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RANK=$SLURM_PROCID

echo "JOB ID: $SLURM_JOBID"
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"

export CUDA_LAUNCH_BLOCKING=1
source $VIRTUAL_ENV/bin/activate

nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > logs/gpu_usage_$SLURM_JOBID.log 2>&1 &
NVIDIA_SMI_PID=$!

while true; do
  myjobs -j $SLURM_JOBID >> logs/task_monitor_$SLURM_JOBID.log 2>&1
  sleep 1
done &
CPU_FREE_PID=$!

# 👇 Replace with your script
python3 /cluster/home/askrika/ML/plmfit/cross_validation/cross_validate.py

kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID
