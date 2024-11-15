#!/bin/bash
#SBATCH --cpus-per-task=1

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
# Export the global rank using SLURM_PROCID
export RANK=$SLURM_PROCID
echo "JOB ID: $SLURM_JOBID"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

export CUDA_LAUNCH_BLOCKING=1

source venv/bin/activate

#nvidia-smi
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > ${11}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
#NVIDIA_SMI_PID=$!

#while true; do
#  myjobs -j $SLURM_JOBID >> ${11}/task_monitor.log 2>&1
#  sleep 1
#done &
#CPU_FREE_PID=$!
 
python3 plmfit --function fine_tuning --ft_method feature_extraction --head_config paratope_linear_token_classification_head_config.json \
        --data_type paratope --split sampled --plm esm2_t6_8M_UR50D --layer last --reduction None --embeddings_path paratope_experiment_dir1\
        --experiment_dir paratope_experiment_dir1 --experiment_name paratope_experiment1 --gpus rtx_2080_ti:1 --beta True --experimenting True

#kill $NVIDIA_SMI_PID
#kill $CPU_FREE_PID