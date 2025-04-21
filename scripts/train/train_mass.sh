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
source $VIRTUAL_ENV/bin/activate

nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > ${7}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
NVIDIA_SMI_PID=$!

while true; do
  myjobs -j $SLURM_JOBID >> ${7}/task_monitor.log 2>&1
  sleep 1
done &
CPU_FREE_PID=$!

#python3 plmfit --function $1 --head_config $2 \
        #--data_type $4 --split $5 --ray_tuning $3 \
        #--output_dir ${6} --experiment_dir ${7} --experiment_name ${8} \
        #--gpus ${9} --nodes ${10} --beta True --experimenting ${11}

python3 plmfit --function "$1" --head_config "$2" \
        --data_type "$4" --split "$5" --ray_tuning "$3" \
        --output_dir "$6" --experiment_dir "$7" --experiment_name "$8" \
        --gpus "$9" --nodes "${10}" --beta True --experimenting "${11}" \
        --model_path "${12}" --experiment_name "${13}" \
        --evaluate True

kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID