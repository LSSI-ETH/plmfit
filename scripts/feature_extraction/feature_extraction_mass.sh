#!/bin/bash
echo "JOB ID: $SLURM_JOBID"
module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1
set -a && source .env && set +a
source $VIRTUAL_ENV/bin/activate

nvcc --version
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > ${11}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
NVIDIA_SMI_PID=$!

while true; do
  myjobs -j $SLURM_JOBID >> ${11}/task_monitor.log 2>&1
  sleep 1
done &
CPU_FREE_PID=$!

python3 plmfit --function $1 --ft_method $2 --head_config $3 --ray_tuning $4 \
        --data_type $5 --split $6 --plm $7 --layer $8 --reduction $9 \
        --output_dir ${10} --experiment_dir ${11} --experiment_name ${12} --beta True

kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID