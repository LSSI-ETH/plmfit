#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=70:00:00
echo "JOB ID: $SLURM_JOBID"

export HF_HOME='/cluster/scratch/dshlesin/'
export HF_HUB_CACHE='/cluster/scratch/dshlesin/'

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1
source /cluster/home/dshlesin/envs/myenv_plm/bin/activate
export DATA_DIR='/cluster/scratch/dshlesin/Manuscript/plmfit/plmfit'
nvcc --version
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used --format=csv -l 1 > ${7}/gpu_usage.log 2>&1 &
# Store the PID of the nvidia-smi background process
NVIDIA_SMI_PID=$!

while true; do
  myjobs -j $SLURM_JOBID >> ${7}/task_monitor.log 2>&1
  sleep 1
done &
CPU_FREE_PID=$!

python3 plmfit --function $1 --layer $5 --reduction $4 \
         --data_type $2 --plm $3 --output_dir $6 --experiment_dir $7 --experiment_name $8

kill $NVIDIA_SMI_PID
kill $CPU_FREE_PID