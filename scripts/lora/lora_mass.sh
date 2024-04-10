#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

export DATA_DIR='/cluster/home/estamkopoulo/plmfit_workspace/plmfit/plmfit'
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONFAULTHANDLER=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Export the global rank using SLURM_PROCID
export RANK=$SLURM_PROCID
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2
module load cuda/12.1.1

nvcc --version
nvidia-smi
watch -n 10 nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> ${9}/gpu_usage.log 2>&1 &
srun python3 plmfit.py --function $1 --ft_method $2 --head_config $3 \
        --data_type $4 --plm $5 --layer $6 --reduction $7 \
        --output_dir ${8} --experiment_dir ${9} --experiment_name ${10} --gpus ${11} --nodes ${12} --beta True
kill $(pgrep watch)
