#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

export DATA_DIR='/cluster/home/srenwick/plmfit_workspace/plmfit/plmfit'
module load stack/2024-05
module load gcc/13.2.0
module load python/3.9.18_cuda

nvcc --version
nvidia-smi
python3 plmfit.py --function $1 --layer $5 --reduction $4 \
         --data_type $2 --plm $3 --output_dir $6 --experiment_dir $7 --experiment_name $8
