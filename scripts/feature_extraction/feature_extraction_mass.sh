#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2

nvcc --version
nvidia-smi
python3 plmfit.py --function $1 --ft_method $2 --head_config $3 \
        --data_type $4 --plm $5 --layer $6 --reduction $7 \
        --output_dir ${8} --experiment_dir ${9} --experiment_name ${10}
