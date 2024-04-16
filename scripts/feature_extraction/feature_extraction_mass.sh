#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=16:00:00

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2

export DATA_DIR='/cluster/home/estamkopoulo/plmfit_workspace/plmfit/plmfit'

nvcc --version
nvidia-smi
python3 plmfit.py --function $1 --ft_method $2 --head_config $3 \
        --data_type $5 --plm $6 --layer $7 --reduction $8 \
        --output_dir ${9} --experiment_dir ${10} --experiment_name ${11}
