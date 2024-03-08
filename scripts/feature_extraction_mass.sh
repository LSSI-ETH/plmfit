#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2
python3 plmfit.py --function $1 --ft_method $2 \
        --head $5 --head_config $6 \
        --layer $7 --reduction $8 --data_type $3 --plm $4 \
        --batch_size $9 --epochs ${10} --lr ${11} --weight_decay ${12} \
        --optimizer ${13} --loss_f ${14} --embs ${15} --output_dir ${15}
