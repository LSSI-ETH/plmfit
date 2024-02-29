#!/bin/bash
#SBATCH --job-name=feature_extraction    # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16g      # total memory per node (8 GB per cpu-core is default)
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:16g        #gpu memory
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=experiments/config_%j/out.out
#SBATCH --error=experiments/config_%j/error.err

module load gcc/8.2.0  python_gpu/3.11.2

python3 plmfit.py --function fine_tuning --ft_method lora \
        --head logistic_regression --head_config config_logistic_regression.json \
        --data_type aav --plm progen2-small \
        --batch_size 128 --epochs 5 --lr 1e-4 --weight_decay 1e-6 \
        --optimizer adam --loss_f mse