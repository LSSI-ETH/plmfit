#!/bin/bash
#SBATCH --job-name=hyperparameter_tuning    # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8g      # total memory per node (8 GB per cpu-core is default)
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g        #gpu memory
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=experiments/config_%a/out.out
#SBATCH --error=experiments/config_%a/error.err

module load gcc/8.2.0  python_gpu/3.11.2

python3 plmfit.py --function extract_embeddings --layer first --reduction mean --data_type meltome --plm progen2-small --output_dir $SCRATCH