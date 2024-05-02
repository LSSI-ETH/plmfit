#!/bin/bash
#SBATCH --job-name=berk_playground   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000      # total memory per node (4 GB per cpu-core is default)
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:40g        #gpu memory
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=out.out
#SBATCH --error=error.err

#env2lmod
#module load gcc/8.2.0  python_gpu/3.11.2

python3 plmfit --function extract_embeddings --layer 48 --reduction mean --data_type rbd --plm ankh-large