#!/bin/bash
#SBATCH --job-name=accelerate   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12g      # total memory per node (8 GB per cpu-core is default)
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gres=gpumem:24g        #gpu memory
#SBATCH --gpus-per-node=2
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=accel_out.out
#SBATCH --error=accel_error.err

echo $SLURM_GPUS
nvidia-smi

python3 multi_gpu_inference.py

