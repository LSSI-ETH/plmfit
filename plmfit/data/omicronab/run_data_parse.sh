#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=1:00:00
#SBATCH --output=output.out
#SBATCH --error=error.err

# Load required modules
module load gcc/8.2.0
module load python_gpu/3.11.2

# Activate your virtual environment if necessary
# source activate <your_virtual_environment>

# Run the Python script
python /cluster/home/srenwick/plmfit/plmfit/data/omicronab/data_parse.py

echo "Script execution completed"