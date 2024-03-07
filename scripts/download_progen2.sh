#!/bin/bash

# Load necessary modules, if wget is not available by default
# module load wget

#SBATCH --output=experiments/config_%a/out.out
#SBATCH --error=experiments/config_%a/error.err
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal

module load eth_proxy

MODEL_NAME="progen2-xlarge"

# Define the target directory for the download
TARGET_DIR="./plmfit/language_models/progen2/checkpoints"

# Define the URL of the file to download
FILE_URL="https://storage.googleapis.com/sfr-progen-research/checkpoints/${MODEL_NAME}.tar.gz"

# Define the directory name to extract into
EXTRACT_DIR="${TARGET_DIR}/${MODEL_NAME}"

# Create the target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Navigate to the target directory
cd "${TARGET_DIR}"

# Download the file
echo "Downloading ${MODEL_NAME}.tar.gz..."
wget "${FILE_URL}"

# Extract the contents into the ${MODEL_NAME} directory
echo "Extracting contents..."
mkdir -p "${MODEL_NAME}"
tar -xzvf "${MODEL_NAME}.tar.gz" -C "${MODEL_NAME}"

# Clean up the downloaded tar.gz file
echo "Cleaning up..."
rm "${MODEL_NAME}.tar.gz"

echo "Done."
