#!/bin/bash

# Path to the CSV file
csv_file="./scripts/commands.csv"

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type plm reduction layer output_dir gpus gres mem_per_cpu
do
  sbatch --mem-per-cpu="$mem_per_cpu" --gpus="$gpus" --gres="gpumem:$gres" scripts/extract_embeddings_mass.sh "$function" "$data_type" "$plm" "$reduction" "$layer"
done
