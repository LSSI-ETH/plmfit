#!/bin/bash

# Path to the CSV file
csv_file="./scripts/experiments_setup.csv"

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type plm reduction layer output_dir gpus gres mem_per_cpu
do
  experiment_dir="$output_dir/$function/${data_type}_${plm}_layer-${layer}_${reduction}"
  sbatch --output="$experiment_dir/euler_output.out" --error="$experiment_dir/euler_error.err" --mem-per-cpu="$mem_per_cpu" --gpus="$gpus" --gpus-per-node="$gpus" --gres="gpumem:$gres" scripts/extract_embeddings_mass.sh "$function" "$data_type" "$plm" "$reduction" "$layer" "$output_dir"
done
