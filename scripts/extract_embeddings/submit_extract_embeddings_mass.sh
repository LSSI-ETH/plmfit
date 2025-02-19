#!/bin/bash

# Path to the CSV file
csv_file="./scripts/extract_embeddings/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type plm reduction layer output_dir gpus gres mem_per_cpu batch_size
do
  #experiment_dir="$output_dir/$function/${data_type}_${plm}_layer-${layer}_${reduction}"
  output_dir="$output_dir"
  experiment_name="${data_type}_${plm}_embs_${layer}_${reduction}"
  experiment_dir="$output_dir/$function/$experiment_name"
  sbatch --job-name="extract_embs_${plm}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --gpus="$gpus" \
         --gpus-per-node="$gpus" \
         --gres="gpumem:$gres" \
         scripts/extract_embeddings/extract_embeddings_mass.sh \
         "$function" "$data_type" "$plm" "$reduction" "$layer" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$batch_size"
done
