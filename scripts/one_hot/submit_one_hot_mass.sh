#!/bin/bash

# Path to the CSV file
csv_file="./scripts/one_hot/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type split head task head_config ray_tuning output_dir gpus gres mem_per_cpu nodes
do
  output_dir="$output_dir"
  experiment_name="${data_type}_${head}_${task}"
  experiment_dir="$output_dir/$function/$experiment_name/$uid"
  sbatch --job-name="one_hot" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nice=1 \
         scripts/one_hot/one_hot_mass.sh \
         "$function" "$split" "$head_config" "$ray_tuning" "$data_type" "$output_dir" "$experiment_dir" "$experiment_name"
  sleep 2
done