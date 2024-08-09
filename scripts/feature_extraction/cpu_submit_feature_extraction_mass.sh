#!/bin/bash

# Path to the CSV file
csv_file="./scripts/feature_extraction/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method data_type plm head task head_config ray_tuning layer reduction output_dir gpus gres mem_per_cpu
do
  output_dir="$output_dir"
  experiment_name="${data_type}_${plm}_${ft_method}_${layer}_${reduction}_${head}_${task}"
  experiment_dir="$output_dir/$function/${ft_method}/$experiment_name/$uid"
  sbatch --job-name="feature_extraction_${uid}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         scripts/feature_extraction/cpu_ray_tuning_mass.sh \
         "$function" "$ft_method" "$head_config" "$ray_tuning" "$data_type" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name"
  sleep 2
done