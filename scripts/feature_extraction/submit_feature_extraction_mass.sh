#!/bin/bash

# Path to the CSV file
csv_file="./scripts/feature_extraction/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method data_type split plm head task head_config ray_tuning layer reduction output_dir gpus gres mem_per_cpu nodes run_time
do
  output_dir="$output_dir"
  experiment_name="${data_type}_${plm}_${ft_method}_${layer}_${reduction}_${head}_${task}"
  experiment_dir="$output_dir/$function/${ft_method}/$experiment_name/$uid"
  total_gpus="$((${gpus}*${nodes}))"
  sbatch --job-name="feature_extraction_${data_type}_${plm}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes="$nodes" \
         --ntasks="$nodes" \
         --cpus-per-task=1 \
         --tasks-per-node=1 \
         --gpus-per-node=1 \
         --gres=gpumem:${gres} \
         --time="${run_time}:00:00" \
         --nice=1 \
         scripts/feature_extraction/ray_tuning_mass.sh \
         "$function" "$ft_method" "$head_config" "$ray_tuning" "$data_type" "$split" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus"
done
