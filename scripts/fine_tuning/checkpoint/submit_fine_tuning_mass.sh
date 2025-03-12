#!/bin/bash

select_gpus() {
  bash ./scripts/find_configuration.sh "$gres" "$gpus" "$mem_per_cpu"
}

# Path to the CSV file
csv_file="./scripts/fine_tuning/checkpoint/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method target_layers data_type split plm head task head_config layer reduction output_dir experiment_dir gpus gres mem_per_cpu nodes run_time experimenting checkpoint
do
  # read -r node_name gres gpus <<< "$(select_gpus)"
  # echo "$node_name $gres $gpus"
  experiment_name="${data_type}_${split}_${plm}_${ft_method}_${layer}_${reduction}_${head}_${task}"
  total_gpus="$((${gpus}*${nodes}))"
  
  sbatch --job-name="${ft_method}_${data_type}_${plm}_${task}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes=$nodes \
         --ntasks=$total_gpus \
         --ntasks-per-node=$gpus \
         --gpus-per-node=$gres:$gpus \
         --time=$run_time:00:00 \
         scripts/fine_tuning/checkpoint/fine_tuning_mass.sh \
         "$function" "$ft_method" "$target_layers" "$head_config" "$data_type" "$split" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting" "$checkpoint"
done
