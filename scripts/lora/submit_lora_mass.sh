#!/bin/bash

select_gpus() {
  bash ./scripts/find_configuration.sh "$gres" "$gpus" "$mem_per_cpu"
}

# Path to the CSV file
csv_file="./scripts/lora/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method data_type plm head task head_config layer reduction output_dir gpus gres mem_per_cpu nodes
do
  # read -r node_name gres gpus <<< "$(select_gpus)"
  # echo "$node_name $gres $gpus"
  output_dir="$output_dir"
  experiment_name="${data_type}_${plm}_${ft_method}_${layer}_${reduction}_${head}_${task}"
  experiment_dir="$output_dir/$function/$ft_method/$experiment_name/$uid"
  total_gpus="$((${gpus}*${nodes}))"
  
  sbatch --job-name="lora_${uid}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes=$nodes \
         --ntasks=$total_gpus \
         --ntasks-per-node=$gpus \
         --gpus-per-node=$gres:$gpus \
         scripts/lora/lora_mass.sh \
         "$function" "$ft_method" "$head_config" "$data_type" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes"
done
