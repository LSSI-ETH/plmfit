#!/bin/bash

select_gpus() {
  bash ./scripts/find_configuration.sh "$gres" "$gpus" "$mem_per_cpu"
}

# Path to the CSV file
csv_file="./scripts/domain_adaptation/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method target_layers data_type split plm head task head_config layer output_dir gpus gres mem_per_cpu nodes run_time experimenting
do
  # read -r node_name gres gpus <<< "$(select_gpus)"
  # echo "$node_name $gres $gpus"
  output_dir="$output_dir"
  experiment_name="${data_type}_${split}_${plm}_${ft_method}_${layer}_${head}_${task}"
  experiment_dir="$output_dir/$function/${ft_method}_${target_layers}/$experiment_name/$uid"
  total_gpus="$((${gpus}*${nodes}))"
  
  sbatch --job-name="dom_adapt_${data_type}_${plm}_${task}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes=$nodes \
         --ntasks=$total_gpus \
         --ntasks-per-node=$gpus \
         --gpus-per-node=$gres:$gpus \
         --time=$run_time:00:00 \
         --nice=1 \
         scripts/domain_adaptation/single_job.sh \
         "$function" "$ft_method" "$target_layers" "$head_config" "$data_type" "$split" "$plm" "$layer" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting"
done
