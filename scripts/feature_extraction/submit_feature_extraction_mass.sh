#!/bin/bash

# Path to the CSV file
csv_file="./scripts/feature_extraction/experiments_setup.csv"


# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type split plm head task head_config ray_tuning layer reduction output_dir gpus gres mem_per_cpu nodes run_time experimenting embeddings_path
do
  echo "embeddings_path from CSV: $embeddings_path"
  uid=$(date +%Y%m%d_%H%M%S)_$RANDOM
  output_dir="$output_dir"
  experiment_name="${data_type}_${split}_${plm}_${function}_${layer}_${reduction}_${head}_${task}"
  experiment_dir="$output_dir/$function/$experiment_name/$uid"
  total_gpus="$((${gpus}*${nodes}))"
  # Print all variables to debug
  echo "function: $function"
  echo "data_type: $data_type"
  echo "split: $split"
  echo "plm: $plm"
  echo "head: $head"
  echo "task: $task"
  echo "head_config: $head_config"
  echo "ray_tuning: $ray_tuning"
  echo "layer: $layer"
  echo "reduction: $reduction"
  echo "output_dir: $output_dir"
  echo "experiment_name: $experiment_name"
  echo "experiment_dir: $experiment_dir"
  echo "gpus: $gpus"
  echo "gres: $gres"
  echo "mem_per_cpu: $mem_per_cpu"
  echo "nodes: $nodes"
  echo "run_time: $run_time"
  echo "experimenting: $experimenting"
  echo "embeddings_path: $embeddings_path"
  
  sbatch --job-name="feature_extraction_${data_type}_${plm}" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes=$nodes \
         --ntasks=$total_gpus \
         --ntasks-per-node=$gpus \
         --gpus-per-node=$gres:$gpus \
         --time=$run_time:00:00 \
         scripts/feature_extraction/feature_extraction_mass.sh \
        "$function" "$head_config" "$ray_tuning" "$data_type" "$split" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting" "$embeddings_path"
done
