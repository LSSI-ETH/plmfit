#!/bin/bash

# Path to the CSV file
csv_file="./scripts/commands.csv"

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method data_type plm head head_config layer reduction batch_size epochs lr weight_decay optimizer loss_f output_dir gpus gres mem_per_cpu
do
  output_dir="$output_dir"
  experiment_name="${data_type}_${plm}_${ft_method}_${layer}_${reduction}_${head}"
  experiment_dir="$output_dir/$function/$experiment_name"
  sbatch --output="$experiment_dir/euler_output.out" --error="$experiment_dir/euler_error.err" --mem-per-cpu="$mem_per_cpu" --gpus="$gpus" --gres="gpumem:$gres" scripts/feature_extraction_mass.sh "$function" "$ft_method" "$data_type" "$plm" "$head" "$head_config" "$layer" "$reduction" $batch_size $epochs $lr $weight_decay "$optimizer" "$loss_f" "$output_dir" "$experiment_dir" "$experiment_name"
done
