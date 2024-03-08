#!/bin/bash

# Path to the CSV file
csv_file="./scripts/commands.csv"

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method data_type plm head head_config layer reduction batch_size epochs lr weight_decay optimizer loss_f embs gpus gres mem_per_cpu
do
  sbatch --mem-per-cpu="$mem_per_cpu" --gpus="$gpus" --gres="gpumem:$gres" scripts/feature_extraction_mass.sh "$function" "$ft_method" "$data_type" "$plm" "$head" "$head_config" "$layer" "$reduction" $batch_size $epochs $lr $weight_decay "$optimizer" "$loss_f" $embs
done
