#!/bin/bash

select_gpus() {
  bash ./scripts/find_configuration.sh "$gres" "$gpus" "$mem_per_cpu"
}

# Path to the CSV file
csv_file="./scripts/fine_tuning/experiments_setup.csv"

uid=$(date +%Y%m%d_%H%M%S)

# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function ft_method target_layers data_type split plm head task head_config layer reduction output_dir gpus gres mem_per_cpu nodes run_time experimenting
do
  # read -r node_name gres gpus <<< "$(select_gpus)"
  # echo "$node_name $gres $gpus"
  output_dir="$output_dir"
  experiment_name="${data_type}_${split}_${plm}_${ft_method}_${layer}_${reduction}_${head}_${task}"
  experiment_dir="$output_dir/$function/${ft_method}_${target_layers}/$experiment_name/$uid"
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
         scripts/fine_tuning/fine_tuning_mass.sh \
         "$function" "$ft_method" "$target_layers" "$head_config" "$data_type" "$split" "$plm" "$layer" "$reduction" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting"
done

#fine_tuning	lora	all	RBDDSFFT	default	esmc_300m	linear	classification	lora_last_linear_classification_head_config.json	last	mean	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDGFFT	default	esmc_300m	linear	multilabel_classification	lora_linear_multilabel_classification_head_config.json	last	mean	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False


#fine_tuning	lora	all	RBDHCPWF	split	esmc_300m	linear	classification	lora_last_linear_classification_head_config_HC.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDDSPWF	split	esmc_300m	linear	classification	lora_last_linear_classification_head_config_DS.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDGPWF	split	esmc_300m	linear	multilabel_classification	lora_linear_multilabel_classification_head_config_G.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDHCBF	split	esmc_300m	linear	classification	lora_last_linear_classification_head_config_B.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDDSBF	split	esmc_300m	linear	classification	lora_last_linear_classification_head_config_B.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
#fine_tuning	lora	all	RBDGBF	split	esmc_300m	linear	multilabel_classification	lora_linear_multilabel_classification_head_config_BG.json	last	bos	/cluster/scratch/askrika	6	rtx_4090	20g	1	72	False
