#!/bin/bash

# Path to the CSV file
csv_file="./scripts/train/experiments_setup.csv"


# Skip the header line
tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type split head task head_config ray_tuning output_dir gpus gres mem_per_cpu nodes run_time experimenting
#tail -n +2 "$csv_file" | while IFS=$'\t' read -r function data_type split head task head_config ray_tuning output_dir model_path evaluate gpus gres mem_per_cpu nodes run_time experimenting

do
  uid=$(date +%Y%m%d_%H%M%S)_$RANDOM
  output_dir="$output_dir"
  experiment_name="${data_type}_${split}_${head}_${task}"
  experiment_dir="$output_dir/$function/$experiment_name/$uid"
  total_gpus="$((${gpus}*${nodes}))"
  
  sbatch --job-name="train" \
         --output="$experiment_dir/euler_output.out" \
         --error="$experiment_dir/euler_error.err" \
         --mem-per-cpu="$mem_per_cpu" \
         --nodes=$nodes \
         --ntasks=$total_gpus \
         --ntasks-per-node=$gpus \
         --gpus-per-node=$gres:$gpus \
         --time=$run_time:00:00 \
         scripts/train/train_mass.sh \
         "$function" "$head_config" "$ray_tuning" "$data_type" "$split" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting"
  sleep 2
done
<<<<<<< Updated upstream
=======

#one_hot	RBDDS	default	mlp	classification	RBD_mlp_classification_head_config_weighted.json	True	/cluster/scratch/askrika/DS	1	rtx_4090	100g	1	72	False
#one_hot	RBDG	split	mlp	classification	RBD_mlp_multilabel_classification_head_config_weighted.json	False	/cluster/scratch/askrika/G	1	rtx_4090	100g	1	1	False
#function	data_type	split	head	task	head_config	ray_tuning	output_dir	gpus	gres	mem-per-cpu	nodes	run_time	experimenting
#one_hot	RBDHC	split	mlp	classification	RBD_mlp_classification_head_config_weighted.json	True	/cluster/scratch/askrika/HC	1	rtx_4090	100g	1	24	False
#one_hot	RBDDS	split	mlp	classification	RBD_mlp_classification_head_config_weighted.json	True	/cluster/scratch/askrika/DS	1	rtx_4090	100g	1	24	False
#one_hot	RBDG	split	mlp	classification	RBD_mlp_multilabel_classification_head_config_weighted.json	True	/cluster/scratch/askrika/G	1	rtx_4090	100g	1	24	False
#"$function" "$head_config" "$ray_tuning" "$data_type" "$split" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting" 
#one_hot	RBDHC	split	mlp	classification	RBDDS_split_classification_head_config.json	False	/cluster/scratch/askrika/DSE	/cluster/scratch/askrika/DS/one_hot/RBDDS_split_mlp_classification/20250423_132523_17128/lightning_logs/best_model.ckpt	True	1	rtx_4090	100g	1	24	False
#one_hot	RBDDS	split	mlp	classification	RBDHC_split_classification_head_config.json	False	/cluster/scratch/askrika/HCE	/cluster/scratch/askrika/HC/one_hot/RBDHC_split_mlp_classification/20250423_132519_27114/lightning_logs/best_model.ckpt	True	1	rtx_4090	100g	1	24	False
#function	data_type	split	head	task	head_config	ray_tuning	output_dir	model_path	evaluate	gpus	gres	mem-per-cpu	nodes	run_time	experimenting
#one_hot	RBDHC	split	mlp	classification	RBDDS_split_classification_head_config.json	False	/cluster/scratch/askrika/DSE	/cluster/scratch/askrika/DS/one_hot/RBDDS_split_mlp_classification/20250423_132523_17128/lightning_logs/best_model.ckpt	True	1	rtx_4090	100g	1	24	False
#one_hot	RBDDS	split	mlp	classification	RBDHC_split_classification_head_config.json	False	/cluster/scratch/askrika/HCE	/cluster/scratch/askrika/HC/one_hot/RBDHC_split_mlp_classification/20250423_132519_27114/lightning_logs/best_model.ckpt	True	1	rtx_4090	100g	1	24	False
#"$function" "$head_config" "$ray_tuning" "$data_type" "$split" "$output_dir" "$experiment_dir" "$experiment_name" "$gpus" "$nodes" "$experimenting" "$model_path" "$evaluate"
>>>>>>> Stashed changes
