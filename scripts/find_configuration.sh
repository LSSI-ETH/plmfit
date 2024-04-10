#!/bin/bash

# Requirements
required_gpu_mem_per_gpu=24 # Minimum memory required per GPU in GiB
required_total_gpu_mem=100 # Total GPU memory required in GiB
required_cpu_mem=20000 # Required CPU memory in MiB

# Check if arguments are provided and override defaults
if [ $# -eq 3 ]; then
    required_gpu_mem_per_gpu=$1
    required_total_gpu_mem=$2
    required_cpu_mem=$3
fi

# Define the GPUs and their memory from the table
declare -A gpu_mem_map
gpu_mem_map["gtx_1080_ti"]=11
gpu_mem_map["rtx_2080_ti"]=11
gpu_mem_map["rtx_3090"]=24
gpu_mem_map["rtx_4090"]=24
gpu_mem_map["titan_rtx"]=24
gpu_mem_map["quadro_rtx_6000"]=24
gpu_mem_map["v100"]=32
gpu_mem_map["a100-pcie-40gb"]=40
gpu_mem_map["a100_80gb"]=80

select_configuration() {
  # Get node info and check each node
  while read -r node_info; do
    node_name=$(echo "$node_info" | grep -oP "NodeName=\K\S+")
    real_mem=$(echo "$node_info" | grep -oP "RealMemory=\K[0-9]+")
    gres=$(echo "$node_info" | grep -oP "Gres=\K[^ ]+")
    state=$(echo "$node_info" | grep -oP "State=\K\w+")

    if [ "$real_mem" -ge "$required_cpu_mem" ] && [[ "$state" == "IDLE" ]]; then
      local gpu_mem_needed="$required_total_gpu_mem"
      local selected_gpus=()

      for gpu_type in "${!gpu_mem_map[@]}"; do
        if [[ $gres =~ $gpu_type:([0-9]+) ]]; then
          local num_gpus=${BASH_REMATCH[1]}
          local gpu_mem=${gpu_mem_map[$gpu_type]}

          # Only consider GPUs that meet the minimum memory requirement per GPU
          if [ "$gpu_mem" -ge "$required_gpu_mem_per_gpu" ]; then
            # Calculate how many of this type of GPU are needed to satisfy the memory requirement
            while [ "$num_gpus" -gt 0 ] && [ "$gpu_mem_needed" -gt 0 ]; do
              selected_gpus+=("$gpu_type")
              gpu_mem_needed=$((gpu_mem_needed - gpu_mem))
              num_gpus=$((num_gpus - 1))
            done
          fi
        fi
      done

      if [ "$gpu_mem_needed" -le 0 ]; then
        selected_gpus_number="${#selected_gpus[@]}"
        echo "$node_name ${selected_gpus[0]} $selected_gpus_number"
        return 0 # Exit as soon as the first node meeting requirements is found
      fi
    fi
  done < <(scontrol -o show node)
  return 1
}

select_configuration
