#!/bin/bash
#SBATCH --job-name=ray_workload    # create a short name for your job
#SBATCH --nodes=1          # node count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2

export DATA_DIR='/cluster/home/estamkopoulo/plmfit_workspace/plmfit/plmfit'

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 5

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block --verbose &
    sleep 5

    # Wait for the worker node to be up and ready
    while true; do
        if ray status | grep -q "1 node"; then
            echo "Worker $i is up and ready."
            break
        else
            echo "Waiting for worker $i to initialize..."
            sleep 10
        fi
    done
done

python3 -u plmfit.py --function $1 --ft_method $2 --head_config $3 --ray_tuning $4 \
        --data_type $5 --split $6 --plm $7 --layer $8 --reduction $9 \
        --output_dir ${10} --experiment_dir ${11} --experiment_name ${12}
