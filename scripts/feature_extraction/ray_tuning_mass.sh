#!/bin/bash
#SBATCH --job-name=ray_workload    # create a short name for your job
#SBATCH --nodes=2             # node count
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)

module load eth_proxy
module load gcc/8.2.0  python_gpu/3.11.2

export DATA_DIR='/cluster/home/estamkopoulo/plmfit_workspace/plmfit/plmfit'

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(hostname)
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
echo "cpus per task $SLURM_CPUS_PER_TASK"
echo "gpus $SLURM_GPUS"
echo "gpus per task $SLURM_GPUS_PER_NODE"
echo "nodes $SLURM_NNODES"
echo "Starting HEAD at $head_node"

echo "head node $head_node"
ray start --head --node-ip-address=$head_node_ip \
    --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &

    sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1))


for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $ip_head"
    ray start --address $ip_head \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &
    sleep 5
done
echo 'ray OK'

python3 plmfit.py --function $1 --ft_method $2 --head_config $3 --ray_tuning $4 \
        --data_type $5 --plm $6 --layer $7 --reduction $8 \
        --output_dir ${9} --experiment_dir ${10} --experiment_name ${11}
