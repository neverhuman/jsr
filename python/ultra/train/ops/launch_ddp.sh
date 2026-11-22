#!/usr/bin/env bash
# Launch a data-parallel trainer with torchrun (one node). GPU throttling is the trainer's own
# synchronised step-boundary sleep (--set ddp_util_target=...), never fleet_guard's SIGSTOP.
#   launch_ddp.sh <run_dir> <cpu_list> <gpu_list e.g. 0,1> -- <train.trainer arguments without --run-dir>
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT in the environment on every node.
# Writes <run_dir>/{PID,PGID,stdout.log}.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$4" != "--" ]; then
  echo "usage: $0 <run_dir> <cpu_list> <gpu_list> -- <trainer args>" >&2
  exit 2
fi
run_dir=$1 cpus=$2 gpus=$3
shift 4
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
py="${PY:-$repo/.venv/bin/python}"
n=$(awk -F, '{print NF}' <<< "$gpus")
mkdir -p "$run_dir"
cd "$repo/python/ultra"
export CUDA_VISIBLE_DEVICES=$gpus NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4
setsid nohup taskset -c "$cpus" nice -n 10 "$py" -m torch.distributed.run \
  --nnodes "${NNODES:-1}" --node_rank "${NODE_RANK:-0}" --nproc_per_node "$n" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29511}" \
  -m train.trainer --run-dir "$run_dir" "$@" >> "$run_dir/stdout.log" 2>&1 < /dev/null &
pid=$!
sleep 2
echo "$pid" > "$run_dir/PID"
ps -o pgid= -p "$pid" | tr -d ' ' > "$run_dir/PGID"
echo "host=$(hostname) torchrun_pid=$pid nproc=$n gpus=$gpus run_dir=$run_dir"
