#!/usr/bin/env bash
# Fixed ranks for a fleet with different GPU counts per host (e.g. 2+2+1+1).
# On each host set the same MASTER_ADDR (private RDZV host), MASTER_PORT, and WORLD_SIZE.
# Pass the global offset for this host and disjoint CPU masks in local GPU order.
#   launch_ddp_ranks.sh RUN_DIR RANK_OFFSET GPU_LIST CPU_MASK[:CPU_MASK...] -- TRAINER_ARGS
# Keep this supervisor running for the life of the local group. Use the trainer's
# synchronized utilization throttle on GPUs. A worker failure stops its local peers.
set -euo pipefail

if [[ $# -lt 6 || $5 != -- ]]; then
    printf 'Usage: %s RUN_DIR RANK_OFFSET GPU_LIST CPU_MASK[:CPU_MASK...] -- TRAINER_ARGS\n' "$0" >&2
    exit 2
fi
run_dir=$1 offset=$2 visible=$3 affinity=$4
shift 5
: "${MASTER_ADDR:?Set the rendezvous address on the fleet network}"
: "${MASTER_PORT:?Set the rendezvous TCP port}"
: "${WORLD_SIZE:?Set the global worker count}"
if [[ ! $offset =~ ^(0|[1-9][0-9]*)$ || ! $WORLD_SIZE =~ ^[1-9][0-9]*$ ||
      ! $visible =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ || ! $affinity =~ ^[0-9,-]+(:[0-9,-]+)*$ ]]; then
    printf 'Invalid rank geometry or affinity\n' >&2
    exit 2
fi
IFS=, read -ra gpus <<< "$visible"
IFS=: read -ra cpus <<< "$affinity"
n=${#gpus[@]}
if ((n != ${#cpus[@]} || offset + n > WORLD_SIZE || WORLD_SIZE < 2)); then
    printf 'Rank range must fit WORLD_SIZE and every GPU needs one CPU mask\n' >&2
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
if [[ ! -f $repo/python/ultra/train/dist.py ]]; then
    printf 'This launcher requires the distributed trainer (JSR PR #10)\n' >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
exec 9>"$run_dir/ddp-launch.lock"
flock -n 9 || { printf 'A local group already owns %s\n' "$run_dir" >&2; exit 1; }
lease_dir=${JSR_DATA:-$repo/target}/gpu-leases
mkdir -p "$lease_dir"
declare -A used=()
for gpu in "${gpus[@]}"; do
    if [[ -v used[$gpu] ]]; then
        printf 'GPU %s appears more than once\n' "$gpu" >&2
        exit 2
    fi
    used[$gpu]=1
    exec {lease_fd}>"$lease_dir/gpu-$gpu.lock"
    if ! flock -n "$lease_fd"; then
        printf 'GPU %s is already leased by another trainer\n' "$gpu" >&2
        exit 1
    fi
    tenants=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n $tenants ]]; then
        printf 'GPU %s has active compute processes\n' "$gpu" >&2
        exit 1
    fi
done

py=${PY:-$repo/.venv/bin/python}
cd "$repo/python/ultra"
export MASTER_ADDR MASTER_PORT WORLD_SIZE LOCAL_WORLD_SIZE=$n
export CUDA_VISIBLE_DEVICES=$visible CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

declare -A live=()
cleanup() {
    local pid
    for pid in "${!live[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${!live[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

for ((local_rank=0; local_rank<n; local_rank++)); do
    rank=$((offset + local_rank))
    log="$run_dir/rank-$rank.stdout.log"
    RANK=$rank LOCAL_RANK=$local_rank taskset -c "${cpus[local_rank]}" nice -n 10 \
        "$py" -m train.trainer --run-dir "$run_dir" "$@" >> "$log" 2>&1 9>&- &
    pid=$!
    live[$pid]=$rank
    printf 'host=%s rank=%s gpu=%s pid=%s log=%s\n' "$(hostname)" "$rank" "${gpus[local_rank]}" "$pid" "$log"
done

status=0
while ((${#live[@]})); do
    wait -n -p finished_pid "${!live[@]}" || status=$?
    if [[ -n ${finished_pid:-} ]]; then
        unset 'live[$finished_pid]'
    fi
    if ((status != 0)); then
        exit "$status"  # EXIT terminates only the remaining child processes.
    fi
done
