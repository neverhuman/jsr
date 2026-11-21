#!/usr/bin/env bash
# Launch a trainer run in its own session (setsid), pinned and niced, and put it under the fleet
# guard straight away (owner rule: no xbabe host above 90% mean CPU/GPU utilisation).
#
#   launch_guarded.sh <run_dir> <cpu_list> <gpu_index> -- <train.trainer arguments without --run-dir>
#
# Environment: JSR_DATA (data root on this host), PY (python, default <repo>/.venv/bin/python),
# GPU_TARGET (0.74), CPU_TARGET (0.85), MIN_DUTY (0.0).
# Writes <run_dir>/{PID,PGID,GUARD_PID,stdout.log,fleet_guard.jsonl,guard.log}.
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$4" != "--" ]; then
  echo "usage: $0 <run_dir> <cpu_list> <gpu_index> -- <trainer args>" >&2
  exit 2
fi
run_dir=$1 cpus=$2 gpu=$3
shift 4
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
py="${PY:-$repo/.venv/bin/python}"
mkdir -p "$run_dir"
cd "$repo/python/ultra"

setsid nohup taskset -c "$cpus" nice -n 10 "$py" -m train.trainer --run-dir "$run_dir" "$@" \
  >>"$run_dir/stdout.log" 2>&1 < /dev/null &
pid=$!
sleep 2
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
if [ -z "$pgid" ]; then
  echo "trainer exited immediately; see $run_dir/stdout.log" >&2
  exit 1
fi
echo "$pid" > "$run_dir/PID"
echo "$pgid" > "$run_dir/PGID"

setsid nohup nice -n 5 "$py" -m ultra.fleet_guard --pgid "$pgid" --gpu "$gpu" \
  --gpu-target "${GPU_TARGET:-0.74}" --cpu-target "${CPU_TARGET:-0.85}" --min-duty "${MIN_DUTY:-0.0}" \
  --log "$run_dir/fleet_guard.jsonl" >>"$run_dir/guard.log" 2>&1 < /dev/null &
echo $! > "$run_dir/GUARD_PID"
echo "host=$(hostname) pid=$pid pgid=$pgid guard_pid=$(cat "$run_dir/GUARD_PID") run_dir=$run_dir"
