#!/usr/bin/env bash
# H50 energy / cap sidecar, one per host for the life of a run (start it in the run's process group,
# stop it by its PID):  gpu_sampler.sh RUN_DIR GPU_LIST   e.g. gpu_sampler.sh $R 0,1
# Writes RUN_DIR/gpu_<host>.csv (1 Hz: timestamp,index,uuid,power.draw,utilization.gpu,memory.used,
# clocks.sm,temperature.gpu) and RUN_DIR/cpu_<host>.log (vmstat every 5 s, UTC stamped).
set -euo pipefail
run_dir=$1 gpus=$2 h=$(hostname)
nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits -i "$gpus" -lms 1000 >> "$run_dir/gpu_$h.csv" &
p1=$!
TZ=UTC vmstat -n -t 5 >> "$run_dir/cpu_$h.log" &
p2=$!
echo "$p1 $p2" > "$run_dir/sampler_$h.pids"
trap 'kill $p1 $p2 2>/dev/null' EXIT TERM INT
wait
