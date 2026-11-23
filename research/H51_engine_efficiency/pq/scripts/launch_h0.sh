#!/usr/bin/env bash
# H0: N single-thread engine_sens workers in ONE process group (nice 10) under fleet_guard (CPU-only).
# usage: launch_h0.sh <tag> <s> <R> <workers> [extra engine_sens args...]
set -euo pipefail
tag=$1; s=$2; R=$3; n=$4; shift 4
cd /mnt/xbabe3-external/jsr/h51-pq
mkdir -p /dev/shm/h51-pq/logs "results/engine_sens/$tag"
run="/dev/shm/h51-pq/logs/$tag.h0.run.sh"
pids="/dev/shm/h51-pq/logs/$tag.h0.pids"
rm -f "$pids"
{
  echo "cd /mnt/xbabe3-external/jsr/h51-pq"
  echo "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
  for i in $(seq 0 $((n - 1))); do
    printf 'nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag %q --s %q --R %q --shard %d/%d' "$tag" "$s" "$R" "$i" "$n"
    for x in "$@"; do printf ' %q' "$x"; done
    printf ' > /dev/shm/h51-pq/logs/%s.h0.%d.log 2>&1 &\n' "$tag" "$i"
    echo "echo \$! >> $pids"
  done
  echo "wait"
} > "$run"
setsid bash "$run" < /dev/null > /dev/null 2>&1 &
leader=$!
sleep 1
pgid=$(ps -o pgid= -p "$leader" | tr -d ' ')
echo "$pgid" > "/dev/shm/h51-pq/logs/$tag.h0.pgid"
PYTHONPATH=/mnt/xbabe3-external/jsr/h51-pq/src/python/ultra nohup nice -n 5 /home/ubuntu/jsr/.venv/bin/python -m ultra.fleet_guard \
  --pgid "$pgid" --gpu -1 --cpu-target 0.80 --log "/dev/shm/h51-pq/logs/$tag.h0.fleet_guard.jsonl" \
  > "/dev/shm/h51-pq/logs/$tag.h0.fleet_guard.out" 2>&1 &
echo "$!" > "/dev/shm/h51-pq/logs/$tag.h0.guard_pid"
sleep 1
echo "tag=$tag pgid=$pgid guard=$(cat /dev/shm/h51-pq/logs/$tag.h0.guard_pid) workers=$(tr '\n' ' ' < $pids)"
