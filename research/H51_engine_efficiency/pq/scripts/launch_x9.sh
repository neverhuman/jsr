#!/usr/bin/env bash
# x9: launch an engine_sens sweep as N single-thread workers in ONE new process group (nice 19),
# guarded by ultra.fleet_guard (CPU target 0.85, CPU-only). Records PGID / PIDs.
# usage: launch_x9.sh <tag> <s> <R> <workers> [extra engine_sens args...]
set -euo pipefail
tag=$1; s=$2; R=$3; n=$4; shift 4
H=$HOME/h51-pq
cd "$H"
export H51_ROOT=$H/data H51_SRC=$H/src/python/ultra H51_BPE=$H/data/bpe.model \
  H51_ENGINE=$H/bin/ultra-engine-eval-musl H51_SCORE=$H/bin/ultra-score H51_OUT=$H/results/engine_sens \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p logs "results/engine_sens/$tag"
extra=""
if [ $# -gt 0 ]; then extra=$(printf '%q ' "$@"); fi
setsid bash -c "
  for i in \$(seq 0 $((n - 1))); do
    nice -n 19 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag $tag --s $s --R $R --shard \$i/$n $extra > logs/$tag.\$i.log 2>&1 &
    echo \$! >> logs/$tag.pids
  done
  wait
" < /dev/null > /dev/null 2>&1 &
leader=$!
sleep 1
pgid=$(ps -o pgid= -p "$leader" | tr -d ' ')
echo "$pgid" > "logs/$tag.pgid"
PYTHONPATH=$H/src/python/ultra nohup nice -n 5 /home/ubuntu/jsr/.venv/bin/python -m ultra.fleet_guard --pgid "$pgid" --gpu -1 --cpu-target ${CPU_TARGET:-0.85} \
  --log "logs/$tag.fleet_guard.jsonl" > "logs/$tag.fleet_guard.out" 2>&1 &
echo "$!" > "logs/$tag.guard_pid"
echo "tag=$tag pgid=$pgid guard_pid=$(cat logs/$tag.guard_pid) workers=$(cat logs/$tag.pids | tr '\n' ' ')"
