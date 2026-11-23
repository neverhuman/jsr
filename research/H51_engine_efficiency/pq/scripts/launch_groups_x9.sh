#!/usr/bin/env bash
# x9: joint (group) PQ runs, one single-thread worker per group, ONE process group (nice 19),
# guarded by ultra.fleet_guard (CPU 0.85). usage: launch_groups_x9.sh <tag> <s> <R> [extra args]
set -euo pipefail
tag=$1; s=$2; R=$3; shift 3
H=$HOME/h51-pq
cd "$H"
mkdir -p logs "results/engine_sens/$tag"
rm -f "logs/$tag.pids"
run="logs/$tag.run.sh"
{
  echo "cd $H"
  echo "export H51_ROOT=$H/data H51_SRC=$H/src/python/ultra H51_BPE=$H/data/bpe.model H51_ENGINE=$H/bin/ultra-engine-eval-musl H51_SCORE=$H/bin/ultra-score H51_OUT=$H/results/engine_sens OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
  i=0
  if [ -n "${REGEXES:-}" ]; then read -r -a RXS <<< "$REGEXES"; else RXS=('ff[12]\.w1$' 'ff[12]\.w2$' 'att\.qkv$' 'att\.out$' 'conv\.pw1$' 'conv\.pw2$' '^(encoder\.stem\.out|joiner\..*)$' '^encoder\.'); fi
  for re in "${RXS[@]}"; do
    printf 'nice -n 19 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag %q --s %q --R %q --layers %q --rows-suffix g%d --no-distortion' "$tag" "$s" "$R" "group:$re" "$i"
    for x in "$@"; do printf ' %q' "$x"; done
    printf ' > logs/%s.g%d.log 2>&1 &\n' "$tag" "$i"
    echo "echo \$! >> logs/$tag.pids"
    i=$((i + 1))
  done
  echo "wait"
} > "$run"
setsid bash "$run" < /dev/null > /dev/null 2>&1 &
leader=$!
sleep 1
pgid=$(ps -o pgid= -p "$leader" | tr -d ' ')
echo "$pgid" > "logs/$tag.pgid"
PYTHONPATH=$H/src/python/ultra nohup nice -n 5 /home/ubuntu/jsr/.venv/bin/python -m ultra.fleet_guard --pgid "$pgid" --gpu -1 --cpu-target 0.85 \
  --log "logs/$tag.fleet_guard.jsonl" > "logs/$tag.fleet_guard.out" 2>&1 &
echo "$!" > "logs/$tag.guard_pid"
sleep 1
echo "tag=$tag pgid=$pgid guard_pid=$(cat logs/$tag.guard_pid) workers=$(cat logs/$tag.pids 2>/dev/null | tr '\n' ' ')"
