#!/usr/bin/env bash
# H46: encoder pass of the acoustic model over full dev (train.lm_decode encode: EMA weights, batch 8
# in length order, the trained chunk), one single-threaded worker per pinned core, then merge.
#   CODE=<python/ultra of the pinned commit> CKPT=<slim .pt> OUT=<cache dir> CORES="16 ... 29" h46_encode.sh
set -euo pipefail
: "${CODE:?}" "${CKPT:?}" "${OUT:?}" "${CORES:?}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
LOGS=${LOGS:-$OUT/logs}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
read -r -a cores <<< "$CORES"
n=${#cores[@]}
mkdir -p "$OUT" "$LOGS"
: > "$LOGS/encode.pids"
cd "$CODE"
for i in "${!cores[@]}"; do
  taskset -c "${cores[$i]}" nice -n 10 "$PY" -m train.lm_decode encode --checkpoint "$CKPT" \
    --out "$OUT" --shard "$i/$n" > "$LOGS/encode.$i.log" 2>&1 &
  echo $! >> "$LOGS/encode.pids"
done
wait
taskset -c "${cores[0]}" "$PY" -m train.lm_decode merge --out "$OUT" > "$LOGS/merge.log" 2>&1
echo "encode + merge done: $(date -u +%H:%M:%SZ)"
