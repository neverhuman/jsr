#!/usr/bin/env bash
# Run train.teacher_cache over every (split, shard, speed) task on pinned single cores, sharing a
# task list through mkdir locks (several runners may share it). Speed 1.0 first, then 0.9 / 1.1.
#   teacher_cache_runner.sh <jsr_data> <out_dir> <speeds e.g. "1.0 0.9 1.1"> <core...>
# SPLITS (env, default all three train splits) limits the splits, so hosts can take disjoint slices.
set -uo pipefail
DATA=$1 OUT=$2 SPEEDS=$3
shift 3
W=$HOME/jsr/.agent/worktrees/trainer/python/ultra
PY=$DATA/venvs/train/bin/python
M=$DATA/models/zipformer-en-2023-06-26
export JSR_DATA=$DATA
mkdir -p "$OUT/locks" "$OUT/logs"
tasks=()
for sp in $SPEEDS; do
  for split in ${SPLITS:-train-clean-100 train-clean-360 train-other-500}; do
    n=$(cd "$W" && "$PY" -c "from train.teacher_cache import shard_files; print(len(shard_files('$split')))" 2>/dev/null)
    for ((i = 0; i < ${n:-0}; i++)); do tasks+=("$split $i $sp"); done
  done
done
run_core() {
  local core=$1 t split shard sp name
  for t in "${tasks[@]}"; do
    read -r split shard sp <<< "$t"
    name=$(printf '%s_%04d_sp%s' "$split" "$shard" "$sp")
    [ -f "$OUT/$name.npz" ] && continue
    mkdir "$OUT/locks/$name" 2>/dev/null || continue
    (cd "$W" && taskset -c "$core" nice -n 10 "$PY" -m train.teacher_cache --model-dir "$M" \
      --split "$split" --shard "$shard" --speed "$sp" --out "$OUT" > "$OUT/logs/$name.log" 2>&1) \
      || rmdir "$OUT/locks/$name"
  done
}
for c in "$@"; do run_core "$c" & done
wait
