#!/usr/bin/env bash
# H46: Rust engine (`ultra-engine-eval --npz`) over full dev, cut into per-core npz shards
# (train/ops/h46_npz_shards.py), one single-threaded engine process per claimed core taking jobs
# from a shared queue (atomic mkdir lock per job), so a rerun resumes and more cores can join.
#
#   BIN=<dir with ultra-engine-eval> WEIGHTS=<prepared int8 dir> SHARDS=<dir of s00..sNN> \
#   OUT=<dir> CORES="14 15 ..." h46_engine_fulldev.sh <runs.tsv>
#
# runs.tsv columns: name, beam, lm pack ("-" for none), lm_weight, lodr_weight, blank_penalty.
# Each job writes <OUT>/<name>/<split>.<shard>/engine_eval.json; worker pids go to <OUT>/pids.
set -euo pipefail
RUNS=${1:?runs.tsv}
: "${BIN:?}" "${WEIGHTS:?}" "${SHARDS:?}" "${OUT:?}" "${CORES:?}"
read -r -a cores <<< "$CORES"
mkdir -p "$OUT/locks"
jobs=()
while IFS=$'\t' read -r name beam lm w lw pen; do
  [ -n "$name" ] || continue
  for d in "$SHARDS"/s*; do
    for npz in "$d"/dev-*.npz; do
      jobs+=("$name|$beam|$lm|$w|$lw|$pen|$npz")
    done
  done
done < "$RUNS"
worker() {
  local core=$1 j name beam lm w lw pen npz split shard out lmargs
  for j in "${jobs[@]}"; do
    IFS='|' read -r name beam lm w lw pen npz <<< "$j"
    split=$(basename "$npz" .npz); shard=$(basename "$(dirname "$npz")")
    out="$OUT/$name/$split.$shard"
    [ -s "$out/engine_eval.json" ] && continue
    mkdir "$OUT/locks/$name.$split.$shard" 2>/dev/null || continue
    lmargs=()
    if [ "$lm" != "-" ]; then
      lmargs=(--lm "$lm" --lm-weight "$w" --lodr-weight "$lw" --blank-penalty "$pen")
    fi
    mkdir -p "$OUT/$name"
    taskset -c "$core" nice -n 10 "$BIN/ultra-engine-eval" --weights "$WEIGHTS" --precision int8 \
      --npz "$npz" --limit 1000000 --streams 16 --beam "$beam" --beam-max-symbols 1 \
      "${lmargs[@]}" --out "$out" > "$OUT/$name/$split.$shard.log" 2>&1 \
      || echo "FAILED $name $split $shard" >> "$OUT/FAILED"
  done
}
: > "$OUT/pids"
for c in "${cores[@]}"; do worker "$c" & echo $! >> "$OUT/pids"; done
wait
echo "engine full-dev runs done: $(date -u +%H:%M:%SZ)"
