#!/usr/bin/env bash
# Pinned CPU evaluation queue. Job file lines: <tag> <run_dir> <checkpoint> <evaluate args...>
#   eval_queue.sh <python> <ultra_dir> <jobfile> <out_dir> <slot cores...>
# Jobs are spread round-robin over the core slots (4 threads each); each slot runs its jobs one
# after another; a job whose <out>/<tag>.json is already non-empty is skipped (restartable).
set -uo pipefail
PY=$1 W=$2 JOBS=$3 OUT=$4
shift 4
slots=("$@")
mkdir -p "$OUT"
cd "$W" || exit 1
mapfile -t lines < <(grep -v '^\s*$' "$JOBS")
run_slot() {
  local cores=$1; shift
  local line tag run ck rest
  for line in "$@"; do
    read -r tag run ck rest <<< "$line"
    [ -s "$OUT/$tag.json" ] && continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="" taskset -c "$cores" nice -n 10 "$PY" -m train.evaluate --run-dir "$run" \
      --checkpoint "$ck" --threads 4 --dump "$OUT/$tag.jsonl" $rest 2>"$OUT/$tag.err" \
      | grep '^{' > "$OUT/$tag.json"
  done
}
ns=${#slots[@]}
for s in $(seq 0 $((ns - 1))); do
  mine=()
  for k in "${!lines[@]}"; do [ $((k % ns)) -eq "$s" ] && mine+=("${lines[$k]}"); done
  run_slot "${slots[$s]}" "${mine[@]}" &
done
wait
echo DONE > "$OUT/DONE"
