#!/usr/bin/env bash
# xbabe0 CPU evaluation jobs (cores 0-47, taskset, nice 10). Each job: <cores> <tag> <run_dir>
# <checkpoint> <evaluate args...>, written one per line to a job file:
#   xbabe0_eval_jobs.sh <jobfile> <out_dir>
# Every job runs `train.evaluate --threads "${THREADS:-8}" --dump <out>/<tag>.jsonl` and writes <out>/<tag>.json.
set -uo pipefail
JOBS=$1 OUT=$2
export JSR_DATA=$HOME/jsr-data
W=$HOME/jsr/.agent/worktrees/trainer/python/ultra
PY=$HOME/jsr-data/venvs/train/bin/python
mkdir -p "$OUT"
cd "$W" || exit 1
while read -r cores tag run ck rest; do
  [ -z "$cores" ] && continue
  # shellcheck disable=SC2086
  ( CUDA_VISIBLE_DEVICES="" taskset -c "$cores" nice -n 10 "$PY" -m train.evaluate --run-dir "$run" \
      --checkpoint "$ck" --threads "${THREADS:-8}" --dump "$OUT/$tag.jsonl" $rest 2>"$OUT/$tag.err" \
      | grep '^{' > "$OUT/$tag.json" ) &
done < "$JOBS"
wait
echo DONE > "$OUT/DONE"
