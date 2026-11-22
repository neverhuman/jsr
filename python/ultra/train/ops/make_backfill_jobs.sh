#!/usr/bin/env bash
# Build eval_queue.sh jobs for every 5k snapshot of one run: full dev-clean and dev-other with
# --beam 4 (greedy and beam in one pass) for model-<step>.pt and for the last-3 average
# (step-10k, step-5k, step), creating the averages first (avg3-<step>.pt next to the snapshots).
#   make_backfill_jobs.sh <python> <ultra_dir> <run_dir> <label> > jobs.txt
set -uo pipefail
PY=$1 W=$2 R=$3 LABEL=$4
cd "$W" || exit 1
mapfile -t steps < <(ls "$R" | sed -n 's/^model-0*\([0-9]\+\)\.pt$/\1/p' | sort -n)
for s in "${steps[@]}"; do
  f=$(printf '%s/model-%08d' "$R" "$s")
  [ -f "$f.sha256" ] || continue
  for split in dev-clean dev-other; do
    echo "${LABEL}_${s}_snap_${split} $R $f.pt --split $split --limit-all --beam 4"
  done
  a=$(printf '%s/model-%08d' "$R" $((s - 5000))); b=$(printf '%s/model-%08d' "$R" $((s - 10000)))
  if [ -f "$a.sha256" ] && [ -f "$b.sha256" ]; then
    avg=$R/avg3-$s.pt
    if [ ! -f "${avg%.pt}.sha256" ]; then
      nice -n 10 "$PY" -m train.average --out "$avg" "$b.pt" "$a.pt" "$f.pt" > /dev/null 2>&1 || continue
    fi
    for split in dev-clean dev-other; do
      echo "${LABEL}_${s}_avg3_${split} $R $avg --split $split --limit-all --beam 4"
    done
  fi
done
