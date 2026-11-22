#!/usr/bin/env bash
# Backfill on the host that holds a run's snapshots (local disk): every 5k snapshot and its
# last-3 average, full dev with --beam 4, latest step first, using an MKL CPU torch.
#   backfill_local.sh <python> <ultra_dir> <jsr_data> <cpu_torch_dir|-> <run_dir> <label> <tag> <slot cores...>
# Several instances (different <tag>, different cores) share one job list via eval_queue locks.
set -uo pipefail
PY=$1 W=$2 DATA=$3 CPUT=$4 R=$5 LABEL=$6 TAG=$7
shift 7
export JSR_DATA=$DATA
[ "$CPUT" != - ] && export PYTHONPATH=$CPUT:$W
O=$R/backfill
mkdir -p "$O"
AVG_CREATE=1 nice -n 10 bash "$W/train/ops/make_backfill_jobs.sh" "$PY" "$W" "$R" "$LABEL" \
  | awk '{a[NR]=$0} END {for (i = NR; i > 0; i--) print a[i]}' > "$O/jobs_$TAG.txt"
bash "$W/train/ops/eval_queue.sh" "$PY" "$W" "$O/jobs_$TAG.txt" "$O" "$@"
