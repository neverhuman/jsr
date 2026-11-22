#!/usr/bin/env bash
# H0 D2 backfill from the internal-NVMe stage (stage_h0_cache.sh), never the USB data drive:
# every staged snapshot and its last-3 average, full dev with beam 4, latest step first.
#   backfill_h0_local.sh <out_tag> <slot cores...>
# Several invocations may share the job list (each job is skipped once its output exists), so
# more slots can be added later by starting another instance with other cores.
set -uo pipefail
TAG=$1; shift
L=/home/ubuntu/jsr/.agent/worktrees/trainer
C=/home/ubuntu/jsr-local-cache
R=$C/runs/train/D2_S_960h
O=$R/backfill
export JSR_DATA=$C PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu:$L/python/ultra
mkdir -p "$O"
AVG_CREATE=1 nice -n 10 bash "$L/python/ultra/train/ops/make_backfill_jobs.sh" "$L/.venv/bin/python" \
  "$L/python/ultra" "$R" D2 | tac > "$O/jobs_$TAG.txt"
bash "$L/python/ultra/train/ops/eval_queue.sh" "$L/.venv/bin/python" "$L/python/ultra" "$O/jobs_$TAG.txt" "$O" "$@"
