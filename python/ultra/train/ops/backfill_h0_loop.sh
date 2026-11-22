#!/usr/bin/env bash
# H0: D2 backfill (every 5k snapshot + last-3 average, full dev, beam 4) on cores 16-43 and
# 80-103 (never the 44-47 / 108-111 timing fence), MKL CPU torch, repeated as snapshots land.
set -uo pipefail
L=/home/ubuntu/jsr/.agent/worktrees/trainer
R=/mnt/xbabe3-external/jsr/runs/train/D2_S_960h
O=$R/backfill
export JSR_DATA=/mnt/xbabe3-external/jsr PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu:$L/python/ultra
mkdir -p "$O"
while :; do
  AVG_CREATE=1 taskset -c 43 nice -n 10 bash "$L/python/ultra/train/ops/make_backfill_jobs.sh" \
    "$L/.venv/bin/python" "$L/python/ultra" "$R" D2 > /dev/null &
  AVG_CREATE=0 bash "$L/python/ultra/train/ops/make_backfill_jobs.sh" "$L/.venv/bin/python" "$L/python/ultra" "$R" D2 > "$O/jobs.txt"
  bash "$L/python/ultra/train/ops/eval_queue.sh" "$L/.venv/bin/python" "$L/python/ultra" "$O/jobs.txt" "$O" \
    16-19 20-23 24-27 28-31 32-35 36-39 40-43 80-83 84-87 88-91 92-95 96-99 100-103
  wait
  ps -p "$(cat "$R/PID")" -o args= | grep -q train.trainer || break
  sleep 300
done
