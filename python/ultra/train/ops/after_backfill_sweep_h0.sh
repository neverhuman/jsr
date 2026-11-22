#!/usr/bin/env bash
# H0: once the local D2 backfill queues (pgids given) have exited, run the DEV decode sweep on the
# staged D2 avg(55k, 60k, 65k) (avg3-65000), from internal NVMe, on the same core slots.
#   after_backfill_sweep_h0.sh <pgid...>
set -uo pipefail
for pg in "$@"; do while kill -0 -- "-$pg" 2>/dev/null; do sleep 60; done; done
L=/home/ubuntu/jsr/.agent/worktrees/trainer
C=/home/ubuntu/jsr-local-cache
R=$C/runs/train/D2_S_960h
export JSR_DATA=$C PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu:$L/python/ultra
bash "$L/python/ultra/train/ops/decode_sweep.sh" "$L/.venv/bin/python" "$L/python/ultra" "$R" "$R/avg3-65000.pt" \
  "$L/docs/ultra/data/dev_selection_panel.json" "$R/decode_sweep_avg65" 0/1 \
  16-19 20-23 24-27 28-31 32-35 36-39 40-43 80-83 84-87 88-91 92-95 96-99 100-103
