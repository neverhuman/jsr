#!/usr/bin/env bash
# Repeat backfill_local.sh every 20 minutes while the run's trainer is alive, so each new 5k
# snapshot gets evaluated where it is written. Same arguments as backfill_local.sh.
set -uo pipefail
R=$5
while :; do
  bash "$2/train/ops/backfill_local.sh" "$@"
  rm -f "$R/backfill/DONE"
  P=$(cat "$R/PID" 2>/dev/null)
  ps -p "${P:-0}" -o args= 2>/dev/null | grep -q train.trainer || break
  sleep 1200
done
