#!/usr/bin/env bash
# Start one H41 arm (h41_replay.sh) on a card as soon as it is ready, then arm its end-of-run decodes.
#   h41_when_ready.sh <jsr_data> <cpu_list> <gpu> <arm> [key=value ...]
# Ready means, in order (each condition optional, set by environment):
#   AFTER_RUN=<run dir>      that run's trainer has exited (its guard is stopped)
#   AFTER_DECODE=<run name>  its burst decodes wrote <name>.DONE or .FAILED in $BURST_DIR (default <jsr_data>/evals/burst/h41)
#   TEACHER=<cache dir>      the R0 teacher cache is complete (>= TEACHER_SHARDS .npz, default 126) and SHA256SUMS verifies
#   and the card's GPU lease is free. PY / GPU_TARGET / INIT_FROM / RUN_NAME pass through to h41_replay.sh.
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3 ARM=$4
OPS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${AFTER_RUN:-}" ]; then
  until ! kill -0 "$(cat "$AFTER_RUN/PID")" 2>/dev/null; do sleep 5; done
  sleep 3; kill "$(cat "$AFTER_RUN/GUARD_PID")" 2>/dev/null
  echo "$(date -u +%T) $(basename "$AFTER_RUN") has ended"
fi
if [ -n "${AFTER_DECODE:-}" ]; then
  BD=${BURST_DIR:-$DATA/evals/burst/h41}
  until [ -s "$BD/$AFTER_DECODE.DONE" ] || [ -s "$BD/$AFTER_DECODE.FAILED" ]; do sleep 10; done
  echo "$(date -u +%T) decodes of $AFTER_DECODE finished"
fi
if [ -n "${TEACHER:-}" ]; then
  until [ "$(ls "$TEACHER"/*.npz 2>/dev/null | wc -l)" -ge "${TEACHER_SHARDS:-126}" ] && [ -s "$TEACHER/SHA256SUMS" ] \
        && (cd "$TEACHER" && sha256sum --quiet -c SHA256SUMS); do sleep 20; done
  echo "$(date -u +%T) teacher cache verified"
fi
mkdir -p "$DATA/gpu-leases"
until flock --nonblock "$DATA/gpu-leases/gpu-$GPU.lock" true; do sleep 5; done
GPU_TARGET=${GPU_TARGET:-0.87} bash "$OPS/h41_replay.sh" "$@" || exit 1
sleep 5
R=$DATA/runs/train/${RUN_NAME:-H41_$ARM}
setsid nohup taskset -c "$CPUS" bash "$OPS/h41_after_end.sh" "$R" "$GPU" "$CPUS" > "$DATA/logs/fulldev_$(basename "$R").log" 2>&1 < /dev/null &
echo "$(date -u +%T) launched $(basename "$R")"
