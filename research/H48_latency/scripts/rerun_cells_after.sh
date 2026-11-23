#!/usr/bin/env bash
# Re-run the decode cells stopped for the VRAM cap, once the run's in-flight set has finished.
# RERUN_ALONE=1: alone on the card with the GPU lease, <= 3 cells (fraction 0.29 each); otherwise beside a trainer (no lease),
# <= 2 cells at 0.245 each (fulldev.sh defaults). Expandable segments limit allocator fragmentation in every cell.
#   rerun_cells_after.sh <run_name> <gpu> <cpus> "<chunks>"
set -u
N=$1 G=$2 C=$3 CH=$4; O=$HOME/jsr-data/evals/burst/h41
until [ -e "$O/$N.DONE" ] || [ -e "$O/$N.FAILED" ]; do sleep 15; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${RERUN_ALONE:-0}" = 1 ]; then
  sleep 5
  BURST_CHUNKS="$CH" BURST_MAXJ=3 bash "$HOME/fulldev.sh" "$HOME/jsr-data/runs/train/$N" "$G" "$C"
else
  sleep 120
  BURST_NO_LEASE=1 BURST_CHUNKS="$CH" BURST_MAXJ=2 bash "$HOME/fulldev.sh" "$HOME/jsr-data/runs/train/$N" "$G" "$C"
fi
