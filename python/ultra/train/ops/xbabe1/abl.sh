#!/usr/bin/env bash
# xbabe1: two 5k-step ablations at lr 8e-4 (amplifies the qkv runaway; D3MR reached 190 @5k),
# then D3MRc (fix on). Each run under launch_guarded.sh (fleet_guard 0.74/0.85/min-duty 0).
set -uo pipefail
export JSR_DATA=/mnt/amphora-backup/jsr-data
I=$JSR_DATA/runs/train/D1MR_S_MR_tc100/ckpt-00040000.pt
L=~/jsr/python/ultra/train/launch_guarded.sh
COMMON="--splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 --init-from $I"
SETS="budget_s=400.0 max_batch=96 workers=8 total_steps=5000 warmup=1000 lr=0.0008 clip=5.0 specaug_start=2000 eval_every=0 ckpt_every_steps=5000 log_every=50 diverge_after=100000"
run() { name=$1; shift; R=$JSR_DATA/runs/train/$name; $L $R 0-15,64-79 0 -- "$@"; while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 60; done; }
run ABL_A_noI8S_sp --preset S_MR $COMMON --set $SETS speed_perturb=0.9,1.0,1.1
run ABL_B_I8S_nosp --preset S_MR_I8S $COMMON --set $SETS
echo ABL_DONE > $JSR_DATA/runs/train/ABL_DONE
