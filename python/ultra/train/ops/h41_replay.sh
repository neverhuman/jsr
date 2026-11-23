#!/usr/bin/env bash
# H41 replay gate (research/H41_replay_gates): fork H22ctl model-00060000, 200 warmup steps to that step's
# LR (2.476e-4), cosine to 3e-5 at step 8,000, budget 650 s / max batch 144 (fits 24 GB), 960 h with speed
# perturbation, seed 1234: identical batches in every arm unless the arm's own key=value overrides say so.
#   h41_replay.sh <jsr_data> <cpu_list> <gpu> <arm> [key=value ...]
# INIT_FROM / RUN_NAME override the fork checkpoint and the run directory name (trunk steps such as A1L).
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3 ARM=$4; shift 4
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$DATA/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.85} CPU_TARGET=0.85 MIN_DUTY=0.0
SETS="seed=1234 budget_s=650.0 max_batch=144 optimizer=adamw lr=0.0002476 warmup=200 total_steps=8000 min_lr_ratio=0.1212 rnnt_pieces=500 ctc_pieces=500 tokenizer=$DATA/models/zipformer-en-2023-06-26/bpe.model clip=5.0 specaug_start=0 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995 qkv_cap_from=100 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=1000 eval_every=2000 eval_each_epoch=false dev_limit=400 ckpt_every_steps=1000 weights_every=2000 log_every=50 loader=process workers=12 window_groups=16"
R=$DATA/runs/train/${RUN_NAME:-H41_$ARM}
mkdir -p "$R"
# shellcheck disable=SC2086
exec "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 \
  --init-from "${INIT_FROM:-$DATA/seeds/H41/model-00060000.pt}" --set $SETS "$@"
