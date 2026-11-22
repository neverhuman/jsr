#!/usr/bin/env bash
# F1 fast trunk (research/H37_trunks_a1_f1): S_MR, R0 BPE-500, from scratch, H22ctl's recipe (AdamW, lr 1e-3,
# warmup 2000, pruned from 2000, SpecAugment from 4000, speed perturbation, EMA), single A100 under fleet_guard,
# budget 2,000 s / max batch 400 (L2's S_MR batch on a 40 GB A100), 54,000 steps = 30k audio-h.
# qkv cap from step 100: the cap is relative to the step-100 (near-initial) row norms (~20), so the ratio is
# 6.0 (cap ~120-135, where healthy runs plateau: H22ctl 118, L3 114-117); ratio 1.5 would pin them at ~33.
#   f1_launch.sh <jsr_data> <cpu_list> <gpu>
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$DATA/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.85} CPU_TARGET=0.85 MIN_DUTY=0.0
R=$DATA/runs/train/F1_S_MR_r0bpe_960h
mkdir -p "$R"
exec "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- --preset S_MR \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --set seed=1234 \
  budget_s=2000.0 max_batch=400 optimizer=adamw lr=0.001 warmup=2000 total_steps=54000 rnnt_pieces=500 ctc_pieces=500 \
  tokenizer=$DATA/models/zipformer-en-2023-06-26/bpe.model clip=5.0 specaug_start=4000 speed_perturb=0.9,1.0,1.1 \
  rnnt_loss=pruned prune_range=5 prune_warmup=2000 ema_decay=0.9995 qkv_cap_from=100 qkv_cap_ratio=6.0 \
  qkv_guard_ratio=8.0 diverge_after=5500 eval_every=2500 eval_each_epoch=false dev_limit=400 \
  ckpt_every_steps=2500 weights_every=2500 log_every=50 loader=process workers=20 window_groups=16
