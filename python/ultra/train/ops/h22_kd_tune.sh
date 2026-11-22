#!/usr/bin/env bash
# H22 kd_weight tuning (research/H22_r0_distillation, "weight tuning"), fork design approved
# 2026-11-22: each arm forks H22ctl's ckpt-00030000 weights and trains 3k steps on train-clean-100
# (the first complete speed-1.0 teacher-cache split) with KD at kd_weight w; KL is masked on the
# 0.9/1.1 speed copies (no teacher entry). Arms run one after another on one GPU, each under
# fleet_guard; pick = lowest dev-clean400 WER at 3k (DEV only). w=0 is the same-fork reference.
# Half of H22ctl's global batch (650 s / 144) to fit 24 GB; LR = H22ctl's LR at 30k x sqrt(1/2).
#   h22_kd_tune.sh <jsr_data> <cpu_list> <gpu> <w...>
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3; shift 3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$HOME/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.83} CPU_TARGET=0.85 MIN_DUTY=0.0
SETS="budget_s=650.0 max_batch=144 lr=0.00074 warmup=100 total_steps=3000 rnnt_pieces=500 ctc_pieces=500
 tokenizer=$DATA/models/zipformer-en-2023-06-26/bpe.model seed=1234 clip=5.0 specaug_start=0
 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 ema_decay=0.9995 qkv_cap_from=100
 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=500 eval_every=1000 dev_limit=400 ckpt_every_steps=500
 weights_every=1000 log_every=50 loader=process workers=12 window_groups=8 kd_dir=$DATA/teacher_cache_R0_tc100"
for w in "$@"; do
  R=$DATA/runs/train/H22t_kd$w
  [ -f "$R/DONE" ] && continue
  mkdir -p "$R"
  # shellcheck disable=SC2086
  "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- --preset S --splits train-clean-100 \
    --device cuda --threads 8 --init-from "$DATA/seeds/H22/ckpt-00030000.pt" --set $SETS kd_weight=$w || exit 1
  while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 30; done
  kill "$(cat "$R/GUARD_PID")" 2>/dev/null
  if grep -q '"kind": "train", "step": 3000' "$R/train.jsonl"; then touch "$R/DONE"; else echo "H22t_kd$w stopped early"; exit 1; fi
done
echo H22_KD_TUNE_DONE
