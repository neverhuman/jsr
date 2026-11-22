#!/usr/bin/env bash
# H34 (research/H34_delay_penalty): 3k-step fine-tunes from D2c final (ckpt-00010193) with the pruned
# RNN-T delay_penalty lambda; arms run back to back on one GPU under fleet_guard; everything but
# lambda is matched (preset S, bpe4096, 960 h, speed perturbation, budget 650 s / 144, lr 1.1e-4).
#   h34_delay.sh <jsr_data> <cpu_list> <gpu> <lambda...>
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3; shift 3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$HOME/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.83} CPU_TARGET=0.85 MIN_DUTY=0.0
SETS="budget_s=650.0 max_batch=144 lr=0.00011 warmup=100 total_steps=3000 seed=1234 clip=5.0 specaug_start=0
 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995
 qkv_cap_from=100 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=500 eval_every=1000 dev_limit=400
 ckpt_every_steps=500 weights_every=1000 log_every=50 loader=process workers=12 window_groups=8"
for lam in "$@"; do
  R=$DATA/runs/train/H34_dp$lam
  [ -f "$R/DONE" ] && continue
  mkdir -p "$R"
  # shellcheck disable=SC2086
  "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- --preset S \
    --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 \
    --init-from "$DATA/seeds/H34/ckpt-00010193.pt" --set $SETS delay_penalty=$lam || exit 1
  while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 30; done
  kill "$(cat "$R/GUARD_PID")" 2>/dev/null
  if grep -q '"kind": "train", "step": 3000' "$R/train.jsonl"; then touch "$R/DONE"; else echo "H34_dp$lam stopped early"; exit 1; fi
done
echo H34_DONE
