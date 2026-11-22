#!/usr/bin/env bash
# H13b/c/d matched 100 h controls (research/H13_vocab_and_pruned_loss): the given arms run one after
# another on one GPU (in practice one arm per GPU: H13b/H13c on xbabe7's A100s, H13d on xbabe6's
# A10), each under its own fleet_guard (GPU target 0.88). Pinned: seed 1234, budget 300 s,
# max batch 64, lr 1e-3, warmup 2000, 24k steps, clip 5, SpecAugment from 4k, no speed perturbation,
# EMA 0.9995, dev-clean400 + eval-only snapshot every 2000 steps.
#   h13_controls.sh <jsr_data> <cpu_list> <gpu_index> <arm...>   (arms: H13b H13c H13d)
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3
shift 3
want=" $* "
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=$DATA/venvs/train/bin/python GPU_TARGET=0.88 CPU_TARGET=0.85 MIN_DUTY=0.0
COMMON="--preset S --splits train-clean-100 --device cuda --threads 8"
SETS="seed=1234 budget_s=300.0 max_batch=64 workers=6 lr=0.001 warmup=2000 total_steps=24000 clip=5.0 specaug_start=4000 ema_decay=0.9995 eval_every=2000 dev_limit=400 ckpt_every_steps=2000 weights_every=2000 log_every=50 diverge_after=5000"
for arm in "H13b_exact4096 rnnt_loss=exact rnnt_pieces=4096 ctc_pieces=1024" \
           "H13c_pruned4096 rnnt_loss=pruned prune_range=5 prune_warmup=2000 rnnt_pieces=4096 ctc_pieces=1024" \
           "H13d_pruned512 rnnt_loss=pruned prune_range=5 prune_warmup=2000 rnnt_pieces=512 ctc_pieces=512"; do
  read -r name extra <<< "$arm"
  [[ "$want" == *" ${name%%_*} "* ]] || continue
  R=$DATA/runs/train/$name
  [ -f "$R/DONE" ] && continue
  ln -sfn "$R" "$DATA/runs/train/H13_current_gpu$GPU"  # guard_ceiling follows the running arm
  # shellcheck disable=SC2086
  "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- $COMMON --set $SETS $extra || exit 1
  while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 60; done
  kill "$(cat "$R/GUARD_PID")" 2>/dev/null
  grep -q '"kind": "train", "step": 24000' "$R/train.jsonl" && touch "$R/DONE" || { echo "$name did not finish"; exit 1; }
done
