#!/usr/bin/env bash
# H23 optimiser bake-off (research/H23_optimizer_bakeoff): matched 100 h arms, one arm per GPU, each
# under its own fleet_guard (GPU target 0.88). Protocol pinned to the H13d winner (preset S,
# SentencePiece-512 both heads, pruned RNN-T loss): seed 1234, budget 300 s, max batch 64, 24k
# steps, SpecAugment from 4k, no speed perturbation, EMA 0.9995, dev-clean400 + eval-only snapshot
# every 2000 steps. Only the optimiser / LR schedule / clipping differ per arm (README table).
#   h23_bakeoff.sh <jsr_data> <cpu_list> <gpu_index> <arm...>   (arms: H23a H23b H23c H23d H23e)
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3
shift 3
want=" $* "
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=$DATA/venvs/train/bin/python GPU_TARGET=0.88 CPU_TARGET=0.85 MIN_DUTY=0.0
COMMON="--preset S --splits train-clean-100 --device cuda --threads 8"
SETS="seed=1234 budget_s=300.0 max_batch=64 workers=6 loader=process total_steps=24000 clip=5.0 specaug_start=4000 ema_decay=0.9995 eval_every=2000 dev_limit=400 ckpt_every_steps=2000 weights_every=2000 log_every=50 diverge_after=5000 rnnt_loss=pruned prune_range=5 prune_warmup=2000 rnnt_pieces=512 ctc_pieces=512"
for arm in "H23a_adamw optimizer=adamw lr=0.001 warmup=2000" \
           "H23b_scaledadam optimizer=scaled_adam lr=0.045 warmup=500 eden_lr_batches=7500 eden_lr_epochs=3.5 eden_steps_per_epoch=1200 grad_clip_mode=adaptive clip_k=2.0" \
           "H23c_schedfree optimizer=schedulefree_adamw lr=0.002 warmup=2000 sf_beta2=0.98" \
           "H23d_muon optimizer=muon lr=0.001 warmup=2000 muon_momentum=0.95" \
           "H23e_adamw_aclip optimizer=adamw lr=0.001 warmup=2000 grad_clip_mode=adaptive clip_k=2.0 clip_window=200"; do
  read -r name extra <<< "$arm"
  [[ "$want" == *" ${name%%_*} "* ]] || continue
  R=$DATA/runs/train/$name
  [ -f "$R/DONE" ] && continue
  mkdir -p "$R" && ln -sfn "$R" "$DATA/runs/train/H23_current_gpu$GPU"  # guard_ceiling follows the running arm
  # shellcheck disable=SC2086
  "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- $COMMON --set $SETS $extra || exit 1
  while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 60; done
  kill "$(cat "$R/GUARD_PID")" 2>/dev/null
  grep -q '"kind": "train", "step": 24000' "$R/train.jsonl" && touch "$R/DONE" || { echo "$name did not finish"; exit 1; }
done
