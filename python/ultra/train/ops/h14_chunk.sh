#!/usr/bin/env bash
# H14 (research/H14_chunk_latency): 4k-step fine-tunes from D2e final (ckpt-00016457, raw) at a smaller
# attention chunk (25 Hz frames per chunk: 2 = 80 ms vs the default 4 = 160 ms); the chunk=4 arm is the
# matched control. Arms run back to back under fleet_guard; only `chunk` differs.
#   h14_chunk.sh <jsr_data> <cpu_list> <gpu> <chunk...>
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3; shift 3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$HOME/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.85} CPU_TARGET=0.85 MIN_DUTY=0.0
SETS="budget_s=650.0 max_batch=144 lr=0.0001 warmup=200 total_steps=4000 seed=1234 clip=5.0 specaug_start=0
 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995
 qkv_cap_from=100 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=500 eval_every=1000 dev_limit=400
 ckpt_every_steps=500 weights_every=1000 log_every=50 loader=process workers=12 window_groups=16"
for c in "$@"; do
  R=$DATA/runs/train/H14_chunk$c
  [ -f "$R/DONE" ] && continue
  mkdir -p "$R"
  # shellcheck disable=SC2086
  "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- --preset S \
    --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 \
    --init-from "$DATA/seeds/H14/ckpt-00016457.pt" --set $SETS chunk=$c || exit 1
  while kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 30; done
  kill "$(cat "$R/GUARD_PID")" 2>/dev/null
  if grep -q '"kind": "train", "step": 4000' "$R/train.jsonl"; then touch "$R/DONE"; else echo "H14_chunk$c stopped early"; exit 1; fi
done
echo H14_DONE
