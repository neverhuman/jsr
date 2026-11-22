#!/usr/bin/env bash
# xbabe5 GPU0: D2c = warm start of D2 avg(60/65/70k) on the A100, D2's 4096/1024 tokenizers, exact -> pruned
set -uo pipefail
D=$HOME/jsr-data; W=$HOME/jsr/.agent/worktrees/trainer
until grep -q READY $D/logs/d2c-corpus.log 2>/dev/null; do sleep 30; done
export JSR_DATA=$D PY=$D/venvs/train/bin/python GPU_TARGET=0.88
$W/python/ultra/train/launch_guarded.sh $D/runs/train/D2c_S_warm_pruned_960h 0-27 0 -- --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 \
  --init-from $D/seeds/D2c_seed_pruned_lin.pt --set seed=1234 budget_s=750.0 max_batch=160 lr=0.00022 warmup=1000 \
  total_steps=20000 clip=5.0 specaug_start=0 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=1000 prune_hold=200 \
  ema_decay=0.9995 qkv_cap_from=1000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=1500 eval_every=2500 \
  dev_limit=400 ckpt_every_steps=2500 weights_every=2500 log_every=50 loader=process workers=22 window_groups=16
