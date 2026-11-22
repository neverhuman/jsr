#!/usr/bin/env bash
# xbabe7: wait for the 960 h sha check, then launch L3 (GPU1) and H22-ctl (GPU0), each guarded at 0.88
set -uo pipefail
D=$HOME/jsr-data; W=$HOME/jsr/.agent/worktrees/trainer; O=$W/python/ultra/train/ops
mkdir -p $D/models/zipformer-en-2023-06-26 && mv $D/r0.sha256 $D/models/zipformer-en-2023-06-26/SHA256SUMS 2>/dev/null
bash $O/r0_fetch.sh $D
until grep -q "SHA_OK\|SHA_FAIL" $D/logs/node_setup.log 2>/dev/null; do sleep 60; done
grep -q SHA_OK $D/logs/node_setup.log || exit 1
export JSR_DATA=$D PY=$D/venvs/train/bin/python GPU_TARGET=0.88
SPL="train-clean-100 train-clean-360 train-other-500"
REC="seed=1234 warmup=2000 clip=5.0 specaug_start=4000 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=2000 ema_decay=0.9995 qkv_cap_from=30000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=5500 eval_every=5000 dev_limit=400 ckpt_every_steps=5000 weights_every=5000 log_every=50 workers=24"
# shellcheck disable=SC2086
$W/python/ultra/train/launch_guarded.sh $D/runs/train/L3_M_MR_spm512_pruned_960h 30-57 1 -- --preset M_MR --splits $SPL \
  --device cuda --threads 4 --set budget_s=1200.0 max_batch=256 lr=0.001 total_steps=86400 rnnt_pieces=512 ctc_pieces=512 $REC
# shellcheck disable=SC2086
$W/python/ultra/train/launch_guarded.sh $D/runs/train/H22ctl_S_r0bpe_pruned_960h 0-27 0 -- --preset S --splits $SPL \
  --device cuda --threads 4 --set budget_s=1300.0 max_batch=288 lr=0.00147 total_steps=79800 rnnt_pieces=500 ctc_pieces=500 \
  tokenizer=$D/models/zipformer-en-2023-06-26/bpe.model $REC
