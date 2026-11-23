#!/usr/bin/env bash
# D2c canonical, 3-rank cross-host DDP: node0 xbabe5 GPU0, node1 xbabe9 GPU0, node2 xbabe8 GPU0 (per-rank 750 s, bf16 comm)
set -uo pipefail
NR=$1
D=$HOME/jsr-data; R=$D/runs/train/D2c_S_warm_pruned_960h
export JSR_DATA=$D PY=$D/venvs/train/bin/python NNODES=3 NODE_RANK=$NR MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR (see local fleet hosts file)} MASTER_PORT=29650 NCCL_SOCKET_IFNAME=eno1
bash $HOME/jsr/.agent/worktrees/trainer/python/ultra/train/ops/launch_ddp.sh "$R" 0-27 0 -- --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --set seed=1234 \
  budget_s=2250.0 max_batch=480 lr=0.00038 warmup=1000 total_steps=10806 clip=5.0 specaug_start=0 \
  speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=1000 prune_hold=200 ema_decay=0.9995 \
  qkv_cap_from=1000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=1500 eval_every=2500 dev_limit=400 \
  ckpt_every_steps=2500 weights_every=2500 log_every=50 loader=process workers=20 window_groups=16 \
  ddp_util_target=0.85 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true
