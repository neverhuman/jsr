#!/usr/bin/env bash
# D2d (warm from D2c final), 4-rank cross-host DDP: node0 xbabe5 GPU0, node1 xbabe5 GPU1 (own dir), node2 xbabe9, node3 xbabe8
#   d2c_ddp4.sh <node_rank> <gpu> <cpus> <run_dir_name>
set -uo pipefail
NR=$1 GPU=$2 CPUS=$3 NAME=$4
D=$HOME/jsr-data; R=$D/runs/train/$NAME
export JSR_DATA=$D PY=$D/venvs/train/bin/python NNODES=4 NODE_RANK=$NR MASTER_ADDR=172.27.124.200 MASTER_PORT=29670 NCCL_SOCKET_IFNAME=eno1
bash $HOME/jsr/.agent/worktrees/trainer/python/ultra/train/ops/launch_ddp.sh "$R" "$CPUS" "$GPU" -- --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --init-from $D/seeds/D2d_seed.pt --set seed=1234 \
  budget_s=2800.0 max_batch=600 lr=0.0003 warmup=386 total_steps=18514 clip=5.0 specaug_start=0 \
  speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995 \
  qkv_cap_from=100 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=1500 eval_every=2500 dev_limit=400 \
  ckpt_every_steps=2500 weights_every=2500 log_every=50 loader=process workers=16 window_groups=16 \
  ddp_util_target=0.85 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true
