#!/usr/bin/env bash
# A1 accuracy trunk (research/H37_trunks_a1_f1): dense S, R0 BPE-500, warm from H22ctl ckpt-00079800 (EMA
# weights, seeds/A1_seed_ema.pt), AdamW, 1,000 warmup steps to 1e-3, one cosine; per-rank 1,300 s / 288 (H22ctl's
# batch), window_groups 96, chunk sampling {4,8,16}, dev eval at chunk 4 and 8, hottest-rank throttle 0.83.
# 4-rank DDP: node0 xbabe5 GPU0, node1 xbabe5 GPU1 (own dir), node2 xbabe7 GPU0, node3 xbabe9.
#   a1_ddp.sh <node_rank> <gpu> <cpus> <run_dir_name> [key=value ...]   env: NNODES (4), BUDGET, MAXB, TOTAL
# Trailing key=value pairs override the settings below (e.g. total_steps re-derived from the measured rate).
set -uo pipefail
NR=$1 GPU=$2 CPUS=$3 NAME=$4
shift 4
D=$HOME/jsr-data; R=$D/runs/train/$NAME
export JSR_DATA=$D PY=$D/venvs/train/bin/python NNODES=${NNODES:-4} NODE_RANK=$NR MASTER_ADDR=172.27.124.200 MASTER_PORT=${MASTER_PORT:-29710} NCCL_SOCKET_IFNAME=eno1
bash $HOME/jsr/.agent/worktrees/trainer/python/ultra/train/ops/launch_ddp.sh "$R" "$CPUS" "$GPU" -- --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --init-from $D/seeds/A1_seed_ema.pt --set seed=1234 \
  budget_s=${BUDGET:-5200.0} max_batch=${MAXB:-1152} optimizer=adamw lr=0.001 warmup=1000 total_steps=${TOTAL:-36000} \
  rnnt_pieces=500 ctc_pieces=500 tokenizer=$D/models/zipformer-en-2023-06-26/bpe.model clip=5.0 specaug_start=0 \
  speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995 \
  qkv_cap_from=100 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=8000 diverge_wer_points=15.0 \
  eval_every=2500 eval_each_epoch=false dev_limit=400 ckpt_every_steps=2500 weights_every=2500 log_every=50 \
  loader=process workers=16 window_groups=96 chunk_modes=4,8,16 eval_chunks=8 \
  ddp_util_target=0.83 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true "$@"
