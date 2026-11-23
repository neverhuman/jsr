#!/usr/bin/env bash
# H53 RC1 (Amendment 7; release cooldown), world 6, node 1 of 4: a NEW run from S2b's cut checkpoint (raw weights; one file
# copied to every node under one hash), fresh AdamW, 200 warmup steps back to S2b's stable 4.30e-4, cosine to 2.19e-5
# (min_lr_ratio 0.051) at total_steps = D (max(10,000 audio-h, 0.15 E_main) at the measured audio per step), EMA tau 0.25 D
# (0.999134 at 7,800 s), seed 7919; otherwise S2b's recipe (0.5-capped mix, same bw and window_groups, dp 0.005, chunks 4/8/16).
# Required env: RC1_INIT, RC1_INIT_SHA256, RC1_STEPS, S2B_BW, S2B_MEDIUM_SUM16, S2B_WG (from S2b's launch).
set -euo pipefail
export PIN_COMMIT=90b7f4d56f1046ff6e12e38780028fd14fa9b9d8
export PIN_TREE_SHA256=3780b4518a3bb02614316f61fce8a2ae393280321591558b31417b9d61487631
export INIT=${RC1_INIT:?} INIT_SHA256=${RC1_INIT_SHA256:?}
export TREE=$HOME/jsr/.agent/worktrees/trainer-s2 NCCL_PROFILE=a1env MASTER_PORT=29757
SM=$HOME/jsr-data/librilight/shards/r0_v11/librilight.small; MD=$HOME/jsr-data/librilight/shards/r0_v11/librilight.medium
[ "$(sha256sum ~/h53/h53_trunk.sh | cut -c1-16)" = e783ea450cd928a8 ] || { echo "REFUSED: launcher digest"; exit 1; }
[ "$(sha256sum ~/h53/v3_trunk_s1.set | cut -c1-16)" = 72dc74e7148cc481 ] || { echo "REFUSED: set-file digest"; exit 1; }
[ "$(sha256sum $SM/SHA256SUMS | cut -c1-16)" = 8f371a650591ff06 ] || { echo "REFUSED: small manifest"; exit 1; }
[ "$(sha256sum $MD/SHA256SUMS | cut -c1-16)" = "${S2B_MEDIUM_SUM16:?}" ] || { echo "REFUSED: medium manifest"; exit 1; }
exec bash ~/h53/h53_trunk.sh 1 4 0,1 0-55 H53_RC1_s7919 ${X5_LAN:?} ~/h53/v3_trunk_s1.set \
  budget_s=7800 max_batch=1728 lr=0.00043 warmup=200 total_steps=${RC1_STEPS:?} min_lr_ratio=0.051 ema_decay=0.999134 \
  workers=20 delay_penalty=0.005 extra_splits=$SM,$MD extra_base_weight=${S2B_BW:?} window_groups=${S2B_WG:?}
