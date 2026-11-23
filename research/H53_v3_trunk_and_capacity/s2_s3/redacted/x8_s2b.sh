#!/usr/bin/env bash
# H53 S2b (Amendment 6; medium segment), world 6, node 2 of 4: a NEW run initialised from S2a's window-boundary checkpoint
# (raw weights; one file copied to every node under one hash), fresh optimizer, 500 warmup steps back to 4.30e-4, WSD-stable;
# 960 h + small v1.1 + medium v1.1 with extra_base_weight from the measured row-group audio (Libri-Light cap 0.5).
# Required env: S2B_INIT (path, identical on all nodes), S2B_INIT_SHA256, S2B_BW, S2B_MEDIUM_SUM16, S2B_WG (window_groups
# from the Amendment 6 per-batch audit: the smallest value passing median 0.45-0.55, p95 <= 0.65; trainer-lane 11:09Z).
set -euo pipefail
export PIN_COMMIT=90b7f4d56f1046ff6e12e38780028fd14fa9b9d8
export PIN_TREE_SHA256=3780b4518a3bb02614316f61fce8a2ae393280321591558b31417b9d61487631
export INIT=${S2B_INIT:?} INIT_SHA256=${S2B_INIT_SHA256:?}
export TREE=$HOME/jsr/.agent/worktrees/trainer-s2 NCCL_PROFILE=a1env MASTER_PORT=29756
SM=$HOME/jsr-data/librilight/shards/r0_v11/librilight.small; MD=$HOME/jsr-data/librilight/shards/r0_v11/librilight.medium
[ "$(sha256sum ~/h53/h53_trunk.sh | cut -c1-16)" = e783ea450cd928a8 ] || { echo "REFUSED: launcher digest"; exit 1; }
[ "$(sha256sum ~/h53/v3_trunk_s1.set | cut -c1-16)" = 72dc74e7148cc481 ] || { echo "REFUSED: set-file digest"; exit 1; }
[ "$(sha256sum $SM/SHA256SUMS | cut -c1-16)" = 8f371a650591ff06 ] || { echo "REFUSED: small manifest"; exit 1; }
[ "$(sha256sum $MD/SHA256SUMS | cut -c1-16)" = "${S2B_MEDIUM_SUM16:?}" ] || { echo "REFUSED: medium manifest"; exit 1; }
exec bash ~/h53/h53_trunk.sh 2 4 0 0-27 H53_S2b_medium_s7919 ${X5_LAN:?} ~/h53/v3_trunk_s1.set \
  budget_s=7800 max_batch=1728 lr=0.00043 warmup=500 ema_decay=0.999398 workers=20 delay_penalty=0.005 \
  extra_splits=$SM,$MD extra_base_weight=${S2B_BW:?} window_groups=${S2B_WG:?}
