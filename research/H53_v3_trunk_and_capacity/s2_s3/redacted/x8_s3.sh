#!/usr/bin/env bash
# H53 S3 (Amendment 8; trunk + Libri-Light large tier-A subset), world 6, node 2 of 4: a NEW run initialised from the trunk's
# window-boundary checkpoint (raw weights; one file copied to every node under one hash), fresh AdamW, 500 warmup steps back to
# 4.30e-4, WSD-stable; 960 h + small v1.1 + medium v1.1 + the large subset, LibriSpeech 0.5 of the audio (extra_base_weight from
# exact per-row FLAC frames), window_groups 96, loader workers 24 (Amendment 8 addendum 1: the data-wait fix point).
# Required env: S3_INIT, S3_INIT_SHA256, S3_BW, S3_LARGE_SUM16, S2B_MEDIUM_SUM16.
set -euo pipefail
export PIN_COMMIT=90b7f4d56f1046ff6e12e38780028fd14fa9b9d8
export PIN_TREE_SHA256=3780b4518a3bb02614316f61fce8a2ae393280321591558b31417b9d61487631
export INIT=${S3_INIT:?} INIT_SHA256=${S3_INIT_SHA256:?}
export TREE=$HOME/jsr/.agent/worktrees/trainer-s2 NCCL_PROFILE=a1env MASTER_PORT=29758
V=$HOME/jsr-data/librilight/shards/r0_v11; SM=$V/librilight.small; MD=$V/librilight.medium; LG=$V/librilight.large.s3
[ "$(sha256sum ~/h53/h53_trunk.sh | cut -c1-16)" = e783ea450cd928a8 ] || { echo "REFUSED: launcher digest"; exit 1; }
[ "$(sha256sum ~/h53/v3_trunk_s1.set | cut -c1-16)" = 72dc74e7148cc481 ] || { echo "REFUSED: set-file digest"; exit 1; }
[ "$(sha256sum $SM/SHA256SUMS | cut -c1-16)" = 8f371a650591ff06 ] || { echo "REFUSED: small manifest"; exit 1; }
[ "$(sha256sum $MD/SHA256SUMS | cut -c1-16)" = "${S2B_MEDIUM_SUM16:?}" ] || { echo "REFUSED: medium manifest"; exit 1; }
[ "$(sha256sum $LG/SHA256SUMS | cut -c1-16)" = "${S3_LARGE_SUM16:?}" ] || { echo "REFUSED: large subset manifest"; exit 1; }
exec bash ~/h53/h53_trunk.sh 2 4 0 0-27 H53_S3_large_s7919 ${X5_LAN:?} ~/h53/v3_trunk_s1.set \
  budget_s=7800 max_batch=1728 lr=0.00043 warmup=500 ema_decay=0.999398 workers=24 delay_penalty=0.005 \
  extra_splits=$SM,$MD,$LG extra_base_weight=${S3_BW:?} window_groups=96
