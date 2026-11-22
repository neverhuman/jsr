#!/usr/bin/env bash
# Verified VRAM-maximizing (85-92% VRAM) and 5-minute step-boundary throttle profiles
# for ULTRA training across xbabe1-xbabe9. Source this file or pass profile names to print
# recommended `--set` arguments and `PYTORCH_CUDA_ALLOC_CONF`.
#
# Usage:
#   bash vram_profiles.sh a100_ddp3   # print --set overrides for 3x A100-40GB DDP (D2e)
#   bash vram_profiles.sh a100_ddp6   # print --set overrides for 6x A100-40GB DDP (elastic V2)
#   bash vram_profiles.sh a100_single # print --set overrides for 1x A100-40GB (H22-kd / H22ctl)
#   bash vram_profiles.sh rtx4090     # print --set overrides for 1x RTX 4090 24GB (xbabe1)
#   bash vram_profiles.sh rtx3090     # print --set overrides for 1x RTX 3090 24GB (xbabe2/3)
#   bash vram_profiles.sh a10_24g     # print --set overrides for 1x A10 24GB (xbabe6)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.85}"

profile="${1:-}"
case "$profile" in
  a100_ddp3)
    # 3x A100-40GB (xbabe5 x2 + xbabe9): ~1,200 s/rank (~35-37.5 GB VRAM with expandable_segments)
    echo "budget_s=3600.0 max_batch=780 loader=process workers=16 window_groups=64 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true"
    ;;
  a100_ddp4)
    # 4x A100-40GB (xbabe5 x2 + xbabe8 + xbabe9): ~1,150 s/rank (~35-37.5 GB VRAM)
    echo "budget_s=4600.0 max_batch=1000 loader=process workers=16 window_groups=64 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true"
    ;;
  a100_ddp6)
    # 6x A100-40GB (xbabe5 x2 + xbabe7 x2 + xbabe8 + xbabe9 over 100GbE): ~1,150 s/rank
    echo "budget_s=6900.0 max_batch=1500 loader=process workers=16 window_groups=64 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82 ddp_global_batches=true ddp_native=true ddp_bf16_comm=true"
    ;;
  a100_single)
    # 1x A100-40GB (xbabe8 H22-kd / xbabe7 H22ctl): ~35-37.5 GB VRAM, step-boundary 5-min throttle
    echo "budget_s=1450.0 max_batch=320 loader=process workers=16 window_groups=64 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82"
    ;;
  rtx4090)
    # 1x RTX 4090 24GB (xbabe1): ~20.5-22.0 GB VRAM, step-boundary 5-min throttle
    echo "budget_s=950.0 max_batch=208 loader=process workers=12 window_groups=32 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82"
    ;;
  rtx3090)
    # 1x RTX 3090 24GB (xbabe2 / xbabe3): ~20.5-22.0 GB VRAM, pruned RNN-T (~2.5x faster than exact)
    echo "budget_s=950.0 max_batch=208 loader=process workers=12 window_groups=32 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82"
    ;;
  a10_24g)
    # 1x A10 24GB (xbabe6 H23 / V2): ~19.5-21.5 GB VRAM (up from 9.7 GB / 300 s!)
    echo "budget_s=780.0 max_batch=176 loader=process workers=8 window_groups=16 rnnt_loss=pruned ddp_util_target=0.85 cpu_util_target=0.82"
    ;;
  *)
    echo "usage: $0 {a100_ddp3|a100_ddp4|a100_ddp6|a100_single|rtx4090|rtx3090|a10_24g}" >&2
    exit 2
    ;;
esac
