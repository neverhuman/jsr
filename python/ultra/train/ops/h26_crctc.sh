#!/usr/bin/env bash
# H26 CR-CTC 100 h gate (research/H26_kd_audit_smbr_and_cr_ctc, "CR-CTC 100 h gate arms"): exactly the
# H23a / H13d-winner protocol (preset S, SentencePiece-512 both heads, pruned RNN-T, AdamW lr 1e-3, warmup
# 2000, seed 1234, budget 300 s / max batch 64, 24k steps, SpecAugment from 4k, EMA, dev-clean400 every 2k)
# plus cr_ctc_weight=w. H23a_adamw is the control. One arm per call, under fleet_guard (target 0.83).
#   h26_crctc.sh <jsr_data> <cpu_list> <gpu_index> <cr_ctc_weight>
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3 W=$4
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export JSR_DATA=$DATA PY=${PY:-$DATA/venvs/train/bin/python} GPU_TARGET=${GPU_TARGET:-0.83} CPU_TARGET=0.85 MIN_DUTY=0.0
COMMON="--preset S --splits train-clean-100 --device cuda --threads 8"
SETS="seed=1234 budget_s=300.0 max_batch=64 workers=6 loader=process window_groups=4 total_steps=24000 clip=5.0 specaug_start=4000 ema_decay=0.9995 eval_every=2000 dev_limit=400 ckpt_every_steps=2000 weights_every=2000 log_every=50 diverge_after=5000 rnnt_loss=pruned prune_range=5 prune_warmup=2000 rnnt_pieces=512 ctc_pieces=512 optimizer=adamw lr=0.001 warmup=2000"
R=$DATA/runs/train/H26_crctc$W
mkdir -p "$R"
# shellcheck disable=SC2086
exec "$REPO/python/ultra/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- $COMMON --set $SETS cr_ctc_weight=$W
