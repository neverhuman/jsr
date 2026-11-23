#!/usr/bin/env bash
# Launch matched D0 synchronous control on xbabe1 4090 (exact full 650s global batch, world=1).
# Usage: launch_d0_control.sh --seed 1234 [--max-steps 4000] [--dry-run]
set -euo pipefail
SEED=1234; MAX_STEPS=4000; DRY_RUN=0
DIR="$(cd "$(dirname "$0")" && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SET_ARGS=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  SET_ARGS+=" --set $line"
done < "$DIR/common_d0d1_24g.set"

RUN_DIR="\$HOME/jsr-data/runs/h50_d0d1/D0_sync_s${SEED}_xbabe1"
TOK="\$HOME/jsr-data/models/zipformer-en-2023-06-26/bpe.model"
INIT="\$HOME/jsr-data/seeds/H41/model-00060000.pt"

CMD="mkdir -p $RUN_DIR && export JSR_DATA=\$HOME/jsr-data; \
cd ~/jsr-d0d1; PYTHONPATH=python/ultra nohup ~/jsr/.venv/bin/python -m train.trainer \
  --run-dir $RUN_DIR --preset S --device cuda --threads 4 --max-steps $MAX_STEPS \
  --init-from $INIT \
  --splits train-clean-100 train-clean-360 train-other-500 \
  $SET_ARGS \
  --set tokenizer=$TOK \
  --set seed=$SEED --set data_rank=0 --set data_world=0 \
  > $RUN_DIR/train.out 2>&1 < /dev/null & echo \$! > $RUN_DIR/train.pid && cat $RUN_DIR/train.pid"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] ssh xbabe1 \"$CMD\""
else
  ssh xbabe1 "$CMD"
fi
