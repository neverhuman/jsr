#!/usr/bin/env bash
# Launch D1 two-island synchronous DiLoCo on xbabe1 (data_rank=0) + xbabe2 (data_rank=1).
# Usage: launch_d1_islands.sh --seed 1234 [--every 250] [--max-steps 4000] [--port 8778] [--dry-run]
set -euo pipefail
SEED=1234; EVERY=250; MAX_STEPS=4000; PORT=8778; DRY_RUN=0
DIR="$(cd "$(dirname "$0")" && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SET_ARGS=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  SET_ARGS+=" $line"
done < "$DIR/common_d0d1_24g.set"

TOKEN_FILE="\$HOME/jsr-data/runs/h50_d0d1/.diloco_token"
TUNNEL_CMD="ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:${PORT}:127.0.0.1:${PORT} xbabe2"

launch_island() {
  local HOST="$1" RANK="$2" WID="$3"
  local RUN_DIR="\$HOME/jsr-data/runs/h50_d0d1/D1_H${EVERY}_s${SEED}_${WID}"
  local TOK="\$HOME/jsr-data/models/zipformer-en-2023-06-26/bpe.model"
  local INIT="\$HOME/jsr-data/seeds/H41/model-00060000.pt"
  local CMD="mkdir -p $RUN_DIR && export JSR_DATA=\$HOME/jsr-data && export DILOCO_AUTH_TOKEN=\$(cat $TOKEN_FILE); \
cd ~/jsr-d0d1; PYTHONPATH=python/ultra nohup ~/jsr/.venv/bin/python -m train.trainer \
  --run-dir $RUN_DIR --preset S --device cuda --threads 4 --max-steps $MAX_STEPS \
  --init-from $INIT \
  --splits train-clean-100 train-clean-360 train-other-500 \
  --set $SET_ARGS \
    tokenizer=$TOK \
    seed=$SEED data_rank=$RANK data_world=2 \
    diloco_hub=http://127.0.0.1:${PORT} diloco_worker=$WID \
    diloco_every=$EVERY diloco_dtype=float32 \
  > $RUN_DIR/train.out 2>&1 < /dev/null & echo \$! > $RUN_DIR/train.pid && cat $RUN_DIR/train.pid"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] ssh $HOST \"$CMD\""
  else
    ssh "$HOST" "$CMD"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Copy 0600 token xbabe2 -> xbabe1 and open loopback tunnel:"
  echo "[DRY-RUN] ssh xbabe1 'mkdir -p ~/jsr-data/runs/h50_d0d1 && ssh xbabe2 cat ~/jsr-data/runs/h50_d0d1/.diloco_token > ~/jsr-data/runs/h50_d0d1/.diloco_token && chmod 600 ~/jsr-data/runs/h50_d0d1/.diloco_token'"
  echo "[DRY-RUN] ssh xbabe1 \"$TUNNEL_CMD\""
else
  ssh xbabe1 "mkdir -p ~/jsr-data/runs/h50_d0d1 && ssh xbabe2 cat ~/jsr-data/runs/h50_d0d1/.diloco_token > ~/jsr-data/runs/h50_d0d1/.diloco_token && chmod 600 ~/jsr-data/runs/h50_d0d1/.diloco_token"
  ssh xbabe1 "$TUNNEL_CMD"
fi

launch_island xbabe1 0 xbabe1-4090
launch_island xbabe2 1 xbabe2-3090
