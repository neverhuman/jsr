#!/usr/bin/env bash
# Launch hardened safetensors+HMAC DiLoCoHub on xbabe2 (loopback 127.0.0.1:8768 ONLY).
# Usage: launch_hub_xb2.sh --seed 1234 [--every 250] [--port 8768] [--dry-run]
set -euo pipefail
SEED=1234; EVERY=250; PORT=8768; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
# hub_ema_decay = 0.9995 ** EVERY (matches D0 per-step ema_decay=0.9995)
HUB_EMA=$(python3 -c "print(f'{0.9995 ** int($EVERY):.6f}')")
WORK_DIR="\$HOME/jsr-data/runs/h50_d0d1/hub_D1_H${EVERY}_s${SEED}"
TOKEN_FILE="\$HOME/jsr-data/runs/h50_d0d1/.diloco_token"
INIT_ANCHOR="\$HOME/jsr-data/seeds/H41/model-00060000.pt"

CMD="mkdir -p $WORK_DIR && if [ ! -f $TOKEN_FILE ]; then openssl rand -hex 32 > $TOKEN_FILE && chmod 600 $TOKEN_FILE; fi; \
cd ~/jsr-d0d1; PYTHONPATH=python/ultra nohup ~/jsr/.venv/bin/python -m train.diloco_http \
  --cohort H50_D1_H${EVERY}_s${SEED} \
  --init-anchor $INIT_ANCHOR \
  --work-dir $WORK_DIR \
  --host 127.0.0.1 --port $PORT \
  --roster xbabe1-4090,xbabe2-3090 \
  --auth-token-file $TOKEN_FILE \
  --outer-lr 0.7 --outer-momentum 0.9 \
  --hub-ema-decay $HUB_EMA \
  --keep-last-k 4 --keep-every-m 4 \
  > $WORK_DIR/hub.log 2>&1 < /dev/null & echo \$! > $WORK_DIR/hub.pid && cat $WORK_DIR/hub.pid"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] ssh xbabe2 \"$CMD\""
else
  ssh xbabe2 "$CMD"
fi
