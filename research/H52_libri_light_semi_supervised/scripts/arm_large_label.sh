#!/usr/bin/env bash
# H52 large tier A labels on xbabe6 (v1.1 code only: kdcode/r0_gpu.py 7474f083): wait for the first NNNN.ready from
# pass 2, then start 2 GPU servers + 2 feature fronts (own process groups, fleet_guard: GPU <= 88%, CPU <= 80%) and
# the per-shard driver (claim by mkdir NNNN.claim; labels + frames + NNNN.labeled beside each shard).
set -euo pipefail
B=$HOME/jsr-data/h52-labeler; C=$HOME/jsr-data/librilight/shards/raw/librilight.large
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
test "$(sha256sum $B/kdcode/r0_gpu.py | cut -c1-64)" = 7474f0835051f13a695346fbb0defdebdc13d364f5301e42f661b1b3cedb2168 \
  || { log "kdcode/r0_gpu.py is not v1.1; refusing"; exit 3; }
until ls $C/*.ready > /dev/null 2>&1; do sleep 30; done
log "first large shard ready: $(ls $C/*.ready | head -1)"
mkdir -p $B/L-src $B/L-stage $B/L-feats $B/L-out $B/large-state
export H52LAB_CODE=$B/code-next
bash $B/code-next/launch_worker.sh $B xL-cons gpu 0 0.88 0.80 -- serve --feats $B/L-feats --out $B/L-out \
  --max-frames 800000 --mem-gb 6 --torch-frac 0.12 --frames 1 --beam 8
bash $B/code-next/launch_worker.sh $B xL-cons2 gpu 0 0.88 0.80 -- serve --feats $B/L-feats --out $B/L-out \
  --max-frames 800000 --mem-gb 6 --torch-frac 0.12 --frames 1 --beam 8
bash $B/code-next/launch_worker.sh $B xL-prod front -1 0.88 0.80 -- serve --inbox $B/L-src --stage $B/L-stage \
  --workers 8 --max-unshipped 3
bash $B/code-next/launch_worker.sh $B xL-prod2 front -1 0.88 0.80 -- serve --inbox $B/L-src --stage $B/L-stage \
  --workers 8 --max-unshipped 3
cd $B/logs
setsid nohup /usr/bin/python3 $B/code-next/lab_drive_large.py --contract $C --workers $B/large-workers.json \
  --state $B/large-state > $B/logs/drive_large.log 2>&1 < /dev/null &
log "driver pid $! started; servers / fronts in logs/pids.jsonl"
