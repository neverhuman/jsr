#!/usr/bin/env bash
# H52: finalise small on xbabe5 (verify hashes, pre-registered filter, global shuffle) -> shards/r0/librilight.small
set -euo pipefail
L=$HOME/jsr-data/librilight; cd $L
EXTRA=${1:-}
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise small start (extra: ${EXTRA:-none})"
nice -n 19 ionice -c3 $L/venv/bin/python -I bin/ll_finalize.py --raw shards/raw/librilight.small --out shards/r0 \
  --subset small --rows-per-shard 2000 --guard-runs "~/jsr-data/runs/train/*/train.jsonl" $EXTRA
sha256sum -c --quiet shards/r0/librilight.small/SHA256SUMS --strict 2>&1 | tail -2 || true
( cd shards/r0/librilight.small && sha256sum -c --quiet SHA256SUMS && echo "SHA256SUMS verified: $(wc -l < SHA256SUMS) shards" )
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise small done"
