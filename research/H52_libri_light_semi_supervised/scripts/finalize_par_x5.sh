#!/usr/bin/env bash
# H52: parallel finaliser on x5 (ll_finalize_par.py = ll_finalize.py's filter + output, passes spread over workers).
# usage: finalize_par_x5.sh <subset> <out-root> <workers> [extra ll_finalize options...]
# Disk: buckets peak ~ one copy of the kept data; each bucket's parts are deleted once its final shard is verified.
set -euo pipefail
L=$HOME/jsr-data/librilight; cd $L
SUB=$1; OUT=$2; W=$3; shift 3
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise $SUB (parallel x$W) start (extra: $*)"
nice -n 19 ionice -c3 $L/venv/bin/python -I bin/ll_finalize_par.py --raw shards/raw/librilight.$SUB --out $OUT \
  --subset $SUB --rows-per-shard 2000 --guard-runs "~/jsr-data/runs/train/*/train.jsonl" --guard-mode v2 --max-wait-pct 8 \
  --rate-drop 0.05 --min-free-gb 60 --workers $W "$@"
cd $OUT/librilight.$SUB
split -n l/$W -d SHA256SUMS .sumpart.
ls .sumpart.* | xargs -P $W -I{} nice -n 19 ionice -c3 sha256sum -c --quiet {}
rm -f .sumpart.*
echo "SHA256SUMS verified: $(wc -l < SHA256SUMS) shards"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise $SUB done"
