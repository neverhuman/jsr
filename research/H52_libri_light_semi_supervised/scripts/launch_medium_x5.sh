#!/usr/bin/env bash
# H52 medium on xbabe5: wait for pass 1's verified index, overlap check, skip list, then pass 2 (local shards).
set -euo pipefail
L=$HOME/jsr-data/librilight; PY=$L/venv/bin/python; cd $L
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
until [ -f meta/medium/receipt.json ]; do sleep 20; done
$PY -I -c "import json,sys; r=json.load(open('meta/medium/receipt.json')); sys.exit(0 if r['md5_matches_official'] else 3)" \
  || { log "medium receipt does NOT match the official md5; stopping"; exit 3; }
log "medium index verified: $(cat meta/medium/receipt.json | tr -d '\n' | cut -c1-400)"
$PY -I bin/ll_overlap.py meta/librispeech analysis/overlap_medium medium:meta/medium/json_meta.jsonl
$PY -I - <<'PYEOF'
import os
d = 'analysis/overlap_medium'
rows, seen = [], set()
for fn in ('excluded_medium.tsv', 'train_dup_medium.tsv'):
    for line in open(os.path.join(d, fn)).read().splitlines()[1:]:
        n = line.split('\t')[0]
        if n not in seen:
            seen.add(n); rows.append(line)
open(os.path.join(d, 'skip_medium.tsv'), 'w').write('name\tspeaker\tproject\treasons\tvad_s\n' + '\n'.join(rows) + '\n')
print('skip_medium rows', len(rows), 'vad h %.1f' % (sum(float(r.split('\t')[4]) for r in rows) / 3600))
PYEOF
echo 40 > $L/pass2_medium_throttle
setsid nohup nice -n 19 ~/jsr/.venv/bin/python -I bin/ll_guard.py --runs "~/jsr-data/runs/train/*/train.jsonl" \
  --throttle-file $L/pass2_medium_throttle --max 40 --start 40 --interval 30 --log $L/logs/guard_pass2_medium.log \
  > $L/logs/guard_pass2_medium.out 2>&1 < /dev/null &
log "pass 2 medium starting"
exec nice -n 19 $PY -I bin/ll_pass2.py --index meta/medium --exclude analysis/overlap_medium/skip_medium.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/medium.tar --subset medium --out $L/shards/raw --handoff \
  --workers 20 --k 8 --rows-per-group 100 --rows-per-shard 800 --throttle-file $L/pass2_medium_throttle
