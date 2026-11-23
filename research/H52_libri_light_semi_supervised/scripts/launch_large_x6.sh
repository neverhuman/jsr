#!/usr/bin/env bash
# H52 large on xbabe6: verified index -> overlap/dup check -> pre-registered selection (v1) -> wait for GO
# (H0 posts the list sha256 first) -> tier A segmentation on local disk.
set -euo pipefail
L=$HOME/jsr-data/librilight; PY=$L/venv/bin/python; cd $L
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
until [ -f meta/large/receipt.json ]; do sleep 60; done
$PY -I -c "import json,sys; r=json.load(open('meta/large/receipt.json')); sys.exit(0 if r['md5_matches_official'] else 3)" \
  || { log "large receipt does NOT match the official md5; stopping"; exit 3; }
log "large index verified: $(tr -d '\n' < meta/large/receipt.json | cut -c1-300)"
$PY -I bin/ll_overlap.py meta/librispeech analysis/overlap_large large:meta/large/json_meta.jsonl
$PY -I bin/ll_select_large.py meta/large analysis/overlap_large meta/large/select
log "selection done; waiting for GO (H0 posts the list sha256 first)"
until [ -f meta/large/select/GO ]; do sleep 30; done
# tier A only: exclude every FLAC member not in tier A
$PY -I - <<'PYEOF'
sel = {l.split('\t')[0] for l in open('meta/large/select/select_large_v1.tsv').read().splitlines()[1:] if l.endswith('\tA')}
with open('meta/large/select/not_tierA.tsv', 'w') as f:
    f.write('name\n')
    for line in open('meta/large/members.tsv').read().splitlines()[1:]:
        n, k = line.split('\t')[:2]
        if k == 'reg' and n.endswith('.flac') and n not in sel:
            f.write(n + '\n')
print('tier A files', len(sel))
PYEOF
# x6 CPU priority: h52-labeler feature workers (lead 09:2xZ); my student engines stay at 0
S=$L/student
for g in $(pgrep -f "^/home/ubuntu/jsr/.venv/bin/python -I bin/ll_guard.py"); do kill -TERM $g; done
echo 0 > $S/src_max
(cd $S && setsid nohup nice -n 19 ~/jsr/.venv/bin/python -I bin/ll_guard2.py --throttle-file $S/src_max --max 0 --start 0 \
   --interval 30 --discover 60 --log $S/logs/guard2.log > $S/logs/guard2.out 2>&1 < /dev/null &)
echo 4 > $L/pass2_large_throttle
log "pass 2 large tier A starting (in-flight cap 4: labeller first on x6)"
exec nice -n 19 $PY -I bin/ll_pass2.py --index meta/large --exclude meta/large/select/not_tierA.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/large.tar --subset large --out $L/shards/raw --handoff \
  --workers 10 --k 8 --rows-per-group 100 --rows-per-shard 800 --throttle-file $L/pass2_large_throttle
