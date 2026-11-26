#!/bin/bash
# Locked final-pass replay on a new acoustic INT8 model: native c16 B16 M5 n-best -> locked 11.7M L2R (no retune) -> pinned vs own final and vs locked H141 4.84.
# locked_replay_model.sh TAG MODEL_DIR
# (decode only; scoring and replay are in locked_replay_list.sh, which can also be run on an existing list)
set -euo pipefail
TAG=$1; MODEL=$2; H=/mnt/xbabe3-external/jsr/wer-push/H68; cd $H
T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools; PY=/home/ubuntu/jsr/.venv/bin/python
SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score; O=rescore/locked_$TAG; mkdir -p $O
sha256sum $MODEL/* > $O/model.SHA256SUMS
(cd /home/ubuntu/jsr/.agent/worktrees/wer-codex && PYTHONPATH=python/ultra $PY -m train.ops.dev_eval --weights $MODEL --decoder /mnt/xbabe3-external/jsr/wer-push/H92/ultra-dev-worker --pcm-dir $H/pcm_tail320 --out $H/nbest_native/${TAG}_c16_b16_m5_tail --chunk 16 --beam 16 --nbest --lm /mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/M5.ulm --lm-weight 0.8 --lodr 0.6 --workers xbabe0=32) > $O/nbest.log 2>&1
$PY - $TAG <<'PY'
import json, numpy as np, glob, sys
tag=sys.argv[1]; uids=[]
for s in ('dev-clean','dev-other'): uids += [str(u) for u in np.load(f'pcm_tail320/{s}.npz')['uids']]
R={r['uid']:r for r in map(json.loads,open('/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl'))}
rows=[]; d=f'nbest_native/{tag}_c16_b16_m5_tail'
for f in glob.glob(f'{d}/xbabe*.json'): rows+=json.load(open(f))['rows']
si={r['id']:r['hyp'] for r in map(json.loads,open(f'{d}/score-input.jsonl'))}
bad=0
with open(f'nbest_native/{tag}_final_b16.nbest.jsonl','w') as out:
    for r in sorted(rows,key=lambda r:r['index']):
        u=uids[r['index']]; bad+=r['nbest'][0]['hyp']!=si[u]
        out.write(json.dumps({'uid':u,'ref':R[u]['ref'],'nbest':[[c['tokens'],c['score'],c['hyp']] for c in r['nbest']]})+'\n')
assert len(rows)==2782 and bad==0, (len(rows),bad)
PY
exec $H/locked_replay_list.sh $TAG $H/nbest_native/${TAG}_final_b16.nbest.jsonl $H/nbest_native/${TAG}_c16_b16_m5_tail/score-input.jsonl 16
