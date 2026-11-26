#!/bin/bash
# Locked final-pass replay on an existing n-best list ({uid,ref,nbest:[[tokens,score,text],...]}, top-1 == own final).
# locked_replay_list.sh TAG NBEST OWN_FINAL_SCOREINPUT [TOPK=16]
# Scores L2R on the GPU when it has room (else CPU), applies the LOCKED config (11.7M L2R, w 0.6, M5-sub 0.6,
# skip 2.7629) with no retune, and reports pinned paired deltas vs its own final and vs the locked H141 4.84.
set -euo pipefail
TAG=$1; N=$2; OWN=$3; K=${4:-16}; H=/mnt/xbabe3-external/jsr/wer-push/H68; cd $H
T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools; PY=/home/ubuntu/jsr/.venv/bin/python
SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score; O=rescore/locked_$TAG; mkdir -p $O
$PY - "$N" "$OWN" <<'PY'
import json,sys
f={r['id']:r['hyp'] for r in map(json.loads,open(sys.argv[2]))}; rows=[json.loads(x) for x in open(sys.argv[1])]
bad=sum(r['nbest'][0][2]!=f[r['uid']] for r in rows); assert len(rows)==2782 and bad==0, (len(rows),bad); print('top1 == own final 2782/2782')
PY
PYTHONPATH=/mnt/xbabe3-external/jsr/wer-push/shared/pylib $PY $T/m5_feats.py $N /mnt/xbabe3-external/jsr/lm/models/token/tok5.p00149.trie.bin $O/m5.npy $K > /dev/null
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -gt 4000 ]; then DEVENV=(); else DEVENV=(env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16); fi
"${DEVENV[@]}" $PY $T/score_nbest.py --nbest $N --topk $K --out $O --l2r snap/m12_l2r_32k.pt --m5 $O/m5.npy --family engine --sub-grid 0.6 > $O/score.log 2>&1
$PY $T/locked_replay.py $N $O/feats.npz $O/locked.score-input.jsonl 0.6 0.6 2.762935630679131 $K
$SC $OWN $O/locked.score-input.jsonl --md-out $O/vs_own.md > /dev/null
$SC lock_h141_m12l/locked.score-input.jsonl $O/locked.score-input.jsonl --md-out $O/vs_locked.md > /dev/null
echo "== $TAG"; grep -E "A \| \*\*pooled|B \| \*\*pooled|B \| dev|Delta" $O/vs_own.md; grep Delta $O/vs_locked.md
