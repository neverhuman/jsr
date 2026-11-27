#!/bin/bash
# engine_fullS.sh RUN TAG : wait for runs/RUN done -> snapshot -> offline CV (weights) -> .utfr export -> native engine full S (xbabe0, 24 niced shards) -> pinned score vs engine lock
set -e; H=/mnt/xbabe3-external/jsr/wer-push/H68; cd $H; R=$1; G=$2
PY=/home/ubuntu/jsr/.venv/bin/python; T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools; SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score
until grep -q '"done"' runs/$R.log; do grep -q Traceback runs/$R.log && { echo "$R crashed"; exit 1; }; sleep 60; done
[ -f snap/$G.pt ] || cp runs/$R/latest.pt snap/$G.pt
O=rescore/kd/h141_$G
[ -f $O/summary.json ] || $PY $T/score_nbest.py --nbest nbest_native/h141_final_b16.nbest.jsonl --m5 nbest_native/h141_final_b16.m5.npy --family engine --sub-grid 0,0.2,0.4,0.6,0.8 --out $O --l2r snap/$G.pt >/dev/null
read W S < <(python3 -c "import json;d=json.load(open('$O/summary.json'));a,b=d['cv_weights']['fold0_tuned'],d['cv_weights']['fold1_tuned'];print(a[0],a[3]) if a==b else print(round((a[0]+b[0])/2,2),round((a[3]+b[3])/2,2))")
echo "$G offline CV edits $(python3 -c "import json;print(json.load(open('$O/summary.json'))['cv_edits'])") weights w=$W s=$S"
PYTHONPATH=/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY $T/export_utfr.py snap/$G.pt lock_kd12l24/$G.utfr nbest_native/h141_final_b16.nbest.jsonl 160 | cut -c1-200
rsync -a lock_kd12l24/$G.utfr xbabe0:wer-push-claude/lock/
ssh -o BatchMode=yes xbabe0 "cd ~/wer-push-claude/lock && for k in \$(seq -w 0 23); do nice -n 10 ./ultra-engine-commit --model model --commit-chunk 16 --tail-pad-ms 320 --commit-beam 16 --commit-lm final.ulm --lm-scale 0.8 --lodr-scale 0.6 --threads 1 --precision int8 --rescore-skip-margin 2.762935630679131 --rescore-l2r $G.utfr --rescore-weight $W --rescore-m5-sub $S --manifest s_shards/s\$k.npz --out fullS/${G}_\$k.jsonl >/dev/null 2>fullS/${G}_\$k.err & done; wait"
rsync -a xbabe0:wer-push-claude/lock/fullS/ engine_fullS/
python3 - $G <<'PY'
import json,glob,sys
g=sys.argv[1]; R={r['uid']:r for r in map(json.loads,open('/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl'))}
rows=[x for f in sorted(glob.glob(f'/mnt/xbabe3-external/jsr/wer-push/H68/engine_fullS/{g}_*.jsonl')) for x in map(json.loads,filter(str.strip,open(f))) if 'uid' in x]
assert len(rows)==2782,len(rows)
with open(f'/mnt/xbabe3-external/jsr/wer-push/H68/engine_fullS/{g}.score-input.jsonl','w') as fh:
    for x in rows:
        r=R[x['uid']]; fh.write(json.dumps({"id":x['uid'],"speaker":str(r['speaker']),"dataset":r['subset'],"ref":r['ref'],"hyp":x['committed'],"duration_s":r['samples']/16000})+"\n")
PY
for B in lock t2w8; do $SC engine_fullS/$B.score-input.jsonl engine_fullS/$G.score-input.jsonl --md-out engine_fullS/${G}_vs_$B.md >/dev/null; echo "$G (w $W s $S) engine S: $(grep -E 'B \| \*\*pooled' engine_fullS/${G}_vs_$B.md | cut -c15-24) vs engine $B: $(grep -oE 'B - A\): [-+0-9.]+ points.*\]' engine_fullS/${G}_vs_$B.md | sed 's/(.*relative), //')"; done
grep valid_nll runs/$R.log | tail -1 | cut -c1-110; TZ=America/Denver date +%H:%M
