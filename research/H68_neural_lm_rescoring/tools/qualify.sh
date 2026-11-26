#!/bin/bash
# H68 x H104 qualification: qualify.sh TAG L2R_CKPT [R2L_CKPT]
# 1) engine-family 2-fold speaker CV on the A2 final B16 n-best with native M5, pinned ultra-score vs H104,
#    hard-5/rest-31; 2) INT8 .utfr export + parity; 3) ABAB x2 one-core commit CPU on the 147-utt panel
#    (A2 INT8 weights, shipped commit settings) base vs rescored at the CV weights. Results -> qual/TAG/.
set -euo pipefail
TAG=$1; L2R=$2; R2L=${3:-}
H=/mnt/xbabe3-external/jsr/wer-push/H68; W=/mnt/xbabe3-external/jsr/wer-push
TOOLS=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools
PY=/home/ubuntu/jsr/.venv/bin/python; SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score
O=$H/qual/$TAG; mkdir -p $O; cd $H
R2LARG=(); [ -n "$R2L" ] && R2LARG=(--r2l "$R2L")
$PY $TOOLS/score_nbest.py --nbest nbest_native/a2_final_b16.nbest.jsonl --out $O/cv --l2r "$L2R" "${R2LARG[@]}" \
  --m5 nbest_native/a2_final_b16.m5native.npy --family engine --sub-grid 0,0.2,0.4,0.6,0.8 > $O/cv.log 2>&1
$SC $W/H104/dev-s-final/score-input.jsonl $O/cv/cv.score-input.jsonl --md-out $O/vs_h104.md > $O/vs_h104.json
$PY - "$O" <<'PY'
import json,sys
sys.path.insert(0,'/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra')
from ultra.wer import normalise, edit_counts
O=sys.argv[1]; H5={'1650','1585','7697','5543','2506'}; e={1:0,0:0}; n={1:0,0:0}
for r in map(json.loads,open(f'{O}/cv/cv.score-input.jsonl')):
    k=int(r['speaker'] in H5); rw=normalise(r['ref']); e[k]+=sum(edit_counts(rw,normalise(r['hyp']))); n[k]+=len(rw)
json.dump({'hard5':100*e[1]/n[1],'rest31':100*e[0]/n[0]},open(f'{O}/split.json','w'))
PY
read WT MS < <($PY -c "import json;w=json.load(open('$O/cv/summary.json'))['full_tuned_weights'];print(w[0],w[3])")
EXP=(); for d in l2r r2l; do
  ck=$L2R; [ $d = r2l ] && ck=$R2L; [ -z "$ck" ] && continue
  PYTHONPATH=/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    $PY $TOOLS/export_utfr.py "$ck" $O/$d.utfr nbest_native/a2_final_b16.nbest.jsonl 160 > $O/export_$d.json
  EXP+=(--rescore-$d $O/$d.utfr)
done
B=$H/cpu/ultra-engine-commit.h68k; M=$W/H104/ema-int8
COMMON=(--model $M --manifest panel147.npz --commit-chunk 16 --tail-pad-ms 320 --commit-beam 16 --commit-lm $W/shared/isosyll/isosyll-model/lm/final.ulm --lm-scale 0.8 --lodr-scale 0.6 --threads 1 --precision int8)
CORE=${CORE:-100}
for i in 1 2; do
  /usr/bin/time -f "base$i %U" taskset -c $CORE $B "${COMMON[@]}" --out $O/base$i.jsonl >/dev/null 2>>$O/time.log
  /usr/bin/time -f "resc$i %U" taskset -c $CORE $B "${COMMON[@]}" "${EXP[@]}" --rescore-weight $WT --rescore-m5-sub $MS --rescore-r2l-top-k ${R2LK:-4} --out $O/resc$i.jsonl >/dev/null 2>>$O/time.log
done
$PY - "$O" <<'PY'
import json,sys,re
O=sys.argv[1]
def rtf(f):
    r=[json.loads(x) for x in open(f'{O}/{f}.jsonl') if x.strip()]; r=[x for x in r if 'uid' in x]
    return sum(x['commit_wall_ms'] for x in r)/1000/sum(x['audio_s'] for x in r)
t=dict(l.split()[0:2] for l in open(f'{O}/time.log') if re.match(r'^(base|resc)\d ',l))
b=(rtf('base1')+rtf('base2'))/2; c=(rtf('resc1')+rtf('resc2'))/2
ub=(float(t['base1'])+float(t['base2']))/2; uc=(float(t['resc1'])+float(t['resc2']))/2
s=json.load(open(f'{O}/cv/summary.json')); sp=json.load(open(f'{O}/split.json'))
md=open(f'{O}/vs_h104.md').read(); pooled=re.findall(r'\| B \| \*\*pooled\*\* \| ([0-9.]+)',md); delta=re.findall(r'Delta macro WER.*',md)
res={'cv_edits':s['cv_edits'],'top1_edits':s['top1_edits'],'weights':s['cv_weights'],'pooled_B':pooled,'delta':delta,**sp,
     'commit_rtf_base':round(b,5),'commit_rtf_resc':round(c,5),'commit_cpu_pct':round(100*(c/b-1),1),'process_user_cpu_pct':round(100*(uc/ub-1),1)}
json.dump(res,open(f'{O}/qual.json','w'),indent=1); print(json.dumps(res))
PY
