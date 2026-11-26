#!/bin/bash
# H68 generic qualification with margin-skip operating points.
# qual2.sh TAG MODEL_DIR NBEST M5NPY BASE_FINAL_SCOREINPUT L2R_CKPT [R2L_CKPT]
# -> qual/TAG/{cv/, vs_base.md, split.json, *.utfr, time.log, gates.json, gate15/25.*, summary.json}
set -euo pipefail
TAG=$1; MODEL=$2; NBEST=$3; M5=$4; BASEF=$5; L2R=$6; R2L=${7:-}
H=/mnt/xbabe3-external/jsr/wer-push/H68; W=/mnt/xbabe3-external/jsr/wer-push
T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools
PY=/home/ubuntu/jsr/.venv/bin/python; SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score
O=$H/qual/$TAG; mkdir -p $O; cd $H; CORE=${CORE:-101}
R2LARG=(); [ -n "$R2L" ] && R2LARG=(--r2l "$R2L")
$PY $T/score_nbest.py --nbest $NBEST --out $O/cv --l2r "$L2R" "${R2LARG[@]}" --m5 $M5 --family engine --sub-grid 0,0.2,0.4,0.6,0.8 > $O/cv.log 2>&1
$SC $BASEF $O/cv/cv.score-input.jsonl --md-out $O/vs_base.md > /dev/null
read WT MS < <($PY -c "import json;w=json.load(open('$O/cv/summary.json'))['full_tuned_weights'];print(w[0],w[3])")
EXP=(); for d in l2r r2l; do ck=$L2R; [ $d = r2l ] && ck=$R2L; [ -z "$ck" ] && continue
  PYTHONPATH=/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    $PY $T/export_utfr.py "$ck" $O/$d.utfr $NBEST 160 > $O/export_$d.json; EXP+=(--rescore-$d $O/$d.utfr); done
B=$H/cpu/ultra-engine-commit.h68g
COMMON=(--model $MODEL --manifest panel147.npz --commit-chunk 16 --tail-pad-ms 320 --commit-beam 16 --commit-lm $W/shared/isosyll/isosyll-model/lm/final.ulm --lm-scale 0.8 --lodr-scale 0.6 --threads 1 --precision int8)
RS=("${EXP[@]}" --rescore-weight $WT --rescore-m5-sub $MS)
run() { /usr/bin/time -f "$1 %U" taskset -c $CORE $B "${COMMON[@]}" "${@:2}" --out $O/$1.jsonl >/dev/null 2>>$O/time.log; }
for i in 1 2; do run base$i; run resc$i "${RS[@]}"; done
$PY $T/gate_sim.py $NBEST $O/cv $O 15 25 > $O/gate_sim.log
for b in 15 25; do th=$($PY -c "import json;print(json.load(open('$O/gates.json'))['$b']['engine_threshold'])")
  for i in 1 2; do run gbase${b}_$i; run g${b}_$i "${RS[@]}" --rescore-skip-margin $th; done; done
for v in cv/cv gate15 gate25; do $SC $BASEF $O/$v.score-input.jsonl --md-out $O/$(basename $v).vs_base.md > /dev/null; done
$PY - "$O" "$BASEF" <<'PY'
import json,sys,re
sys.path.insert(0,'/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra')
from ultra.wer import normalise, edit_counts
O,BF=sys.argv[1:3]; H5={'1650','1585','7697','5543','2506'}
def split(p):
    e={1:0,0:0}; n={1:0,0:0}
    for r in map(json.loads,open(p)):
        k=int(r['speaker'] in H5); rw=normalise(r['ref']); e[k]+=sum(edit_counts(rw,normalise(r['hyp']))); n[k]+=len(rw)
    return round(100*e[1]/n[1],2), round(100*e[0]/n[0],2)
def rtf(f):
    r=[json.loads(x) for x in open(f'{O}/{f}.jsonl') if x.strip()]; r=[x for x in r if 'uid' in x]
    return sum(x['commit_wall_ms'] for x in r)/1000/sum(x['audio_s'] for x in r)
def md(f):
    t=open(f'{O}/{f}').read()
    return re.findall(r'\| B \| \*\*pooled\*\* \| ([0-9.]+)',t)[0], re.findall(r'Delta macro WER \(B - A\): ([-+0-9.]+) points.*bounds \[([-+0-9.]+), ([-+0-9.]+)\]',t)[0]
ut=dict(l.split()[:2] for l in open(f'{O}/time.log'))
def pct(a,b,n=2): 
    x=sum(rtf(f'{a}{i}') for i in range(1,n+1)); y=sum(rtf(f'{b}{i}') for i in range(1,n+1)); return round(100*(y/x-1),1)
def upct(a,b): return round(100*(sum(float(ut[f'{b}{i}']) for i in (1,2))/sum(float(ut[f'{a}{i}']) for i in (1,2))-1),1)
g=json.load(open(f'{O}/gates.json'))
res={'base':{'split':split(BF)},
     'full':{'wer':md('cv.vs_base.md'),'split':split(f'{O}/cv/cv.score-input.jsonl'),'commit_cpu_pct':pct('base','resc'),'proc_cpu_pct':upct('base','resc')}}
for b in ('15','25'):
    res['gate'+b]={'wer':md(f'gate{b}.vs_base.md'),'split':split(f'{O}/gate{b}.score-input.jsonl'),'commit_cpu_pct':pct(f'gbase{b}_',f'g{b}_'),
                   'proc_cpu_pct':upct(f'gbase{b}_',f'g{b}_'),'threshold':g[b]['engine_threshold'],'rescored_frac':g[b]['rescored_utt_frac']}
res['cv_weights']=json.load(open(f'{O}/cv/summary.json'))['cv_weights']
json.dump(res,open(f'{O}/summary.json','w'),indent=1); print(json.dumps(res))
PY
