#!/bin/bash
# eval_l2r_h141.sh RUN TAG : wait for runs/RUN done, snapshot, CV + locked-cfg drop-in on H141 B16 vs locked
cd /mnt/xbabe3-external/jsr/wer-push/H68; R=$1; G=$2
until grep -q '"done"' runs/$R.log; do grep -q Traceback runs/$R.log && { echo "$R crashed"; exit 1; }; sleep 60; done
cp runs/$R/latest.pt snap/$G.pt
PY=/home/ubuntu/jsr/.venv/bin/python; T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools; SC=/mnt/xbabe3-external/jsr/releases/v3.1.0-fresh-download/ultra-score
O=rescore/kd/h141_$G
$PY $T/score_nbest.py --nbest nbest_native/h141_final_b16.nbest.jsonl --m5 nbest_native/h141_final_b16.m5.npy --family engine --sub-grid 0,0.2,0.4,0.6,0.8 --out $O --l2r snap/$G.pt | tail -1 | cut -c1-260
$PY $T/locked_replay.py nbest_native/h141_final_b16.nbest.jsonl $O/feats.npz $O/lockedcfg.score-input.jsonl 0.6 0.6 2.762935630679131 16 >/dev/null
for f in cv lockedcfg; do $SC lock_h141_m12l/locked.score-input.jsonl $O/$f.score-input.jsonl --md-out $O/${f}_vs_locked.md >/dev/null; echo "$G $f: $(grep -E 'B \| \*\*pooled' $O/${f}_vs_locked.md | cut -c1-28) $(grep -oE 'B - A\): [-+0-9.]+ points.*\]' $O/${f}_vs_locked.md | sed 's/(.*relative), //')"; done
$SC rescore/kd/h141_kd12l24/lockedcfg.score-input.jsonl $O/lockedcfg.score-input.jsonl --md-out $O/vs_kd24.md >/dev/null; echo "$G lockedcfg vs KD24 drop-in: $(grep -oE 'B - A\): [-+0-9.]+ points.*\]' $O/vs_kd24.md | sed 's/(.*relative), //')"
grep valid_nll runs/$R.log | tail -1 | cut -c1-120; TZ=America/Denver date +%H:%M
