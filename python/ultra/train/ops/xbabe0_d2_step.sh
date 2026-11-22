#!/usr/bin/env bash
# xbabe0: for D2 snapshot <step>, average the last 3 snapshots (step-10k, step-5k, step), then
# full-dev greedy+beam-4 of the snapshot, and beam-4 of the average on the 730 panel and full dev.
# Cores 24-47 (4 threads per job); cores 0-23 are left to the decode sweep.
#   xbabe0_d2_step.sh <step>
set -uo pipefail
S=$1
R=$HOME/jsr-data/runs/train/D2_S_960h
PANEL=$HOME/jsr/.agent/worktrees/trainer/docs/ultra/data/dev_selection_panel.json
PY=$HOME/jsr-data/venvs/train/bin/python
OPS=$HOME/jsr/.agent/worktrees/trainer/python/ultra/train/ops
O=$R/x0_step$S
mkdir -p "$O"
m() { printf '%s/model-%08d.pt' "$R" "$1"; }
avg=$R/avg-last3-$S.pt
(cd "$HOME/jsr/.agent/worktrees/trainer/python/ultra" && JSR_DATA=$HOME/jsr-data taskset -c 24-31 nice -n 10 \
  "$PY" -m train.average --out "$avg" "$(m $((S - 10000)))" "$(m $((S - 5000)))" "$(m "$S")") > "$O/avg.log" 2>&1 \
  || { echo "average failed"; exit 1; }
cat > "$O/jobs.txt" <<JOBS
24-27 dev-clean_${S}_beam4 $R $(m "$S") --split dev-clean --limit-all --beam 4
28-31 dev-other_${S}_beam4 $R $(m "$S") --split dev-other --limit-all --beam 4
32-35 panel-clean_avg3_${S}_beam4 $R $avg --panel $PANEL --panel-split dev-clean --beam 4
36-39 panel-other_avg3_${S}_beam4 $R $avg --panel $PANEL --panel-split dev-other --beam 4
40-43 dev-clean_avg3_${S}_beam4 $R $avg --split dev-clean --limit-all --beam 4
44-47 dev-other_avg3_${S}_beam4 $R $avg --split dev-other --limit-all --beam 4
JOBS
THREADS=4 "$OPS/xbabe0_eval_jobs.sh" "$O/jobs.txt" "$O"
