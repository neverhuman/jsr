#!/usr/bin/env bash
# xbabe0 (cores 0-31): D2 model-55000 full dev greedy + beam 4 (cores 0-15), then A3 averaging of
# model-45000/50000/55000 (last 3) on the 730-utterance DEV selection panel (cores 16-31).
# (The last-5 average, 35k-55k, was run the same way by hand.) As run 2026-11-21 on xbabe0.
set -uo pipefail
export JSR_DATA=$HOME/jsr-data
W=$HOME/jsr/.agent/worktrees/trainer/python/ultra
PY=$HOME/jsr-data/venvs/train/bin/python
R=$JSR_DATA/runs/train/D2_S_960h
O=$R/x0_eval; mkdir -p "$O"
PANEL=$HOME/jsr/.agent/worktrees/trainer/docs/ultra/data/dev_selection_panel.json
cd "$W" || exit 1
ev() { local cores=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES="" taskset -c "$cores" nice -n 10 "$PY" -m train.evaluate --run-dir "$R" \
    --threads 8 --dump "$O/$tag.jsonl" "$@" 2>"$O/$tag.err" | grep '^{' > "$O/$tag.json"; }
ev 0-7 dev-clean_55k_beam4 --checkpoint "$R/model-00055000.pt" --split dev-clean --limit-all --beam 4 &
ev 8-15 dev-other_55k_beam4 --checkpoint "$R/model-00055000.pt" --split dev-other --limit-all --beam 4 &
taskset -c 16-23 nice -n 10 "$PY" -m train.average --out "$R/avg-45-55k.pt" \
  "$R/model-00045000.pt" "$R/model-00050000.pt" "$R/model-00055000.pt" > "$O/avg.log" 2>&1
for ck in model-00055000 avg-45-55k; do
  ev 16-23 "panel-clean_$ck" --checkpoint "$R/$ck.pt" --panel "$PANEL" --panel-split dev-clean &
  pc=$!
  ev 24-31 "panel-other_$ck" --checkpoint "$R/$ck.pt" --panel "$PANEL" --panel-split dev-other
  wait "$pc"
done
wait
echo X0_D2_DONE > "$O/DONE"
