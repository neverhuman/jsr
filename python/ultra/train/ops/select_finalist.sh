#!/usr/bin/env bash
# select_finalist.sh <label> <local_dir_with_snapshots> <final_checkpoint.pt>
# B2/C1: score the final checkpoint and the averages of the last 3 and last 5 eval-only snapshots
# on the DEV selection panel (origin/main docs/ultra/data/dev_selection_panel.json; dev-clean and
# dev-other separately, pooled computed from S/D/I) and on first-400 dev-clean (continuity).
# H0 CPU cores 12-15 (MKL torch); the caller runs it under a fleet guard.
set -uo pipefail
label=$1 D=$2 FINAL=$3
REPO=/home/ubuntu/jsr/.agent/worktrees/trainer
PANEL=$D/dev_selection_panel.json
git -C /home/ubuntu/jsr fetch -q origin
git -C /home/ubuntu/jsr show origin/main:docs/ultra/data/dev_selection_panel.json > $PANEL || { echo "panel not on origin/main"; exit 2; }
cd $REPO/python/ultra
E="env CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu taskset -c 12-15,76-79 nice -n 10 $REPO/.venv/bin/python"
snaps=$(ls $D/model-*.pt | sort | tail -5)
$E -m train.average --out $D/avg-last5.pt $snaps
$E -m train.average --out $D/avg-last3.pt $(echo $snaps | tr ' ' '\n' | tail -3)
OUT=$D/selection.jsonl
tag() { python3 -c "import sys,json; r=json.loads(sys.stdin.read()); r.update(candidate='$1', panel='$2'); print(json.dumps(r))"; }
for ck in $FINAL $D/avg-last3.pt $D/avg-last5.pt; do
  n=$(basename $ck .pt)
  for sp in dev-clean dev-other; do
    $E -m train.evaluate --run-dir $D --checkpoint $ck --panel $PANEL --panel-split $sp --threads 6 --dump $D/sel_panel_${sp}_$n.jsonl 2>/dev/null | grep '^{' | tag $n selection:$sp >> $OUT
  done
  $E -m train.evaluate --run-dir $D --checkpoint $ck --split dev-clean --limit 400 --threads 6 --dump $D/sel_first400_$n.jsonl 2>/dev/null | grep '^{' | tag $n first400 >> $OUT
done
echo SELECTION_DONE >> $D/selection.log
