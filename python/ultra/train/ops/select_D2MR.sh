#!/usr/bin/env bash
# After A3: pull D2MR's last 5 eval-only snapshots to the H0 copy, wait for the selection panel on
# main, then run select_finalist (guarded CPU job on cores 12-15) and post the rows.
set -uo pipefail
H=/mnt/xbabe3-external/jsr/runs/train/D2MR_S_MR_960h_copy
R=/home/ubuntu/jsr-data/runs/train/D2MR_S_MR_960h
T=/mnt/xbabe3-external/jsr/runs/train
until [ -f $H/FINISH_DONE ]; do sleep 120; done
for s in $(ssh -n xbabe2 "ls $R | grep -E '^model-[0-9]+\.pt$' | sort | tail -5"); do
  ionice -c3 nice -n 10 rsync -t xbabe2:$R/$s xbabe2:$R/${s%.pt}.sha256 $H/
  [ "$(sha256sum $H/$s | cut -d' ' -f1)" = "$(cat $H/${s%.pt}.sha256)" ] || { echo "BAD $s"; exit 1; }
done
until git -C /home/ubuntu/jsr fetch -q origin && git -C /home/ubuntu/jsr cat-file -e origin/main:docs/ultra/data/dev_selection_panel.json 2>/dev/null; do sleep 300; done
bash /mnt/xbabe3-external/jsr/logs/trainer/start_guarded_cpu.sh /mnt/xbabe3-external/jsr/logs/trainer/select_D2MR $T/select_finalist.sh D2MR $H $H/ckpt-00050000.pt
until grep -q SELECTION_DONE $H/selection.log 2>/dev/null; do sleep 60; done
python3 $T/post_selection.py D2MR $H/selection.jsonl
