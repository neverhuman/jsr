#!/usr/bin/env bash
# H0 driver: every round, copy new 5k snapshots of D3MRc (xbabe1), H13a (xbabe2) and D4M (xbabe4)
# to xbabe0 (sha256 checked; relayed through H0's data drive), then on xbabe0 build backfill jobs
# (make_backfill_jobs.sh) and run them on cores 0-63 (eval_queue.sh, 16 slots x 4 threads).
# Completed jobs are skipped, so each round only evaluates new snapshots.
set -uo pipefail
RELAY=/mnt/xbabe3-external/jsr/tmp/relay
X=jsr-data/runs/train
OPS=jsr/.agent/worktrees/trainer/python/ultra/train/ops
W=jsr/.agent/worktrees/trainer/python/ultra
PY=jsr-data/venvs/train/bin/python
runs=("D3MRc xbabe1 /mnt/amphora-backup/jsr-data/runs/train/D3MRc_S_MR_I8S_960h D3MRc_S_MR_I8S_960h"
      "H13a xbabe2 /home/ubuntu/jsr-data/runs/train/H13a_S_spm512_960h H13a_S_spm512_960h"
      "D4M xbabe4 /Users/agent/jsr-train/data/runs/train/D4M_M_MR_960h D4M_M_MR_960h")
slots="0-3 4-7 8-11 12-15 16-19 20-23 24-27 28-31 32-35 36-39 40-43 44-47 48-51 52-55 56-59 60-63"
mkdir -p "$RELAY"
while :; do
  for r in "${runs[@]}"; do
    read -r label host rdir name <<< "$r"
    ssh -n xbabe0 "mkdir -p $X/$name"
    have=$(ssh -n xbabe0 "ls $X/$name 2>/dev/null")
    for f in $(ssh -n "$host" "ls $rdir | grep -E '^model-[0-9]+\.sha256$'"); do
      m=${f%.sha256}
      grep -qx "$f" <<< "$have" && continue
      rsync -t "$host:$rdir/$m.pt" "$host:$rdir/$f" "$RELAY/" || continue
      [ "$(sha256sum "$RELAY/$m.pt" | cut -c1-64)" = "$(cut -c1-64 "$RELAY/$f")" ] || { echo "bad sha $label $m"; continue; }
      rsync -t "$RELAY/$m.pt" "xbabe0:$X/$name/" && rsync -t "$RELAY/$f" "xbabe0:$X/$name/"
      rm -f "$RELAY/$m.pt" "$RELAY/$f"
      echo "$(date -u +%FT%TZ) copied $label $m"
    done
  done
  ssh -n xbabe0 "export JSR_DATA=\$HOME/jsr-data; (for r in D3MRc:D3MRc_S_MR_I8S_960h H13a:H13a_S_spm512_960h D4M:D4M_M_MR_960h; do AVG_CREATE=1 taskset -c 63 nice -n 10 $OPS/make_backfill_jobs.sh \$HOME/$PY \$HOME/$W \$HOME/$X/\${r#*:} \${r%%:*} > /dev/null; done) & : > $X/backfill_jobs.txt; for r in D3MRc:D3MRc_S_MR_I8S_960h H13a:H13a_S_spm512_960h D4M:D4M_M_MR_960h; do AVG_CREATE=0 $OPS/make_backfill_jobs.sh \$HOME/$PY \$HOME/$W \$HOME/$X/\${r#*:} \${r%%:*} >> $X/backfill_jobs.txt; done; $OPS/eval_queue.sh \$HOME/$PY \$HOME/$W $X/backfill_jobs.txt \$HOME/$X/backfill $slots; wait"
  echo "$(date -u +%FT%TZ) round done"
  sleep 600
done
