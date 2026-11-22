#!/usr/bin/env bash
# H0: as D2 writes snapshots <steps...> (default 60000 65000), copy each to xbabe0 (sha256 checked),
# run xbabe0_d2_step.sh there (full-dev greedy+beam-4 of the snapshot; last-3 average on the panel
# and full dev with beam 4) and post the rows to AGENT_CHAT. Waits for earlier xbabe0 jobs first.
set -uo pipefail
D2=/mnt/xbabe3-external/jsr/runs/train/D2_S_960h
X=jsr-data/runs/train/D2_S_960h
OPS=jsr/.agent/worktrees/trainer/python/ultra/train/ops
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(60000 65000)
until ssh -n xbabe0 "test -f jsr-data/runs/train/x0_stack/DONE"; do sleep 120; done
for S in "${steps[@]}"; do
  f=$(printf 'model-%08d' "$S")
  until [ -f "$D2/$f.sha256" ] && [ -f "$D2/$f.pt" ] && \
        [ "$(sha256sum "$D2/$f.pt" | cut -c1-64)" = "$(cut -c1-64 "$D2/$f.sha256")" ]; do sleep 300; done
  for k in $((S - 10000)) $((S - 5000)) "$S"; do
    g=$(printf 'model-%08d' "$k")
    ionice -c3 nice -n 10 rsync -t "$D2/$g.pt" "$D2/$g.sha256" "xbabe0:$X/" || exit 1
    ssh -n xbabe0 "cd $X && [ \"\$(sha256sum $g.pt | cut -c1-64)\" = \"\$(cut -c1-64 $g.sha256)\" ]" \
      || { echo "sha mismatch $g on xbabe0"; exit 1; }
  done
  ssh -n xbabe0 "$OPS/xbabe0_d2_step.sh $S" || { echo "step $S failed"; exit 1; }
  rows=$(ssh -n xbabe0 "cd $X/x0_step$S && for j in *.json; do python3 -c \"import json,sys; r=json.load(open('\$j')); b=r.get('beam_wer',{}).get('wer'); print(f'| \$j | {r[\\\"wer\\\"]} | {b} |')\"; done")
  printf '\n## %s | trainer-lane | D2 step %s on xbabe0 (auto): greedy / beam-4 WER, snapshot and last-3 average\n| job | greedy WER | beam-4 WER |\n|---|---:|---:|\n%s\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" "$S" "$rows" >> "$CHAT"
done
