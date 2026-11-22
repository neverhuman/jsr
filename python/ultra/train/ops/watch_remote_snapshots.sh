#!/usr/bin/env bash
# H0: for a run on another host, evaluate each new 5k snapshot there with snapshot_eval.sh
# (panel greedy + beam 4, snapshot and last-3 average) and post the rows to AGENT_CHAT.
#   watch_remote_snapshots.sh <label> <host> <run_dir> <jsr_data> <python> <ultra_dir> <panel> <first_step> <slotA> <slotB> <slotC> <slotD>
# Stops when the remote trainer has exited and no newer snapshot exists.
set -uo pipefail
LABEL=$1 HOST=$2 RUN=$3 DATA=$4 PY=$5 W=$6 PANEL=$7 S=$8
shift 8
SLOTS="$*"
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
while :; do
  f=$(printf 'model-%08d' "$S")
  if ssh -n "$HOST" "cd $RUN && test -f $f.sha256 && [ \"\$(sha256sum $f.pt | cut -c1-64)\" = \"\$(cut -c1-64 $f.sha256)\" ]"; then
    ssh -n "$HOST" "JSR_DATA=$DATA $W/train/ops/snapshot_eval.sh $PY $W $RUN $S $PANEL $SLOTS" || true
    rows=$(ssh -n "$HOST" "cd $RUN/snap_eval/$S && for j in snap_dev-clean snap_dev-other avg3_dev-clean avg3_dev-other; do [ -s \$j.json ] && python3 -c \"import json; r=json.load(open('\$j.json')); print('| \$j | ' + str(r['wer']) + ' | ' + str(r['beam_wer']['wer']) + ' |')\"; done")
    printf '\n## %s | trainer-lane | %s step %s (auto, panel 730): greedy / beam-4 WER\n| job | greedy | beam 4 |\n|---|---:|---:|\n%s\n' \
      "$(date -u +%Y-%m-%dT%H:%MZ)" "$LABEL" "$S" "$rows" >> "$CHAT"
    S=$((S + 5000))
    continue
  fi
  if ! ssh -n "$HOST" "P=\$(cat $RUN/PID); ps -p \$P -o args= | grep -q train.trainer"; then
    break
  fi
  sleep 300
done
