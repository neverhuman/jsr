#!/usr/bin/env bash
# h45-wins: poll the pad-0 INT8 parity outputs (x9 partial + xbabe2 pad0_rest_* / pad0_x2b_* + xbabe1 pad0_back_*) until
# all 5,567 dev utterances have a row; then stop xbabe1's pad-0 group (its command line checked first; xbabe2 moves on
# to the 320 ms wave by itself) and score the union with int8_parity.py.
D=/mnt/xbabe3-external/jsr/h45-wins/results/int8_parity
mkdir -p $D/x2_partial $D/x1_partial
while true; do
  scp -q 'xbabe2:jsr-data/h45-wins/out/int8/pad0_*' $D/x2_partial/ 2>/dev/null
  scp -q 'xbabe1:/mnt/amphora-backup/h45-wins/out/int8/pad0_back_*' $D/x1_partial/ 2>/dev/null
  n=$(python3 - <<'PY'
import json, glob
D = "/mnt/xbabe3-external/jsr/h45-wins/results/int8_parity"
done = set()
for p in glob.glob(f"{D}/commit_int8.pad0.*.jsonl") + glob.glob(f"{D}/x2_partial/*") + glob.glob(f"{D}/x1_partial/*"):
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "uid" in r:
            done.add(r["uid"])
print(len(done))
PY
)
  echo "$(date -u +%T) covered $n / 5567"
  [ "$n" -ge 5567 ] && break
  sleep 60
done
ssh -o BatchMode=yes -n xbabe1 'ps -o args= -p 3085055 | grep -q x1_int8.sh && kill -TERM -- -3085055 && echo "stopped xbabe1 int8 pgid 3085055"'
echo "$(date -u +%FT%TZ) full pad-0 coverage reached; xbabe1 int8 3085055 stopped if still running" >> /mnt/xbabe3-external/jsr/h45-wins/logs/pids.txt
python3 /mnt/xbabe3-external/jsr/h45-wins/bin/int8_parity.py full_pad0 $D $D/commit_int8.pad0.dev-clean.jsonl $D/commit_int8.pad0.dev-other.jsonl $D/x2_partial/* $D/x1_partial/*
