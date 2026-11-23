#!/usr/bin/env bash
# one status line for large tier A labelling on x6 (v2: monitored PIDs in large_stat.pids)
cd ~/jsr-data/librilight/shards/raw/librilight.large || exit 1
r=$(ls *.ready 2>/dev/null | wc -l); l=$(ls *.labeled 2>/dev/null | wc -l)
h=$(python3 - <<'PY'
import glob, json
s = 0.0
for f in glob.glob("*.labeled"):
    try:
        s += json.load(open(f))["stats"]["audio_s"] or 0
    except Exception:
        pass
print(round(s / 3600, 1))
PY
)
g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
P=$(cat ~/jsr-data/h52-labeler/large_stat.pids 2>/dev/null); n=0; a=0
for p in $P; do n=$((n+1)); [ -e /proc/$p ] && a=$((a+1)); done
st=OK; [ "$a" -lt "$n" ] && st="DOWN($a/$n)"
fr=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
echo "STAT ready=$r labeled=$l labeled_h=$h gpu=$g alive=$st x6_free=${fr}GB"
grep -h -E "collect_error|dispatch_error|Traceback" ~/jsr-data/h52-labeler/logs/drive_large.log 2>/dev/null | tail -2 | cut -c1-200 | sed 's/^/ERR /'
