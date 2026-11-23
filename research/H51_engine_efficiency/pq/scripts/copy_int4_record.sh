#!/usr/bin/env bash
# copy Addendum A results into the research record (no hypotheses / score-input files, no binaries)
set -euo pipefail
S=/mnt/xbabe3-external/jsr/h51-pq/results/int4
D=/mnt/xbabe3-external/jsr/h51-pq/research/H51_engine_efficiency/pq/results/int4
mkdir -p "$D"
for f in "$S"/*.json "$S"/*.score.md "$S"/bench_*.jsonl "$S"/CODE_int4.sha256; do
  [ -e "$f" ] && cp "$f" "$D/"
done
# fold the uptime receipts into one file
for f in "$S"/*.uptime_before; do b=${f%.uptime_before}; echo "$(basename "$b"): before $(cat "$f") | after $(cat "$b.uptime_after" 2>/dev/null)"; done > "$D/bench_load_receipts.txt"
ls -la "$D" | awk '{print $5, $9}'
