#!/usr/bin/env bash
# P04 timing suite in the nemotron core block (physical 16-31, siblings 80-95; AGENT_CHAT CORE PARTITION).
# Holds the nemotron bench lease for the whole suite; each run records loadavg before/after.
# Usage: nemotron_suite.sh headline|batch|other
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/home/ubuntu/jsr/.venv/bin/python
RUN="$PY $ROOT/python/ultra/adapters/nemotron_run.py"
RUNS=/mnt/xbabe3-external/jsr/runs/nemotron
LOGS=/mnt/xbabe3-external/jsr/logs/nemotron
LEASE=/mnt/xbabe3-external/jsr/BENCH_LEASE.nemotron
mkdir -p "$RUNS" "$LOGS"

headline() {
  # Uncontended-as-possible c1 cost: every 4th test-clean utterance (655 utts), one process per
  # operating point, on different CCDs (cores 16 and 24), SMT siblings idle.
  $RUN --split test-clean --shard 0/4 --chunk-ms 160 --cpus 16 --tag headline --out "$RUNS/headline.jsonl" > "$LOGS/hl_160.log" 2>&1 &
  $RUN --split test-clean --shard 0/4 --chunk-ms 560 --cpus 24 --tag headline --out "$RUNS/headline.jsonl" > "$LOGS/hl_560.log" 2>&1 &
  wait
}

batch() {
  # Unpaced streams-per-core screen on DEV (dev-clean, 128 utts), 1 core each, one process per CCX.
  local cores=(16 20 24 28) k=0
  for spec in 160:1 160:8 160:32 560:1 160:16 160:64 560:8 560:32; do
    local cm=${spec%%:*} c=${spec##*:} core=${cores[$((k % 4))]}
    $RUN --split dev-clean --limit 128 --chunk-ms "$cm" --concurrency "$c" --cpus "$core" --tag batch \
      --out "$RUNS/batch.jsonl" > "$LOGS/batch_${cm}_c${c}.log" 2>&1 &
    k=$((k + 1))
    if (( k % 4 == 0 )); then wait; fi
  done
  wait
}

other() {
  # test-other WER (+ CPU receipts), 4 shards per operating point, 8 processes on cores 16-31.
  local k=0
  for cm in 160 560; do
    for s in 0 1 2 3; do
      $RUN --split test-other --shard "$s/4" --chunk-ms "$cm" --cpus $((16 + 2 * k)) \
        --out "$RUNS/results.jsonl" > "$LOGS/to_${cm}_s$s.log" 2>&1 &
      k=$((k + 1))
    done
  done
  wait
}

uptime | tee -a "$LOGS/suite_uptime.log"
exec 9>>"$LEASE"
flock 9
"$1"
uptime | tee -a "$LOGS/suite_uptime.log"
