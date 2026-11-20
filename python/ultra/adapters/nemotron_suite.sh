#!/usr/bin/env bash
# P04 timing suite in the nemotron core block (physical 16-31, siblings 80-95; AGENT_CHAT CORE PARTITION).
# Holds the nemotron bench lease for the whole suite; each run records loadavg before/after.
# Usage: nemotron_suite.sh headline|batch|timing|other|quiet_then_other  (TAG=... names the headline run)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/home/ubuntu/jsr/.venv/bin/python
RUN="$PY $ROOT/python/ultra/adapters/nemotron_run.py"
RUNS=/mnt/xbabe3-external/jsr/runs/nemotron
LOGS=/mnt/xbabe3-external/jsr/logs/nemotron
LEASE=/mnt/xbabe3-external/jsr/BENCH_LEASE.nemotron
mkdir -p "$RUNS" "$LOGS"

headline() {
  # Single-stream c1 cost: every 8th test-clean utterance (328 utts), one process per operating
  # point, on different CCDs (cores 22 and 28), SMT siblings idle.
  $RUN --split test-clean --shard 0/8 --chunk-ms 160 --cpus 22 --tag "${TAG:-headline}" --out "$RUNS/headline.jsonl" > "$LOGS/hl_160_${TAG:-headline}.log" 2>&1 &
  $RUN --split test-clean --shard 0/8 --chunk-ms 560 --cpus 28 --tag "${TAG:-headline}" --out "$RUNS/headline.jsonl" > "$LOGS/hl_560_${TAG:-headline}.log" 2>&1 &
  wait
}

batch() {
  # Unpaced streams-per-core screen on DEV (dev-clean, first 128 utts), 1 core per process,
  # c in {1,8,32}; one operating point at a time on cores 20, 24, 28.
  for cm in 160 560; do
    local k=0 cores=(20 24 28)
    for c in 1 8 32; do
      $RUN --split dev-clean --limit 128 --chunk-ms "$cm" --concurrency "$c" --cpus "${cores[$k]}" --tag batch \
        --out "$RUNS/batch.jsonl" > "$LOGS/batch_${cm}_c${c}.log" 2>&1 &
      k=$((k + 1))
    done
    wait
  done
}

timing() { headline; batch; }
quiet_then_other() { TAG=headline_quiet headline; other; }

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
