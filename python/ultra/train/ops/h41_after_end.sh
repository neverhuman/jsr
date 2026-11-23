#!/usr/bin/env bash
# Wait for an H41 arm to end, then run its full-dev EMA decodes on the same GPU (h41_burst_decode.sh).
# Start it in its own session so the decode's fleet guard stops only this process group:
#   setsid nohup taskset -c <cpus> bash h41_after_end.sh <run_dir> <gpu> <cpus> > <log> 2>&1 < /dev/null &
R=$1 GPU=$2 CPUS=$3
until ! kill -0 "$(cat "$R/PID")" 2>/dev/null; do sleep 15; done
sleep 10
kill "$(cat "$R/GUARD_PID")" 2>/dev/null
grep -q '"kind": "train", "step": 8000' "$R/train.jsonl" || { echo "run did not reach 8000; no decode"; exit 1; }
bash "$(dirname "${BASH_SOURCE[0]}")/h41_burst_decode.sh" "$R" "$GPU" "$CPUS"
