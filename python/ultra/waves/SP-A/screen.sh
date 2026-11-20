#!/usr/bin/env bash
# SP-A throughput screen: unpaced, one pinned core, single vs batch over N streams.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
split=${SPLIT:-train-clean-100}; limit=${LIMIT:-400}; core=${CORE:-62}; prec=${PREC:-int8}
out=${OUT:-/mnt/xbabe3-external/jsr/runs/sp_a/screen_$(date -u +%Y%m%dT%H%M%SZ).jsonl}
run() { PYTHONPATH=python/ultra taskset -c "$core" .venv/bin/python -m ultra.sp_a --split "$split" --limit "$limit" --precision "$prec" --out "$out" "$@" > /dev/null; }
run --streams 1 --decode single
for n in 8 32 64 128; do run --streams "$n" --decode batch; done
run --streams 32 --decode single
echo "$out"
