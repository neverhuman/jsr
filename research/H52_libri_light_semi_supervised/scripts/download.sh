#!/usr/bin/env bash
# Resumable, rate-limited download supervisor (H52). Never truncates: always curl -C -.
# usage: download.sh <url> <out> <expected_bytes> [rate]
set -uo pipefail
url=$1; out=$2; want=$3; rate=${4:-40M}
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "START url=$url out=$out want=$want rate=$rate pid=$$"
while :; do
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$have" -ge "$want" ]; then break; fi
  # current rate override file (lets me retune without restarting the supervisor)
  [ -s "$out.rate" ] && rate=$(cat "$out.rate")
  nice -n 19 ionice -c2 -n7 curl -fsSL --limit-rate "$rate" -C - \
       --speed-limit 102400 --speed-time 180 --connect-timeout 30 \
       -o "$out" "$url"
  rc=$?
  log "curl exit $rc at $(stat -c %s "$out" 2>/dev/null) bytes (rate $rate)"
  [ "$rc" -ne 0 ] && sleep 30
done
log "DONE size=$(stat -c %s "$out")"
