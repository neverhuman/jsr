#!/bin/bash
# h52-labeler memory watchdog: stop one of OUR process groups (by PGID) if host MemAvailable < floor; requeue its jobs.
#   memwatch.sh <pgid> <floor_mb> <inbox dir> <log>
pg=$1; floor=$2; inbox=$3; log=$4
while kill -0 -"$pg" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$floor" ]; then
    echo "$(date -u +%FT%TZ) MemAvailable ${avail} MB < ${floor} MB: stopping pgid $pg" >> "$log"
    kill -TERM -"$pg"; sleep 5
    for f in "$inbox"/*.taken; do [ -e "$f" ] && mv "$f" "${f%.taken}.go"; done
    exit 0
  fi
  sleep 10
done
