#!/usr/bin/env bash
# H52 (x6): pause large pass 2 (throttle 0) while free disk < 150 GB; restore the cap (4) once free > 200 GB.
T=$HOME/jsr-data/librilight/pass2_large_throttle; paused=0
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
while :; do
  free=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
  if [ "$paused" = 0 ] && [ "$free" -lt 150 ]; then echo 0 > $T; paused=1; log "PAUSE pass 2: free ${free} GB < 150 GB"; fi
  if [ "$paused" = 1 ] && [ "$free" -gt 200 ]; then echo 4 > $T; paused=0; log "RESUME pass 2: free ${free} GB"; fi
  sleep 60
done
