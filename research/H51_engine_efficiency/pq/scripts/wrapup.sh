#!/usr/bin/env bash
# Stop every h51-pq background job by recorded PID / process group, pull the last rows, keep the
# derived data on the data drive, then free RAM scratch. Prints what it did.
set -u
W=/mnt/xbabe3-external/jsr/h51-pq
# 1. xbabe2 sweep (process group + guard), then its rows
ssh -o BatchMode=yes xbabe2 'cd ~/h51-pq && G=$(cat logs/s4r2.guard_pid); PG=$(cat logs/s4r2.pgid); for p in $(cat logs/s4r2.pids); do kill $p 2>/dev/null; done; sleep 1; for p in $(ps -eo pid=,pgid=,args= | awk -v g=$PG "\$2==g && \$3 ~ /h51-pq\/bin\/ultra-engine-eval/ {print \$1}"); do kill $p; done; kill $G 2>/dev/null; sleep 2; echo "xbabe2 left in pgid: $(ps -eo pgid | awk -v g=$PG "\$1==g" | wc -l)"; rm -rf data/variants/*'
rsync -a --exclude '*h0.jsonl' xbabe2:h51-pq/results/engine_sens/s4r2/ $W/results/engine_sens/s4r2/
# 2. H0 sync loop and H0 sweep group + guard
kill "$(cat /dev/shm/h51-pq/logs/sync_s4r2.pid)" 2>/dev/null
PG=$(cat /dev/shm/h51-pq/logs/s4r2.h0.pgid); G=$(cat /dev/shm/h51-pq/logs/s4r2.h0.guard_pid)
for p in $(cat /dev/shm/h51-pq/logs/s4r2.h0.pids); do kill $p 2>/dev/null; done
sleep 1
for p in $(ps -eo pid=,pgid=,args= | awk -v g=$PG '$2==g && $3 ~ /ultra-engine-eval/ {print $1}'); do kill $p; done
kill $G 2>/dev/null
sleep 2
echo "H0 sweep left in pgid: $(ps -eo pgid | awk -v g=$PG '$1==g' | wc -l)"
# 3. keep derived data (Sigma / held rows / fit caches) on the data drive
mkdir -p $W/derived
rsync -a /dev/shm/h51-pq/trace /dev/shm/h51-pq/fits /dev/shm/h51-pq/panel730.npz /dev/shm/h51-pq/train300.npz /dev/shm/h51-pq/panel73.npz $W/derived/ && echo "derived data copied"
mkdir -p $W/logs/run_logs && rsync -a /dev/shm/h51-pq/logs/ $W/logs/run_logs/ && cp /dev/shm/h51-pq/*.log $W/logs/run_logs/ 2>/dev/null; echo "run logs copied"
echo "scratch before cleanup: $(du -sh /dev/shm/h51-pq | cut -f1)"
