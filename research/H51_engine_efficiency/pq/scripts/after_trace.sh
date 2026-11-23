#!/usr/bin/env bash
# Merge the trace parts, ship Sigma / held rows to x9, free the parts, launch the 2-bit sweep on x9.
set -euo pipefail
cd /mnt/xbabe3-external/jsr/h51-pq
n=$(ls /dev/shm/h51-pq/trace_parts/part_*.pt | wc -l)
[ "$n" -eq 6 ] || { echo "only $n parts"; exit 1; }
nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/merge_trace.py /dev/shm/h51-pq/trace_parts /dev/shm/h51-pq/trace results/trace > logs/merge_trace.log 2>&1
ns=$(ls /dev/shm/h51-pq/trace/sigma.*.npy | wc -l); nh=$(ls /dev/shm/h51-pq/trace/held.*.npy | wc -l)
echo "sigma files $ns, held files $nh"
[ "$ns" -ge 131 ] && [ "$nh" -ge 131 ] || { echo "merge incomplete"; exit 1; }
rsync -a /dev/shm/h51-pq/trace xbabe9:h51-pq/data/
rm -f /dev/shm/h51-pq/trace_parts/part_*.pt
ssh xbabe9 'rm -f ~/h51-pq/logs/s4r2.pids; bash ~/h51-pq/scripts/launch_x9.sh s4r2 4 2 7'
