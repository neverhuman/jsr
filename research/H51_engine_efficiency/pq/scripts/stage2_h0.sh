#!/usr/bin/env bash
# After the LN-input fitters finish: fit the post-SiLU layers (4 procs), then full distortion (3 procs).
cd /mnt/xbabe3-external/jsr/h51-pq
for f in /dev/shm/h51-pq/logs/fit_s4r1.pids /dev/shm/h51-pq/logs/fit_s4r2.pids; do
  for p in $(cat $f); do while kill -0 $p 2>/dev/null; do sleep 10; done; done
done
echo "LN fitters done $(date -u +%T)"
RX='(ff[12]\.w2|conv\.pw2)$|^encoder\.stem\.out$'
: > /dev/shm/h51-pq/logs/fit_post.pids
for i in 0 1 2 3; do
  OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/fit_only.py 4 2 "$RX" $i/4 > /dev/shm/h51-pq/logs/fit_post_$i.log 2>&1 &
  echo $! >> /dev/shm/h51-pq/logs/fit_post.pids
done
wait
echo "post-SiLU fits done $(date -u +%T)"
mkdir -p results/distortion
: > /dev/shm/h51-pq/logs/dist.pids
for i in 0 1 2; do
  OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/distortion_all.py 4 2 $i/3 results/distortion/s4r2.part$i.jsonl > /dev/shm/h51-pq/logs/dist_$i.log 2>&1 &
  echo $! >> /dev/shm/h51-pq/logs/dist.pids
done
wait
echo "distortion done $(date -u +%T)"
