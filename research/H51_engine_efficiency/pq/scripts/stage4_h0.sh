#!/usr/bin/env bash
# GPTQ fits for the post-SiLU layers, then the all-encoder 2-bpw joint with GPTQ fits.
cd /mnt/xbabe3-external/jsr/h51-pq
RX='(ff[12]\.w2|conv\.pw2)$|^encoder\.stem\.out$'
: > /dev/shm/h51-pq/logs/fit_gptq_post.pids
for i in 0 1 2; do
  OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/fit_only.py 4 2 "$RX" $i/3 --gptq > /dev/shm/h51-pq/logs/fit_gptq_post_$i.log 2>&1 &
  echo $! >> /dev/shm/h51-pq/logs/fit_gptq_post.pids
done
wait
echo "GPTQ post-SiLU fits done $(date -u +%T)"
OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag joint_s4r2_gptq --s 4 --R 2 --gptq --layers 'group:^encoder\.' --no-distortion --rows-suffix enc129 > /dev/shm/h51-pq/logs/joint_s4r2_gptq_enc129.log 2>&1 &
echo "$! joint_s4r2_gptq_enc129" > /dev/shm/h51-pq/logs/joint3.pids
wait
echo "stage4 done $(date -u +%T)"
