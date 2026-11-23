#!/usr/bin/env bash
# After the first joint runs: GPTQ fits for the LN-input layers (1 and 2 bpw), then joint GPTQ runs,
# then (after the post-SiLU k-means fits exist) the all-encoder and post-SiLU-only 2-bpw joint runs.
cd /mnt/xbabe3-external/jsr/h51-pq
for p in $(awk '{print $1}' /dev/shm/h51-pq/logs/joint.pids); do while kill -0 $p 2>/dev/null; do sleep 15; done; done
echo "first joint runs done $(date -u +%T)"
LN='(ff[12]\.w1|att\.qkv|att\.out|conv\.pw1)$'
: > /dev/shm/h51-pq/logs/fit_gptq.pids
for R in 1 2; do for i in 0 1; do
  OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/fit_only.py 4 $R "$LN" $i/2 --gptq > /dev/shm/h51-pq/logs/fit_gptq_r${R}_$i.log 2>&1 &
  echo $! >> /dev/shm/h51-pq/logs/fit_gptq.pids
done; done
wait
echo "GPTQ fits done $(date -u +%T)"
: > /dev/shm/h51-pq/logs/joint2.pids
for R in 1 2; do
  OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag joint_s4r${R}_gptq --s 4 --R $R --gptq --layers "group:$LN" --no-distortion --rows-suffix lnin80 > /dev/shm/h51-pq/logs/joint_s4r${R}_gptq.log 2>&1 &
  echo "$! joint_s4r${R}_gptq" >> /dev/shm/h51-pq/logs/joint2.pids
done
# post-SiLU k-means fits are produced by stage2_h0.sh; wait for its fitters
for p in $(cat /dev/shm/h51-pq/logs/fit_post.pids 2>/dev/null); do while kill -0 $p 2>/dev/null; do sleep 15; done; done
OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag joint_s4r2 --s 4 --R 2 --layers 'group:^encoder\.' --no-distortion --rows-suffix enc129 > /dev/shm/h51-pq/logs/joint_s4r2_enc129.log 2>&1 &
echo "$! joint_s4r2_enc129" >> /dev/shm/h51-pq/logs/joint2.pids
OMP_NUM_THREADS=1 nice -n 10 /home/ubuntu/jsr/.venv/bin/python -I scripts/engine_sens.py --tag joint_s4r2 --s 4 --R 2 --layers 'group:(ff[12]\.w2|conv\.pw2)$|^encoder\.stem\.out$' --no-distortion --rows-suffix postsilu49 > /dev/shm/h51-pq/logs/joint_s4r2_postsilu49.log 2>&1 &
echo "$! joint_s4r2_postsilu49" >> /dev/shm/h51-pq/logs/joint2.pids
wait
echo "stage3 done $(date -u +%T)"
