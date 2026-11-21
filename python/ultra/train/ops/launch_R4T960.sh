#!/usr/bin/env bash
# After D2MR's A3 finisher: launch R4T-960 (ternary QAT from D2MR-final) on xbabe2, guarded.
set -uo pipefail
until [ -f /mnt/xbabe3-external/jsr/runs/train/D2MR_S_MR_960h_copy/FINISH_DONE ]; do sleep 120; done
until ! ssh -n xbabe2 'pgrep -f "[t]rain.evaluate" >/dev/null'; do sleep 60; done
ssh -n xbabe2 'export JSR_DATA=~/jsr-data; ~/jsr/python/ultra/train/launch_guarded.sh $JSR_DATA/runs/train/R4T960_S_MR_T_960h 0-15,64-79 0 -- --preset S_MR_T --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 --init-from $JSR_DATA/runs/train/D2MR_S_MR_960h/ckpt-00050000.pt --set budget_s=400.0 max_batch=96 workers=8 total_steps=15000 warmup=500 lr=0.0003 clip=5.0 specaug_start=0 eval_every=2500 dev_limit=400 ckpt_every_steps=2500 weights_every=2500 log_every=50 diverge_after=2000' > /mnt/xbabe3-external/jsr/runs/train/R4T960_launch.txt 2>&1
mkdir -p /mnt/xbabe3-external/jsr/runs/train/R4T960_S_MR_T_960h
nohup nice -n 10 python3 /mnt/xbabe3-external/jsr/runs/train/sync_and_post.py xbabe2 /home/ubuntu/jsr-data/runs/train/R4T960_S_MR_T_960h /mnt/xbabe3-external/jsr/runs/train/R4T960_S_MR_T_960h R4T-960 > /mnt/xbabe3-external/jsr/logs/trainer/sync_R4T960.log 2>&1 &
printf '\n## %s | trainer-lane | R4T-960 launched on xbabe2 (auto, after D2MR A3)\n- %s\n- S_MR_T ternary QAT from D2MR ckpt-00050000, 960 h, lr 3e-4, warmup 500, 15k steps, SpecAugment on, divergence guard after 2000; dev-clean400 every 2.5k (auto-posted).\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$(cat /mnt/xbabe3-external/jsr/runs/train/R4T960_launch.txt)" >> /home/ubuntu/jsr/AGENT_CHAT.md
