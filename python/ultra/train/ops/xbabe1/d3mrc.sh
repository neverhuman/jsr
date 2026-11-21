#!/usr/bin/env bash
# After the ablations: D3MRc = D3 recipe + attention cap (qkv rows / LN gain <= 1.5x their step-2000 value)
# + qkv-norm guard (stop if max_qkv_norm > 2x its step-2000 value), from D1MR-40k, lr 3e-4.
set -uo pipefail
export JSR_DATA=/mnt/amphora-backup/jsr-data
until [ -f $JSR_DATA/runs/train/ABL_DONE ]; do sleep 120; done
~/jsr/python/ultra/train/launch_guarded.sh $JSR_DATA/runs/train/D3MRc_S_MR_I8S_960h 0-15,64-79 0 -- --preset S_MR_I8S --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 --init-from $JSR_DATA/runs/train/D1MR_S_MR_tc100/ckpt-00040000.pt --set budget_s=400.0 max_batch=96 workers=8 total_steps=140000 warmup=1000 lr=0.0003 clip=5.0 specaug_start=2000 speed_perturb=0.9,1.0,1.1 eval_every=5000 dev_limit=400 ckpt_every_steps=5000 weights_every=5000 log_every=50 diverge_after=3500 qkv_cap_from=2000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 > $JSR_DATA/logs/d3mrc_launch.txt 2>&1
