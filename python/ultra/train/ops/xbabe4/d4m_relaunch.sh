#!/bin/zsh
# Relaunch D4M (xbabe4, MPS) from its latest checkpoint with the qkv cap/guard from step 30k, plus mac_guard.
# Used 2026-11-21 after the old trainer was stopped with SIGTERM (which writes a checkpoint: ckpt-00000823).
set -u
R=/Users/agent/jsr-train/data/runs/train/D4M_M_MR_960h
PY=/Users/agent/jsr-train/venv/bin/python
export JSR_DATA=/Users/agent/jsr-train/data PYTORCH_ENABLE_MPS_FALLBACK=1
kill $(cat $R/GUARD_PID) 2>/dev/null
cd /Users/agent/jsr-train/jsr/python/ultra
nohup $PY -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" nice -n 15 $PY -m train.trainer --run-dir $R --preset M_MR --splits train-clean-100 train-clean-360 train-other-500 --device mps --threads 8 --set budget_s=300.0 max_batch=64 workers=8 total_steps=140000 warmup=2000 lr=0.0005 clip=5.0 specaug_start=4000 speed_perturb=0.9,1.0,1.1 pad_frames=200 pad_tokens=16 eval_every=5000 dev_limit=400 ckpt_every_steps=5000 weights_every=5000 log_every=50 diverge_after=5500 qkv_cap_from=30000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 >> $R/stdout.log 2>&1 < /dev/null &
P=$!; sleep 3; echo $P > $R/PID; echo $P > $R/PGID
nohup $PY -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" nice -n 10 $PY -m train.mac_guard --pgid $P --gpu-target 0.74 --cpu-target 0.85 --min-duty 0.0 --log $R/mac_guard.jsonl >> $R/guard.log 2>&1 < /dev/null &
echo $! > $R/GUARD_PID
echo "restarted pid=$P guard=$(cat $R/GUARD_PID)"
