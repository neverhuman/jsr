#!/usr/bin/env bash
# H41 arm 8 (research/H41_replay_gates, amendment 2): minimum-WER fine-tune with runtime-lane's
# train/mwer.py (PR #35) from H22ctl's final raw weights, a fixed number of steps at a constant LR.
# lam=0 is its control: the same loop, data order and optimiser without the MWER term.
#   h41_mwer.sh <jsr_data> <cpu_list> <gpu> <name> <lam> [steps=2000]
set -uo pipefail
DATA=$1 CPUS=$2 GPU=$3 NAME=$4 LAM=$5 STEPS=${6:-2000}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY=${PY:-$DATA/venvs/train/bin/python}
R=$DATA/runs/train/H41_$NAME
mkdir -p "$R"
export JSR_DATA=$DATA CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO/python/ultra" || exit 1
setsid nohup taskset -c "$CPUS" nice -n 10 "$PY" -m train.mwer \
  --checkpoint "$DATA/seeds/H41final/model-00079800.pt" --out "$R/mwer-final.pt" \
  --splits train-clean-100 train-clean-360 train-other-500 --steps "$STEPS" --beam 4 --lam "$LAM" \
  --tau 1.0 --lr 2e-5 --budget-s 120 --max-batch 16 --device cuda --seed 1234 \
  >> "$R/stdout.log" 2>&1 < /dev/null &
pid=$!
sleep 2
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
[ -n "$pgid" ] || { echo "mwer exited immediately; see $R/stdout.log" >&2; exit 1; }
echo "$pid" > "$R/PID"
echo "$pgid" > "$R/PGID"
setsid nohup nice -n 5 "$PY" -m ultra.fleet_guard --pgid "$pgid" --gpu "$GPU" \
  --gpu-target "${GPU_TARGET:-0.85}" --cpu-target 0.85 --min-duty 0.0 \
  --log "$R/fleet_guard.jsonl" >> "$R/guard.log" 2>&1 < /dev/null &
echo $! > "$R/GUARD_PID"
echo "host=$(hostname) pid=$pid pgid=$pgid guard_pid=$(cat "$R/GUARD_PID") run_dir=$R"
