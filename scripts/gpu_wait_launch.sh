#!/usr/bin/env bash
# Wait for the shared RTX 3090 to have enough free VRAM, then launch the D1 run
# (S model, train-clean-100) once. Polls nvidia-smi every POLL_S seconds at low priority.
# Never stops or touches other GPU jobs. Logs to the data drive.
#
#   nohup nice -n 10 scripts/gpu_wait_launch.sh >/dev/null 2>&1 &
#
# Environment overrides:
#   NEED_MIB (20480)  free VRAM required for two consecutive polls before launching
#   POLL_S   (300)    seconds between polls
#   RUN_NAME (D1_S_tc100_gpu)
#   JSR_DATA (/mnt/xbabe3-external/jsr)
#   EXTRA_SET ("")    extra TrainConfig overrides passed to --set, e.g. "budget_s=600.0"
set -euo pipefail

NEED_MIB="${NEED_MIB:-20480}"
POLL_S="${POLL_S:-300}"
RUN_NAME="${RUN_NAME:-D1_S_tc100_gpu}"
JSR_DATA="${JSR_DATA:-/mnt/xbabe3-external/jsr}"
EXTRA_SET="${EXTRA_SET:-budget_s=600.0 max_batch=96}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$JSR_DATA/runs/train/$RUN_NAME"
log_dir="$JSR_DATA/logs/trainer"
mkdir -p "$run_dir" "$log_dir"
wait_log="$log_dir/gpu_wait_${RUN_NAME}.log"
lock="$run_dir/.launcher.lock"

exec 9>"$lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another launcher holds $lock; exiting" >>"$wait_log"
  exit 1
fi

say() { echo "$(date -u +%FT%TZ) $*" >>"$wait_log"; }

free_mib() {
  # Minimum free MiB across GPU 0 (the only GPU); empty output on nvidia-smi failure.
  nice -n 19 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null \
    | head -n1 | tr -d ' '
}

say "waiting: need ${NEED_MIB} MiB free on GPU 0 for two consecutive polls (every ${POLL_S}s)"
ok_polls=0
while true; do
  f="$(free_mib || true)"
  if [[ "$f" =~ ^[0-9]+$ ]]; then
    if (( f >= NEED_MIB )); then
      ok_polls=$((ok_polls + 1))
      say "free=${f} MiB (ok ${ok_polls}/2)"
    else
      ok_polls=0
      say "free=${f} MiB (waiting)"
    fi
  else
    ok_polls=0
    say "nvidia-smi failed or returned '${f}'"
  fi
  if (( ok_polls >= 2 )); then
    break
  fi
  sleep "$POLL_S"
done

say "launching D1 in $run_dir"
cd "$repo/python/ultra"
status=0
# shellcheck disable=SC2086
nice -n 5 "$repo/.venv/bin/python" -m train.trainer \
  --run-dir "$run_dir" --preset S --splits train-clean-100 --device cuda --threads 8 \
  --set $EXTRA_SET >>"$run_dir/stdout.log" 2>&1 || status=$?
say "D1 exited with status $status"
exit "$status"
