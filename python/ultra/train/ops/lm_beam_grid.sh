#!/usr/bin/env bash
# H42: run `train.lm_decode beam` for a list of decode configs, one shard per pinned core.
#
#   CORES="16 17 ... 29" lm_beam_grid.sh <configs.json> <out-dir> [log-dir]
#
# LM_BEAM_ARGS adds options to every worker (e.g. LM_BEAM_ARGS=--fast for weight sweeps).
# Every core gets one single-threaded worker that decodes its shard of dev-clean + dev-other for
# each config in turn (finished shard files are skipped, so a rerun resumes). Worker pids are
# written to <log-dir>/pids so they can be stopped by pid, never by pattern.
set -euo pipefail
CONFIGS=${1:?configs.json}
OUT=${2:?output dir}
LOGS=${3:-$OUT/logs}
: "${CORES:?space-separated CPU list}"
: "${LM_CACHE:?encoder cache dir (train.lm_decode encode)}"
: "${LM_CKPT:?checkpoint .pt}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
read -r -a cores <<< "$CORES"
# shellcheck disable=SC2086  # LM_BEAM_ARGS is a list of options, split on purpose
n=${#cores[@]}
mkdir -p "$OUT" "$LOGS"
: > "$LOGS/pids"
cd "$ROOT"
for i in "${!cores[@]}"; do
  taskset -c "${cores[$i]}" "$PY" -m train.lm_decode beam --cache "$LM_CACHE" \
    --checkpoint "$LM_CKPT" --configs "$CONFIGS" --shard "$i/$n" --out "$OUT" \
    ${LM_BEAM_ARGS:-} > "$LOGS/shard$i.log" 2>&1 &
  echo $! >> "$LOGS/pids"
done
wait
echo "all $n shards done: $(date -u +%H:%M:%SZ)"
