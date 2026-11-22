#!/usr/bin/env bash
# Launch an elastic 80%-CPU-targeted teacher-cache slice on a single host via ops/cpu_pool.py.
# Coordinates disjoint (split, speed) slices so multiple xbabe hosts can generate R0 teacher
# cache shards in parallel without duplicating work or breaching the 85%/90% host CPU cap.
#
# Usage:
#   launch_cluster_cpu_cache.sh <jsr_data> <out_dir> <cores> <splits> <speeds> [min_free_ram_gb]
# Example:
#   bash launch_cluster_cpu_cache.sh /home/ubuntu/jsr-data /home/ubuntu/jsr-data/teacher_cache \
#     0-31,48-63 "train-clean-360" "1.0 0.9 1.1" 12
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: $0 <jsr_data> <out_dir> <cores> <splits> <speeds> [min_free_ram_gb]" >&2
  exit 2
fi

DATA=$1
OUT=$2
CORES=$3
SPLITS=$4
SPEEDS=$5
MIN_RAM_GB=${6:-12}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${PY:-$DATA/venvs/train/bin/python}"
[ -x "$PY" ] || PY="$REPO/.venv/bin/python"
RUNNER="$REPO/python/ultra/train/ops/teacher_cache_runner.sh"

mkdir -p "$OUT/logs"
export SPLITS="$SPLITS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CMD="bash $RUNNER $DATA $OUT \"$SPEEDS\" {core}"
setsid nohup "$PY" "$REPO/python/ultra/train/ops/cpu_pool.py" \
  --cores "$CORES" \
  --cmd "$CMD" \
  --cpu-low 0.75 \
  --cpu-high 0.83 \
  --min-free-ram-gb "$MIN_RAM_GB" \
  --nice 15 \
  --log "$OUT/logs/cpu_pool_$(hostname -s).jsonl" \
  >> "$OUT/logs/cpu_pool_$(hostname -s).stdout.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$OUT/logs/cpu_pool_$(hostname -s).PID"
echo "host=$(hostname -s) cpu_pool_pid=$pid cores=$CORES splits='$SPLITS' speeds='$SPEEDS' out=$OUT"
