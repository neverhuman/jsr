#!/bin/bash
# H12 diagnostic on one checkpoint (CPU): greedy vs modified beam 4 at the trained chunk, and
# greedy with larger chunks and with full (future-including) attention. Full dev-clean/dev-other.
# h12_diag.sh <run_dir> <checkpoint.pt> <out_dir> <python> [parallel jobs, default 4]
# Breakdowns (per length, final word) come from the beam run's dumps: train/diagnose.py.
set -uo pipefail
RUN=$1 CK=$2 OUT=$3 PY=$4 JOBS=${5:-4}
cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$OUT"
jobs_list=()
for split in dev-clean dev-other; do
  jobs_list+=("$split base --beam 4")
  jobs_list+=("$split full --full-context")
  jobs_list+=("$split chunk8 --chunk 8")
  jobs_list+=("$split chunk16 --chunk 16")
done
run() {
  local split=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" -m train.evaluate --run-dir "$RUN" --checkpoint "$CK" \
    --split "$split" --limit-all --threads 4 --dump "$OUT/${split}_$tag.jsonl" "$@" \
    2>"$OUT/${split}_$tag.err" | grep '^{' > "$OUT/${split}_$tag.json"
}
for j in "${jobs_list[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
done
wait
echo H12_DIAG_DONE > "$OUT/DONE"
