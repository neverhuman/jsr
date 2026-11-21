#!/usr/bin/env bash
# H12: impact of the multirate padding-before-pairing fix (ac30d53) on one MR checkpoint.
# mrpad_impact.sh <run_dir> <checkpoint.pt> <out_dir> <old_code_dir (a train/ package copy with the
# pre-fix model.py)>. Full dev-clean and dev-other, CPU, for {old, fixed} x {batch 8 length-sorted,
# batch 8 shuffled} plus batch 1 (old == fixed at batch 1: nothing is padded). Run it guarded.
set -uo pipefail
RUN=$1 CK=$2 OUT=$3 OLD=$4
REPO=/home/ubuntu/jsr/.agent/worktrees/trainer
PY=$REPO/.venv/bin/python
CPU=/mnt/xbabe3-external/jsr/venvs/torch270-cpu
mkdir -p "$OUT"
core=16
for split in dev-clean dev-other; do
  for v in old-b8-length fixed-b8-length old-b8-shuffled fixed-b8-shuffled fixed-b1-length; do
    code=${v%%-*}; rest=${v#*-}; b=${rest%%-*}; b=${b#b}; order=${rest#*-}
    if [ "$code" = old ]; then dir=$OLD; pp=$OLD:$CPU:$REPO/python/ultra; else dir=$REPO/python/ultra; pp=$CPU:$REPO/python/ultra; fi
    ( cd "$dir" || exit 1
      CUDA_VISIBLE_DEVICES="" PYTHONPATH=$pp taskset -c "$core-$((core + 2))" nice -n 10 "$PY" -m train.evaluate \
        --run-dir "$RUN" --checkpoint "$CK" --split "$split" --limit-all --threads 3 --batch "$b" \
        --order "$order" --dump "$OUT/${split}_$v.jsonl" 2>"$OUT/${split}_$v.err" | grep '^{' > "$OUT/${split}_$v.json" ) &
    core=$((core + 3))
  done
done
wait
echo MRPAD_DONE > "$OUT/DONE"
