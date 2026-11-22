#!/usr/bin/env bash
# DEV decode operating-point sweep (H12/R1 freeze): beam {2,4,8} x blank penalty {0,0.5,1,1.5}
# x length norm {off,on} on the 730-utterance selection panel (dev-clean 400 + dev-other 330).
#   decode_sweep.sh <python> <ultra_dir> <run_dir> <checkpoint> <panel.json> <out_dir> <part>/<parts> <slot cores...>
# The 48 jobs are numbered; this host takes those with index % parts == part, spread round-robin
# over the given core slots (4 threads each), each slot running its jobs one after another.
set -uo pipefail
PY=$1 W=$2 RUN=$3 CK=$4 PANEL=$5 OUT=$6 PART=${7%/*} PARTS=${7#*/}
shift 7
slots=("$@")
mkdir -p "$OUT"
cd "$W" || exit 1
jobs=()
n=0
for beam in 2 4 8; do
  for bp in 0.0 0.5 1.0 1.5; do
    for ln in off on; do
      for split in dev-clean dev-other; do
        if [ $((n % PARTS)) -eq "$PART" ]; then jobs+=("$beam $bp $ln $split"); fi
        n=$((n + 1))
      done
    done
  done
done
run_slot() {
  local cores=$1; shift
  local j beam bp ln split tag extra
  for j in "$@"; do
    read -r beam bp ln split <<< "$j"
    tag="b${beam}_bp${bp}_ln${ln}_${split}"
    extra=""; [ "$ln" = on ] && extra="--length-norm"
    [ -s "$OUT/$tag.json" ] && continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="" taskset -c "$cores" nice -n 10 "$PY" -m train.evaluate --run-dir "$RUN" \
      --checkpoint "$CK" --panel "$PANEL" --panel-split "$split" --threads 4 --beam "$beam" \
      --blank-penalty "$bp" $extra --dump "$OUT/$tag.jsonl" 2>"$OUT/$tag.err" | grep '^{' > "$OUT/$tag.json"
  done
}
ns=${#slots[@]}
for s in $(seq 0 $((ns - 1))); do
  mine=()
  for k in "${!jobs[@]}"; do [ $((k % ns)) -eq "$s" ] && mine+=("${jobs[$k]}"); done
  run_slot "${slots[$s]}" "${mine[@]}" &
done
wait
echo SWEEP_DONE > "$OUT/DONE.$PART"
