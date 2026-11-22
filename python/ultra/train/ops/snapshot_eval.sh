#!/usr/bin/env bash
# One training snapshot on CPU (run on the training host's spare cores): the 730-utterance DEV
# panel with greedy + beam 4 for model-<step>.pt and, when the two previous 5k snapshots exist,
# for their last-3 average. 4 jobs x 4 threads on the given core slots.
#   snapshot_eval.sh <python> <ultra_dir> <run_dir> <step> <panel.json> <slotA> <slotB> <slotC> <slotD>
set -uo pipefail
PY=$1 W=$2 RUN=$3 S=$4 PANEL=$5
shift 5
slots=("$@")
OUT=$RUN/snap_eval/$S
mkdir -p "$OUT"
cd "$W" || exit 1
m() { printf '%s/model-%08d.pt' "$RUN" "$1"; }
cks=("$(m "$S")")
tags=("snap")
if [ -f "$(m $((S - 5000)))" ] && [ -f "$(m $((S - 10000)))" ]; then
  taskset -c "${slots[0]}" nice -n 10 "$PY" -m train.average --out "$OUT/avg3.pt" \
    "$(m $((S - 10000)))" "$(m $((S - 5000)))" "$(m "$S")" > "$OUT/avg.log" 2>&1 \
    && { cks+=("$OUT/avg3.pt"); tags+=("avg3"); }
fi
k=0
for i in "${!cks[@]}"; do
  for split in dev-clean dev-other; do
    CUDA_VISIBLE_DEVICES="" taskset -c "${slots[$k]}" nice -n 10 "$PY" -m train.evaluate \
      --run-dir "$RUN" --checkpoint "${cks[$i]}" --panel "$PANEL" --panel-split "$split" \
      --threads 4 --beam 4 2>"$OUT/${tags[$i]}_$split.err" | grep '^{' > "$OUT/${tags[$i]}_$split.json" &
    k=$((k + 1))
  done
done
wait
rm -f "$OUT/avg3.pt"
echo DONE > "$OUT/DONE"
