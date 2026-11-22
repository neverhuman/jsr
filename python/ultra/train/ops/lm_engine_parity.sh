#!/usr/bin/env bash
# H42: the engine side of the fusion parity check. Packs token ARPA files with `ultra-lm-pack`
# (ultra.lm-ngram/v1, q8) and runs `ultra-engine-eval` on a verified DEV panel: one run without
# an LM and one per (name, lm, lodr, lm_weight, lodr_weight, blank_penalty) line of <runs.tsv>,
# each pinned to its own core, all in parallel.
#
#   BIN=<dir with ultra-engine-eval, ultra-lm-pack> WEIGHTS=<prepared int8 dir> PANEL=<panel.json> \
#   CACHE_DIR=<dir holding dev-clean.npz, dev-other.npz> CORES="16 17 18" BEAM=4 \
#     lm_engine_parity.sh <runs.tsv> <out-dir>
#
# runs.tsv columns: name, lm.arpa, lodr.arpa, lm_weight, lodr_weight, blank_penalty.
# The first core runs the no-LM pass; run i uses core i+1.
set -euo pipefail
RUNS=${1:?runs.tsv}
OUT=${2:?out dir}
: "${BIN:?dir with the engine binaries}" "${WEIGHTS:?prepared int8 weights dir}"
: "${PANEL:?dev panel json}" "${CACHE_DIR:?dir of the dev npz caches}" "${CORES:?space-separated CPU list}"
BEAM=${BEAM:-4}
read -r -a cores <<< "$CORES"
mkdir -p "$OUT/lm"
# the panel verifies its manifest and the caches' sha256 itself
common=(--weights "$WEIGHTS" --precision int8 --panel "$PANEL" --cache-dir "$CACHE_DIR"
        --streams 16 --beam "$BEAM" --qualifying)
taskset -c "${cores[0]}" "$BIN/ultra-engine-eval" "${common[@]}" --out "$OUT/nolm_b$BEAM" \
  > "$OUT/nolm_b$BEAM.log" 2>&1 &
echo "$!" > "$OUT/pids"
i=1
while IFS=$'\t' read -r name lm lodr w lw pen; do
  [ -n "$name" ] || continue
  pack="$OUT/lm/$name.ulm"
  [ -s "$pack" ] || taskset -c "${cores[$i]}" "$BIN/ultra-lm-pack" --arpa "$lm" --lodr "$lodr" \
    --vocab 501 --out "$pack" > "$OUT/lm/$name.pack.log" 2>&1
  taskset -c "${cores[$i]}" "$BIN/ultra-engine-eval" "${common[@]}" --lm "$pack" \
    --lm-weight "$w" --lodr-weight "$lw" --blank-penalty "$pen" --out "$OUT/${name}_b$BEAM" \
    > "$OUT/${name}_b$BEAM.log" 2>&1 &
  echo "$!" >> "$OUT/pids"
  i=$((i + 1))
done < "$RUNS"
wait
echo "engine parity runs done: $(date -u +%H:%M:%SZ)"
