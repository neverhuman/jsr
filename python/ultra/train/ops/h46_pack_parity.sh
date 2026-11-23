#!/usr/bin/env bash
# H46 item 2: ultra.lm-ngram v1 against its `ultra-lm-pack --repack` v2 twin on the 730-utterance
# DEV panel (--qualifying), one engine run per (LM, pack version), each on its own core.
#
#   BIN=<dir with ultra-engine-eval, ultra-lm-pack> WEIGHTS=<prepared int8 dir> PANEL=<panel.json>
#   CACHE_DIR=<dir with dev-clean.npz, dev-other.npz> LMDIR=<dir with <name>.ulm (v1)> OUT=<dir>
#   CORES="c0 c1 ..." h46_pack_parity.sh <runs.tsv>
#
# runs.tsv columns: name (the v1 pack is LMDIR/<name>.ulm), lm_weight, lodr_weight, blank_penalty.
# Core i*2 runs <name> v1, core i*2+1 its v2 twin. A no-LM panel run takes the last core.
set -euo pipefail
RUNS=${1:?runs.tsv}
: "${BIN:?}" "${WEIGHTS:?}" "${PANEL:?}" "${CACHE_DIR:?}" "${LMDIR:?}" "${OUT:?}" "${CORES:?}"
read -r -a cores <<< "$CORES"
mkdir -p "$OUT"
common=(--weights "$WEIGHTS" --precision int8 --panel "$PANEL" --cache-dir "$CACHE_DIR"
        --streams 16 --beam 4 --beam-max-symbols 1 --qualifying)
: > "$OUT/pids"
i=0
while IFS=$'\t' read -r name w lw pen; do
  [ -n "$name" ] || continue
  v2="$LMDIR/$name.v2.ulm"
  [ -s "$v2" ] || taskset -c "${cores[$i]}" "$BIN/ultra-lm-pack" --repack "$LMDIR/$name.ulm" \
    --out "$v2" > "$OUT/$name.repack.log" 2>&1
  for ver in v1 v2; do
    pack="$LMDIR/$name.ulm"; off=0
    if [ "$ver" = v2 ]; then pack="$v2"; off=1; fi
    core=${cores[$((i * 2 + off))]}
    taskset -c "$core" nice -n 10 "$BIN/ultra-engine-eval" "${common[@]}" --lm "$pack" \
      --lm-weight "$w" --lodr-weight "$lw" --blank-penalty "$pen" --out "$OUT/${name}_$ver" \
      > "$OUT/${name}_$ver.log" 2>&1 &
    echo "$!" >> "$OUT/pids"
  done
  i=$((i + 1))
done < "$RUNS"
taskset -c "${cores[$((i * 2))]}" nice -n 10 "$BIN/ultra-engine-eval" "${common[@]}" \
  --out "$OUT/nolm" > "$OUT/nolm.log" 2>&1 &
echo "$!" >> "$OUT/pids"
wait
echo "pack parity runs done: $(date -u +%H:%M:%SZ)"
