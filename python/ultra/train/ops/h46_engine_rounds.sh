#!/usr/bin/env bash
# H46 amendment 4: engine cost per config measured in ROUNDS. Each round runs every config at once
# (one single-threaded process per config, each on its own core) on the same utterance subset,
# so all configs share the host's load of that moment; the next round takes the next subset.
#
#   BIN=<dir with ultra-engine-eval> WEIGHTS=<int8 dir> SHARDS=<dir of sNN> OUT=<dir> LIMIT=50 \
#   CORES="16 ... 23" SUBSETS="s00/dev-clean s00/dev-other ..." h46_engine_rounds.sh <runs.tsv>
#
# runs.tsv columns as h46_engine_fulldev.sh. GNU time reports go next to each engine_eval.json.
set -euo pipefail
RUNS=${1:?runs.tsv}
: "${BIN:?}" "${WEIGHTS:?}" "${SHARDS:?}" "${OUT:?}" "${CORES:?}" "${SUBSETS:?}"
read -r -a cores <<< "$CORES"
mapfile -t cfgs < <(grep -v '^\s*$' "$RUNS")
[ "${#cfgs[@]}" -le "${#cores[@]}" ] || { echo "need one core per config" >&2; exit 2; }
for sub in $SUBSETS; do
  i=0
  for line in "${cfgs[@]}"; do
    IFS=$'\t' read -r name beam lm w lw pen <<< "$line"
    out="$OUT/$name/${sub//\//.}"
    lmargs=()
    [ "$lm" != "-" ] && lmargs=(--lm "$lm" --lm-weight "$w" --lodr-weight "$lw" --blank-penalty "$pen")
    mkdir -p "$OUT/$name"
    taskset -c "${cores[$i]}" /usr/bin/time -v -o "$out.time" "$BIN/ultra-engine-eval" \
      --weights "$WEIGHTS" --precision int8 --npz "$SHARDS/$sub.npz" --limit "${LIMIT:-50}" \
      --streams 16 --beam "$beam" --beam-max-symbols 1 "${lmargs[@]}" --out "$out" \
      > "$out.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  echo "round $sub done: $(date -u +%H:%M:%SZ)"
done
