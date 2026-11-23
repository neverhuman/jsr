#!/usr/bin/env bash
# H46: fusion sweep of one grid on the selection half of eval-lane's dev split v2.
# H42's lm_sweep.sh with two changes: the grid is given (not generated), and the held-out choice
# that drives the pre-registered edge rule is the SELECTION-HALF choice (train.h46_tune select), so
# no confirmation-half hypothesis influences the candidate set.
#   decodes on pinned cores (FastBeam, train/ops/lm_beam_grid.sh --fast)
#   -> train.h46_tune select -> train.lm_tune extend -> decodes of the added points -> ...
#   until no family's selection-half choice sits on an edge (at most MAX_ROUNDS extensions).
#
#   CODE=<frozen python/ultra> CORES="16 ..." LM_CACHE=<enc cache> LM_CKPT=<slim .pt> RUNS=<dir> \
#   SPLIT=<dev_split_v2.json> h46_sweep.sh <name> <grid.json> <work-dir>
set -euo pipefail
NAME=${1:?sweep name}
GRID_IN=${2:?grid json}
WORK=${3:?work dir}
: "${CODE:?frozen python/ultra}" "${LM_CACHE:?}" "${LM_CKPT:?}" "${CORES:?}" "${RUNS:?}" "${SPLIT:?}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
grid="$WORK/grids/$NAME.json"
mkdir -p "$WORK/grids" "$RUNS" "$WORK/results" "$WORK/logs"
[ -s "$grid" ] || cp "$GRID_IN" "$grid"
taskset -cp "${CORES// /,}" $$ > /dev/null
export LM_CACHE LM_CKPT CORES PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
lm_tune() { (cd "$CODE" && "$PY" -m train.lm_tune "$@"); }
batch_file=$grid
round=0
while :; do
  PY="$PY" LM_BEAM_ARGS=--fast bash "$CODE/train/ops/lm_beam_grid.sh" "$batch_file" "$RUNS" \
    "$WORK/logs/$NAME.r$round"
  (cd "$CODE" && "$PY" -m train.h46_tune select --cache "$LM_CACHE" --runs "$RUNS" \
    --grid "$grid" --split "$SPLIT" --out "$WORK/results/select_$NAME.json" \
    --procs "${PROCS:-8}") > "$WORK/logs/select_$NAME.log"
  echo "round $round of $NAME tuned on the selection half: $(date -u +%H:%M:%SZ)"
  round=$((round + 1))
  batch_file="$WORK/grids/$NAME.ext$round.json"
  n=$(lm_tune extend --grid "$grid" --results "$WORK/results/select_$NAME.json" --out "$batch_file")
  if [ "$n" -eq 0 ]; then echo "edge rule: every selection-half choice is interior"; break; fi
  if [ "$round" -gt "${MAX_ROUNDS:-10}" ]; then
    echo "edge rule: CAPPED after ${MAX_ROUNDS:-10} rounds, $n configs still on an edge (not decoded)"
    break
  fi
  echo "edge rule: $n configs added in round $round"
  "$PY" -c 'import json, sys; g = json.load(open(sys.argv[1])); g += json.load(open(sys.argv[2])); json.dump(g, open(sys.argv[1], "w"), indent=1)' "$grid" "$batch_file"
done
echo "sweep $NAME done: $(date -u +%H:%M:%SZ)"
