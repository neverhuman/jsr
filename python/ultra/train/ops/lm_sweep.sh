#!/usr/bin/env bash
# H42: the fusion sweep of one token LM at one beam width: grid -> decodes on pinned cores (NumPy
# beam, train.lm_fusion.FastBeam) -> held-out choice of weights and paired bootstrap -> the
# pre-registered edge rule (train.lm_tune extend) -> decodes of the added points -> ... until no
# fold's choice sits on an edge (at most MAX_ROUNDS extensions).
#
#   CORES="16 17 ..." LM_CACHE=<enc cache> LM_CKPT=<slim .pt> \
#     lm_sweep.sh <tag> <lm.bin> <lodr.bin> <full|reduced> <beam> <work-dir>
#
# Writes <work-dir>/grids/<tag>_b<beam>.json (grown by the extensions), results/fusion_<...>.json,
# results/hyps/ (held-out hypotheses in the input format of ultra-score) and logs/. Decode outputs
# go to RUNS (default <work-dir>/runs; a tmpfs is much faster for the statistics step).
# BASELINE names the no-LM run the families are compared with (default b4, must exist in RUNS).
set -euo pipefail
TAG=${1:?tag}
LM=${2:?token LM binary}
LODR=${3:?LODR LM binary}
KIND=${4:?full|reduced}
BEAM=${5:?beam width}
WORK=${6:?work dir}
: "${LM_CACHE:?encoder cache dir}"
: "${CORES:?space-separated CPU list}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
RUNS=${RUNS:-$WORK/runs}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
name="${TAG}_b$BEAM"
grid="$WORK/grids/$name.json"
mkdir -p "$WORK/grids" "$RUNS" "$WORK/results" "$WORK/logs"
# Everything this script starts stays on the claimed cores (the decode workers get one core each).
taskset -cp "${CORES// /,}" $$ > /dev/null
lm_tune() { (cd "$ROOT" && "$PY" -m train.lm_tune "$@"); }
lm_tune grid --tag "$TAG" --lm "$LM" --lodr "$LODR" --beam "$BEAM" --kind "$KIND" --out "$grid"
batch_file=$grid
round=0
while :; do
  PY="$PY" LM_BEAM_ARGS=--fast bash "$HERE/lm_beam_grid.sh" "$batch_file" "$RUNS" "$WORK/logs/$name.r$round"
  lm_tune fusion --cache "$LM_CACHE" --runs "$RUNS" --baseline "${BASELINE:-b4}" --grid "$grid" \
    --dump "$WORK/results/hyps" --out "$WORK/results/fusion_$name.json" --procs "${PROCS:-8}" \
    > "$WORK/logs/tune_$name.log"
  echo "round $round of $name tuned: $(date -u +%H:%M:%SZ)"
  round=$((round + 1))
  batch_file="$WORK/grids/$name.ext$round.json"
  n=$(lm_tune extend --grid "$grid" --results "$WORK/results/fusion_$name.json" --out "$batch_file")
  if [ "$n" -eq 0 ]; then echo "edge rule: every chosen value is interior"; break; fi
  if [ "$round" -gt "${MAX_ROUNDS:-4}" ]; then
    echo "edge rule: CAPPED after ${MAX_ROUNDS:-4} rounds, $n configs still on an edge (not decoded)"
    break
  fi
  echo "edge rule: $n configs added in round $round"
  "$PY" -c 'import json, sys; g = json.load(open(sys.argv[1])); g += json.load(open(sys.argv[2])); json.dump(g, open(sys.argv[1], "w"), indent=1)' "$grid" "$batch_file"
done
echo "sweep $name done: $(date -u +%H:%M:%SZ)"
