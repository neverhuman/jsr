#!/usr/bin/env bash
# H42: decode the configs a sweep chose again with the reference search (train.beam + the fusion
# hook), one config at a time on all claimed cores, so each config's decode time and LM contexts
# are its own; a no-LM config of the same beam width runs in the same batch as the timing
# reference. Then score them with the sweep's fold assignment (train.lm_tune reference).
#
#   CORES="16 17 ..." LM_CACHE=<enc cache> LM_CKPT=<slim .pt> RUNS=<runs dir> \
#     lm_reference.sh <sweep-name> <work-dir>          e.g. lm_reference.sh M5_b4 /data/lm/work
set -euo pipefail
NAME=${1:?sweep name, e.g. M5_b4}
WORK=${2:?work dir}
: "${LM_CACHE:?encoder cache dir}"
: "${CORES:?space-separated CPU list}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
RUNS=${RUNS:-$WORK/runs}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
taskset -cp "${CORES// /,}" $$ > /dev/null
lm_tune() { (cd "$ROOT" && "$PY" -m train.lm_tune "$@"); }
chosen="$WORK/grids/ref_$NAME.json"
lm_tune chosen --results "$WORK/results/fusion_$NAME.json" --grids "$WORK/grids/$NAME.json" \
  --prefix ref_ --out "$chosen"
beam=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[0]["beam"])' "$chosen")
n=$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$chosen")
for i in $(seq 0 "$n"); do
  one="$WORK/grids/ref_$NAME.$i.json"
  if [ "$i" -eq "$n" ]; then  # the no-LM timing reference, same beam, same batch
    echo "[{\"name\": \"ref_${NAME}_nolm_timing\", \"beam\": $beam, \"blank_penalty\": 0.0}]" > "$one"
  else
    "$PY" -c 'import json, sys; json.dump([json.load(open(sys.argv[1]))[int(sys.argv[2])]], open(sys.argv[3], "w"), indent=1)' "$chosen" "$i" "$one"
  fi
  PY="$PY" bash "$HERE/lm_beam_grid.sh" "$one" "$RUNS" "$WORK/logs/ref_$NAME.$i"
done
lm_tune reference --cache "$LM_CACHE" --runs "$RUNS" --baseline "${BASELINE:-b4}" \
  --results "$WORK/results/fusion_$NAME.json" --out "$WORK/results/ref_$NAME.json" \
  --dump "$WORK/results/hyps" --procs "${PROCS:-8}" --timing "ref_${NAME}_nolm_timing" \
  > "$WORK/logs/ref_tune_$NAME.log"
echo "reference $NAME done: $(date -u +%H:%M:%SZ)"
