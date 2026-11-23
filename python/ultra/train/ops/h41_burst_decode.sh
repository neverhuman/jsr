#!/usr/bin/env bash
# Full-dev EMA decodes of one finished H41 arm on the GPU it trained on, under the GPU
# lease and the fleet guard at 0.87. The six decodes are the command lines of eval-lane's burst_decode.sh (pinned
# d5f231e evaluator, eval-lane's checks and file layout), run concurrently instead of one after another.
#   h41_burst_decode.sh <run_dir> <gpu> <cpu_list>   (h41_after_end.sh starts it in its own session)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
R=${1%/}; G=$2; CPUS=$3
PY=$HOME/jsr-data/venvs/train/bin/python; [ -x "$PY" ] || PY=$HOME/venvs/train/bin/python
B=$HOME/jsr-data/evals/burst; SRC=$B/src-d5f231e/python/ultra; OUT=${BURST_OUT:-$B/h41}; mkdir -p "$OUT"
RUN=$(basename "$R"); CK=$R/model-00008000.pt; J=${JSR_DATA:-$HOME/jsr-data}
[ -s "$CK" ] && [ -s "$R/model-00008000.sha256" ] || { echo "no finished model-00008000 in $R"; exit 2; }
[ "$(sha256sum "$CK" | cut -d' ' -f1)" = "$(tr -d ' \n' < "$R/model-00008000.sha256")" ] || { echo "checkpoint hash mismatch"; exit 3; }
for p in "dev-clean bfb27f5857411d7cea5fd8a307bab47793955425fbcb7ce5db5b428a919644b6" "dev-other 2944dbff8f3f3158e1192143a4aa6e6b4205681efbb12f1d1fd1c433b21d03c8"; do
  set -- $p; [ "$(sha256sum "$J/cache/librispeech_pcm/$1.npz" | cut -d' ' -f1)" = "$2" ] || { echo "$1 cache differs from eval-lane's"; exit 4; }
done
mkdir -p "$J/gpu-leases"; exec 9> "$J/gpu-leases/gpu-$G.lock"
flock --nonblock 9 || { echo "$(date -u +%T) GPU $G lease is held by another process; no decode"; exit 6; }
pgid=$(ps -o pgid= -p $$ | tr -d ' ')
(cd "$REPO/python/ultra" && exec setsid nohup nice -n 5 "$PY" -m ultra.fleet_guard --pgid "$pgid" \
   --gpu "$G" --gpu-target 0.87 --cpu-target 0.85 --min-duty 0.0 --log "$OUT/$RUN.guard.jsonl" >> "$OUT/$RUN.guard.log" 2>&1 < /dev/null 9>&-) &
GP=$!; echo "$GP" > "$OUT/$RUN.GUARD_PID"
trap 'kill $GP 2>/dev/null' EXIT
LIM="--limit-all"; DEV=cuda; [ -n "${BURST_TEST:-}" ] && { LIM="--limit 3"; DEV=cpu; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G JSR_DATA=$J PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SRC" || exit 5
pids=()
for c in 4 8 16; do for s in dev-clean dev-other; do
  T=$OUT/$RUN-8000.ema.c$c.ms8.$s; [ -s "$T.json" ] && continue
  ( "$PY" -m train.evaluate --run-dir "$R" --checkpoint "$CK" --device $DEV --threads 2 --split $s $LIM --max-symbols 8 --chunk $c --ema \
     --dump "$T.jsonl" 2> "$T.err" | grep '^{' > "$T.json.tmp" && [ -s "$T.json.tmp" ] && mv "$T.json.tmp" "$T.json" \
     && echo "$(date -u +%T) $RUN chunk $c $s: $(grep -o '"wer": *[0-9.]*' "$T.json")" || echo "$(date -u +%T) $RUN chunk $c $s FAILED (see $T.err)" ) &
  pids+=($!)
done; done
for p in "${pids[@]}"; do wait "$p"; done
sha256sum "$CK" > "$OUT/$RUN.checkpoint.sha256"; date -u +%FT%TZ > "$OUT/$RUN.DONE"; echo "$RUN done: $OUT"
