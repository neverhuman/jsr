#!/usr/bin/env bash
# eval-lane helper: checkpoint -> engine fp32 dir -> prepared INT8 dir, the way the dashboard promotion did it for
# D2e-16457 (python/ultra/tools/dashboard_asr_models.py evaluate()): r1_to_safetensors.py convert --bundle, then
# ultra-engine-prepare. The bundle (tokens.txt / bpe.model / mel_fbank.f32 / feature spec) is written first with the
# trainer lane's train.export --no-int8, because no existing bundle carries the 500- or 512-piece tokenizer.
#   export_model.sh <checkpoint.pt (with .sha256 marker)> <fresh out dir> <cpu list>
# Tools (read/execute only): converter python/ultra/tools/r1_to_safetensors.py (sha256 25be8bd7..., identical in the
# main checkout and the lanes); ultra-engine-prepare from e2ef608 sources ($PREPARE).
set -euo pipefail
CK=$1 OUT=$2 CPUS=$3
LANE=/dev/shm/eval-lane/src-d5f231e  # pinned read-only snapshot of lane commit d5f231e (the trainer lane itself moves under us)
PY=/home/ubuntu/jsr/.venv/bin/python
PREPARE=${PREPARE:-/mnt/xbabe3-external/jsr/evals/eval-lane/bin/engine-tools/ultra-engine-prepare.energy-debug-2225}  # private copy, sha 818644c7; reproduces the D2e prepared INT8 (5633a93e) byte for byte
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu
export JSR_DATA=${JSR_DATA:-/dev/shm/eval-lane/jsr_data} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GIT_OPTIONAL_LOCKS=0
CONVERTER=${CONVERTER:-/home/ubuntu/jsr/python/ultra/tools/r1_to_safetensors.py}  # the copy that exported D2e-16457 (commit 3a53c4d, sha256 25be8bd7...)
[ -e "$OUT" ] && { echo "$OUT exists; exports go to a fresh directory"; exit 2; }
mkdir -p "$OUT"
T="taskset -c $CPUS nice -n 10"
echo "== bundle (train.export --no-int8)"; ( cd $LANE/python/ultra && $T $PY -m train.export --checkpoint "$CK" --out "$OUT/bundle" --no-int8 --threads 2 ) > "$OUT/bundle.log" 2>&1 || { tail -5 "$OUT/bundle.log"; echo "BUNDLE FAILED"; exit 3; }
echo "== convert"; $T $PY $CONVERTER convert --checkpoint "$CK" --bundle "$OUT/bundle" --out "$OUT/fp32" > "$OUT/convert.log" 2>&1 || { tail -5 "$OUT/convert.log"; echo "CONVERT FAILED"; exit 4; }
echo "== prepare int8 ($(sha256sum $PREPARE | cut -c1-16))"; $T $PREPARE --weights "$OUT/fp32" --out "$OUT/int8" > "$OUT/prepare.log" 2>&1 || { tail -5 "$OUT/prepare.log"; echo "PREPARE FAILED"; exit 5; }
( cd "$OUT" && sha256sum $PREPARE > tools.sha256 && sha256sum $CONVERTER >> tools.sha256 && sha256sum fp32/* int8/* bundle/* > SHA256SUMS )
echo "== done"; cat "$OUT/SHA256SUMS"
