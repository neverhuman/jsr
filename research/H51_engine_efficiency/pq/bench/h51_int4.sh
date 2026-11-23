#!/usr/bin/env bash
# H51 Addendum A quiet assay: int4-weight x int8-activation small-M kernels (K1 naive unpack, K2
# offset-binary + i16 deferred widening) vs the engine's INT8 gemm_i8 on A1-49000's encoder shapes:
# cache-hot per shape and a whole-encoder sweep (129 distinct matrices, DRAM-streamed), M = 1 / 2 / 4.
# Synthetic weights / activations (timing does not depend on values); bit-exactness asserted.
# usage: h51_int4.sh BIN OUT_DIR CPU [REPS]     env: WRAP="meter ..." prefixes the run.
set -euo pipefail
BIN=$1; OUT=$2; CPU=$3; REPS=${4:-200}
[[ -x $BIN ]] || { echo "usage: h51_int4.sh BIN OUT_DIR CPU [REPS]" >&2; exit 2; }
[[ -e $OUT ]] && { echo "OUT_DIR $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
sum() { sha256sum "$1" | awk '{print $1}'; }
{
  echo "schema h51.int4-assay/v1"
  echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ) role ${ROLE_LABEL:-x0} cpu $CPU reps $REPS"
  echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "bin_sha256 $(sum "$BIN") (posted build: 5a07473c396938fcd486b719544c255aedd1892d3194073215a480f9e68f8fa5)"
} > "$OUT/receipt.txt"
# shellcheck disable=SC2086
${WRAP:-} taskset -c "$CPU" "$BIN" "$REPS" > "$OUT/int4_bench.jsonl"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/receipt.txt"
grep sweep_step "$OUT/int4_bench.jsonl" >> "$OUT/receipt.txt"
cat "$OUT/receipt.txt"
