#!/usr/bin/env bash
# H51-PQ quiet engine phase profile (exploratory, DEV only): the Amdahl split of one stream.
#
# ultra-engine-eval --profile (A1-49000 EMA INT8, greedy, 1 stream, unpaced, packets of one chunk)
# on 73 DEV utterances (every 10th of the 730-utt panel; 9.56 audio-min) at chunk 4, 2 and 1, plus
# chunk 1 with the INT8 KV cache. Output: one engine_eval.json per run (cpu_s_per_audio_s and
# phase_s_per_audio_s: encoder gemm_i8 / attention / depthwise_conv / layer_norm / activation /
# quantize / stem layer 1, decoder gemm_i8 / greedy) and a receipt.
#
# usage: h51_prof.sh BIN WEIGHTS_DIR NPZ OUT_DIR CPU [REPS]
#   BIN          ultra-engine-eval from origin/main e61121e (sha256 below, or record yours)
#   WEIGHTS_DIR  A1-49000-ema/int8 (weights.int8.safetensors 4701787c..., arch.json b1581cbb...)
#   NPZ          panel73.npz (sha256 edfc06ac...)
#   OUT_DIR      new directory
#   CPU          one CPU to pin every run to (runs are sequential)
#   REPS         repetitions (default 1); rep r alternates the order (4,2,1,1kv / 1kv,1,2,4)
# env: WRAP="meter ..." prefixes every run.
set -euo pipefail
BIN=$1; W=$2; NPZ=$3; OUT=$4; CPU=$5; REPS=${6:-1}
[[ -x $BIN && -d $W && -f $NPZ ]] || { echo "usage: h51_prof.sh BIN WEIGHTS_DIR NPZ OUT_DIR CPU [REPS]" >&2; exit 2; }
[[ -e $OUT ]] && { echo "OUT_DIR $OUT exists; pass a new directory" >&2; exit 2; }
mkdir -p "$OUT"
sum() { sha256sum "$1" | awk '{print $1}'; }
[[ $(sum "$NPZ") == edfc06acd49726ff37973fc6a77e3cfdfc2bf74788212ddd651c69f32b91748f ]] || { echo "npz hash mismatch" >&2; exit 1; }
[[ $(sum "$W/weights.int8.safetensors") == 4701787c94d320953c42367bf0cf3be1ce2378e66bc811935af9f0f135b1e285 ]] || { echo "weights.int8 hash mismatch" >&2; exit 1; }
[[ $(sum "$W/arch.json") == b1581cbb1f22707ed16b146b0b04bdbcc181cdb6def5a68ad1108cc9af9db39f ]] || { echo "arch.json hash mismatch" >&2; exit 1; }
{
  echo "schema h51.prof/v1"
  echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "role ${ROLE_LABEL:-x0} cpu $CPU reps $REPS"
  echo "kernel $(uname -r)"
  echo "governor $(cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor" 2>/dev/null || echo n/a)"
  echo "bin_sha256 $(sum "$BIN") (posted build: 968d3ea766d93fdd5e3698c71e877020ec45001898ec21d530e919c4dbc86ca0)"
  echo "wrap ${WRAP:-none}"
} > "$OUT/receipt.txt"
run() {  # rep config
  local r=$1 cfg=$2 c=${2%kv} extra=""
  [[ $cfg == *kv ]] && extra="--kv-int8"
  local t0; t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # shellcheck disable=SC2086
  ${WRAP:-} taskset -c "$CPU" "$BIN" --weights "$W" --npz "$NPZ" --limit 73 --precision int8 --streams 1 \
    --chunk "$c" --packet-ms $((40 * c)) --profile $extra --out "$OUT/r$r.c$cfg" > "$OUT/r$r.c$cfg.log" 2>&1
  echo "run r$r.c$cfg start $t0 end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); e=d["phase_s_per_audio_s"]["encoder"]; print("cpu_s_per_audio_s %.5f enc_gemm_i8 %.5f wer %.3f" % (d["cpu_s_per_audio_s"], e["gemm_i8"], d["wer_pct"]))' "$OUT/r$r.c$cfg/engine_eval.json")" >> "$OUT/receipt.txt"
}
for r in $(seq 1 "$REPS"); do
  if (( r % 2 == 1 )); then order=(4 2 1 1kv); else order=(1kv 1 2 4); fi
  for cfg in "${order[@]}"; do run "$r" "$cfg"; done
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/receipt.txt"
cat "$OUT/receipt.txt"
