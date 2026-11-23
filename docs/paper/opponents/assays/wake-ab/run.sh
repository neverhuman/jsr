#!/usr/bin/env bash
# bench-open 14o (lead 11:11:19Z: #105's per-chunk wakes stay opt-in until bench-open's paired quiet-xbabe0 receipt;
# v3-engine-reviewer 10:42:32Z item 3): one live B0 stream (A1-49000 EMA INT8) paced at 1x in 20 ms packets, the same
# binary (wake-kit/ultra, built from main with #105 merged), --wake-per-packet (the default) against --wake-per-chunk, on
# 300 s of speech-only DEV audio and 300 s of pink noise at -30 dBFS (the first halves of 14f's inputs). Sequence: idle
# 120 s, four 300 s phases, idle 120 s; odd rounds packet / chunk on speech then chunk / packet on pink, even rounds the
# mirror image (ABBA inside each hold and across holds). Per phase OUT gets the stream receipt and events (design-clock
# event times and committed text are compared packet vs chunk) and phases.tsv for the join with energy's 1 Hz log.
#   run.sh ROUND CPU OUT          (run by open-run.sh with OPEN_PANEL=assay)
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
ultra=$opp/bench-open/assays/wake-kit/ultra
model=${WAKE_MODEL:-"$opp/models/A1-49000-ema/int8"}
inputs=${WAKE_INPUTS:-"$opp/bench-open/assays/h47-a0"}
mkdir -p "$out"
python3 - "$inputs/speech600.wav" "$out/speech300.wav" "$inputs/pink-30_600.wav" "$out/pink300.wav" <<'PY'
import sys, wave
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    with wave.open(src) as r, wave.open(dst, "wb") as w:
        w.setparams(r.getparams())
        w.writeframes(r.readframes(300 * r.getframerate()))
PY
sha256sum "${BASH_SOURCE[0]}" "$ultra" "$model/arch.json" "$model/weights.int8.safetensors" "$out/speech300.wav" \
    "$out/pink300.wav" > "$out/inputs.sha256"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
live() {  # NAME WAV packet|chunk
    phase "$1_start"
    taskset -c "$cpu" "$ultra" stream --candidate "$model" --input "$2" --pace 1.0 "--wake-per-$3" \
        --receipt "$out/$1.receipt.json" --events "$out/$1.events.jsonl" > "$out/$1.stdout" 2> "$out/$1.stderr"
    phase "$1_end"
}
phase idle1_start; sleep "${WAKE_IDLE_S:-120}"; phase idle1_end
if (( round % 2 == 1 )); then
    live speechpacket "$out/speech300.wav" packet; live speechchunk "$out/speech300.wav" chunk
    live pinkchunk "$out/pink300.wav" chunk; live pinkpacket "$out/pink300.wav" packet
else
    live speechchunk "$out/speech300.wav" chunk; live speechpacket "$out/speech300.wav" packet
    live pinkpacket "$out/pink300.wav" packet; live pinkchunk "$out/pink300.wav" chunk
fi
phase idle2_start; sleep "${WAKE_IDLE_S:-120}"; phase idle2_end
