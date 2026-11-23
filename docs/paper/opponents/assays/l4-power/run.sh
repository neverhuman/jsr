#!/usr/bin/env bash
# bench-open 14l: power hold for the streaming-L4 fallback, the twin of 14f (B0's pair 1) with the L4 runner
# (h46-native-l4-paced, d561d65d...; B0 INT8 4701787c... + M5.ulm 72a621c2..., chunk 4, beam 4, 0.8 / 0.6 / 0) as ONE
# stream paced at 1x (20 ms packets) on the same 600 s of speech-only DEV audio (100% speech duty) and 600 s of pink
# noise at -30 dBFS (0% speech duty) as 14f. Sequence: idle 180 s, X 600 s, Y 600 s, idle 180 s; odd rounds X =
# speech first, even rounds Y = pink first (ABBA across holds). Phase times go to OUT/phases.tsv (epoch seconds) for
# the join with energy's 1 Hz power log. The runner reads a folder of uNNNNN.wav files, so each input is hard-linked
# alone into its own folder (speech = u90000, pink = u90001). Run by open-run.sh (OPEN_PANEL=assay).
#   run.sh ROUND CPU OUT
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
kit=$opp/bench-open/assays/l4-kit
model=${L4_MODEL:-"$opp/models/A1-49000-ema/int8"}
inputs=${L4_POWER_INPUTS:-"$opp/bench-open/assays/h47-a0"}
mkdir -p "$out/speech-in" "$out/pink-in"
ln "$inputs/speech600.wav" "$out/speech-in/u90000.wav"
ln "$inputs/pink-30_600.wav" "$out/pink-in/u90001.wav"
sha256sum "$kit/h46-native-l4-paced" "$kit/M5.ulm" "$model"/*.* "$inputs/speech600.wav" "$inputs/pink-30_600.wav" > "$out/inputs.sha256"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
live() {  # NAME FOLDER
    phase "$1_start"
    /usr/bin/time -f '%U %S %e %M' -o "$out/$1.time" taskset -c "$cpu" "$kit/h46-native-l4-paced" "$model" "$kit/M5.ulm" \
        "$2" "$out/$1.predictions.jsonl" "$out/$1.trace.jsonl" > "$out/$1.stdout" 2> "$out/$1.stderr"
    phase "$1_end"
}
phase idle1_start; sleep "${L4_POWER_IDLE_S:-180}"; phase idle1_end
if (( round % 2 == 1 )); then live speech "$out/speech-in"; live pink "$out/pink-in"
else live pink "$out/pink-in"; live speech "$out/speech-in"; fi
phase idle2_start; sleep "${L4_POWER_IDLE_S:-180}"; phase idle2_end
