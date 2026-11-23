#!/usr/bin/env bash
# bench-open 14k, B0 arm (the L4 arm's paired control): B0 INT8 (A1-49000 EMA, weights.int8 4701787c...), greedy,
# chunk 4, with eval-lane's paced runner ultra-cold-wav.paced-chunk-h41arm10 (beb61399...; the binary behind every
# H43 B0 trace, same trace schema as h46-native-l4-paced), physically paced at 1x (20 ms packets) over the same
# panel73 WAVs. One process on CPU. OUT as for l4-stream. Run by open-run.sh (OPEN_PANEL=assay).
#   run.sh ROUND CPU OUT
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
kit=$opp/bench-open/assays/l4-kit
model=${L4_MODEL:-"$opp/models/A1-49000-ema/int8"}
mkdir -p "$out"
echo "round $round cpu $cpu" > "$out/arm"
sha256sum "$kit/ultra-cold-wav.paced-chunk-h41arm10" "$model"/*.* "$kit"/p73/*.wav > "$out/inputs.sha256"
printf 'run_start\t%s\n' "$(date +%s.%N)" >> "$out/phases.tsv"
/usr/bin/time -f '%U %S %e %M' -o "$out/time" taskset -c "$cpu" "$kit/ultra-cold-wav.paced-chunk-h41arm10" r1-engine-int8 \
    "$model" "$kit/p73" "$out/predictions.jsonl" --paced "$out/trace.jsonl" --chunk 4 > "$out/stdout" 2> "$out/stderr"
printf 'run_end\t%s\n' "$(date +%s.%N)" >> "$out/phases.tsv"
