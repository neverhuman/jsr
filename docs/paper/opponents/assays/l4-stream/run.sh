#!/usr/bin/env bash
# bench-open 14k, L4 arm (lead 09:55Z; codex-full-fleet 08:16:53Z / 08:44:49Z): the streaming-L4 fallback = B0 INT8
# (A1-49000 EMA, weights.int8 4701787c...) + M5.ulm (72a621c2...) with LODR, run by codex-full-fleet's
# h46-native-l4-paced (d561d65d...; e61121e engine library; chunk 4, beam 4, max 1 symbol, 0.8 / 0.6 / 0 fixed in
# the binary), physically paced at 1x (20 ms packets) over panel73 (73 utterances, 597.1 s, staged under their
# full-dev ids so eval-lane joins the trace to its locked MFA times). One process on CPU. OUT gets predictions, the
# paced trace (packet arrival and text emission, ns from each utterance's start), the runner's peak RSS (stderr),
# its user / sys / wall / max-RSS (time) and the phase times (phases.tsv). Run by open-run.sh (OPEN_PANEL=assay).
#   run.sh ROUND CPU OUT
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
kit=$opp/bench-open/assays/l4-kit
model=${L4_MODEL:-"$opp/models/A1-49000-ema/int8"}
mkdir -p "$out"
echo "round $round cpu $cpu" > "$out/arm"
sha256sum "$kit/h46-native-l4-paced" "$kit/M5.ulm" "$model"/*.* "$kit"/p73/*.wav > "$out/inputs.sha256"
printf 'run_start\t%s\n' "$(date +%s.%N)" >> "$out/phases.tsv"
/usr/bin/time -f '%U %S %e %M' -o "$out/time" taskset -c "$cpu" "$kit/h46-native-l4-paced" "$model" "$kit/M5.ulm" \
    "$kit/p73" "$out/predictions.jsonl" "$out/trace.jsonl" > "$out/stdout" 2> "$out/stderr"
printf 'run_end\t%s\n' "$(date +%s.%N)" >> "$out/phases.tsv"
