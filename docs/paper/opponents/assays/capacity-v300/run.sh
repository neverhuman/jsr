#!/usr/bin/env bash
# bench-open TIMING 14s (lead 19:20:13Z item 3): the v3.0.0 live stream's always-on capacity on one core, energy's
# capacity test (TIMING 4b, H22ctl) repeated on the release: N = 1, 4, 8, 16 live streams of `ultra stream` built at the
# v3.0.0 tag (9c8c05e) on RC1-EMA INT8 (weights.int8 edac19d8), each paced at 1x on paced120.wav (DEV panel u00000-u00021,
# 128 s, byte-identical to energy's file), one process per stream, all N pinned to the assay's one CPU. Run by open-run.sh
# under the TIMING lock with OPEN_PANEL=assay, one repetition per call:
#   run.sh ROUND CPU OUT
# Phases (OUT/phases.tsv): idle 60 s, then for each N the run (all N streams start together) and idle 60 s. The H0-side
# join reads watts from energy's continuous 1 Hz power log (the service core's register against the idle floor around
# it) and the stable-word lag from each stream's own events against the panel's forced-aligned word ends.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
cpu=$2 out=$3
bin=$here/ultra
model=$opp/models/candidate-rc/int8
wav=$here/paced120.wav
idle=${CAPACITY_IDLE_S:-60}
mkdir -p "$out"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
sha256sum "$bin" "$wav" "$model"/* > "$out/inputs.sha256"
phase idle0_start; sleep "$idle"; phase idle0_end
for n in ${CAPACITY_STREAMS:-1 4 8 16}; do
    pids=()
    phase "run_n${n}_start"
    for ((w = 0; w < n; w++)); do
        taskset -c "$cpu" "$bin" stream --candidate "$model" --input "$wav" --pace 1.0 \
            --receipt "$out/n$n-w$w.receipt.json" --events "$out/n$n-w$w.events.jsonl" \
            > "$out/n$n-w$w.stdout" 2> "$out/n$n-w$w.stderr" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    phase "run_n${n}_end"
    (( failed == 0 )) || { echo "n=$n: $failed stream(s) failed" >&2; exit 1; }
    phase "idle_n${n}_start"; sleep "$idle"; phase "idle_n${n}_end"
done
