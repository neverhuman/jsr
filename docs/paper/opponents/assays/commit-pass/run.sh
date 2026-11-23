#!/usr/bin/env bash
# bench-open 14m (lead 10:00:14Z / 10:35:26Z; owner 09:50Z release focus): the release decoder's quiet cost on xbabe0.
# One process, one GEMM thread, on CPU: the release ultra-engine-commit (commit-kit/, the build of #106 posted by
# runtime-lane) decodes the 730-utterance panel (5,259.56 s, the paper's cost workload) unpaced, 20 ms packets: the
# chunk-4 greedy stream (provisional text) and, at each utterance's endpoint, the chunk-16 final pass over the segment
# with a 320 ms synthetic tail pad. COMMIT_VARIANT picks the final pass (the wrappers commit-greedy/, commit-lm4/ and
# commit-lm16/ set it):
#   commit-greedy : greedy;
#   commit-lm4    : beam 4 + M5.ulm / LODR 0.65 / 0.45;
#   commit-lm16   : beam 16 + M5.ulm / LODR 0.80 / 0.60 (the release configuration).
# Idle COMMIT_IDLE_S (60 s) before and after, so the run joins energy's 1 Hz power log as its own phase. The binary
# reports per-utterance stream and final-pass wall times (commit.jsonl, summary on stdout); OUT also gets
# /usr/bin/time (user / sys / wall / max RSS), phases.tsv and every input's sha256. Run by open-run.sh
# (OPEN_PANEL=assay):  run.sh ROUND CPU OUT        (COMMIT_MODEL: model folder; default B0 = A1-49000 EMA INT8)
set -euo pipefail
round=$1 cpu=$2 out=$3
variant=${COMMIT_VARIANT:?COMMIT_VARIANT is set by the commit-greedy / commit-lm4 / commit-lm16 wrappers}
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
bin=$opp/bench-open/assays/commit-kit/ultra-engine-commit
lm=${COMMIT_LM:-"$opp/bench-open/assays/l4-kit/M5.ulm"}
model=${COMMIT_MODEL:-"$opp/models/A1-49000-ema/int8"}
panel=${COMMIT_PANEL:-"$opp/panel"}
case $variant in
    commit-greedy) final=() ;;
    commit-lm4) final=(--commit-lm "$lm" --commit-beam 4 --lm-scale 0.65 --lodr-scale 0.45 --token-bonus 0.0) ;;
    commit-lm16) final=(--commit-lm "$lm" --commit-beam 16 --lm-scale 0.80 --lodr-scale 0.60 --token-bonus 0.0) ;;
    *) echo "unknown variant $variant" >&2; exit 2 ;;
esac
mkdir -p "$out"
echo "$variant round $round cpu $cpu" > "$out/arm"
python3 - "$panel" "$out/panel730.manifest.jsonl" <<'PY'
import json, sys
panel, out = sys.argv[1], sys.argv[2]
with open(out, "w") as w:
    for row in json.load(open(panel + "/audio.json")):
        w.write(json.dumps({"uid": row["id"], "wav": f"{panel}/wav/{row['id']}.wav"}) + "\n")
PY
sha256sum "${BASH_SOURCE[0]}" "$bin" "$lm" "$model"/*.* "$panel/audio.json" "$out/panel730.manifest.jsonl" > "$out/inputs.sha256"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
phase idle1_start; sleep "${COMMIT_IDLE_S:-60}"; phase idle1_end
phase run_start
/usr/bin/time -f '%U %S %e %M' -o "$out/time" taskset -c "$cpu" "$bin" --model "$model" \
    --manifest "$out/panel730.manifest.jsonl" --commit-chunk 16 --tail-pad-ms 320 --threads 1 "${final[@]}" \
    --out "$out/commit.jsonl" > "$out/stdout" 2> "$out/stderr"
phase run_end
phase idle2_start; sleep "${COMMIT_IDLE_S:-60}"; phase idle2_end
