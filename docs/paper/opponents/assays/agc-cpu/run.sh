#!/usr/bin/env bash
# bench-open 14n (lead 09:34:43Z, warm-AGC acceptance criterion 4: "<= 1% added engine CPU, quiet xbabe0"; 10:29:58Z /
# 10:35:26Z order): #104's warm AGC, on vs off, in the same binary. ultra-engine-agc (agc-kit/; 1faea9cc..., runtime-lane's
# release build at d70d5ed = merged main adc3180) on B0 INT8 (A1-49000 EMA, 4701787c...) decodes every utterance of
# panel73 (73 DEV utterances, 597.1 s; every load re-verifies all model hashes, ~0.4 s, so the 730 panel would cost
# ~16 min per round) twice on one CPU, one process per decode: --warm-agc --tau-s 5.0 --preroll-ms 500
# (the fixed release setting) and --no-warm-agc (bit-exact passthrough). The arm that goes first alternates by
# utterance and round (ABBA), so drift cancels in the paired sum. The binary times its own decode loop (thread CPU and
# wall, model load excluded): each decode's JSON goes to OUT/dec/<id>.<on|off>.json, a summary (paired totals, added
# CPU %) to OUT/agc_cpu.json after the loop. OUT also gets /usr/bin/time over the whole loop and phases.tsv.
# Run by open-run.sh (OPEN_PANEL=assay):  run.sh ROUND CPU OUT        (AGC_MODEL: model folder; default B0)
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
bin=$opp/bench-open/assays/agc-kit/ultra-engine-agc
model=${AGC_MODEL:-"$opp/models/A1-49000-ema/int8"}
panel=${AGC_PANEL:-"$opp/bench-cross/workloads/panel73"}
mkdir -p "$out/dec"
echo "agc-cpu round $round cpu $cpu" > "$out/arm"
python3 -c 'import json, sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1]))))' "$panel/audio.json" > "$out/ids"
sha256sum "${BASH_SOURCE[0]}" "$bin" "$model"/*.* "$panel/audio.json" > "$out/inputs.sha256"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
decode() {  # ID on|off
    local flags=(--no-warm-agc)
    [[ $2 == on ]] && flags=(--warm-agc --tau-s 5.0 --preroll-ms 500)
    taskset -c "$cpu" "$bin" --weights "$model" --wav "$panel/wav/$1.wav" --precision int8 "${flags[@]}" \
        > "$out/dec/$1.$2.json" 2> "$out/dec/$1.$2.stderr"
}
loop() {
    local i=0 id
    while read -r id; do
        if (( (i + round) % 2 == 0 )); then decode "$id" on; decode "$id" off; else decode "$id" off; decode "$id" on; fi
        i=$((i + 1))
    done < "$out/ids"
}
phase run_start
export -f decode loop
export out cpu bin model panel round
/usr/bin/time -f '%U %S %e %M' -o "$out/time" bash -c loop
phase run_end
python3 - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ids = (out / "ids").read_text().split()
pairs, texts_differ = [], 0
for i in ids:
    on = json.loads((out / "dec" / f"{i}.on.json").read_text())
    off = json.loads((out / "dec" / f"{i}.off.json").read_text())
    assert on["warm_agc"] is True and off["warm_agc"] is False, i
    pairs.append((on["cpu_s"], off["cpu_s"], on["audio_s"]))
    texts_differ += on["text"] != off["text"]
cpu_on, cpu_off = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
audio = sum(p[2] for p in pairs)
summary = {"utterances": len(pairs), "audio_s": round(audio, 3), "cpu_on_s": round(cpu_on, 4),
           "cpu_off_s": round(cpu_off, 4), "added_cpu_pct": round(100 * (cpu_on - cpu_off) / cpu_off, 4),
           "cpu_per_audio_s_on": round(cpu_on / audio, 6), "cpu_per_audio_s_off": round(cpu_off / audio, 6),
           "texts_differ": texts_differ}
(out / "agc_cpu.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary))
PY
