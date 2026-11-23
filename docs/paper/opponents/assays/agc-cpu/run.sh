#!/usr/bin/env bash
# bench-open 14n (lead 09:34:43Z, warm-AGC acceptance criterion 4: "<= 1% added engine CPU, quiet xbabe0"; 11:51:13Z: C4 on the
# warm-AGC v2 binary after 14m): #108's ultra-engine-agc (agc-kit/; 02e1fa4c..., runtime-lane's build of the PR head ee613d5,
# the binary eval-lane runs C1-C3 on) on B0 INT8 (A1-49000 EMA, 4701787c...) decodes the 730-utterance panel (5,259.56 s) in
# the shipping mode: one process per arm with --manifest (the AGC level carried across utterances, the decoder reset per clip),
# --warm-agc --tau-s 5.0 against --no-warm-agc (bit-exact passthrough), on one CPU, the first arm alternating by round (ABBA
# over the holds). The binary times each utterance's decode loop (thread CPU and wall, model load excluded); after both arms
# OUT/agc_cpu.json pairs them per utterance (totals, added CPU %, a paired 95% interval). OUT also gets /usr/bin/time per arm
# and phases.tsv.   Run by open-run.sh (OPEN_PANEL=assay):  run.sh ROUND CPU OUT   (AGC_MODEL / AGC_PANEL override)
set -euo pipefail
round=$1 cpu=$2 out=$3
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
bin=$opp/bench-open/assays/agc-kit/ultra-engine-agc
model=${AGC_MODEL:-"$opp/models/A1-49000-ema/int8"}
panel=${AGC_PANEL:-"$opp/panel"}
mkdir -p "$out"
echo "agc-cpu round $round cpu $cpu" > "$out/arm"
python3 - "$panel/audio.json" "$out/manifest.jsonl" <<'PY'
import json, sys
with open(sys.argv[2], "w") as w:
    for r in json.load(open(sys.argv[1])):
        w.write(json.dumps({"id": r["id"], "wav": r["id"] + ".wav"}) + "\n")
PY
sha256sum "${BASH_SOURCE[0]}" "$bin" "$model"/*.* "$panel/audio.json" "$out/manifest.jsonl" > "$out/inputs.sha256"
phase() { printf '%s\t%s\n' "$1" "$(date +%s.%N)" >> "$out/phases.tsv"; }
decode() {  # on|off
    local flags=(--no-warm-agc)
    [[ $1 == on ]] && flags=(--warm-agc --tau-s 5.0)
    phase "$1_start"
    /usr/bin/time -f '%U %S %e %M' -o "$out/$1.time" taskset -c "$cpu" "$bin" --weights "$model" \
        --manifest "$out/manifest.jsonl" --wav-root "$panel/wav" --precision int8 --chunk 4 "${flags[@]}" \
        --out "$out/$1.jsonl" > "$out/$1.stdout" 2> "$out/$1.stderr"
    phase "$1_end"
}
if (( round % 2 == 1 )); then decode on; decode off; else decode off; decode on; fi
python3 - "$out" <<'PY'
import json, math, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = {}
for arm in ("on", "off"):
    rows[arm] = {}
    for line in (out / f"{arm}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[arm][r["id"]] = r
ids = [i for i in rows["off"] if i in rows["on"]]
assert len(ids) == len(rows["on"]) == len(rows["off"]), (len(ids), len(rows["on"]), len(rows["off"]))
assert all(rows["on"][i]["warm_agc"] is True and rows["off"][i]["warm_agc"] is False for i in ids)
on = sum(rows["on"][i]["cpu_s"] for i in ids); off = sum(rows["off"][i]["cpu_s"] for i in ids)
audio = sum(rows["off"][i]["audio_s"] for i in ids)
d = [rows["on"][i]["cpu_s"] - rows["off"][i]["cpu_s"] for i in ids]
half = 1.96 * statistics.stdev(d) * math.sqrt(len(d))  # 95% half-width of the summed paired difference
summary = {"utterances": len(ids), "audio_s": round(audio, 3), "cpu_on_s": round(on, 4), "cpu_off_s": round(off, 4),
           "added_cpu_pct": round(100 * (on - off) / off, 4),
           "added_cpu_pct_95": [round(100 * (on - off - half) / off, 4), round(100 * (on - off + half) / off, 4)],
           "cpu_per_audio_s_on": round(on / audio, 6), "cpu_per_audio_s_off": round(off / audio, 6),
           "texts_differ": sum(rows["on"][i]["text"] != rows["off"][i]["text"] for i in ids)}
(out / "agc_cpu.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary))
PY
