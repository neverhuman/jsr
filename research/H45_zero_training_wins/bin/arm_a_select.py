#!/usr/bin/env python3
"""h45-wins Arm A: rank on the selection half S only and name the finalist (lowest S pooled WER at chunk 4 among the
13 pre-registered candidates; ties -> fewer ingredients). Reads out/a/scores_S.jsonl and out/a/gs/greedy_soup.jsonl."""
import json, sys
d = sys.argv[1] if len(sys.argv) > 1 else "/mnt/xbabe3-external/jsr/h45-wins/out/a"
rows = [json.loads(l) for l in open(f"{d}/scores_S.jsonl")]
sc = {r["system"].split("/")[-1].split(".S.")[0]: r for r in rows}
gs = [json.loads(l) for l in open(f"{d}/gs/greedy_soup.jsonl")]
final = [g for g in gs if "final" in g][-1]
ningr = {"A-E2": 2, "A-E3": 3, "A-E4": 4, "A-E5": 5, "A-R2": 2, "A-R3": 3, "A-R4": 4, "A-R5": 5, "A-P4": 5, "A-P8": 5,
         "A-M10": 10, "A-L50": 2, "A-L25": 2, "A-GS": len(final["final"])}
cands = {k: sc[k]["pooled"]["wer"] for k in ningr if k in sc}
cands["A-GS"] = final["s_wer"]
b0 = sc["E49000"]["pooled"]["wer"]
order = sorted(cands, key=lambda k: (cands[k], ningr[k]))
print("| candidate | S pooled WER | vs B0 (S) | ingredients |")
print("|---|---:|---:|---:|")
for k in order:
    print(f"| {k} | {cands[k]:.4f} | {cands[k] - b0:+.4f} | {ningr[k]} |")
print()
print("| single | S pooled WER | vs B0 (S) |")
print("|---|---:|---:|")
for k in ("E49000", "R49000", "E47500", "R47500", "E45000", "R45000", "E42500", "R42500", "E40000", "R40000", "A1LE8000", "A1LR8000"):
    print(f"| {k} | {sc[k]['pooled']['wer']:.4f} | {sc[k]['pooled']['wer'] - b0:+.4f} |")
print()
print("greedy soup:", " -> ".join(f"{g.get('candidate', g.get('start'))}:{g.get('s_wer')}{'*' if g.get('kept') else ''}" for g in gs if 'final' not in g))
print(json.dumps({"finalist": order[0], "s_wer": cands[order[0]], "b0_s_wer": b0, "delta_s": round(cands[order[0]] - b0, 4),
                  "gs_final": final["final"]}))
