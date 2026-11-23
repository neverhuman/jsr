#!/usr/bin/env python3
"""eval-lane: canonical scoring of the R0 Zipformer INT8 full-dev control (predictions with staged ids u00000..u05566).
  r0_fulldev_score.py PREDICTIONS.all.jsonl OUTDIR
Writes OUTDIR/R0-int8.fulldev.score-input.jsonl and runs the pinned ultra-score (10,000 speaker-cluster replicates, seed 20261122,
one-sided 95%) paired against B0 INT8 (engine e2ef608) and B0 PyTorch EMA greedy at 160 ms, both on the same 5,567 ids and references.
Also scores the 730-utterance panel subset for comparison with the paper's R0 row (5.253 / 3.034 / 8.427)."""
import json, os, subprocess, sys
first_json = lambda b: json.JSONDecoder().raw_decode(b.decode())[0]   # ultra-score may append a markdown table
E = "/mnt/xbabe3-external/jsr/evals/eval-lane"; SC = "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score"
pred, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
index = json.load(open(f"{E}/export/fulldev-wav.index.json"))
hyps = {}
for line in open(pred):
    r = json.loads(line)
    if r["id"] in hyps:
        sys.exit(f"duplicate {r['id']}")
    hyps[r["id"]] = r["hyp"]
ids = {r["id"] for r in index}
if ids != hyps.keys():
    sys.exit(f"wrong population: missing {len(ids - hyps.keys())}, extra {len(hyps.keys() - ids)}")
r0 = f"{out}/R0-int8.fulldev.score-input.jsonl"
with open(r0, "w") as f:
    for ref in index:
        row = {k: v for k, v in ref.items() if k not in {"id", "uid"}}
        f.write(json.dumps({**row, "id": ref["uid"], "hyp": hyps[ref["id"]]}) + "\n")
comparators = {"B0-int8-e2ef608": f"{E}/export/A1-49000-ema/fulldev-int8/vs-pytorch.engine.score-input.jsonl",
               "B0-pytorch-ema-c4": f"{E}/curve/A1-49000.ema.c4.ms8.score-input.jsonl"}
mine = {json.loads(l)["id"]: json.loads(l) for l in open(r0)}
summary = {"label": "exploratory", "r0_score_input": r0, "comparisons": {}}
for name, path in comparators.items():
    other = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    if other.keys() != mine.keys() or any(other[k]["ref"] != mine[k]["ref"] for k in mine):
        sys.exit(f"{name}: ids or references differ")
    res = first_json(subprocess.run([SC, path, r0, "--seed", "20261122", "--replicates", "10000", "--alpha", "0.05"], capture_output=True, check=True).stdout)
    json.dump(res, open(f"{out}/R0-vs-{name}.score.json", "w"), indent=1)
    summary["comparisons"][name] = res
panel = f"{E}/export/A1-30000-ema/panel-int8/panel730.engine.score-input.jsonl"
pids = [json.loads(l)["id"] for l in open(panel)]
sub = f"{out}/R0-int8.panel730.score-input.jsonl"
with open(sub, "w") as f:
    for k in pids:
        f.write(json.dumps(mine[k]) + "\n")
res = first_json(subprocess.run([SC, sub, "--seed", "20261122", "--replicates", "1000", "--alpha", "0.05"], capture_output=True, check=True).stdout)
summary["panel730"] = res.get("system_a", res)
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
def fmt(s):
    d = s["datasets"]
    return " / ".join(f"{100*d[x]['wer']:.3f}" for x in ("dev-clean", "dev-other")) + f" / pooled {100*s['pooled_wer']:.3f}"
for name, res in summary["comparisons"].items():
    c = res["comparison"]["pooled_delta"]
    print(f"R0 {fmt(res['system_b'])} | {name} {fmt(res['system_a'])} | R0 - {name} = {100*c['delta']:+.3f} [{100*c['lower']:+.3f}, {100*c['upper']:+.3f}]")
print("R0 panel730:", fmt(summary["panel730"]), "(paper row r0-int8-zen2-t1: 3.034 / 8.427 / pooled 5.253)")
