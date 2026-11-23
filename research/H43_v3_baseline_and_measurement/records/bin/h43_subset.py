#!/usr/bin/env python3
"""eval-lane H43: aggregate (same definitions as h43_latency.aggregate) over a subset of a per-utterance latency file, e.g. the
100 MFA clips inside a full-dev run, and optionally write the filtered rows for `h43_latency.py compare`.
  h43_subset.py UTTS.jsonl --ids-from OTHER.utts.jsonl|ids.txt [--write OUT.utts.jsonl] [--dataset dev-clean|dev-other]"""
import argparse, importlib.util, json
spec = importlib.util.spec_from_file_location("h43l", "/mnt/xbabe3-external/jsr/evals/eval-lane/bin/h43_latency.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ap = argparse.ArgumentParser(); ap.add_argument("utts"); ap.add_argument("--ids-from"); ap.add_argument("--write"); ap.add_argument("--dataset")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.utts)]
if a.ids_from:
    if a.ids_from.endswith(".jsonl"):
        ids = {json.loads(l)["uid"] for l in open(a.ids_from)}
    else:
        ids = {l.strip() for l in open(a.ids_from) if l.strip()}
    rows = [r for r in rows if r["uid"] in ids]
if a.dataset:
    rows = [r for r in rows if r["dataset"] == a.dataset]
if a.write:
    with open(a.write, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
for r in rows:                      # JSON round trip turned the integer deadline keys into strings
    for k in ("cq_stable", "cq_prov"):
        if k in r:
            r[k] = {int(x): v for x, v in r[k].items()}
print(json.dumps({"n": len(rows), **m.aggregate(rows)}))
