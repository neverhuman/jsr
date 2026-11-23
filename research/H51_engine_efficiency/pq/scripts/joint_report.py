"""Compact report of all joint / group engine runs (rows with more than one layer)."""
from __future__ import annotations

import glob
import json
import os

base = "/mnt/xbabe3-external/jsr/h51-pq/results/engine_sens"
out = []
for f in sorted(glob.glob(os.path.join(base, "*", "rows.*.jsonl"))):
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if len(r["layers"]) < 2:
            continue
        c = r["score"]["comparison"]
        pq_b = sum(x["pq_bytes"]["total"] for x in r["layers"])
        i8_b = sum(x["int8_bytes"] for x in r["layers"])
        out.append({"tag": r["tag"], "group": r["key"].replace("group:", ""), "layers": len(r["layers"]),
                    "gptq": r.get("gptq", False), "int8_MB": round(i8_b / 1e6, 2), "pq_MB": round(pq_b / 1e6, 2),
                    "wer_pooled": round(100 * c["pooled_delta"]["observed_b"], 3),
                    "d_pooled": round(100 * c["pooled_delta"]["delta"], 3),
                    "lo": round(100 * c["pooled_delta"]["lower"], 3), "hi": round(100 * c["pooled_delta"]["upper"], 3),
                    "d_clean": round(100 * c["per_dataset"]["dev-clean"]["delta"], 3),
                    "d_other": round(100 * c["per_dataset"]["dev-other"]["delta"], 3)})
for o in out:
    print(f"{o['tag']:18s} gptq={str(o['gptq'])[0]} {o['layers']:3d} layers  INT8 {o['int8_MB']:6.2f} MB -> PQ {o['pq_MB']:5.2f} MB  "
          f"WER {o['wer_pooled']:.3f}  d {o['d_pooled']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]  clean {o['d_clean']:+.3f} other {o['d_other']:+.3f}  | {o['group'][:60]}")
json.dump(out, open(os.path.join(base, "joint_summary.json"), "w"), indent=1)
