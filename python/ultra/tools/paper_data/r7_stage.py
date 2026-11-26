#!/usr/bin/env python3
"""Stage the R7 receipts: RAW_DIR (a copy of runs/r7 from the benchmark host: b<N>-<arm>.json plus
b<N>-<arm>.unclean<K>.json attempts) -> RECEIPT_ROOT/block<N>/<arm>.json (the attempt used) and
RECEIPT_ROOT/attempts/ (every attempt, byte-identical), with RECEIPT_ROOT/selection.json.

Rule (fixed before reading any selection outcome, it looks at disturbance, never at the energy value):
a core-clean attempt beats an unclean one; among clean attempts the latest; among unclean attempts the one
with the smallest sibling_busy_fraction_run + elsewhere_busy_cpus_run / 64.
"""
import json
import shutil
import sys
from pathlib import Path

raw, root = Path(sys.argv[1]), Path(sys.argv[2])
groups = {}
for p in sorted(raw.glob("b[123]-*.json")):
    stem = p.name[:-5]
    if ".failed" in stem:
        base, tag = stem.split(".failed")[0], "failed"
    elif ".unclean" in stem:
        base, tag = stem.split(".unclean")[0], "unclean" + stem.split(".unclean")[1]
    else:
        base, tag = stem, "final"
    groups.setdefault(base, []).append((tag, p))


def disturbance(p):
    r = json.loads(p.read_text())["reps"][0]["busy"]
    return r["sibling_busy_fraction_run"] + r["elsewhere_busy_cpus_run"] / 64


selection = {}
for base, items in groups.items():
    blk, arm = base.split("-", 1)
    cands = []
    for tag, p in items:
        d = json.loads(p.read_text())
        if tag == "failed" or "nonzero_exit" in d["flags"]:
            continue
        cands.append((d["clean_repetitions"]["core"] == 1, tag, p, disturbance(p)))
    if not cands:
        continue
    clean = [c for c in cands if c[0]]
    if clean:
        # latest clean attempt: "final" is the last one written, otherwise the highest unclean number is older
        use = next((c for c in clean if c[1] == "final"), clean[-1])
    else:
        use = min(cands, key=lambda c: c[3])
    (root / ("block" + blk[1:])).mkdir(parents=True, exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)
    shutil.copy2(use[2], root / ("block" + blk[1:]) / f"{arm}.json")
    for tag, p in items:
        shutil.copy2(p, root / "attempts" / p.name)
    selection[base] = {"used": use[1], "core_clean": use[0], "disturbance": round(use[3], 4),
                       "attempts": [{"tag": t, "core_clean": c, "disturbance": round(disturbance(p), 4)}
                                    for c, t, p, _ in cands]}
(root / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
print(len(selection), "receipts staged;", sum(1 for v in selection.values() if not v["core_clean"]), "unclean used")
