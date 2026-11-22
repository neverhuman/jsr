"""Summarise a decode_sweep.sh output directory: pooled panel WER (dev-clean + dev-other, from
S/D/I counts) per (beam, blank penalty, length norm), plus greedy, sorted best first.

  python sweep_summary.py <decode_sweep dir> [<more dirs with the other half>...]
"""
import glob
import json
import os
import re
import sys

TAG = re.compile(r"b(\d+)_bp([\d.]+)_ln(on|off)_(dev-clean|dev-other)\.json$")


def main(dirs):
    cells, greedy = {}, {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            m = TAG.search(p)
            if not m or os.path.getsize(p) == 0:
                continue
            r = json.load(open(p))
            key = (int(m[1]), float(m[2]), m[3])
            bw = r["beam_wer"]
            cells.setdefault(key, {})[m[4]] = bw
            greedy[m[4]] = {k: r[k] for k in ("sub", "del", "ins", "ref_words", "wer")}
    rows = []
    for key, sp in cells.items():
        if set(sp) != {"dev-clean", "dev-other"}:
            continue
        e = sum(v["sub"] + v["del"] + v["ins"] for v in sp.values())
        n = sum(v["ref_words"] for v in sp.values())
        rows.append((round(100 * e / n, 3), key, sp["dev-clean"]["wer"], sp["dev-other"]["wer"]))
    rows.sort()
    out = {"complete_cells": len(rows), "missing": 24 - len(rows)}
    if len(greedy) == 2:
        e = sum(v["sub"] + v["del"] + v["ins"] for v in greedy.values())
        n = sum(v["ref_words"] for v in greedy.values())
        out["greedy"] = {"pooled": round(100 * e / n, 3), "dev-clean": greedy["dev-clean"]["wer"],
                         "dev-other": greedy["dev-other"]["wer"]}
    out["ranked"] = [{"pooled": r[0], "beam": r[1][0], "blank_penalty": r[1][1],
                      "length_norm": r[1][2], "dev-clean": r[2], "dev-other": r[3]} for r in rows]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: sweep_summary.py <dir> [dir...]")
    main(sys.argv[1:])
