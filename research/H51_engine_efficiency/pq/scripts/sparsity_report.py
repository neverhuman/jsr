"""Condense trace_summary.json into the README's sparsity verdict lines (per GEMM-input kind)."""
from __future__ import annotations

import json
import sys

d = json.load(open(sys.argv[1]))
L = d["layers"]
kinds = {}
for n, o in L.items():
    k = n.split(".", 3)[3] if n.startswith("encoder.blocks.") else n.replace("encoder.", "")
    kinds.setdefault(k, []).append(o)


def m(xs, f):
    v = [f(x) for x in xs]
    v = [x for x in v if x == x]
    return sum(v) / max(1, len(v))


def mx(xs, f):
    v = [f(x) for x in xs]
    return max(v) if v else float("nan")


out = {"meta": d["meta"], "kinds": {}}
for k, xs in kinds.items():
    out["kinds"][k] = {
        "layers": len(xs), "k": xs[0]["k"],
        "zero_code_centred_pct": 100 * m(xs, lambda x: x["zero_code_frac"]["centred/none"]),
        "zero_pair_s4_centred_1row_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("centred/none/pair4", float("nan"))),
        "zero_pair_s4_centred_1row_max_pct": 100 * mx(xs, lambda x: x["zero_group_frac"].get("centred/none/pair4", 0.0)),
        "zero_pair_s4_symmetric_1row_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("symmetric/none/pair4", float("nan"))),
        "zero_pair_s4_symmetric_4row_pct": 100 * m(xs, lambda x: x["zero_pair_frac_4row"].get("symmetric/none/s4", float("nan"))),
        "zero_pair_s4_symmetric_16row_pct": 100 * m(xs, lambda x: x.get("zero_pair_frac_16row_random", {}).get("symmetric/none/s4", float("nan"))),
        "zero_pair_s4_centred_fwht_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("centred/fwht16/pair4", float("nan"))),
        "zero_pair_s4_centred_randorth_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("centred/randorth/pair4", float("nan"))),
        "zero_pair_s4_symmetric_fwht_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("symmetric/fwht16/pair4", float("nan"))),
        "zero_pair_s4_symmetric_randorth_pct": 100 * m(xs, lambda x: x["zero_group_frac"].get("symmetric/randorth/pair4", float("nan"))),
        "crest_none": m(xs, lambda x: x["crest"]["none"]),
        "crest_fwht16": m(xs, lambda x: x["crest"]["fwht16"]),
        "crest_randorth": m(xs, lambda x: x["crest"]["randorth"]),
        "centre_over_half": m(xs, lambda x: x["centre_over_half"]),
    }
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out["kinds"].items():
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
