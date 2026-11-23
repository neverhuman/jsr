"""Merge trace parts: Sigma / mean (calibration utterances), held-out rows, sparsity counters.

usage: python -I merge_trace.py <parts_dir> <out_dir> <results_dir>
writes <out_dir>/sigma.<layer>.npy, mean.<layer>.npy, held.<layer>.npy and
<results_dir>/trace_summary.json + trace_sparsity.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/xbabe3-external/jsr/h51-pq/scripts")
import h51lib as L  # noqa: E402

parts_dir, out_dir, res_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
ROTS = ("none", "fwht16", "randorth")
QUANTS = ("centred", "symmetric")
parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.pt")))
metas, merged = [], {}
for pth in parts:
    d = torch.load(pth, weights_only=False)
    metas.append(d["meta"])
    for n, a in d["accs"].items():
        m = merged.setdefault(n, {"k": a["k"], "sigma": torch.zeros_like(a["sigma"]),
                                  "sum": torch.zeros_like(a["sum"]), "rows": 0, "cal_rows": 0,
                                  "held": [], "zero": {}, "inter4": {},
                                  "crest": {r: 0.0 for r in ROTS}, "centre": 0.0,
                                  "zcode": {}, "pz": {q: [] for q in QUANTS}})
        m["sigma"] += a["sigma"]
        m["sum"] += a["sum"]
        m["rows"] += a["rows"]
        m["cal_rows"] += a["cal_rows"]
        if a["held"] is not None:
            m["held"].append(a["held"])
        for key, v in a["zero"].items():
            z = m["zero"].setdefault(key, [0, 0])
            z[0] += v[0]
            z[1] += v[1]
        for key, v in a["inter4"].items():
            z = m["inter4"].setdefault(key, [0, 0])
            z[0] += v[0]
            z[1] += v[1]
        for r in ROTS:
            m["crest"][r] += a["crest"][r]
        m["centre"] += a["centre"]
        for key, v in a["zcode"].items():
            m["zcode"][key] = m["zcode"].get(key, 0.0) + v
        for q in QUANTS:
            if a["pz_samples"][q].shape[0]:
                m["pz"][q].append(a["pz_samples"][q])

rng = np.random.default_rng(5151)
summ = {}
for n, m in merged.items():
    s = L.short(n)
    if m["cal_rows"]:
        np.save(os.path.join(out_dir, f"sigma.{s}.npy"), (m["sigma"] / m["cal_rows"]).float().numpy())
        np.save(os.path.join(out_dir, f"mean.{s}.npy"), (m["sum"] / m["cal_rows"]).float().numpy())
    if m["held"]:
        np.save(os.path.join(out_dir, f"held.{s}.npy"), torch.cat(m["held"]).numpy())
    rows = max(1, m["rows"])
    o = {"k": m["k"], "rows": m["rows"], "cal_rows": m["cal_rows"],
         "held_rows": int(sum(h.shape[0] for h in m["held"])),
         "crest": {r: m["crest"][r] / rows for r in ROTS},
         "centre_over_half": m["centre"] / rows,
         "zero_code_frac": {f"{q}/{r}": v / rows for (q, r), v in m["zcode"].items()},
         "zero_group_frac": {f"{q}/{r}/{u}": v[0] / max(1, v[1]) for (q, r, u), v in sorted(m["zero"].items())},
         "zero_pair_frac_4row": {f"{q}/{r}/s{sz}": v[0] / max(1, v[1]) for (q, r, sz), v in sorted(m["inter4"].items())}}
    for q in QUANTS:
        if m["pz"][q]:
            allp = np.concatenate(m["pz"][q])
            if allp.shape[0] >= 16:
                fr = [allp[rng.choice(len(allp), 16, replace=False)].all(0).mean() for _ in range(400)]
                o.setdefault("zero_pair_frac_16row_random", {})[f"{q}/none/s4"] = float(np.mean(fr))
    summ[n] = o

meta = {"parts": metas, "utts": sum(x["utts"] for x in metas),
        "seconds": sum(x["seconds"] for x in metas)}
json.dump({"meta": meta, "layers": summ}, open(os.path.join(res_dir, "trace_summary.json"), "w"), indent=1)

# Markdown: by layer kind, means over blocks
def kind(n):
    if n.startswith("encoder.blocks."):
        return n.split(".", 3)[3]
    return n.replace("encoder.", "")


kinds = {}
for n, o in summ.items():
    kinds.setdefault(kind(n), []).append(o)
order = ["stem.c2", "stem.out", "ff1.w1", "ff1.w2", "att.qkv", "att.out", "conv.pw1", "conv.pw2",
         "ff2.w1", "ff2.w2", "joiner.enc_proj", "joiner.out"]
mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
lines = [f"# TRAIN activation traces: real group sparsity of the int8 GEMM inputs (A1-49000, chunk 4)", "",
         f"{meta['utts']} TRAIN utterances, {meta['seconds'] / 3600:.3f} h; engine INT8 emulation. "
         "Zero pair = all 2s int8 codes of a pqscan skip unit are 0 (s = 4: 8 values). "
         "1 row = M 1; 4-row = all rows of an aligned 4-frame chunk step (M 4); 16-row = 16 random rows (multi-stream M 16). "
         "Means over the 16 blocks for per-block kinds.", "",
         "| GEMM input | K | zero code % (centred) | zero pair s4, 1 row: centred | symmetric | centred+FWHT16 | centred+rand-orth | symmetric+FWHT16 | 4-row centred | 4-row symmetric | 16-row centred | 16-row symmetric | crest none / FWHT16 / rand | centre / half |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
for kd in order:
    xs = kinds.get(kd)
    if not xs:
        continue
    g = lambda key, sub="zero_group_frac": 100 * mean([x[sub].get(key, float("nan")) for x in xs])  # noqa: E731
    lines.append(
        f"| {kd} | {xs[0]['k']} | {100 * mean([x['zero_code_frac']['centred/none'] for x in xs]):.2f} | "
        f"{g('centred/none/pair4'):.2f} | {g('symmetric/none/pair4'):.2f} | {g('centred/fwht16/pair4'):.2f} | "
        f"{g('centred/randorth/pair4'):.2f} | {g('symmetric/fwht16/pair4'):.2f} | "
        f"{g('centred/none/s4', 'zero_pair_frac_4row'):.2f} | {g('symmetric/none/s4', 'zero_pair_frac_4row'):.2f} | "
        f"{100 * mean([x.get('zero_pair_frac_16row_random', {}).get('centred/none/s4', float('nan')) for x in xs]):.3f} | "
        f"{100 * mean([x.get('zero_pair_frac_16row_random', {}).get('symmetric/none/s4', float('nan')) for x in xs]):.3f} | "
        f"{mean([x['crest']['none'] for x in xs]):.1f} / {mean([x['crest']['fwht16'] for x in xs]):.1f} / {mean([x['crest']['randorth'] for x in xs]):.1f} | "
        f"{mean([x['centre_over_half'] for x in xs]):.2f} |")
lines += ["", "Group widths 2 / 8 (single 's' groups, centred, 1 row), per kind:", "",
          "| GEMM input | zero g2 % | zero g4 % | zero g8 % | zero pair s2 % | zero pair s8 % |", "|---|---:|---:|---:|---:|---:|"]
for kd in order:
    xs = kinds.get(kd)
    if not xs:
        continue
    g = lambda key: 100 * mean([x["zero_group_frac"].get(key, float("nan")) for x in xs])  # noqa: E731
    lines.append(f"| {kd} | {g('centred/none/g2'):.2f} | {g('centred/none/g4'):.2f} | {g('centred/none/g8'):.2f} | "
                 f"{g('centred/none/pair2'):.2f} | {g('centred/none/pair8'):.2f} |")
open(os.path.join(res_dir, "trace_sparsity.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
