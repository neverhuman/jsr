"""Ranked per-layer sensitivity table from engine_sens rows (one tag or several).

usage: python -I aggregate.py <results/engine_sens/<tag>> [more dirs...]
writes <dir>/table.md and <dir>/table.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def kind_of(layer):
    m = re.match(r"encoder\.blocks\.(\d+)\.(.+)$", layer)
    if m:
        return int(m.group(1)), m.group(2).replace(".", "_")
    return -1, layer.replace("encoder.", "").replace(".", "_")


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "rows.*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def main():
    for d in sys.argv[1:]:
        rows = load(d)
        out = []
        for r in rows:
            if len(r["layers"]) != 1:
                continue
            li = r["layers"][0]
            c = r["score"]["comparison"]
            blk, kind = kind_of(li["layer"])
            pb = li["pq_bytes"]
            out.append({
                "layer": li["layer"], "block": blk, "kind": kind, "n": li["n"], "k": li["k"],
                "int8_bytes": li["int8_bytes"], "pq_code_bytes": pb["codes"],
                "pq_total_bytes": pb["total"], "bits_per_weight": pb["bits_per_weight"],
                "rel_mse_int8": li.get("rel_mse_int8"),
                "rel_mse_pq_exact": li.get("rel_mse_pq_exact"),
                "rel_mse_pq_nolut": li.get("rel_mse_pq_nolut"),
                "rel_mse_pq_int8requant": li.get("rel_mse_pq_int8requant"),
                "rel_werr_sigma": li.get("rel_werr_sigma_cal"),
                "rel_werr_sigma_floatcb": li.get("rel_werr_sigma_cal_floatcb"),
                "d_pooled_pp": 100 * c["pooled_delta"]["delta"],
                "d_pooled_lo": 100 * c["pooled_delta"]["lower"],
                "d_pooled_hi": 100 * c["pooled_delta"]["upper"],
                "d_clean_pp": 100 * c["per_dataset"]["dev-clean"]["delta"],
                "d_other_pp": 100 * c["per_dataset"]["dev-other"]["delta"],
                "wer_pooled": 100 * c["pooled_delta"]["observed_b"],
                "errors": r.get("engine", {}).get("errors"),
            })
        out.sort(key=lambda x: (x["d_pooled_pp"], x["rel_mse_pq_exact"] or 0))
        tag = os.path.basename(d.rstrip("/"))
        json.dump(out, open(os.path.join(d, "table.json"), "w"), indent=1)
        L = [f"# Per-layer PQ sensitivity, {tag} (730-utt DEV panel, chunk 4, engine INT8 elsewhere)", "",
             f"{len(out)} layers. ΔWER = variant − INT8 baseline (7.385% pooled), percentage points, "
             "ultra-score paired speaker-cluster bootstrap 95% interval (10,000 replicates, seed 20261122). "
             "rel MSE = held-out TRAIN output error / output variance.", "",
             "| rank | layer | N×K | INT8 B | PQ B (codes) | rel MSE INT8 | rel MSE PQ (exact LUT) | PQ no-LUT | PQ→int8 (engine run) | ΔWER pooled [95%] | Δclean | Δother |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for i, x in enumerate(out):
            f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
            L.append(f"| {i + 1} | {x['layer'].replace('encoder.', '')} | {x['n']}×{x['k']} | {x['int8_bytes']:,} | "
                     f"{x['pq_total_bytes']:,} ({x['pq_code_bytes']:,}) | {f(x['rel_mse_int8'])} | {f(x['rel_mse_pq_exact'])} | "
                     f"{f(x['rel_mse_pq_nolut'])} | {f(x['rel_mse_pq_int8requant'])} | "
                     f"{x['d_pooled_pp']:+.3f} [{x['d_pooled_lo']:+.3f}, {x['d_pooled_hi']:+.3f}] | "
                     f"{x['d_clean_pp']:+.3f} | {x['d_other_pp']:+.3f} |")
        # by kind
        kinds = {}
        for x in out:
            kinds.setdefault(x["kind"], []).append(x)
        L += ["", "## By kind", "", "| kind | layers | INT8 MB | mean ΔWER pp | max ΔWER pp | mean rel MSE PQ |", "|---|---:|---:|---:|---:|---:|"]
        for k, xs in sorted(kinds.items(), key=lambda kv: sum(v["d_pooled_pp"] for v in kv[1]) / len(kv[1])):
            L.append(f"| {k} | {len(xs)} | {sum(v['int8_bytes'] for v in xs) / 1e6:.2f} | "
                     f"{sum(v['d_pooled_pp'] for v in xs) / len(xs):+.3f} | {max(v['d_pooled_pp'] for v in xs):+.3f} | "
                     f"{sum((v['rel_mse_pq_exact'] or 0) for v in xs) / len(xs):.4f} |")
        # cumulative bytes under thresholds
        tot = sum(x["int8_bytes"] for x in out)
        L += ["", "## Bytes coverable at a per-layer ΔWER threshold", "", "| threshold pp | layers | share of INT8 bytes |", "|---:|---:|---:|"]
        for th in (0.0, 0.03, 0.05, 0.10, 0.15, 0.20):
            sel = [x for x in out if x["d_pooled_pp"] <= th]
            L.append(f"| {th:.2f} | {len(sel)} | {sum(x['int8_bytes'] for x in sel) / max(1, tot):.1%} |")
        # depth profile (mean over the 8 kinds of each block)
        L += ["", "## By block depth", "", "| block | mean ΔWER pp | mean rel MSE PQ (exact LUT) | mean rel MSE PQ (no LUT) |", "|---:|---:|---:|---:|"]
        for b in sorted({x["block"] for x in out if x["block"] >= 0}):
            xs = [x for x in out if x["block"] == b]
            L.append(f"| {b} | {sum(v['d_pooled_pp'] for v in xs) / len(xs):+.3f} | "
                     f"{sum((v['rel_mse_pq_exact'] or 0) for v in xs) / len(xs):.4f} | "
                     f"{sum((v['rel_mse_pq_nolut'] or 0) for v in xs) / len(xs):.4f} |")
        # distortion vs WER (Spearman, no scipy)
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for i, j in enumerate(o):
                r[j] = float(i)
            return r
        def spearman(a, b):
            ra, rb = rank(a), rank(b)
            n = len(a)
            ma, mb = sum(ra) / n, sum(rb) / n
            cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
            va = sum((x - ma) ** 2 for x in ra) ** 0.5
            vb = sum((y - mb) ** 2 for y in rb) ** 0.5
            return cov / max(1e-12, va * vb)
        ok = [x for x in out if x["rel_mse_pq_nolut"] is not None]
        if len(ok) > 5:
            L += ["", f"Spearman(ΔWER, rel MSE no-LUT) = {spearman([x['d_pooled_pp'] for x in ok], [x['rel_mse_pq_nolut'] for x in ok]):.2f} over {len(ok)} layers; "
                  f"sum of single-layer ΔWER = {sum(x['d_pooled_pp'] for x in out):+.2f} pp (not additive; see the joint runs)."]
        open(os.path.join(d, "table.md"), "w").write("\n".join(L) + "\n")
        print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
