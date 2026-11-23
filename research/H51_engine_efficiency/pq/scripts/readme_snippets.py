"""Markdown snippets for the README from the per-layer table, distortion file and joint summary."""
import glob
import json
import os

B = "/mnt/xbabe3-external/jsr/h51-pq/results"
tab = json.load(open(f"{B}/engine_sens/s4r2/table.json"))
L = []
n = len(tab)
L.append(f"**{n} of 131 layers measured** (WER); most / least sensitive:")
L.append("")
L.append("| layer | INT8 B → PQ B | rel MSE no-LUT / H24-LUT | ΔWER pooled [95%] |")
L.append("|---|---|---|---|")
for x in tab[::-1][:8] + [None] + tab[:6]:
    if x is None:
        L.append("| … | | | |")
        continue
    L.append(f"| {x['layer'].replace('encoder.', '')} | {x['int8_bytes']:,} → {x['pq_total_bytes']:,} | "
             f"{(x['rel_mse_pq_nolut'] or 0):.3f} / {(x['rel_mse_pq_exact'] or 0):.3f} | "
             f"{x['d_pooled_pp']:+.3f} [{x['d_pooled_lo']:+.3f}, {x['d_pooled_hi']:+.3f}] |")
kinds = {}
for x in tab:
    kinds.setdefault(x["kind"], []).append(x)
L += ["", "| kind | layers | mean ΔWER | mean rel MSE no-LUT | mean rel MSE H24-LUT |", "|---|---:|---:|---:|---:|"]
for k, xs in sorted(kinds.items(), key=lambda kv: sum(v["d_pooled_pp"] for v in kv[1]) / len(kv[1])):
    L.append(f"| {k} | {len(xs)} | {sum(v['d_pooled_pp'] for v in xs) / len(xs):+.3f} | "
             f"{sum((v['rel_mse_pq_nolut'] or 0) for v in xs) / len(xs):.3f} | {sum((v['rel_mse_pq_exact'] or 0) for v in xs) / len(xs):.3f} |")
tot = sum(x["int8_bytes"] for x in tab)
cov = sum(x["int8_bytes"] for x in tab if x["d_pooled_pp"] <= 0.10)
L += ["", f"Coverage (rule b): layers with ΔWER ≤ +0.10 pp hold **{100 * cov / tot:.1f}%** of the measured INT8 bytes."]
dist = []
for f in glob.glob(f"{B}/distortion/s4r2.part*.jsonl"):
    dist += [json.loads(l) for l in open(f) if l.strip()]
if dist:
    m = lambda k: sum(d[k] for d in dist) / len(dist)  # noqa: E731
    L += ["", f"Distortion for {len(dist)} / 131 GEMMs (`results/distortion/s4r2.jsonl`): mean relative output MSE "
          f"INT8 {m('rel_mse_int8'):.5f}, PQ exact products {m('pq_exact_products'):.4f}, H24 LUT {m('pq_lut_h24'):.4f}, "
          f"round-to-nearest LUT {m('pq_lut_nearest7'):.4f}."]
print("\n".join(L))
