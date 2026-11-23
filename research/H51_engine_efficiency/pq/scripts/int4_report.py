"""H51 Addendum A: tabulate the 4-bit screen (WER arms + kernel projection) and apply the stop rules.

usage: python3 int4_report.py   (reads results/int4/*.json, prints markdown, writes results/int4/verdict.json)
"""
import json
import os

R = "/mnt/xbabe3-external/jsr/h51-pq/results/int4"
WER_STOP_PP, SPEED_STOP = 0.30, 1.2
arms = []
for tag in ("int4_gptq", "int4_rtn", "int4_gptq_g128", "int4_rtn_g128"):
    f = os.path.join(R, f"{tag}.json")
    if os.path.exists(f):
        arms.append(json.load(open(f))["summary"])
proj = json.load(open(os.path.join(R, "projection_int4.json")))["projection"]
k2c4 = proj["chunk4"]["K2"]["speedup_vs_measured_step"]

print("| arm | scales | int4 MB | pooled WER | ΔWER vs INT8 (95% CI) | dev-clean Δ | dev-other Δ | held-out rel. MSE (INT8 → int4) | rule |")
print("|---|---|---:|---:|---|---:|---:|---|---|")
for a in arms:
    ok = a["d_pooled_pp"] <= WER_STOP_PP
    print(f"| {a['arm'].upper()} | {'per row' if not a['group'] else 'g' + str(a['group'])} | {a['int4_bytes_mb']:.1f} | "
          f"{a['wer_pooled']:.3f}% | {a['d_pooled_pp']:+.2f} pp [{a['d_lo']:+.2f}, {a['d_hi']:+.2f}] | {a['d_clean']:+.2f} | "
          f"{a['d_other']:+.2f} | {a['mean_rel_mse_int8']:.5f} → {a['mean_rel_mse_int4']:.4f} | {'pass' if ok else 'STOP'} |")
best = min(arms, key=lambda a: a["d_pooled_pp"]) if arms else None
# speed of the best arm's format: per-row -> K2; 128-column groups -> K2g128 (f32 scales, the conservative bytes)
pj = json.load(open(os.path.join(R, "projection_int4.json")))
if best and best["group"] == 128:
    k2c4 = pj["projection_K2g128"]["chunk4"]["f32_scales"]["speedup_vs_measured_step"]
verdict = {
    "wer_rule_pp": WER_STOP_PP, "speed_rule_c4": SPEED_STOP,
    "best_arm": best["tag"] if best else None, "best_d_pooled_pp": best["d_pooled_pp"] if best else None,
    "best_d_ci": [best["d_lo"], best["d_hi"]] if best else None,
    "best_format_chunk4_projected": k2c4, "k2_chunk4_projected": proj["chunk4"]["K2"]["speedup_vs_measured_step"], "k1_chunk4_projected": proj["chunk4"]["K1"]["speedup_vs_measured_step"],
    "k2_chunk2_projected": proj["chunk2"]["K2"]["speedup_vs_measured_step"],
    "k2_chunk1_projected": proj["chunk1"]["K2"]["speedup_vs_measured_step"],
}
verdict["wer_pass"] = bool(best) and best["d_pooled_pp"] <= WER_STOP_PP
verdict["speed_pass"] = k2c4 >= SPEED_STOP
verdict["verdict"] = "GO" if verdict["wer_pass"] and verdict["speed_pass"] else "STOP"
json.dump(verdict, open(os.path.join(R, "verdict.json"), "w"), indent=1)
print()
print(json.dumps(verdict))
