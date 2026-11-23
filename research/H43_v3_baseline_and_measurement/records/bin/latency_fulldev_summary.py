#!/usr/bin/env python3
"""eval-lane: one machine-readable table of every FULL-dev latency run (h43/latency-fulldev/*.json) with its canonical full-dev WER row,
for the money plot and the scoreboard. Design clock = audio sample clock (deterministic). Capture clock = trace host, loaded, not quiet.
  latency_fulldev_summary.py OUT.json"""
import json, glob, os, sys
E = "/mnt/xbabe3-external/jsr/evals/eval-lane"; D = f"{E}/h43/latency-fulldev"
WER = {  # canonical full-dev pooled WER row that matches each latency run (decode path noted)
 "B0-c4": ("export/A1-49000-ema/fulldev-int8/vs-pytorch.pytorch-v-engine.json", "system_b", "B0 INT8 engine e2ef608, 160 ms greedy"),
 "B0-c8": ("curve/A1-49000.ema.c8.ms8.score.json", "system_a", "B0 PyTorch EMA greedy 320 ms (INT8 = PyTorch within 0.03 pp at 160 ms)"),
 "B0-c2": (None, None, "not decoded at 80 ms in PyTorch; engine transcripts in the traces"),
 "A1L-c4": ("h41/A1L_dp0.005-8000.ema.c4.ms8.score.json", "system_a", "A1L PyTorch EMA greedy 160 ms"),
 "H48b-s1234-c4": ("h41/H48b_c2_dp0.0075_s1234-8000.ema.c4.ms8.score.json", "system_a", "PyTorch EMA greedy 160 ms"),
 "H48b-s4321-c4": ("h41/H48b_c2_dp0.0075_s4321-8000.ema.c4.ms8.score.json", "system_a", "PyTorch EMA greedy 160 ms"),
 "H48b-s1234-c2": ("h41/H48b_c2_dp0.0075_s1234-8000.ema.c2.ms8.score.json", "system_a", "PyTorch EMA greedy 80 ms"),
 "H48b-s4321-c2": ("h41/H48b_c2_dp0.0075_s4321-8000.ema.c2.ms8.score.json", "system_a", "PyTorch EMA greedy 80 ms"),
 "R0-320ms": ("r0-fulldev/R0-vs-B0-int8-e2ef608.score.json", "system_b", "R0 Zipformer INT8, 320 ms"),
 "A1C-c4": ("h41/A1C_dp0-8000.ema.c4.ms8.score.json", "system_a", "A1C PyTorch EMA greedy 160 ms"),
 "H48c-a-c4": ("h41/H48c_a_s1234-8000.ema.c4.ms8.score.json", "system_a", "H48c-a PyTorch EMA greedy 160 ms"),
 "B0-L4-c4": ("/home/ubuntu/jsr/target/ultra-v3/h46-native-l4.vs-b0.score.json", "system_b", "native L4 INT8 beam 4 + M5 + LODR (codex-full-fleet receipt)"),
}
ENGINE = {"B0-c2": "B0-A1-49000-c2", "A-L50-c4": "A-L50-c4", "H48c-a-c2": "H48c-a-c2", "A1C-c2": "A1C-c2"}
rows = []
for p in sorted(glob.glob(f"{D}/*.json")):
    name = os.path.basename(p)[:-5]
    if name.endswith(".utts"):
        continue
    d = json.load(open(p))
    if d.get("schema") != "ultra.h43.latency/v1":      # skip paired comparisons and other files
        continue
    a = d["aggregate"]
    w = WER.get(name.replace(".x5", ""))
    wer = None
    src = (w[0] if w and w[0] and w[0].startswith("/") else (f"{E}/{w[0]}" if w and w[0] else None))
    eng = ENGINE.get(name)
    if (not src or not os.path.exists(src)) and eng and os.path.exists(f"{D}/engine-wer/{eng}.score.json"):
        w = (f"h43/latency-fulldev/engine-wer/{eng}.score.json", "system_a", "engine INT8 transcripts from this run's own trace finals"); src = f"{E}/{w[0]}"
    if src and os.path.exists(src):
        s = json.load(open(src))[w[1]]; wer = {"pooled": round(100 * s["pooled_wer"], 3), "dev-clean": round(100 * s["datasets"]["dev-clean"]["wer"], 3), "dev-other": round(100 * s["datasets"]["dev-other"]["wer"], 3), "source": w[0], "path": w[2]}
    pc = a.get("paced_capture_clock_trace_host") or {}
    rows.append({"name": name, "utterances": a["utterances"], "wer_full_dev": wer,
                 "design_clock_ms": {"first_token_end": a["ftl_end"], "first_token_onset": a["ftl_onset"], "first_correct_end": a["fcw_end"], "first_correct_onset": a["fcw_onset"]},
                 "first_word_coverage": a["fcw_coverage"], "correct_by_stable": a["cq_stable"], "first_correct_by_end": a["first_cq_end"],
                 "stable_word_lag_ms": a["stable_lag"], "revision": a["revision"], "capture_clock_trace_host_ms": pc, "bootstrap_95": d.get("bootstrap_95"), "traces": len(d.get("traces", {}))})
out = {"schema": "ultra.h43.latency-fulldev-summary/v1", "label": "exploratory", "clocks": "design = audio sample clock (deterministic); capture = trace host wall clock (loaded hosts, NOT quiet)", "rows": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(len(rows), "rows")
for r in rows:
    dc = r["design_clock_ms"]; print(f"{r['name']:<16} FTL end {dc['first_token_end']['mean_ms']:>6} | FCW end {dc['first_correct_end']['mean_ms']:>6} p95 {dc['first_correct_end']['p95_ms']:>6} | cov {r['first_word_coverage']} | CQ300 {r['correct_by_stable']['300']} | WER {r['wer_full_dev']['pooled'] if r['wer_full_dev'] else '-'}")
