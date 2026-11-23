#!/usr/bin/env python3
"""h45-wins: write the pre-registered merge recipes (Arm A averaging, Arm B task arithmetic) as JSON (xbabe1 paths)."""
import json, os
OUT = "/mnt/xbabe3-external/jsr/h45-wins/recipes"
W = "/mnt/amphora-backup/h45-wins/weights/src"
J = "/mnt/amphora-backup/jsr-data/runs/train"
A1 = {49000: f"{W}/A1-ckpt-00049000.pt", 47500: f"{W}/A1-model-00047500.pt", 45000: f"{W}/A1-model-00045000.pt",
      42500: f"{W}/A1-model-00042500.pt", 40000: f"{W}/A1-model-00040000.pt"}
A1L = f"{W}/A1L-model-00008000.pt"
steps = [49000, 47500, 45000, 42500, 40000]
R = []
def rec(name, terms, note, **kw):
    R.append(dict(name=name, terms=[dict(ckpt=c, key=k, coef=w) for c, k, w in terms], note=note, **kw))
for k in (2, 3, 4, 5):
    rec(f"A-E{k}", [(A1[s], "ema", 1.0 / k) for s in steps[:k]], f"uniform mean of A1 EMA at the last {k} snapshots")
    rec(f"A-R{k}", [(A1[s], "model", 1.0 / k) for s in steps[:k]], f"uniform mean of A1 raw at the last {k} snapshots")
# post-hoc EMA (exponential), coefficients from bin/posthoc_ema.py (nonneg least squares on EMA kernels, sum 1)
P = json.load(open("/mnt/xbabe3-external/jsr/h45-wins/recipes/posthoc_coefs.json"))
for tau in ("4000", "8000"):
    c = P[tau]["coef"]  # ordered 40000 .. 49000
    rec(f"A-P{tau[0]}", [(A1[s], "ema", w) for s, w in zip([40000, 42500, 45000, 47500, 49000], c)],
        f"post-hoc EMA tau={tau} steps at 49k from the five EMA snapshots (fit err {P[tau]['relerr']:.3f})")
rec("A-M10", [(A1[s], key, 0.1) for s in steps for key in ("ema", "model")], "uniform mean of all ten (raw+EMA, five steps)")
rec("A-L50", [(A1[49000], "ema", 0.5), (A1L, "ema", 0.5)], "B0 EMA 0.5 + A1L-final EMA 0.5")
rec("A-L25", [(A1[49000], "ema", 0.75), (A1L, "ema", 0.25)], "B0 EMA 0.75 + A1L-final EMA 0.25")
# Arm B: task arithmetic (EMA primary, raw secondary for B1)
ctl, a10, a11 = f"{J}/H41_control/model-00008000.pt", f"{J}/H41_chunk_low/model-00008000.pt", f"{J}/H41_streamctx/model-00008000.pt"
a7 = f"{J}/H41_crctc_full/model-00008000.pt"
chk = f"{W}/H41_chunk-model-00008000.pt"
a16, a17, a18 = (f"{W}/H41_chunk_predctx4-model-00008000.pt", f"{W}/H41_chunk_left256-model-00008000.pt",
                 f"{W}/H41_chunk_specaug-model-00008000.pt")
for key, tag in (("ema", ""), ("model", "-raw")):
    rec(f"B1-M12{tag}", [(a10, key, 1.0), (a11, key, 1.0), (ctl, key, -1.0)],
        "control + tau10 + tau11 = arm10 + arm11 - control", config_from=a10)
# bundles onto the chunk arm (EMA): theta_chunk + sum_k (theta_k - theta_chunk)
def bundle(name, arms, extra7=False, note=""):
    terms = [(chk, "ema", 1.0)]
    for x in arms:
        terms += [(x, "ema", 1.0), (chk, "ema", -1.0)]
    if extra7:
        terms += [(a7, "ema", 1.0), (ctl, "ema", -1.0)]
    kw = {}
    kw["pred_ctx"] = 4 if a16 in arms else None
    kw["left_context"] = 256 if a17 in arms else 128
    kw["config_from"] = chk
    rec(name, terms, note, **{k: v for k, v in kw.items() if v is not None})
bundle("B2-full", [a16, a17, a18], note="chunk + tau16 + tau17 + tau18")
bundle("B2+7", [a16, a17, a18], extra7=True, note="B2-full + tau7 (CR-CTC full, vs control)")
bundle("B2-16", [a17, a18], note="leave out tau16 (pred_ctx 2, Rust-compatible)")
bundle("B2-17", [a16, a18], note="leave out tau17 (left 128)")
bundle("B2-18", [a16, a17], note="leave out tau18")
# seed check: chunk_s4321 + tau16r + tau18r + tau17(seed 1234)
chk_r, a16r, a18r = (f"{W}/H41_chunk_s4321-model-00008000.pt", f"{W}/H41_chunk_predctx4_s4321-model-00008000.pt",
                     f"{W}/H41_chunk_specaug_s4321-model-00008000.pt")
rec("B2-full-s4321", [(chk_r, "ema", 1.0), (a16r, "ema", 1.0), (chk_r, "ema", -1.0), (a18r, "ema", 1.0),
                      (chk_r, "ema", -1.0), (a17, "ema", 1.0), (chk, "ema", -1.0)],
    "seed check: chunk_s4321 + tau16r + tau18r (vs chunk_s4321) + tau17 (seed 1234)", pred_ctx=4, left_context=256,
    config_from=chk_r)
os.makedirs(OUT, exist_ok=True)
for r in R:
    with open(os.path.join(OUT, r["name"] + ".json"), "w") as fh:
        json.dump(r, fh, indent=1)
print(len(R), "recipes:", " ".join(r["name"] for r in R))
