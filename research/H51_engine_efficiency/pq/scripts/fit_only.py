"""Fill the PQ fit cache in parallel (same seeded fit as engine_sens.get_fit, so results are identical).

usage: python -I fit_only.py <s> <R> <regex> <shard i/n> [--gptq]
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

s, R, rx, shard = int(sys.argv[1]), int(sys.argv[2]), re.compile(sys.argv[3]), sys.argv[4]
gptq = "--gptq" in sys.argv
torch.set_num_threads(1)
ROOT = os.environ.get("H51_ROOT", "/dev/shm/h51-pq")
model, p = L.load_model(os.path.join(ROOT, "A1-49000-ema-as-model.pt"))
names = [n for n in L.target_names(model) if rx.search(n)]
si, sn = map(int, shard.split("/"))
fdir = os.path.join(ROOT, "fits", f"s{s}R{R}k16w{'g' if gptq else ''}")
os.makedirs(fdir, exist_ok=True)
KEYS = ("N", "K", "s", "R", "C", "kc", "codes", "cb_int", "cb_scale", "gain", "cb_float")
for n in names[si::sn]:
    fp = os.path.join(fdir, L.short(n) + ".pt")
    if os.path.exists(fp):
        continue
    w = L.get_mod(model, n).weight.data
    sig = torch.from_numpy(np.load(os.path.join(ROOT, "trace", f"sigma.{L.short(n)}.npy")))
    if n == "encoder.stem.out":
        ch, f2 = 128, 19
        pm = torch.tensor([c * f2 + f for f in range(f2) for c in range(ch)])
        w, sig = w[:, pm], sig[pm][:, pm]
    fit = L.PQFitGPTQ(w, sig, s=s, R=R) if gptq else L.PQFit(w, sig, s=s, R=R, kc=16, weighted=True)
    st = {k: getattr(fit, k) for k in KEYS}
    st["codes"] = fit.codes.to(torch.uint8)
    torch.save(st, fp + ".tmp")
    os.replace(fp + ".tmp", fp)
    print("fit", n, flush=True)
