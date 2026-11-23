"""Compare plain activation-weighted k-means PQ with GPTQ-style error-feedback PQ (same format)
on held-out TRAIN rows: relative output MSE of the exact pqscan arithmetic, per layer.

usage: python -I gptq_test.py <trace_dir> <layers,...> <formats e.g. 4:2,4:1,8:2> <out_json>
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

trace, layers, fmts, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
model, p = L.load_model(os.path.join(os.environ.get("H51_ROOT", "/dev/shm/h51-pq"), "A1-49000-ema-as-model.pt"))
res = []
for name in layers:
    lin = L.get_mod(model, name)
    W = lin.weight.data
    sig = torch.from_numpy(np.load(os.path.join(trace, f"sigma.{L.short(name)}.npy")))
    x = torch.from_numpy(np.load(os.path.join(trace, f"held.{L.short(name)}.npy"))).float()
    with torch.no_grad():
        y = x @ W.float().t() + lin.bias.float()
        var = (y - y.mean(0)).pow(2).sum()
        y8 = L.Int8Emu(lin)(x)
    row = {"layer": name, "rel_mse_int8": float((y8 - y).pow(2).sum() / var)}
    for f in fmts:
        s, R = map(int, f.split(":"))
        for kind in ("kmeans", "gptq"):
            t0 = time.time()
            fit = L.PQFit(W, sig, s=s, R=R) if kind == "kmeans" else L.PQFitGPTQ(W, sig, s=s, R=R)
            pq = L.PQLinear(fit, lin.bias.data)
            with torch.no_grad():
                ypq = pq(x)
            row[f"s{s}R{R}_{kind}"] = float((ypq - y).pow(2).sum() / var)
            row[f"s{s}R{R}_{kind}_s"] = round(time.time() - t0, 1)
    res.append(row)
    print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
