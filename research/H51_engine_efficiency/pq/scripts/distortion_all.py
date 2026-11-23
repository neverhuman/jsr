"""Held-out TRAIN output distortion for every target GEMM (cached seeded fits; fit if missing):
INT8, PQ with exact products, PQ under the H24 LUT (7-bit, truncate + global bias), and PQ under the
proposed round-to-nearest 7-bit LUT. One JSON line per layer.

usage: python -I distortion_all.py <s> <R> <shard i/n> <out_jsonl>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

s, R, shard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
torch.set_num_threads(1)
ROOT = "/dev/shm/h51-pq"
model, p = L.load_model(os.path.join(ROOT, "A1-49000-ema-as-model.pt"))
names = L.target_names(model)
si, sn = map(int, shard.split("/"))
fdir = os.path.join(ROOT, "fits", f"s{s}R{R}k16w")
os.makedirs(fdir, exist_ok=True)
KEYS = ("N", "K", "s", "R", "C", "kc", "codes", "cb_int", "cb_scale", "gain", "cb_float")
done = set()
if os.path.exists(out):
    done = {json.loads(l)["layer"] for l in open(out) if l.strip()}
for n in names[si::sn]:
    if n in done:
        continue
    lin = L.get_mod(model, n)
    w, b = lin.weight.data, lin.bias.data
    sig = torch.from_numpy(np.load(os.path.join(ROOT, "trace", f"sigma.{L.short(n)}.npy")))
    x = torch.from_numpy(np.load(os.path.join(ROOT, "trace", f"held.{L.short(n)}.npy"))).float()
    if n == "encoder.stem.out":
        ch, f2 = 128, 19
        pm = torch.tensor([c * f2 + f for f in range(f2) for c in range(ch)])
        w, sig, x = w[:, pm], sig[pm][:, pm], x[:, pm]
    fp = os.path.join(fdir, L.short(n) + ".pt")
    if os.path.exists(fp):
        st = torch.load(fp, weights_only=False)
        fit = L.PQFit.__new__(L.PQFit)
        for k, v in st.items():
            setattr(fit, k, v)
        fit.codes = fit.codes.long()
    else:
        fit = L.PQFit(w, sig, s=s, R=R)
        st = {k: getattr(fit, k) for k in KEYS}
        st["codes"] = fit.codes.to(torch.uint8)
        torch.save(st, fp + ".tmp")
        os.replace(fp + ".tmp", fp)
    with torch.no_grad():
        y = x @ w.float().t() + b.float()
        var = (y - y.mean(0)).pow(2).sum()
        q = torch.nn.Linear(w.shape[1], w.shape[0])
        q.weight.data, q.bias.data = w.clone(), b.clone()
        r = {"layer": n, "n": int(w.shape[0]), "k": int(w.shape[1]), "rows": int(x.shape[0]),
             "rel_mse_int8": float(((L.Int8Emu(q)(x) - y) ** 2).sum() / var)}
        pq = L.PQLinear(fit, b)
        pq.lut_exact = False
        r["pq_exact_products"] = float(((pq(x) - y) ** 2).sum() / var)
        pq.lut_exact = True
        pq.lut_bits, pq.lut_round = 7, "trunc_bias"
        r["pq_lut_h24"] = float(((pq(x) - y) ** 2).sum() / var)
        pq.lut_round = "nearest"
        r["pq_lut_nearest7"] = float(((pq(x) - y) ** 2).sum() / var)
        r["bytes_int8"] = int(w.shape[0] * w.shape[1])
        r["bytes_pq"] = fit.code_bytes()
    with open(out, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "bytes_pq"}), flush=True)
