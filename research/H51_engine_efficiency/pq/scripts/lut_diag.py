"""Diagnose the pqscan LUT rounding: held-out relative output MSE of one layer for
(a) the PQ weights with exact integer products (no LUT rounding), (b) the exact 7-bit / one-shift
LUT, with and without a block-16 Hadamard rotation of the GEMM input (W -> W H^T, x -> H x),
plus the per-frame shift distribution and how many LUT levels the median subspace gets.

usage: python -I lut_diag.py <trace_dir> <layers,...> <formats e.g. 4:1,4:2> <out_json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

trace, layers, fmts, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
model, p = L.load_model(os.path.join(os.environ.get("H51_ROOT", "/dev/shm/h51-pq"), "A1-49000-ema-as-model.pt"))
H16 = torch.tensor([[1.0]])
for _ in range(4):
    H16 = torch.cat([torch.cat([H16, H16], 1), torch.cat([H16, -H16], 1)], 0)
H16 = H16 / 4.0


def blockH(k):
    return torch.block_diag(*[H16] * (k // 16)).double()


res = []
for name in layers:
    lin = L.get_mod(model, name)
    W = lin.weight.data.double()
    b = lin.bias.data
    sig = torch.from_numpy(np.load(os.path.join(trace, f"sigma.{L.short(name)}.npy"))).double()
    x = torch.from_numpy(np.load(os.path.join(trace, f"held.{L.short(name)}.npy"))).double()
    y = x @ W.t() + b.double()
    var = (y - y.mean(0)).pow(2).sum()
    row = {"layer": name, "rel_mse_int8": float(((L.Int8Emu(lin)(x.float()).double() - y) ** 2).sum() / var)}
    for rot in ("none", "fwht16"):
        if rot == "none":
            Wr, xr, sr = W, x, sig
        else:
            Hb = blockH(W.shape[1])
            Wr, xr, sr = W @ Hb.t(), x @ Hb.t(), Hb @ sig @ Hb.t()
        for f in fmts:
            s, R = map(int, f.split(":"))
            fit = L.PQFit(Wr.float(), sr.float(), s=s, R=R)
            pq = L.PQLinear(fit, b)
            with torch.no_grad():
                pq.lut_exact = False
                ynl = pq(xr.float()).double()
                pq.lut_exact = True
                ylut = pq(xr.float()).double()
            sh = pq.stats["sh_hist"]
            key = f"{rot}/s{s}R{R}"
            row[key] = {"nolut": float(((ynl - y) ** 2).sum() / var),
                        "lut": float(((ylut - y) ** 2).sum() / var),
                        "shift_median": int(np.searchsorted(np.cumsum(sh), sh.sum() / 2)),
                        "shift_hist": {int(i): int(v) for i, v in enumerate(sh) if v}}
            # LUT levels the median subspace actually uses at the frame's shift
            a, sa, ca = L.quant_rows_centred(xr.float())
            v = torch.einsum("tci,rcji->trcj", a.view(a.shape[0], fit.C, fit.s), fit.cb_int.float())
            rng = (v.amax(-1) - v.amin(-1))  # (T, R, C)
            mxr = rng.amax(-1).amax(-1)
            shv = torch.clamp(torch.ceil(torch.log2((mxr + 1) / 128.0)), min=0)
            levels = rng / (2.0 ** shv)[:, None, None]
            row[key]["median_levels_per_subspace"] = float(levels.median())
            row[key]["p10_levels_per_subspace"] = float(levels.flatten().quantile(0.1))
    res.append(row)
    print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
