"""PyTorch A/B on a DEV subset: does the exact 7-bit LUT rounding of pqscan change decoding versus
the LUT-free PQ product (and the engine's PQ->int8 requant path)? All other GEMMs: engine INT8
emulation. Uses the cached PQ fits of a sweep.

usage: python -I lut_ab.py <fit_dir> <layers regex> <npz> <threads> <out_json>
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, "/mnt/xbabe3-external/jsr/h51-pq/scripts")
import torch  # noqa: E402

import h51lib as L  # noqa: E402

fit_dir, rx, npz, threads, out = sys.argv[1], re.compile(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(threads)
model, p = L.load_model("/dev/shm/h51-pq/A1-49000-ema-as-model.pt")
names = L.apply_int8(model)
sel = [n for n in names if rx.search(n)]
utts = L.load_npz_utts(npz)
exs, tok, _ = L.featurize(p, utts)


TRACE = os.environ.get("H51_TRACE", "/dev/shm/h51-pq/trace")
S_, R_ = int(os.environ.get("H51_S", "4")), int(os.environ.get("H51_R", "2"))


def load_fit(n):
    fp = os.path.join(fit_dir, L.short(n) + ".pt")
    if not os.path.exists(fp):  # fit here (same seeded fit as the sweep) and cache
        import numpy as np
        os.makedirs(fit_dir, exist_ok=True)
        w = model._fp32[n].weight.data
        pm = perm_for(n)
        sig = torch.from_numpy(np.load(os.path.join(TRACE, f"sigma.{L.short(n)}.npy")))
        if pm is not None:
            w, sig = w[:, pm], sig[pm][:, pm]
        f = L.PQFit(w, sig, s=S_, R=R_)
        st = {k: getattr(f, k) for k in ("N", "K", "s", "R", "C", "kc", "codes", "cb_int", "cb_scale", "gain", "cb_float")}
        st["codes"] = f.codes.to(torch.uint8)
        torch.save(st, fp)
    st = torch.load(fp, weights_only=False)
    f = L.PQFit.__new__(L.PQFit)
    for k, v in st.items():
        setattr(f, k, v)
    f.codes = f.codes.long()
    return f


def perm_for(n):
    if n != "encoder.stem.out":
        return None
    ch, f2 = 128, 19
    perm = torch.empty(ch * f2, dtype=torch.long)
    for c in range(ch):
        for f in range(f2):
            perm[f * ch + c] = c * f2 + f
    return perm


class Permuted(torch.nn.Module):
    def __init__(self, inner, perm):
        super().__init__()
        self.inner, self.perm = inner, perm

    def forward(self, x):
        return self.inner(x[..., self.perm])


res = {}
t0 = time.time()
hyp_a = L.decode_panel(model, exs, tok, batch=8)
_, res["int8"] = L.wer_rows(utts, hyp_a)
res["int8"]["s"] = round(time.time() - t0, 1)
pqs = {}
for n in sel:
    pq = L.PQLinear(load_fit(n), model._fp32[n].bias.data)
    pm = perm_for(n)
    pqs[n] = pq
    L.set_mod(model, n, Permuted(pq, pm) if pm is not None else pq)
for mode in ("pq_exact_lut", "pq_exact_lut_sym", "pq_nolut"):
    for pq in pqs.values():
        pq.lut_exact = mode != "pq_nolut"
        pq.act = "sym" if mode == "pq_exact_lut_sym" else "centred"
    t0 = time.time()
    hyps = L.decode_panel(model, exs, tok, batch=8)
    _, res[mode] = L.wer_rows(utts, hyps)
    res[mode]["s"] = round(time.time() - t0, 1)
    res[mode]["same_hyps_vs_int8"] = sum(a == b for a, b in zip(hyps, hyp_a))
    if mode == "pq_exact_lut":
        hyp_lut = hyps
    else:
        res[mode]["same_hyps_vs_exact_lut"] = sum(a == b for a, b in zip(hyps, hyp_lut))
res["layers"] = len(sel)
res["npz"] = npz
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v["pooled"]["wer"] if isinstance(v, dict) and "pooled" in v else v) for k, v in res.items()}))
