"""Exact-LUT pqscan error under three activation encodings: deployed centred int8, zero-anchored
symmetric int8, and FWHT16-rotated centred int8 (held-out TRAIN rows, relative output MSE)."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch  # noqa: E401,E402
import h51lib as L  # noqa: E402
trace, layers, fmts, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
model, p = L.load_model(os.path.join(os.environ.get("H51_ROOT", "/dev/shm/h51-pq"), "A1-49000-ema-as-model.pt"))
H16 = torch.tensor([[1.0]])
for _ in range(4):
    H16 = torch.cat([torch.cat([H16, H16], 1), torch.cat([H16, -H16], 1)], 0)
H16 = (H16 / 4.0).double()
res = []
for name in layers:
    lin = L.get_mod(model, name); W = lin.weight.data.double(); b = lin.bias.data
    sig = torch.from_numpy(np.load(os.path.join(trace, f"sigma.{L.short(name)}.npy"))).double()
    x = torch.from_numpy(np.load(os.path.join(trace, f"held.{L.short(name)}.npy"))).double()
    y = x @ W.t() + b.double(); var = (y - y.mean(0)).pow(2).sum()
    row = {"layer": name}
    Hb = torch.block_diag(*[H16] * (W.shape[1] // 16))
    for f in fmts:
        s, R = map(int, f.split(":"))
        for enc in ("centred", "sym", "fwht16_centred", "fwht16_sym"):
            rot = enc.startswith("fwht16")
            Wr, xr, sr = (W @ Hb.t(), x @ Hb.t(), Hb @ sig @ Hb.t()) if rot else (W, x, sig)
            fit = L.PQFit(Wr.float(), sr.float(), s=s, R=R)
            pq = L.PQLinear(fit, b); pq.act = "sym" if enc.endswith("sym") else "centred"
            with torch.no_grad():
                pq.lut_exact = True; yl = pq(xr.float()).double()
                pq.lut_exact = False; yn = pq(xr.float()).double()
            row[f"s{s}R{R}/{enc}"] = {"lut": float(((yl - y) ** 2).sum() / var), "nolut": float(((yn - y) ** 2).sum() / var)}
    res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
