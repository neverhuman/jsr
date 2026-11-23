"""LUT-contract variants x activation encodings: exact relative output MSE on held-out TRAIN rows."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch  # noqa: E401,E402
import h51lib as L  # noqa: E402
trace, layers, fmts, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
model, p = L.load_model("/dev/shm/h51-pq/A1-49000-ema-as-model.pt")
H16 = torch.tensor([[1.0]])
for _ in range(4):
    H16 = torch.cat([torch.cat([H16, H16], 1), torch.cat([H16, -H16], 1)], 0)
H16 = (H16 / 4.0).double()
res = []
for name in layers:
    lin = L.get_mod(model, name); W = lin.weight.data.double(); b = lin.bias.data
    sig = torch.from_numpy(np.load(os.path.join(trace, f"sigma.{L.short(name)}.npy"))).double()
    x = torch.from_numpy(np.load(os.path.join(trace, f"held.{L.short(name)}.npy"))).double()
    if name == "encoder.stem.out":  # engine column order f*ch+c (subspaces follow the deployed order)
        ch, f2 = 128, 19
        pm = torch.tensor([c * f2 + f for f in range(f2) for c in range(ch)])
        W, x, sig = W[:, pm], x[:, pm], sig[pm][:, pm]
    y = x @ W.t() + b.double(); var = (y - y.mean(0)).pow(2).sum()
    Hb = torch.block_diag(*[H16] * (W.shape[1] // 16))
    row = {"layer": name}
    for f in fmts:
        s, R = map(int, f.split(":"))
        for enc in ("centred", "sym", "fwht16_centred"):
            rot = enc.startswith("fwht16")
            Wr, xr, sr = (W @ Hb.t(), x @ Hb.t(), Hb @ sig @ Hb.t()) if rot else (W, x, sig)
            fit = L.PQFit(Wr.float(), sr.float(), s=s, R=R)
            pq = L.PQLinear(fit, b); pq.act = "sym" if enc == "sym" else "centred"
            with torch.no_grad():
                pq.lut_exact = False
                row[f"s{s}R{R}/{enc}/nolut"] = float(((pq(xr.float()).double() - y) ** 2).sum() / var)
                pq.lut_exact = True
                for bits in (7, 8):
                    for rnd in ("trunc_bias", "nearest"):
                        pq.lut_bits, pq.lut_round = bits, rnd
                        row[f"s{s}R{R}/{enc}/{bits}b-{rnd}"] = float(((pq(xr.float()).double() - y) ** 2).sum() / var)
    res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
