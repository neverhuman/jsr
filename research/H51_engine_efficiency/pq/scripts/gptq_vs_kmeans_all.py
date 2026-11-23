"""Held-out relative output MSE (exact products) of cached k-means vs GPTQ fits, LN-input layers."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import h51lib as L
torch.set_num_threads(1)
model, p = L.load_model("/dev/shm/h51-pq/A1-49000-ema-as-model.pt")
rx = re.compile(r"(ff[12]\.w1|att\.qkv|att\.out|conv\.pw1)$")
def load(d, n):
    st = torch.load(os.path.join("/dev/shm/h51-pq/fits", d, L.short(n) + ".pt"), weights_only=False)
    f = L.PQFit.__new__(L.PQFit)
    for k, v in st.items(): setattr(f, k, v)
    f.codes = f.codes.long(); return f
out = []
for n in [x for x in L.target_names(model) if rx.search(x)]:
    lin = L.get_mod(model, n); x = torch.from_numpy(np.load(f"/dev/shm/h51-pq/trace/held.{L.short(n)}.npy")).float()
    y = x @ lin.weight.t() + lin.bias; var = ((y - y.mean(0)) ** 2).sum()
    r = {"layer": n}
    for tag, d in (("k2", "s4R2k16w"), ("g2", "s4R2k16wg"), ("k1", "s4R1k16w"), ("g1", "s4R1k16wg")):
        try:
            pq = L.PQLinear(load(d, n), lin.bias.data); pq.lut_exact = False
            with torch.no_grad(): r[tag] = float(((pq(x) - y) ** 2).sum() / var)
        except FileNotFoundError:
            r[tag] = None
    out.append(r)
json.dump(out, open("results/gptq/gptq_vs_kmeans_lnin.json", "w"), indent=1)
ok = [r for r in out if None not in r.values()]
m = lambda k: sum(r[k] for r in ok) / len(ok)
print(f"{len(ok)} layers: 2 bpw k-means {m('k2'):.4f} GPTQ {m('g2'):.4f} ({100*(1-m('g2')/m('k2')):.0f}% lower); 1 bpw k-means {m('k1'):.4f} GPTQ {m('g1'):.4f} ({100*(1-m('g1')/m('k1')):.0f}% lower)")
