"""TRAIN-only routing scan: total TRAIN word edits and routed fraction vs skip threshold, weights from MODEL.json.
skip_scan.py MODEL.json TRAIN_DIR"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import rerank_x as rx
m = json.load(open(sys.argv[1])); d = sys.argv[2]
owner, pos, X, rows = rx.load(d)
R = {r["uid"]: r for r in map(json.loads, open(f"{d}/refs.jsonl"))}
W = np.array([m["weights"][k] for k in rx.FEATS]); s = X @ W
base = X[:, 0]; nref = 0
G = [np.flatnonzero(owner == u) for u in range(owner.max() + 1)]
e_fit = np.array([rx.ed_of(rows, R, u, idx, s) for u, idx in enumerate(G)])
e_base = np.array([rx.ed_of(rows, R, u, idx, base) for u, idx in enumerate(G)])
gap = np.array([base[idx[0]] - base[idx[1]] if len(idx) > 1 else 1e9 for idx in G])
nref = sum(len(rx.normalise(R[r["uid"]]["ref"])) for r in rows)
for sk in (2.0, 3.0, 3.75, 5.0, 7.0, 10.0, 1e9):
    rt = gap < sk
    print(json.dumps({"skip": sk, "routed_frac": round(rt.mean(), 3), "train_wer": round(100 * (np.where(rt, e_fit, e_base).sum()) / nref, 4)}))
