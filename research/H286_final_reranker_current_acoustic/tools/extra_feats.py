"""H286 cheap extra reranker features (no extra model pass), flat order of feats.npz (first 16 candidates).
extra_feats.py DIR  -> DIR/x_mbr.npy, DIR/x_rank.npy, DIR/x_ins.npy
x_mbr : minus the expected word edit distance to the other candidates under the softmax of the fused first-pass
        score (base); duplicate texts are merged (logaddexp) first, as the engine does. Consensus / MBR signal.
x_rank: log(1 + first-pass rank).
x_ins : number of words not aligned to the posterior-weighted top-1 (inserted words vs top-1), a length-drift signal."""
import json, sys
import numpy as np

d = sys.argv[1]
f = np.load(f"{d}/feats.npz")
owner, pos, base = f["owner"], f["pos"], f["base"].astype(np.float64)
rows = [json.loads(l) for l in open(f"{d}/nbest.jsonl")]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


N = len(owner)
mbr, rank, ins = np.zeros(N), np.log1p(pos.astype(np.float64)), np.zeros(N)
start = np.searchsorted(owner, np.arange(owner.max() + 2))
for u in range(owner.max() + 1):
    a, b = start[u], start[u + 1]
    texts = [str(c[2]).split() for c in rows[u]["nbest"][:16]]
    assert len(texts) == b - a
    key = [" ".join(t) for t in texts]
    uniq = {}
    for i, k in enumerate(key):
        uniq[k] = np.logaddexp(uniq[k], base[a + i]) if k in uniq else base[a + i]
    ks = list(uniq)
    lp = np.array([uniq[k] for k in ks]); p = np.exp(lp - lp.max()); p /= p.sum()
    toks = [k.split() for k in ks]
    D = np.array([[lev(x, y) for y in toks] for x in toks], dtype=np.float64)
    e = D @ p
    idx = {k: i for i, k in enumerate(ks)}
    top = toks[int(p.argmax())]
    for i, k in enumerate(key):
        mbr[a + i] = -e[idx[k]]
        ins[a + i] = max(0, len(texts[i]) - len(top))
np.save(f"{d}/x_mbr.npy", mbr); np.save(f"{d}/x_rank.npy", rank); np.save(f"{d}/x_ins.npy", ins)
print("extra feats", d, N)
