"""H261 reranker: linear score over per-hypothesis features, fit by expected-edit (MWER) minimisation on TRAIN
n-best lists only, then applied unchanged to DEV S at H244's routing (fused top1-top2 < 3.75) with the engine's
duplicate-text logaddexp aggregation.
rerank.py fit TRAIN_DIR OUT.json         (TRAIN_DIR has nbest.jsonl, refs.jsonl, feats.npz, m5.npy, l2r_int8.npy)
rerank.py apply MODEL.json S_DIR OUT.score-input.jsonl [--baseline]"""
import json, sys
import numpy as np, torch
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from ultra.wer import edit_counts, normalise
FEATS = ["base", "l2r", "m5", "nw", "l2r_content", "nw_content", "logfreq_content", "oov_content", "am_c32"]
INIT = {"base": 1.0, "l2r": 0.8, "m5": -0.8}
SKIP = 3.75

def load(d):
    f = np.load(f"{d}/feats.npz")
    X = {k: f[k].astype(np.float64) for k in ("base", "nw", "l2r_content", "nw_content", "logfreq_content", "oov_content")}
    X["l2r"] = np.load(f"{d}/l2r_int8.npy").astype(np.float64)
    X["m5"] = np.load(f"{d}/m5.npy").astype(np.float64)
    am = np.load(f"{d}/am_feats.npz")
    assert (am["owner"] == f["owner"]).all() and (am["pos"] == f["pos"]).all()
    X["am_c32"] = am["am_c32"].astype(np.float64)
    rows = [json.loads(l) for l in open(f"{d}/nbest.jsonl")]
    return f["owner"], f["pos"], np.stack([X[k] for k in FEATS], 1), rows

def groups(owner, base):
    out = []
    for u in range(owner.max() + 1):
        idx = np.flatnonzero(owner == u)
        routed = len(idx) > 1 and base[idx[0]] - base[idx[1]] < SKIP
        out.append((u, idx, routed))
    return out

def fit(d, out, keep=None, lam=None):
    owner, pos, X, rows = load(d)
    R = {r["uid"]: r for r in map(json.loads, open(f"{d}/refs.jsonl"))}
    G = [g for g in groups(owner, X[:, 0]) if g[2] and (keep is None or keep(R[rows[g[0]]["uid"]]["speaker"]))]
    ed = np.zeros(len(owner))
    for u, idx, _ in G:
        ref = normalise(R[rows[u]["uid"]]["ref"])
        for i in idx:
            ed[i] = sum(edit_counts(ref, normalise(str(rows[u]["nbest"][pos[i]][2]))))
    G = [g for g in G if ed[g[1]].max() > ed[g[1]].min()]  # only lists where the choice matters
    mu = X.std(0) + 1e-6
    Xt = torch.tensor(X / mu); E = torch.tensor(ed)
    w0 = torch.tensor([INIT.get(k, 0.0) for k in FEATS]) * torch.tensor(mu)
    w = w0.clone().requires_grad_(True)
    seg = [torch.tensor(g[1]) for g in G]
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
    lam = float(__import__("os").environ.get("LAM", "1e-2")) if lam is None else lam
    def closure():
        opt.zero_grad()
        loss = 0.0
        for s in seg:
            p = torch.softmax(Xt[s] @ w, 0)
            loss = loss + (p * E[s]).sum()
        loss = loss / len(seg) + lam * ((w - w0) ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    W = (w.detach() / torch.tensor(mu)).numpy()
    # report argmax edits on TRAIN: init vs fitted
    def am(Wv):
        s = X @ Wv; return sum(ed[g[1][np.argmax(s[g[1]])]] for g in G)
    Wi = np.array([INIT.get(k, 0.0) for k in FEATS])
    res = {"features": FEATS, "weights": dict(zip(FEATS, map(float, W))), "init": dict(zip(FEATS, map(float, Wi))),
           "train_lists_used": len(G), "train_argmax_edits_init": float(am(Wi)), "train_argmax_edits_fit": float(am(W)),
           "lambda": lam, "skip": SKIP}
    if out: json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
    return W, X, ed, owner, rows, R

def agg(texts, sc):
    g = {}
    for i, (t, x) in enumerate(zip(texts, sc)):
        k = tuple(t.split())
        if k in g:
            a, bt, bs, fi = g[k]; g[k] = (np.logaddexp(a, x), t if x > bs else bt, max(bs, x), fi)
        else:
            g[k] = (x, t, x, i)
    return max(g.values(), key=lambda v: (v[0], -v[3]))[1]

def apply(model, d, out, baseline=False):
    m = json.load(open(model))
    W = np.array([(m["init"] if baseline else m["weights"])[k] for k in FEATS])
    owner, pos, X, rows = load(d)
    R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
    s = X @ W
    with open(out, "w") as fh:
        for u, idx, routed in groups(owner, X[:, 0]):
            r = rows[u]; T = [str(r["nbest"][pos[i]][2]) for i in idx]
            h = agg(T, s[idx] if routed else X[idx, 0])
            rr = R[r["uid"]]
            fh.write(json.dumps({"id": r["uid"], "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"], "hyp": h, "duration_s": rr["samples"] / 16000}) + "\n")

def cv(d):
    """TRAIN-only speaker 2-fold choice of lambda: fit on one speaker half, argmax edits on the other."""
    import zlib
    half = lambda s: zlib.crc32(s.encode()) % 2
    for lam in (1e-2, 1e-3, 1e-4, 1e-5, 0.0):
        tot = 0
        for h in (0, 1):
            W, X, ed, owner, rows, R = fit(d, None, keep=lambda s, h=h: half(s) == h, lam=lam)
            Wi = np.array([INIT.get(k, 0.0) for k in FEATS])
            for Wv, tag in ((W, "fit"),):
                s_ = X @ Wv; si = X @ Wi
                for u, idx, routed in groups(owner, X[:, 0]):
                    if not routed or half(R[rows[u]["uid"]]["speaker"]) == h: continue
                    tot += ed_of(rows, R, u, idx, s_) - ed_of(rows, R, u, idx, si)
        print(json.dumps({"lambda": lam, "heldout_edit_delta_vs_init": tot}), flush=True)

def ed_of(rows, R, u, idx, sc):
    r = rows[u]; T = [str(c[2]) for c in r["nbest"][:16]]
    h = agg([T[i - idx[0]] for i in idx], sc[idx])
    return sum(edit_counts(normalise(R[r["uid"]]["ref"]), normalise(h)))

if __name__ == "__main__":
    if sys.argv[1] == "fit": fit(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "cv": cv(sys.argv[2])
    else: apply(sys.argv[2], sys.argv[3], sys.argv[4], "--baseline" in sys.argv)
