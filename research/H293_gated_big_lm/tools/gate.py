"""H293: confidence-gated 27.5M L2R+R2L rescoring on top of frozen H261.

Score of a candidate = frozen H261 linear score (fit json) for routed rows (H244 skip on the first-pass gap), as in
H275 F3. Gate = H261 margin between the best and second-best unique text (duplicate texts merged by logaddexp, as
the engine does). The lowest-margin fraction f of routed TRAIN utterances defines a threshold; rows under it get
    score + a*big_l2r + b*big_r2l + c*nw
with (a, b, c) picked on TRAIN by a grid (speaker 2-fold held-out delta reported). DEV is applied once per f.

gate.py train TRAIN_DIR FROZEN.json OUT.json        -> thresholds and weights per f
gate.py apply S_DIR FROZEN.json GATE.json OUTPREFIX  -> OUTPREFIX_f{10,20,30,50}.score-input.jsonl + gate stats"""
import json, sys, os, zlib, itertools
import numpy as np
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H286_final_reranker_current_acoustic/tools"]
os.environ.setdefault("EXTRA", "")
import rerank_x as rx

FRACS = (0.10, 0.20, 0.30, 0.50)
A = np.round(np.arange(0, 1.01, 0.1), 2); C = (-1.0, -0.5, 0.0, 0.5, 1.0)


def load(d, frozen):
    owner, pos, X, rows = rx.load(d)
    m = json.load(open(frozen))
    s = X @ np.array([m["weights"][k] for k in rx.FEATS])
    big_l = np.load(f"{d}/x_big_l2r.npy"); big_r = np.load(f"{d}/x_big_r2l.npy")
    nw = X[:, rx.FEATS.index("nw")]
    G = []
    for u, idx, routed in rx.groups(owner, X[:, 0]):
        T = [str(rows[u]["nbest"][pos[i]][2]) for i in idx]
        G.append((u, idx, routed, T))
    return owner, pos, X, rows, s, big_l, big_r, nw, G


def uniq(T, sc):
    g = {}
    for i, (t, x) in enumerate(zip(T, sc)):
        k = tuple(t.split())
        if k in g:
            a, bt, bs, fi = g[k]; g[k] = (np.logaddexp(a, x), t if x > bs else bt, max(bs, x), fi)
        else:
            g[k] = (x, t, x, i)
    return g


def pick(T, sc):
    return max(uniq(T, sc).values(), key=lambda v: (v[0], -v[3]))[1]


def margin(T, sc):
    v = sorted((x[0] for x in uniq(T, sc).values()), reverse=True)
    return v[0] - v[1] if len(v) > 1 else np.inf


def train(d, frozen, out):
    owner, pos, X, rows, s, bl, br, nw, G = load(d, frozen)
    R = {r["uid"]: r for r in map(json.loads, open(f"{d}/refs.jsonl"))}
    half = lambda u: zlib.crc32(R[rows[u]["uid"]]["speaker"].encode()) % 2
    mg = np.array([margin(T, s[idx]) if routed else np.inf for u, idx, routed, T in G])
    routed_m = mg[np.isfinite(mg)]
    res = {"frozen": frozen, "routed_frac": float(np.isfinite(mg).mean()), "fracs": {}}
    ed_cache = {}

    def ed(u, h):
        k = (u, h)
        if k not in ed_cache:
            ed_cache[k] = sum(rx.edit_counts(rx.normalise(R[rows[u]["uid"]]["ref"]), rx.normalise(h)))
        return ed_cache[k]

    for f in FRACS:
        thr = float(np.quantile(routed_m, f))
        gated = [g for g, m_ in zip(G, mg) if m_ <= thr]
        base_e = np.zeros(2)
        for u, idx, _, T in gated:
            base_e[half(u)] += ed(u, pick(T, s[idx]))
        grid = {}
        for a, b, c in itertools.product(A, A, C):
            e = np.zeros(2)
            for u, idx, _, T in gated:
                e[half(u)] += ed(u, pick(T, s[idx] + a * bl[idx] + b * br[idx] + c * nw[idx]))
            grid[(a, b, c)] = e
        best = min(grid, key=lambda k: grid[k].sum())
        # speaker 2-fold: pick on one half, score on the other
        ho = 0.0
        for h in (0, 1):
            bh = min(grid, key=lambda k: grid[k][h])
            ho += grid[bh][1 - h] - base_e[1 - h]
        res["fracs"][str(f)] = {"thr": thr, "n_gated": len(gated), "a_l2r": float(best[0]), "b_r2l": float(best[1]), "c_nw": float(best[2]),
                                "train_delta_edits": float(grid[best].sum() - base_e.sum()), "heldout_delta_edits_2fold": float(ho)}
        print(json.dumps({f: res["fracs"][str(f)]}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def apply(d, frozen, gate, prefix):
    owner, pos, X, rows, s, bl, br, nw, G = load(d, frozen)
    gm = json.load(open(gate))
    R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
    mg = np.array([margin(T, s[idx]) if routed else np.inf for u, idx, routed, T in G])
    stats = {}
    for f, p in gm["fracs"].items():
        n = 0
        with open(f"{prefix}_f{int(float(f) * 100)}.score-input.jsonl", "w") as fh:
            for (u, idx, routed, T), m_ in zip(G, mg):
                if not routed:
                    h = pick(T, X[idx, 0])
                elif m_ <= p["thr"]:
                    n += 1; h = pick(T, s[idx] + p["a_l2r"] * bl[idx] + p["b_r2l"] * br[idx] + p["c_nw"] * nw[idx])
                else:
                    h = pick(T, s[idx])
                rr = R[rows[u]["uid"]]
                fh.write(json.dumps({"id": rows[u]["uid"], "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"], "hyp": h,
                                     "duration_s": rr["samples"] / 16000}) + "\n")
        stats[f] = {"gated": n, "gated_frac_of_all": n / len(G), "gated_frac_of_routed": n / int(np.isfinite(mg).sum())}
    json.dump(stats, open(f"{prefix}_gatestats.json", "w"), indent=1); print(json.dumps(stats))


if __name__ == "__main__":
    if sys.argv[1] == "train": train(*sys.argv[2:5])
    else: apply(*sys.argv[2:6])
