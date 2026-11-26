"""H68: neural-LM n-best rescoring with speaker-disjoint 2-fold CV weight selection.

score = base (fused beam score) + a * L2R + b * R2L + c * n_words   (L2R/R2L include the EOS boundary)
Weights are chosen on one speaker fold and applied to the other (and vice versa) -> CV hypotheses.
Writes ultra-score score-input JSONL for: top-1 baseline, CV-rescored, full-S-tuned (optimistic).
Usage: score_nbest.py --nbest X.jsonl --refs refs.jsonl --out DIR [--l2r ckpt] [--r2l ckpt]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = [os.path.dirname(__file__), "/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from train_lm import load_lm  # noqa: E402
from ultra.wer import edit_counts, normalise  # noqa: E402


@torch.no_grad()
def lm_scores(model, seqs: list[list[int]], reverse: bool, bs: int = 256, prefixes: list[list[int]] | None = None) -> np.ndarray:
    """Sum log P over the hypothesis tokens + closing boundary. With `prefixes` (L2R only), the
    previous committed text is fed as context ([0] prev [0] hyp [0]) and only the hyp part is scored."""
    dev = next(model.parameters()).device
    ctx = model.cfg["ctx"]
    full, skip = [], []
    for i, s in enumerate(seqs):
        body = [0] + (s[::-1] if reverse else list(s)) + [0]
        pre = list(prefixes[i])[-(ctx // 2):] if prefixes is not None else []
        pre = ([0] + pre) if pre else []
        full.append(pre + body)
        skip.append(len(pre))
    out = np.zeros(len(full))
    order = sorted(range(len(full)), key=lambda i: len(full[i]))
    for off in range(0, len(order), bs):
        idx = order[off:off + bs]
        L = min(ctx + 1, max(len(full[i]) for i in idx))
        x = torch.zeros(len(idx), L, dtype=torch.long)
        m = torch.zeros(len(idx), L - 1)
        for r, i in enumerate(idx):
            s = full[i][:L]
            x[r, :len(s)] = torch.tensor(s)
            m[r, skip[i]:len(s) - 1] = 1
        x, m = x.to(dev), m.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lo = model(x[:, :-1]).float()
        lp = F.log_softmax(lo, -1).gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        out[idx] = (lp * m).sum(1).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", required=True)
    ap.add_argument("--refs", default="/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l2r")
    ap.add_argument("--r2l")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--m5", help="precomputed m5_feats.py .npy (same nbest, same topk)")
    ap.add_argument("--sub-grid", default="0,0.2,0.4,0.6", help="multiples of the M5 ln-prob subtracted from base")
    ap.add_argument("--reuse-feats", help="feats.npz from an earlier run on the same nbest (skips LM scoring)")
    ap.add_argument("--cache-grid", default="0,0.5,1,2,4")
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--cache", action="store_true", help="H77: causal chapter name-cache feature (rare words seen in earlier top-1 hyps)")
    ap.add_argument("--family", choices=["free", "tied", "engine"], default="free",
                    help="tied: b = a; engine: b = a and no word bonus (exactly crates/ultra-engine commit/tf.rs)")
    ap.add_argument("--session", action="store_true", help="L2R gets the previous utterance (same chapter) top-1 text as context")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = {r["uid"]: r for r in map(json.loads, open(a.refs))}
    rows = [json.loads(x) for x in open(a.nbest) if x.strip()]
    rows = [r for r in rows if r["uid"] in R]
    assert len(rows) == len(R), (len(rows), len(R))
    flat, owner = [], []
    for ui, r in enumerate(rows):
        for c in r["nbest"][:a.topk]:
            flat.append(c); owner.append(ui)
    owner = np.array(owner)
    base = np.array([float(c[1]) for c in flat])
    nw = np.array([len(str(c[2]).split()) for c in flat], dtype=float)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feats = {"base": base, "nw": nw}
    reuse = np.load(a.reuse_feats) if a.reuse_feats else None
    for name, path in (("l2r", a.l2r), ("r2l", a.r2l)):
        if path and reuse is not None and name in reuse:
            feats[name] = reuse[name]
            continue
        if path:
            m = load_lm(path, dev)
            pref = None
            if a.session and not m.reverse:
                top = {r["uid"]: list(map(int, r["nbest"][0][0])) for r in rows}
                def prev(uid: str) -> list[int]:
                    spk, ch, k = uid.split("-")
                    return top.get(f"{spk}-{ch}-{int(k) - 1:04d}", [])
                pref = [prev(rows[o]["uid"]) for o in owner]
            feats[name] = lm_scores(m, [list(map(int, c[0])) for c in flat], reverse=m.reverse, prefixes=pref)
            del m
    if a.cache:
        from collections import Counter
        wc = Counter()
        for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
            wc.update(normalise(ln))
        order = sorted(range(len(rows)), key=lambda i: tuple(int(x) for x in rows[i]["uid"].split("-")))
        seen: dict[str, set] = {}
        cache_of = {}
        for i in order:
            spk, ch, _ = rows[i]["uid"].split("-")
            key = f"{spk}-{ch}"
            cache_of[i] = set(seen.get(key, set()))
            seen.setdefault(key, set()).update(w for w in normalise(str(rows[i]["nbest"][0][2])) if wc[w] < 100)
        feats["cache"] = np.array([sum(1 for w in normalise(str(c[2])) if w in cache_of[o]) for c, o in zip(flat, owner)], dtype=float)
    if a.m5:
        feats["m5"] = np.load(a.m5)
        assert len(feats["m5"]) == len(flat), (len(feats["m5"]), len(flat))
    np.savez(out / "feats.npz", owner=owner, **feats)
    refw = [normalise(R[r["uid"]]["ref"]) for r in rows]
    ed = np.array([sum(edit_counts(refw[o], normalise(str(c[2])))) for c, o in zip(flat, owner)])
    spk = np.array([str(R[r["uid"]]["speaker"]) for r in rows])
    uspk = sorted(set(spk))
    fold_of_spk = {s: i % 2 for i, s in enumerate(uspk)}
    ufold = np.array([fold_of_spk[s] for s in spk])
    starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]

    pos_in_utt = np.arange(len(owner)) - np.repeat(starts, np.diff(np.r_[starts, len(owner)]))

    def pick(score: np.ndarray) -> np.ndarray:
        # per utterance: highest score, ties -> earliest candidate
        o = np.lexsort((pos_in_utt, -score, owner))
        return o[np.r_[0, np.flatnonzero(np.diff(owner[o])) + 1]]

    grid_a = [0.0] + ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0] if "l2r" in feats else [])
    grid_b = [0.0] + ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0] if "r2l" in feats else [])
    grid_c = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    grid_s = [float(x) for x in a.sub_grid.split(",")] if "m5" in feats else [0.0]
    grid_k = [float(x) for x in a.cache_grid.split(",")] if "cache" in feats else [0.0]
    results = []
    if a.family != "free":
        grid_b = [None]
    if a.family == "engine":
        grid_c = [0.0]
    for wa, wb, wc, ws, wk in itertools.product(grid_a, grid_b, grid_c, grid_s, grid_k):
        if wb is None:
            wb = wa if "r2l" in feats else 0.0
        s = base + wc * nw + (wa * feats["l2r"] if wa else 0) + (wb * feats["r2l"] if wb else 0) - (ws * feats["m5"] if ws else 0) + (wk * feats["cache"] if wk else 0)
        idx = pick(s)
        e = ed[idx]
        results.append(((wa, wb, wc, ws, wk), int(e[ufold == 0].sum()), int(e[ufold == 1].sum()), idx))
    nref = sum(len(w) for w in refw)
    top1 = pick(base)
    bestfull = min(results, key=lambda t: t[1] + t[2])
    b0 = min(results, key=lambda t: t[1])  # tuned on fold 0 -> apply to fold 1
    b1 = min(results, key=lambda t: t[2])
    cv = np.where(ufold == 1, b0[3], b1[3])
    oracle = np.minimum.reduceat(ed, starts).sum()

    def dump(name: str, idx: np.ndarray) -> None:
        with open(out / f"{name}.score-input.jsonl", "w") as fh:
            for ui, r in enumerate(rows):
                rr = R[r["uid"]]
                fh.write(json.dumps({"id": r["uid"], "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"], "hyp": str(flat[idx[ui]][2]), "duration_s": rr["samples"] / 16000}) + "\n")

    dump("top1", top1); dump("cv", cv); dump("full_tuned", bestfull[3])
    summ = {"nbest": a.nbest, "l2r": a.l2r, "r2l": a.r2l, "ref_words_local": nref, "top1_edits": int(ed[top1].sum()),
            "cv_edits": int(ed[cv].sum()), "cv_weights": {"fold0_tuned": b0[0], "fold1_tuned": b1[0]},
            "full_tuned_edits": bestfull[1] + bestfull[2], "full_tuned_weights": bestfull[0], "oracle_edits": int(oracle)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
