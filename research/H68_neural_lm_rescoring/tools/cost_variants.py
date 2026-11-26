"""H68: CV WER of cheaper rescoring variants from saved features (engine formula, speaker 2-fold CV).

Variants: engine cascade (L2R on top-L candidates by beam score, R2L on top-R stage-1 finalists),
L2R only, R2L only. Usage: cost_variants.py NBEST.jsonl FEATS.npz [refs.jsonl]
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from ultra.wer import edit_counts, normalise  # noqa: E402

nb, fz = sys.argv[1], sys.argv[2]
refs = sys.argv[3] if len(sys.argv) > 3 else "/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"
R = {r["uid"]: r for r in map(json.loads, open(refs))}
rows = [json.loads(x) for x in open(nb)]
F = np.load(fz)
owner, base, l2r, r2l, m5 = F["owner"], F["base"], F["l2r"], F["r2l"], F["m5"]
flat = [c for r in rows for c in r["nbest"][:16]]
assert len(flat) == len(owner)
ed = np.array([sum(edit_counts(normalise(R[rows[o]["uid"]]["ref"]), normalise(str(c[2])))) for c, o in zip(flat, owner)])
spk = sorted({str(R[r["uid"]]["speaker"]) for r in rows})
fold = np.array([spk.index(str(R[r["uid"]]["speaker"])) % 2 for r in rows])
starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
pos = np.arange(len(owner)) - np.repeat(starts, np.diff(np.r_[starts, len(owner)]))


def rank_within(score):
    o = np.lexsort((pos, -score, owner))
    r = np.empty(len(o), int)
    r[o] = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(owner)]))
    return r


def pick(score):
    o = np.lexsort((pos, -score, owner))
    return o[np.r_[0, np.flatnonzero(np.diff(owner[o])) + 1]]


base_rank = rank_within(base)
W = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
S = [0.0, 0.2, 0.4, 0.6, 0.8]


def run(name, use_l, use_r, L, Rk):
    res = []
    for w, s in itertools.product(W, S):
        s1 = base - s * m5 + (w * l2r if use_l else 0)
        s1 = np.where(base_rank < L, s1, -np.inf)  # stage 0: only top-L by beam score survive
        if use_r:
            st = s1 + w * r2l
            if use_l and Rk < 16:
                st = np.where(rank_within(s1) < Rk, st, -np.inf)
        else:
            st = s1
        idx = pick(st)
        e = ed[idx]
        res.append(((w, s), int(e[fold == 0].sum()), int(e[fold == 1].sum()), idx))
    b0 = min(res, key=lambda t: t[1]); b1 = min(res, key=lambda t: t[2])
    cv = int(b0[2] + b1[1])
    # hyps scored per utt (proxy for cost): L2R on min(L, n), R2L on min(Rk, n) or all
    n = np.diff(np.r_[starts, len(owner)])
    lh = np.minimum(n, L).sum() if use_l else 0
    rh = (np.minimum(n, Rk if use_l else L).sum()) if use_r else 0
    print(json.dumps({"variant": name, "cv_edits": cv, "w_s": [b0[0], b1[0]], "l2r_hyps_per_utt": round(lh / len(n), 2), "r2l_hyps_per_utt": round(rh / len(n), 2)}), flush=True)


print(json.dumps({"top1_edits": int(ed[pick(base)].sum()), "oracle": int(np.minimum.reduceat(ed, starts).sum())}))
run("engine_L16_R4", 1, 1, 16, 4)
run("all_L16_R16", 1, 1, 16, 16)
for L in (8, 4, 2):
    run(f"L{L}_R4", 1, 1, L, 4)
run("L16_R2", 1, 1, 16, 2)
run("L2R_only_L16", 1, 0, 16, 0)
run("L2R_only_L8", 1, 0, 8, 0)
run("R2L_only_L16", 0, 1, 16, 16)
run("m5sub_only", 0, 0, 16, 0)
