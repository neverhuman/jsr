"""H68 (d): skip commit rescoring when the fused-beam top-1 margin (score[0]-score[1]) exceeds a threshold.

Threshold chosen by speaker 2-fold CV on DEV S (picked on one fold, applied to the other) to minimise
edits subject to a commit-CPU budget; per-utterance rescoring cost is the measured engine delta
(resc - base commit_wall_ms, mean of the ABAB pairs) on the 147-utt panel, extrapolated by tokens.
Usage: gate_sim.py NBEST FEATS_CV_DIR QUAL_DIR [budget_pct ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from ultra.wer import edit_counts, normalise  # noqa: E402

nb, cvdir, qual = sys.argv[1:4]
budgets = [float(x) for x in sys.argv[4:]] or [15.0, 25.0]
gates = {}
R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
rows = [json.loads(x) for x in open(nb)]
top1 = {r["uid"]: r["nbest"][0][2] for r in rows}
margin = {r["uid"]: (r["nbest"][0][1] - r["nbest"][1][1]) if len(r["nbest"]) > 1 else 1e9 for r in rows}
ntok = {r["uid"]: len(r["nbest"][0][0]) for r in rows}
resc = {r["id"]: r["hyp"] for r in map(json.loads, open(f"{cvdir}/cv.score-input.jsonl"))}


def load(f):
    return {x["uid"]: x for x in (json.loads(l) for l in open(f"{qual}/{f}.jsonl") if l.strip()) if "uid" in x}


b = [load("base1"), load("base2")]; c = [load("resc1"), load("resc2")]
pu = list(b[0])
extra = np.array([np.mean([c[i][u]["commit_wall_ms"] - b[i][u]["commit_wall_ms"] for i in range(2)]) for u in pu])
base_ms = np.array([np.mean([b[i][u]["commit_wall_ms"] for i in range(2)]) for u in pu])
toks = np.array([ntok[u] for u in pu])
# rescoring cost model fitted on the panel: ms = k * tokens (through origin), base ms per audio second
k = float((extra * toks).sum() / (toks * toks).sum())
base_per_s = float(base_ms.sum() / sum(b[0][u]["audio_s"] for u in pu))
uids = [r["uid"] for r in rows]
dur = np.array([R[u]["samples"] / 16000 for u in uids])
cost = k * np.array([ntok[u] for u in uids])
base_total = base_per_s * dur.sum()
e_top = np.array([sum(edit_counts(normalise(R[u]["ref"]), normalise(top1[u]))) for u in uids])
e_res = np.array([sum(edit_counts(normalise(R[u]["ref"]), normalise(resc[u]))) for u in uids])
m = np.array([margin[u] for u in uids])
spk = sorted({str(R[u]["speaker"]) for u in uids})
fold = np.array([spk.index(str(R[u]["speaker"])) % 2 for u in uids])
ths = np.r_[np.quantile(m[np.isfinite(m) & (m < 1e8)], np.linspace(0, 1, 101)), np.inf]
print(json.dumps({"panel_fit_ms_per_token": round(k, 4), "full_rescore_cpu_pct": round(100 * cost.sum() / base_total, 1),
                  "top1_edits": int(e_top.sum()), "full_rescore_edits": int(e_res.sum())}))
for bud in budgets:
    pick = {}
    for f in (0, 1):
        sel = fold == f
        best = None
        for t in ths:
            g = m < t  # rescore only low-margin utterances
            pct = 100 * cost[sel & g].sum() / (base_per_s * dur[sel].sum())
            if pct > bud:
                continue
            ed = int(np.where(g, e_res, e_top)[sel].sum())
            if best is None or ed < best[0]:
                best = (ed, t)
        pick[f] = best[1]
    g = np.where(fold == 0, m < pick[1], m < pick[0])  # thresholds transferred across folds
    ed = int(np.where(g, e_res, e_top).sum()); pct = 100 * cost[g].sum() / base_total
    with open(f"{qual}/gate{int(bud)}.score-input.jsonl", "w") as fh:
        for u, gi in zip(uids, g):
            rr = R[u]
            fh.write(json.dumps({"id": u, "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"],
                                 "hyp": resc[u] if gi else top1[u], "duration_s": rr["samples"] / 16000}) + "\n")
    gates[int(bud)] = {"thresholds": [float(pick[0]), float(pick[1])], "engine_threshold": float((pick[0] + pick[1]) / 2),
                       "sim_cpu_pct": round(pct, 1), "cv_edits": ed, "rescored_utt_frac": round(float(g.mean()), 3)}
    json.dump(gates, open(f"{qual}/gates.json", "w"), indent=1)
    print(json.dumps({"budget_pct": bud, "cv_edits": ed, "gain_edits": int(e_top.sum() - ed), "realized_cpu_pct": round(pct, 1),
                      "rescored_utt_frac": round(float(g.mean()), 3), "thresholds": [float(pick[0]), float(pick[1])]}))
