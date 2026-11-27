"""H297: apply a refit H261 model to the gated (wide-beam) DEV rows and the frozen H261 model to all other rows.
apply_gated.py S_DIR FROZEN.json GATED.json ROUTES.json PCT OUT.score-input.jsonl
ROUTES.json is codex H294 routes.json (uids per TRAIN-percent gate). Same routing/aggregation as rerank_x apply."""
import json, os, sys
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H286_final_reranker_current_acoustic/tools"]
os.environ.setdefault("NB", "48")
import numpy as np
import rerank_x as rx

d, frozen, gated, routes, pct, out = sys.argv[1:7]
owner, pos, X, rows = rx.load(d)
Wf = np.array([json.load(open(frozen))["weights"][k] for k in rx.FEATS])
Wg = np.array([json.load(open(gated))["weights"][k] for k in rx.FEATS])
gate = set(json.load(open(routes))[pct])
R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
sf, sg = X @ Wf, X @ Wg
n = 0
with open(out, "w") as fh:
    for u, idx, routed in rx.groups(owner, X[:, 0]):
        r = rows[u]; T = [str(r["nbest"][pos[i]][2]) for i in idx]
        g = r["uid"] in gate
        n += g
        h = rx.agg(T, (sg if g else sf)[idx] if routed else X[idx, 0])
        rr = R[r["uid"]]
        fh.write(json.dumps({"id": r["uid"], "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"], "hyp": h,
                             "duration_s": rr["samples"] / 16000}) + "\n")
print("gated rows", n)
