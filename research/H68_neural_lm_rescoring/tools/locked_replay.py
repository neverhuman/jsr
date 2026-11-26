"""H68: apply a FIXED (locked) engine config to an n-best list; no tuning on the list being scored.

score = base + w * L2R [+ w * R2L] - s * M5; skip (keep beam top-1) when top1 - top2 >= margin.
Usage: locked_replay.py NBEST FEATS.npz OUT.score-input.jsonl W S MARGIN [TOPK=16]
"""
import json
import sys

import numpy as np

nb, fz, out, w, s, mg = sys.argv[1], sys.argv[2], sys.argv[3], *map(float, sys.argv[4:7])
topk = int(sys.argv[7]) if len(sys.argv) > 7 else 16
R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
rows = [json.loads(x) for x in open(nb)]
F = np.load(fz)
sc = F["base"] + w * F["l2r"] - s * F["m5"] + (w * F["r2l"] if "r2l" in F.files else 0)
owner = F["owner"]
pos, n_res = 0, 0
with open(out, "w") as fh:
    for ui, r in enumerate(rows):
        k = min(len(r["nbest"]), topk)
        assert (owner[pos:pos + k] == ui).all()
        nb_ = r["nbest"]
        margin = nb_[0][1] - nb_[1][1] if len(nb_) > 1 else 1e9
        if margin < mg:
            j = int(np.argmax(sc[pos:pos + k])); n_res += 1
        else:
            j = 0
        rr = R[r["uid"]]
        fh.write(json.dumps({"id": r["uid"], "speaker": str(rr["speaker"]), "dataset": rr["subset"], "ref": rr["ref"],
                             "hyp": nb_[j][2], "duration_s": rr["samples"] / 16000}) + "\n")
        pos += k
print(json.dumps({"rescored_frac": round(n_res / len(rows), 4), "out": out}))
