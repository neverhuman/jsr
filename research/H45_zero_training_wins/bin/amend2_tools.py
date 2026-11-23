#!/usr/bin/env python3
"""h45-wins amendment 2: beam width x LM in the commit pass.

  amend2_tools.py collect <lm_decode out dir> <name> <outdir>
      lm_decode beam shards (<dir>/<cfg>/<split>.<i>of<n>.jsonl) -> train.evaluate-style dumps
      <outdir>/<name>.<cfg>.{dev-clean,dev-other}.jsonl (uid, ref, hyp, seconds, sec, lm_contexts); refs and
      durations from eval-lane's B0 chunk-4 dumps (same 5,567 utterances).
  amend2_tools.py select <outdir> <name> <uids S>
      canonical S WER of every config; per beam width the best LM config (ties -> nearest L4 0.8 / 0.6) -> JSON.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

H = "/mnt/xbabe3-external/jsr/h45-wins"
E = "/mnt/xbabe3-external/jsr/evals/eval-lane/curve/A1-49000.ema.c4.ms8"
SETS = ("dev-clean", "dev-other")


def refs():
    out = {}
    for s in SETS:
        for line in open(f"{E}.{s}.jsonl"):
            r = json.loads(line)
            out[r["uid"]] = (s, r["ref"], r["seconds"])
    return out


def collect(d, name, outdir):
    R = refs()
    os.makedirs(outdir, exist_ok=True)
    for cdir in sorted(glob.glob(os.path.join(d, "*"))):
        if not os.path.isdir(cdir) or os.path.basename(cdir) == "logs":
            continue
        cfg = os.path.basename(cdir)
        rows = {}
        for f in glob.glob(os.path.join(cdir, "*.jsonl")):
            if not re.search(r"\.\d+of\d+\.jsonl$", f):
                continue
            for line in open(f):
                r = json.loads(line)
                rows[r["uid"]] = r
        if not rows:
            continue
        for s in SETS:
            with open(os.path.join(outdir, f"{name}.{cfg}.{s}.jsonl"), "w") as fh:
                for uid in sorted(u for u in rows if R[u][0] == s):
                    r = rows[uid]
                    fh.write(json.dumps({"uid": uid, "ref": R[uid][1], "hyp": r["hyp"], "seconds": R[uid][2],
                                         "sec": r.get("sec"), "lm_contexts": r.get("lm_contexts")}) + "\n")
        print(json.dumps({"cfg": cfg, "utterances": len(rows)}))


def score(prefix, uids=None):
    cmd = [sys.executable, f"{H}/bin/h45_score.py", "row", prefix]
    if uids:
        cmd += ["--uids", uids]
    o = subprocess.run(cmd, capture_output=True, text=True, check=True,
                       env=dict(os.environ, H45_TMP="/dev/shm/h45-wins/tmp"))
    return json.loads(o.stdout)


def select(outdir, name, uids):
    res = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"{name}.*.dev-clean.jsonl"))):
        cfg = os.path.basename(f)[len(name) + 1:-len(".dev-clean.jsonl")]
        res[cfg] = score(os.path.join(outdir, f"{name}.{cfg}"), uids)["pooled"]["wer"]
    pick = {}
    for b in (4, 8, 16):
        lm = {c: w for c, w in res.items() if c.startswith(f"b{b}_l") and not c.endswith("_T4")}
        if not lm:
            continue

        def dist(c):
            m = re.match(rf"b{b}_l([\d.]+)_r([\d.]+)", c)
            return abs(float(m.group(1)) - 0.8) + abs(float(m.group(2)) - 0.6)

        best = min(lm, key=lambda c: (round(lm[c], 4), dist(c)))
        pick[f"b{b}"] = {"lm": best, "lm_wer_S": lm[best], "nolm": f"b{b}_nolm", "nolm_wer_S": res.get(f"b{b}_nolm")}
    out = {"tuned_on": os.path.basename(uids), "wer_S": res, "pick": pick}
    json.dump(out, open(os.path.join(outdir, f"{name}.select.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(*sys.argv[2:5])
    elif sys.argv[1] == "select":
        select(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
