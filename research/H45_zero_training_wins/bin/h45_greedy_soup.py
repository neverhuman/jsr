#!/usr/bin/env python3
"""h45-wins Arm A-GS: greedy soup (Wortsman et al.; the rule of train/diloco.greedy_model_soup) on the S half.

  h45_greedy_soup.py --singles singles.json --work <dir> [--chunk 4]

singles.json: [{"tag", "ckpt", "key", "s_wer"}] for the 12 single checkpoints, each already decoded and scored on S.
Ranks them best-first by S pooled WER; starts from the best; for each next ingredient forms the UNIFORM average of the
selected set plus it (exactly diloco.greedy_model_soup's running average), writes a recipe, merges it, decodes S at the
chunk (batch 32), scores it with the pinned ultra-score, and keeps the ingredient only if S WER strictly improves.
Every trial is logged to <work>/greedy_soup.jsonl; the final soup recipe is <work>/A-GS.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

H = "/mnt/amphora-backup/h45-wins"
PY = "/home/ubuntu/jsr/.venv/bin/python"
SRC = f"{H}/src/main-e61121e/python/ultra"
TS = ["taskset", "-c", os.environ.get("H45_CPUS", "40-42,104-106"), "nice", "-n", "10"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--chunk", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    singles = sorted(json.load(open(a.singles)), key=lambda x: (x["s_wer"], x["tag"]))
    log = open(os.path.join(a.work, "greedy_soup.jsonl"), "a")
    sel = [singles[0]]
    best = singles[0]["s_wer"]
    log.write(json.dumps({"step": 0, "start": singles[0]["tag"], "s_wer": best,
                          "ranking": [(x["tag"], x["s_wer"]) for x in singles]}) + "\n"); log.flush()
    for n, cand in enumerate(singles[1:], 1):
        trial = sel + [cand]
        name = f"GS-trial{n:02d}"
        rec = {"name": name, "terms": [{"ckpt": x["ckpt"], "key": x["key"], "coef": 1.0 / len(trial)} for x in trial],
               "note": "greedy-soup trial: uniform mean of " + ",".join(x["tag"] for x in trial)}
        rp = os.path.join(a.work, name + ".json")
        json.dump(rec, open(rp, "w"), indent=1)
        run(TS + [PY, f"{H}/src/h45_merge.py", "--src", SRC, "--out-dir", f"{H}/weights/derived", rp])
        run(TS + [PY, f"{H}/src/h45_decode.py", "--src", SRC, "--jsr-data", f"{H}/jsr_data", "--feat-cache",
                  f"{H}/featcache", "--batch", "32", "--device", "cuda", "--threads", "2", "--out", f"{H}/out/a",
                  "--uids", f"{H}/src/uids_selection.txt", "--subset-tag", ".S", "--chunks", str(a.chunk),
                  "--ckpt", f"{name}={H}/weights/derived/{name}.pt@model"])
        o = json.loads(run([PY, f"{H}/src/h45_score.py", "row", f"{H}/out/a/{name}.S.c{a.chunk}.ms8"]).strip())
        w = o["pooled"]["wer"]
        keep = w < best
        log.write(json.dumps({"step": n, "candidate": cand["tag"], "trial": [x["tag"] for x in trial], "s_wer": w,
                              "best_before": best, "kept": keep}) + "\n"); log.flush()
        if keep:
            sel, best = trial, w
    final = {"name": "A-GS", "terms": [{"ckpt": x["ckpt"], "key": x["key"], "coef": 1.0 / len(sel)} for x in sel],
             "note": "greedy soup on S at chunk %d: %s (S WER %.4f)" % (a.chunk, ",".join(x["tag"] for x in sel), best)}
    json.dump(final, open(os.path.join(a.work, "A-GS.json"), "w"), indent=1)
    log.write(json.dumps({"final": [x["tag"] for x in sel], "s_wer": best}) + "\n"); log.flush()
    print(json.dumps({"final": [x["tag"] for x in sel], "s_wer": best}))


if __name__ == "__main__":
    main()
