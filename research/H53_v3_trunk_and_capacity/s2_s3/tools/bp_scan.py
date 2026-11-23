#!/usr/bin/env python3
"""Run on x5 (node 0). For each run name: cumulative logged audio from the run's own train.jsonl, and the branch points
BPk = the first 500-step checkpoint past k x 10,000 audio-h (H53 Amendment 5), with the state of each BP's file in the
run dir and in ~/jsr-data/protected/<run>/. Also the free space of ~/jsr-data. Prints one JSON line."""
import json, os, shutil, sys
H = os.path.expanduser("~/jsr-data")
out = {"free_gb": shutil.disk_usage(H).free / 2**30, "runs": {}}  # GiB, as df -BG
for n in sys.argv[1:]:
    R, P = f"{H}/runs/train/{n}", f"{H}/protected/{n}"
    tj = f"{R}/train.jsonl"
    if not os.path.exists(tj):
        continue
    cum, k, bps, last, t_last = 0.0, 1, [], None, None
    for l in open(tj):
        if '"kind": "train"' not in l:
            continue
        try:
            r = json.loads(l)
        except ValueError:
            continue  # a partially written last line
        cum += r["audio_s"]; last = r["step"]; t_last = r["time"]
        while cum >= k * 36e6:
            s = r["step"]; bps.append({"k": k, "cross": s, "bp": s if s % 500 == 0 else (s // 500 + 1) * 500}); k += 1
    for b in bps:
        name = "ckpt-%08d" % b["bp"]
        b["name"] = name
        b["run_pt"] = os.path.exists(f"{R}/{name}.pt") and os.path.exists(f"{R}/{name}.sha256")
        pp = f"{P}/{name}.pt"
        b["prot_pt"] = os.path.exists(pp) and os.path.exists(f"{P}/{name}.sha256")
        b["links"] = os.stat(pp).st_nlink if os.path.exists(pp) else 0
        b["marker"] = open(f"{P}/{name}.sha256").read()[:64] if b["prot_pt"] else None
    latest = open(f"{R}/LATEST").read().strip() if os.path.exists(f"{R}/LATEST") else None
    out["runs"][n] = {"cum_h": cum / 3600, "last_step": last, "t_last": t_last, "latest": latest, "bps": bps}
print(json.dumps(out))
