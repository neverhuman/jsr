#!/usr/bin/env python3
"""H53 Amendment 6: extra_base_weight = a_LL / (a_LS + a_LL), with a_X = mean audio seconds per row group of source X (exact,
from every row's FLAC frame count in dur_table.py's tables), a_LL pooled over small + medium by row groups (their natural
share inside the Libri-Light half). Only files in the given dirs count (the loader's globs).
  bw_compute.py <tables,comma> <ls_root> <ll_dir> [<ll_dir> ...]   (Libri-Light dirs in extra_splits order)"""
import glob, json, os, sys
tables, ls_root, lls = sys.argv[1].split(","), sys.argv[2], sys.argv[3:]
T = {}
for t in tables:
    for l in open(t):
        r = json.loads(l); T[r["file"]] = r["groups"]
def src(files):
    g = s = n = 0
    for f in files:
        for grp in T[f]:
            g += 1; n += len(grp); s += sum(x for _, x in grp) / 16000
    return {"files": len(files), "groups": g, "rows": n, "hours": s / 3600, "a": s / g}
ls = src(sorted(f for d in ("train.clean.100", "train.clean.360", "train.other.500") for f in glob.glob(os.path.join(ls_root, d, "*.parquet"))))
ll = {os.path.basename(d): src(sorted(glob.glob(os.path.join(d, "*.parquet")))) for d in lls}
a_ll = sum(v["a"] * v["groups"] for v in ll.values()) / sum(v["groups"] for v in ll.values())
bw = a_ll / (ls["a"] + a_ll)
chk = bw * ls["a"] / (bw * ls["a"] + (1 - bw) * a_ll)
print(json.dumps({"librispeech": ls, **ll, "a_LL_pooled": a_ll, "extra_base_weight": round(bw, 6),
                  "ls_audio_share_check": round(chk, 6)}))
