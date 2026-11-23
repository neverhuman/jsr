"""H52: VAD statistics from Libri-Light JSON metadata + FLAC STREAMINFO (header bytes only).
usage: python -I vad_stats.py <tar> <members.tsv> <json_meta.jsonl> [max_files]"""
import json
import struct
import sys

import numpy as np

tar, members_tsv, jsonl = sys.argv[1], sys.argv[2], sys.argv[3]
maxf = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
mem = {}
for line in open(members_tsv).read().splitlines()[1:]:
    n, k, sz, oh, od = line.split("\t")
    mem[n] = (int(sz), int(od))
f = open(tar, "rb")


def flac_info(name):
    sz, od = mem[name]
    f.seek(od)
    b = f.read(42)
    assert b[:4] == b"fLaC", name
    # METADATA_BLOCK_HEADER (4 bytes) then STREAMINFO (34 bytes)
    si = b[8:42]
    x = int.from_bytes(si[10:18], "big")
    sr = x >> 44
    ch = ((x >> 41) & 0x7) + 1
    bps = ((x >> 36) & 0x1F) + 1
    total = x & ((1 << 36) - 1)
    return sr, ch, bps, total


segs, gaps, tails, heads, durs, speech = [], [], [], [], [], []
srs = set()
n = 0
for line in open(jsonl):
    r = json.loads(line)
    fl = r["name"][:-5] + ".flac"
    if fl not in mem:
        continue
    j = json.loads(r["text"])
    sr, ch, bps, total = flac_info(fl)
    srs.add((sr, ch, bps))
    d = total / sr
    va = sorted(j["voice_activity"])
    durs.append(d)
    speech.append(sum(e - s for s, e in va))
    segs += [e - s for s, e in va]
    gaps += [va[i + 1][0] - va[i][1] for i in range(len(va) - 1)]
    if va:
        heads.append(va[0][0])
        tails.append(d - va[-1][1])
    n += 1
    if n >= maxf:
        break
segs, gaps, tails, heads = map(np.array, (segs, gaps, tails, heads))
q = lambda a: np.percentile(a, [1, 5, 25, 50, 75, 95, 99]).round(2).tolist()
print("files", n, "formats", srs)
print("total dur h %.2f  vad speech h %.2f  ratio %.3f" % (sum(durs) / 3600, sum(speech) / 3600, sum(speech) / max(1, sum(durs))))
print("vad seg len pct", q(segs), "max %.1f" % segs.max(), "n", len(segs))
print("seg len >19.4s: n %d, share of speech %.3f" % ((segs > 19.4).sum(), segs[segs > 19.4].sum() / segs.sum()))
print("seg len >17.4s: n %d, share of speech %.3f" % ((segs > 17.4).sum(), segs[segs > 17.4].sum() / segs.sum()))
print("gap pct", q(gaps), "share gaps <0.3s %.3f" % (gaps < 0.3).mean(), "<0.5 %.3f" % (gaps < 0.5).mean(), "neg %d" % (gaps < 0).sum())
print("head (first vad start) pct", q(heads))
print("tail (file end - last vad end) pct", q(tails), "share <0.3s %.3f" % (tails < 0.3).mean())
