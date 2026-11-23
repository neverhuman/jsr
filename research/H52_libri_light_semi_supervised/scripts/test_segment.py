"""H52 test: segment N files from a (partial) Libri-Light tar; timing + stats. Writes nothing big.
usage: python -I test_segment.py <tar> <json_meta.jsonl> <n_files> [skip]"""
import json, sys, tarfile, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ll_segment as S

tar, jsonl, n_files = sys.argv[1], sys.argv[2], int(sys.argv[3])
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
metas = {}
for line in open(jsonl):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue  # partially flushed last line
    metas[r["name"][:-5]] = json.loads(r["text"])
mem = {}
try:
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isreg() and m.name.endswith(".flac"):
                mem[m.name[:-5]] = (m.offset_data, m.size)
            if len(mem) >= n_files + skip + 5:
                break
except Exception as ex:
    print("tar scan stopped:", type(ex).__name__)
keys = [k for k in mem if k in metas][skip:skip + n_files]
f = open(tar, "rb")
T = {"read": 0, "decode": 0, "plan": 0, "encode": 0}
tot = {"dur": 0, "vad": 0, "utt": 0, "speech": 0, "n": 0, "flac_in": 0, "flac_out": 0}
drops = {}
lens, heads, tails, ecuts = [], [], [], 0
for k in keys:
    od, sz = mem[k]
    t = time.time(); f.seek(od); b = f.read(sz); T["read"] += time.time() - t
    t = time.time(); pcm = S.decode_flac(b); T["decode"] += time.time() - t
    dur = len(pcm) / S.SR
    t = time.time(); utts, st = S.plan(metas[k]["voice_activity"], dur, pcm); T["plan"] += time.time() - t
    t = time.time()
    outb = sum(len(S.encode_flac(S.cut(pcm, u))) for u in utts)
    T["encode"] += time.time() - t
    tot["dur"] += dur; tot["vad"] += st["vad_s"]; tot["n"] += len(utts)
    tot["utt"] += sum(u["t1"] - u["t0"] for u in utts); tot["speech"] += sum(u["speech"] for u in utts)
    tot["flac_in"] += sz; tot["flac_out"] += outb
    for kk, v in st.items():
        if kk.startswith("dropped") or kk in ("energy_splits", "n_long_units"):
            drops[kk] = drops.get(kk, 0) + v
    lens += [u["t1"] - u["t0"] for u in utts]; heads += [u["head"] for u in utts]; tails += [u["tail"] for u in utts]
    ecuts += sum(1 for u in utts if u["eh"] or u["et"])
q = lambda a: np.percentile(a, [1, 5, 25, 50, 75, 95, 99]).round(2).tolist()
print("files", len(keys), "audio h %.2f vad h %.2f utt h %.2f speech-in-utt h %.2f" % (tot["dur"]/3600, tot["vad"]/3600, tot["utt"]/3600, tot["speech"]/3600))
print("kept speech / vad %.3f ; utt audio / file audio %.3f" % (tot["speech"]/tot["vad"], tot["utt"]/tot["dur"]))
print("n utts", tot["n"], "mean len %.2f" % np.mean(lens), "len pct", q(lens), "max %.2f" % max(lens))
print("head pct", q(heads), "tail pct", q(tails), "min tail %.3f" % min(tails))
print("utts touching an energy cut", ecuts, "drops (s)", {k: round(v, 1) for k, v in drops.items()})
print("flac bytes in %.1f MB out %.1f MB ; out MB per utt-hour %.1f ; in MB per file-hour %.1f" % (tot["flac_in"]/1e6, tot["flac_out"]/1e6, tot["flac_out"]/1e6/(tot["utt"]/3600), tot["flac_in"]/1e6/(tot["dur"]/3600)))
cpu = T["decode"] + T["plan"] + T["encode"]
print("timing s", {k: round(v, 2) for k, v in T.items()}, "cpu x realtime %.0f" % (tot["dur"] / cpu))
