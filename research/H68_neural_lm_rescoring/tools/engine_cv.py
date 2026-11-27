"""engine_cv.py TAG... : speaker 2-fold CV over native-engine full-S grid runs (engine_fullS/TAG_w*_s*_NN.jsonl).
Picks (w, s) on one speaker fold by local edits, applies to the other; writes engine_fullS/TAG_cv.score-input.jsonl."""
import glob, json, re, sys
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from ultra.wer import edit_counts, normalise
D = "/mnt/xbabe3-external/jsr/wer-push/H68/engine_fullS"
R = {r["uid"]: r for r in map(json.loads, open("/mnt/xbabe3-external/jsr/wer-push/shared/s_half/refs.jsonl"))}
spk = sorted({str(r["speaker"]) for r in R.values()}); fold = {s: i % 2 for i, s in enumerate(spk)}
for tag in sys.argv[1:]:
    runs = defaultdict(dict)
    for f in glob.glob(f"{D}/{tag}_w*_s*_[0-9][0-9].jsonl"):
        m = re.search(r"_w([0-9.]+)_s([0-9.]+)_\d\d\.jsonl$", f)
        for x in map(json.loads, filter(str.strip, open(f))):
            if "uid" in x: runs[(m[1], m[2])][x["uid"]] = x["committed"]
    runs = {k: v for k, v in runs.items() if len(v) == 2782}
    ed = {k: [0, 0] for k in runs}
    for k, v in runs.items():
        for u, h in v.items(): ed[k][fold[str(R[u]["speaker"])]] += sum(edit_counts(normalise(R[u]["ref"]), normalise(h)))
    b0 = min(ed, key=lambda k: ed[k][0]); b1 = min(ed, key=lambda k: ed[k][1])
    with open(f"{D}/{tag}_cv.score-input.jsonl", "w") as fh:
        for u, r in R.items():
            h = runs[b0][u] if fold[str(r["speaker"])] == 1 else runs[b1][u]
            fh.write(json.dumps({"id": u, "speaker": str(r["speaker"]), "dataset": r["subset"], "ref": r["ref"], "hyp": h, "duration_s": r["samples"] / 16000}) + "\n")
    print(tag, "grid edits", {f"{k[0]}/{k[1]}": sum(v) for k, v in sorted(ed.items())}, "fold0 pick", b0, "fold1 pick", b1)
