"""H261 stage 0: content-word substitution headroom in frozen n-best lists.
headroom.py NBEST FINAL.score-input.jsonl OUT.json
For each final edit (S/D/I, aligned to ref), classify content/function and ask whether the ref word
(for S/D) occurs at that aligned ref position in any n-best hypothesis (alignment per hypothesis)."""
import json, sys
from collections import Counter
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra", "/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H75_shipped_error_anatomy/tools"]
from ultra.wer import normalise
from anatomy import align, FUNC
nb = {r["uid"]: r for r in map(json.loads, open(sys.argv[1]))}
fin = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
wc = Counter()
for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
    wc.update(w.upper() for w in normalise(ln))
st = Counter(); rank = Counter(); conf = Counter(); orc = 0; fe = 0
for u, f in fin.items():
    ref = [w.upper() for w in normalise(f["ref"])]
    ops = align(ref, [w.upper() for w in normalise(f["hyp"])])
    hyps = [[w.upper() for w in normalise(str(c[2]))] for c in nb[u]["nbest"]]
    # per hyp: set of correctly recovered ref positions
    good = []
    for h in hyps:
        good.append({ri for k, rw, hw, ri in align(ref, h) if k == "C"})
    orc += min(sum(1 for k, *_ in align(ref, h) if k != "C") for h in hyps) if hyps else 0
    for k, rw, hw, ri in ops:
        if k == "C": continue
        fe += 1
        w = rw if k in "SD" else hw
        cls = "function" if w in FUNC else ("content-rare<100" if wc[w] < 100 else "content-freq")
        key = (k, cls)
        st[key] += 1
        if k in "SD":
            hit = [i for i, g in enumerate(good) if ri in g]
            if hit:
                st[key + ("in_nbest",)] += 1; rank[(cls, min(hit))] += 1
            if k == "S" and cls == "content-freq": conf[f"{rw}->{hw}"] += 1
res = {"final_edits_local": fe, "oracle_edits_local": orc,
       "by_kind_class": {"/".join(k): v for k, v in sorted(st.items())},
       "content_freq_sub_rank_of_first_correct": {f"{c}@{r}": v for (c, r), v in sorted(rank.items()) if c == "content-freq"},
       "top_content_freq_subs": conf.most_common(40)}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: res[k] for k in ("final_edits_local", "oracle_edits_local", "by_kind_class")}, indent=0))
