"""H286 (1): S/D/I anatomy of a final vs a baseline final, plus remaining headroom in the new n-best lists.
sdi.py BASE.score-input.jsonl NEW.score-input.jsonl NEW_NBEST.jsonl OUT.json
Per ref position: error in base and correct in new = fixed; correct in base and error in new = broken.
Insertions are matched by (kind, preceding ref index, hyp word). Words are classed as function / content-freq / content-rare<100
(train-960 counts). Headroom: for each remaining S/D, whether some hypothesis in the n-best gets that ref position right."""
import json, sys
from collections import Counter
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra", "/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H75_shipped_error_anatomy/tools"]
from ultra.wer import normalise
from anatomy import align, FUNC

base = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
new = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
nb = {r["uid"]: r for r in map(json.loads, open(sys.argv[3]))}
assert base.keys() == new.keys()
wc = Counter()
for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
    wc.update(w.upper() for w in normalise(ln))
W = lambda s: [w.upper() for w in normalise(s)]
cls = lambda w: "function" if w in FUNC else ("content-rare<100" if wc[w] < 100 else "content-freq")


def errs(ref, hyp):
    out, last = {}, -1
    for k, rw, hw, ri in align(ref, hyp):
        if k == "C":
            last = ri
        elif k in "SD":
            out[("SD", ri)] = (k, rw, hw); last = ri
        else:
            out[("I", last, hw)] = (k, rw, hw)
    return out


tot = {"base": Counter(), "new": Counter()}; fixed = Counter(); broken = Counter(); changed = Counter()
fixed_pairs = Counter(); broken_pairs = Counter(); left_pairs = Counter(); head = Counter(); dset = Counter()
nref = 0
for u in base:
    ref = W(base[u]["ref"]); nref += len(ref)
    eb, en = errs(ref, W(base[u]["hyp"])), errs(ref, W(new[u]["hyp"]))
    for name, e in (("base", eb), ("new", en)):
        for k, rw, hw in e.values():
            tot[name][(k, cls(rw if k != "I" else hw))] += 1
    for key in eb.keys() - en.keys():
        k, rw, hw = eb[key]; fixed[(k, cls(rw if k != "I" else hw))] += 1; fixed_pairs[f"{k}:{rw}->{hw}"] += 1
        dset[(base[u]["dataset"], "fixed")] += 1
    for key in en.keys() - eb.keys():
        k, rw, hw = en[key]; broken[(k, cls(rw if k != "I" else hw))] += 1; broken_pairs[f"{k}:{rw}->{hw}"] += 1
        dset[(base[u]["dataset"], "broken")] += 1
    for key in en.keys() & eb.keys():
        if en[key] != eb[key]:
            changed[en[key][0]] += 1
    good = [{ri for k, rw, hw, ri in align(ref, W(str(c[2]))) if k == "C"} for c in nb[u]["nbest"]]
    for key, (k, rw, hw) in en.items():
        c = cls(rw if k != "I" else hw)
        if k == "S" and c == "content-freq":
            left_pairs[f"{rw}->{hw}"] += 1
        if k in "SD":
            head[(k, c, "in_nbest" if any(key[1] in g for g in good) else "absent")] += 1
j = lambda c: {"/".join(map(str, k)): v for k, v in sorted(c.items())}
res = {"ref_words": nref, "base_edits": sum(tot["base"].values()), "new_edits": sum(tot["new"].values()),
       "base": j(tot["base"]), "new": j(tot["new"]), "fixed": j(fixed), "broken": j(broken), "same_pos_changed": dict(changed),
       "by_dataset": j(dset), "remaining_SD_headroom": j(head),
       "top_fixed": fixed_pairs.most_common(25), "top_broken": broken_pairs.most_common(25),
       "top_remaining_content_freq_subs": left_pairs.most_common(30)}
json.dump(res, open(sys.argv[4], "w"), indent=1)
print(json.dumps({k: res[k] for k in ("ref_words", "base_edits", "new_edits", "base", "new", "fixed", "broken", "by_dataset", "remaining_SD_headroom")}, indent=1))
