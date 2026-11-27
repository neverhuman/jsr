"""H302 stage 0b: TRAIN-free alternative lexicon = train-960 vocabulary words within character edit distance <= K of the
top-1 hyp word. Diagnostic only: coverage of DEV gated absent rare-word subs, and mean neighbour count (variant blow-up)."""
import json, sys
from collections import Counter
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra", "/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H75_shipped_error_anatomy/tools"]
from ultra.wer import normalise
from anatomy import align
W = lambda s: [w.upper() for w in normalise(s)]
wc = Counter()
for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
    wc.update(w.upper() for w in normalise(ln))
rare = lambda w: wc[w] < 100
vocab = [w for w in wc if wc[w] < 100]
def lev(a, b, cap):
    if abs(len(a) - len(b)) > cap: return cap + 1
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        if min(cur) > cap: return cap + 1
        prev = cur
    return prev[-1]
C = "/home/ubuntu/jsr/.agent/worktrees/wer-codex/target/h265-nvme/jsr/wer-push/H298/final/h255-step2000"
gate = set(json.load(open(f"{C}/routes.json"))["30"])
st = Counter(); nb = {1: [], 2: []}
for l in open(f"{C}/b48/mixed.nbest.jsonl"):
    r = json.loads(l)
    if r["uid"] not in gate: continue
    ref = W(r["ref"]); hyps = [W(str(c[2])) for c in r["nbest"]]
    good = set().union(*[{ri for k, rw, hw, ri in align(ref, h) if k == "C"} for h in hyps])
    for k, rw, hw, ri in align(ref, hyps[0]):
        if k == "S" and rare(rw) and ri not in good:
            st["absent"] += 1
            for K in (1, 2):
                neigh = [v for v in vocab if lev(hw, v, K) <= K]
                nb[K].append(len(neigh))
                if rw in neigh: st[f"ref_within_ed{K}_of_top1_word"] += 1
print(dict(st), {K: round(sum(v) / max(1, len(v)), 1) for K, v in nb.items()}, "rare vocab", len(vocab))
