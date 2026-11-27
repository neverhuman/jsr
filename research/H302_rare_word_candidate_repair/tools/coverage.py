"""H302 stage 0: can a TRAIN-only confusion lexicon reach DEV rare-word misses at all?
Lexicon: (hyp word -> ref word) substitutions with rare ref word (train-960 count < 100), aligned per n-best hypothesis,
from the H255@2000 (H276/TT) and F3 (H286/TT) MLS TRAIN decodes. Diagnostic on DEV: for gated rows of the H298 B48 list,
each rare ref word that is substituted in the top-1 and absent from every candidate: is the (top-1 word -> ref word)
pair in the lexicon, and is the ref word a lexicon target at all?"""
import json, sys
from collections import Counter, defaultdict
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra", "/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H75_shipped_error_anatomy/tools"]
from ultra.wer import normalise
from anatomy import align
W = lambda s: [w.upper() for w in normalise(s)]
wc = Counter()
for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
    wc.update(w.upper() for w in normalise(ln))
rare = lambda w: wc[w] < 100
P = "/mnt/xbabe3-external/jsr/wer-push"
lex = defaultdict(Counter); allpairs = Counter()
for d in ("H276/TT", "H286/TT"):
    refs = {r["uid"]: r for r in map(json.loads, open(f"{P}/{d}/refs.jsonl"))}
    for l in open(f"{P}/{d}/nbest.jsonl"):
        r = json.loads(l); ref = W(refs[r["uid"]]["ref"])
        for c in r["nbest"][:16]:
            for k, rw, hw, ri in align(ref, W(str(c[2]))):
                if k == "S" and rare(rw):
                    lex[hw][rw] += 1; allpairs[(hw, rw)] += 1
targets = {rw for d in lex.values() for rw in d}
print("lexicon pairs", len(allpairs), "hyp words", len(lex), "rare targets", len(targets))
C = "/home/ubuntu/jsr/.agent/worktrees/wer-codex/target/h265-nvme/jsr/wer-push/H298/final/h255-step2000"
gate = set(json.load(open(f"{C}/routes.json"))["30"]) if isinstance(json.load(open(f"{C}/routes.json")), dict) else set(json.load(open(f"{C}/routes.json")))
st = Counter()
for l in open(f"{C}/b48/mixed.nbest.jsonl"):
    r = json.loads(l)
    if r["uid"] not in gate: continue
    ref = W(r["ref"]); hyps = [W(str(c[2])) for c in r["nbest"]]
    good = set().union(*[{ri for k, rw, hw, ri in align(ref, h) if k == "C"} for h in hyps])
    for k, rw, hw, ri in align(ref, hyps[0]):
        if k == "S" and rare(rw):
            st["rare_sub_top1"] += 1
            if ri in good: st["in_list"] += 1; continue
            st["absent"] += 1
            if rw in targets: st["absent_target_in_lex"] += 1
            if lex.get(hw, {}).get(rw): st["absent_pair_in_lex"] += 1
            if rw in wc: st["absent_seen_in_train960"] += 1
print("gated rows", len(gate), dict(st))
