"""Per-hypothesis full-sentence M5 token n-gram log-prob (ln, BOS, no EOS) for an n-best JSONL, flat order
(rows in file order, first TOPK candidates). Runs where kenlm is installed (xbabe0 train venv).
Usage: m5_feats.py NBEST TRIE OUT.npy [TOPK]"""
import json, sys
import numpy as np, kenlm
nb, trie, out = sys.argv[1:4]
topk = int(sys.argv[4]) if len(sys.argv) > 4 else 16
km = kenlm.Model(trie)
vals = []
for ln in open(nb):
    if ln.strip():
        for c in json.loads(ln)["nbest"][:topk]:
            vals.append(km.score(" ".join(str(int(t)) for t in c[0]), bos=True, eos=False) * 2.302585092994046)
np.save(out, np.array(vals))
print(len(vals))
