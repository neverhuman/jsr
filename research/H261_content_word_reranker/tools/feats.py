"""H261 per-hypothesis features for an n-best JSONL (first 16 candidates, flat order).
feats.py NBEST OUT.npz   -> owner, pos, base, l2r, l2r_content, l2r_function, nw, nw_content, logfreq_content, oov_content
L2R = kd12L_t2_16000 (H68), full sentence incl. closing boundary; per-token log-probs are attributed to the
word the token belongs to (SentencePiece U+2581 starts a word); the boundary token counts as function."""
import json, math, sys
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H68_neural_lm_rescoring/tools",
                "/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra",
                "/home/ubuntu/jsr/.agent/worktrees/wer-claude/research/H75_shipped_error_anatomy/tools"]
from train_lm import load_lm
from ultra.wer import normalise
from anatomy import FUNC
import sentencepiece as spm
nb, out = sys.argv[1:3]
sp = spm.SentencePieceProcessor(model_file="/mnt/xbabe3-external/jsr/wer-push/H234/step2000/ema-int8/bpe.model")
wc = Counter()
for ln in open("/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt", encoding="utf-8"):
    wc.update(w.upper() for w in normalise(ln))
tot = sum(wc.values())
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = load_lm("/mnt/xbabe3-external/jsr/wer-push/H68/snap/kd12L_t2_16000.pt", dev)
flat, owner, pos = [], [], []
for ui, ln in enumerate(open(nb)):
    for k, c in enumerate(json.loads(ln)["nbest"][:16]):
        flat.append(c); owner.append(ui); pos.append(k)
def word_of_tokens(toks):
    # returns per-token word index and the word strings (pieces joined)
    pieces = [sp.id_to_piece(int(t) - 1) for t in toks]
    wid, words, cur = [], [], -1
    for p in pieces:
        if p.startswith("▁") or cur < 0:
            words.append(p.lstrip("▁")); cur += 1
        else:
            words[cur] += p
        wid.append(cur)
    return wid, [w.upper() for w in words]
N = len(flat)
l2r = np.zeros(N); lc = np.zeros(N); lf = np.zeros(N); nw = np.zeros(N); nwc = np.zeros(N); lfq = np.zeros(N); oov = np.zeros(N)
order = sorted(range(N), key=lambda i: len(flat[i][0]))
with torch.no_grad():
    for o in range(0, N, 256):
        idx = order[o:o + 256]
        L = max(len(flat[i][0]) for i in idx) + 2
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            s = [0] + [int(t) for t in flat[i][0]] + [0]; x[r, :len(s)] = torch.tensor(s)
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lo = m(x[:, :-1]).float()
        lp = F.log_softmax(lo, -1).gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        for r, i in enumerate(idx):
            toks = flat[i][0]; n = len(toks)
            wid, words = word_of_tokens(toks)
            iscont = [w not in FUNC for w in words]
            tl = lp[r, :n + 1]
            l2r[i] = tl.sum()
            lc[i] = sum(tl[j] for j in range(n) if iscont[wid[j]])
            lf[i] = l2r[i] - lc[i]
            nw[i] = len(words); nwc[i] = sum(iscont)
            lfq[i] = sum(math.log((wc[w] + 1) / tot) for w, c in zip(words, iscont) if c)
            oov[i] = sum(1 for w, c in zip(words, iscont) if c and wc[w] == 0)
np.savez(out, owner=np.array(owner), pos=np.array(pos), base=np.array([float(c[1]) for c in flat]), l2r=l2r,
         l2r_content=lc, l2r_function=lf, nw=nw, nw_content=nwc, logfreq_content=lfq, oov_content=oov)
print("hyps", N)
