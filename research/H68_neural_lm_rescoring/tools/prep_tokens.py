"""H68: tokenize OpenSLR-11 librispeech-lm-norm (+ train-960 transcripts) with the shipped bpe.model.
Output: uint16 stream, token = sp_id + 1 (ASR convention), 0 = sentence boundary. Lines that exactly
match any DEV transcript (dev-clean + dev-other, both halves) are dropped (leak guard). Every 2000th line -> valid."""
import gzip, json, sys, hashlib
from multiprocessing import Pool
import numpy as np
import sentencepiece as spm
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
BPE = "/mnt/xbabe3-external/jsr/wer-push/shared/isosyll/isosyll-model/bpe.model"
SRC = "/mnt/xbabe3-external/jsr/lm/downloads/openslr11/librispeech-lm-norm.txt.gz.d/librispeech-lm-norm.txt.gz"
OUT = "/mnt/xbabe3-external/jsr/wer-push/H68/data"
_sp = None
def init():
    global _sp; _sp = spm.SentencePieceProcessor(model_file=BPE)
def enc(lines):
    out = []
    for ln in lines:
        ids = _sp.encode(ln)
        out.append(np.asarray([0] + [i + 1 for i in ids], dtype=np.uint16))
    return out
def chunks(it, n=4096):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n: yield buf; buf = []
    if buf: yield buf
def main():
    from ultra.libri import load_split
    dev = {" ".join(u.text.upper().split()) for s in ("dev-clean", "dev-other") for u in load_split(s, None)}
    dropped = 0; n = 0
    def lines():
        nonlocal dropped, n
        with gzip.open(SRC, "rt", encoding="utf-8") as fh:
            for ln in fh:
                t = " ".join(ln.upper().split())
                if not t: continue
                if t in dev: dropped += 1; continue
                n += 1
                yield t
    tr, va = [], []
    i = 0
    with Pool(48, initializer=init) as pool:
        for res in pool.imap(enc, chunks(lines()), chunksize=1):
            for a in res:
                (va if i % 2000 == 0 else tr).append(a); i += 1
            if i % 2_000_000 < 4096: print("lines", i, flush=True)
    for name, lst in (("train", tr), ("valid", va)):
        arr = np.concatenate(lst + [np.zeros(1, np.uint16)])
        arr.tofile(f"{OUT}/{name}.u16")
        print(name, len(lst), "lines", arr.size, "tokens", flush=True)
    json.dump({"src": SRC, "bpe": BPE, "bpe_sha256": hashlib.sha256(open(BPE,'rb').read()).hexdigest(), "kept_lines": n, "dropped_dev_matches": dropped,
               "valid_every": 2000, "token": "sp_id+1, 0=boundary"}, open(f"{OUT}/meta.json", "w"), indent=1)
main()
