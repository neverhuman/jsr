"""H46 item 3: exact bigram and trigram counts of the tokenised LM text (`train.nlm tokens`
output: uint16 tokens, 0 after every sentence), as count-of-counts tables, and the smallest
`lmplz --prune` thresholds that keep a target number of n-grams (amendment 2: thresholds come from
counts and sizes only, never from WER or perplexity).

  cd python/ultra && python -m train.ops.h46_ngram_counts <train.u16> <out.json> \
      [--targets 2:40000,3:0 ...]

Sentence breaks act as <s> / </s>: a bigram is any adjacent pair except (break, break); a trigram
is any window whose middle token is not a break (so <s> a b and a b </s> count, a </s> <s> b does
not), as KenLM counts them. lmplz drops n-grams whose count is <= the threshold.
"""
import json
import sys

import numpy as np

from train.errors import LmEvalError

V = 501


def counts(path: str, chunk: int = 1 << 24) -> tuple[np.ndarray, np.ndarray]:
    """Exact bigram and trigram counts in bounded memory: bigrams by a per-chunk bincount into a
    2 MB table; trigrams by sorting each chunk's codes and adding the per-code counts into one
    persistent int32 table (501^3 < 2^31 bins, 503 MB). No multi-GB temporary is allocated, so
    the pass is cheap on hosts whose page faults are slow and fits beside other jobs."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = data.shape[0]
    c2 = np.zeros(V * V, dtype=np.int64)
    c3 = np.zeros(V * V * V, dtype=np.int32)
    for s in range(0, n - 1, chunk):
        e = min(n, s + chunk + 2)
        x = np.asarray(data[s:e]).astype(np.int64)
        if int(x.max(initial=0)) >= V:
            raise LmEvalError(f"{path}: token >= {V} in [{s}, {e})")
        last = e == n
        # each window is counted in the chunk where it starts: starts s .. s + chunk - 1
        a, b = (x[:-1], x[1:]) if last else (x[:-2], x[1:-1])
        ok2 = ~((a == 0) & (b == 0))
        c2 += np.bincount((a * V + b)[ok2], minlength=V * V)
        p, q, r = x[:-2], x[1:-1], x[2:]
        k = len(p) if last else min(len(p), chunk)
        keep = q[:k] != 0
        code = ((p[:k] * V + q[:k]) * V + r[:k])[keep]
        code.sort()
        u, cnt = np.unique(code, return_counts=True)
        c3[u] += cnt.astype(np.int32)
        if last:
            break
    return c2, c3


def coc(c: np.ndarray) -> dict[int, int]:
    """Count-of-counts up to 1000 (kept n-grams with count > t for each t)."""
    nz = c[c > 0]
    out = {}
    for t in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200,
              300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000):
        out[t] = int((nz > t).sum())
    return out


def main(path: str, out: str) -> None:
    c2, c3 = counts(path)
    res = {"source": path, "bigram_types": int((c2 > 0).sum()), "trigram_types": int((c3 > 0).sum()),
           "kept_bigrams_by_threshold": coc(c2), "kept_trigrams_by_threshold": coc(c3)}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
