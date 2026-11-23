"""H46 item 3: write a uint16 token stream (`train.nlm tokens` output: 0 after every sentence) to
stdout as KenLM `lmplz` input: one sentence per line, tokens as decimal output indices ("17"),
exactly the text H42's `train.lm_decode tokens` writes for the same lines.

  cd python/ultra && python -m train.ops.h46_u16_to_text train.u16 | lmplz -o 3 ...
"""
import sys

import numpy as np

from train.errors import LmEvalError

V = 501


def table():
    """(V, 4) uint8 decimal strings of 1..500 followed by a space, and their lengths."""
    tab = np.zeros((V, 4), dtype=np.uint8)
    lens = np.zeros(V, dtype=np.int64)
    for v in range(1, V):
        s = f"{v} ".encode()
        tab[v, :len(s)] = np.frombuffer(s, dtype=np.uint8)
        lens[v] = len(s)
    return tab, lens


def convert(t: np.ndarray, tab: np.ndarray, lens: np.ndarray) -> bytes:
    """Bytes for a chunk that ends right after a sentence break."""
    t = t.astype(np.int64)
    ln = lens[t]
    keep = np.arange(4)[None, :] < ln[:, None]
    out = tab[t][keep]
    ends = np.cumsum(ln) - 1  # offset of each token's trailing space
    nxt = np.empty_like(t)
    nxt[:-1] = t[1:]
    nxt[-1] = 0
    last = (t != 0) & (nxt == 0)  # the final token of a sentence ends its line
    out[ends[last]] = ord("\n")
    return out.tobytes()


def main(path: str, chunk: int = 1 << 26) -> None:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if data.shape[0] and int(data[-1]) != 0:
        raise LmEvalError(f"{path}: the stream must end with a sentence break (0)")
    tab, lens = table()
    w = sys.stdout.buffer
    s, n = 0, data.shape[0]
    while s < n:
        e = min(n, s + chunk)
        if e < n:  # cut after a sentence break so no line spans two chunks
            brk = np.flatnonzero(np.asarray(data[s:e]) == 0)
            e = s + int(brk[-1]) + 1
        part = np.asarray(data[s:e])
        if int(part.max(initial=0)) >= V:
            raise LmEvalError(f"{path}: token >= {V} at offset >= {s}")
        w.write(convert(part, tab, lens))
        s = e
    w.flush()


if __name__ == "__main__":
    main(sys.argv[1])
