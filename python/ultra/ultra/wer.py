"""Provisional WER for the Python research harness (LibriSpeech style: upper-case, no punctuation).

The scoring authority is the Rust `ultra-eval` crate. This module exists so research runs can
report a number before that lands; results are re-scored by ultra-eval before any claim.
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^a-z0-9' ]+")


def normalise(s: str) -> list[str]:
    return _PUNCT.sub(" ", s.lower()).split()


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a minimum-cost alignment."""
    n, m = len(ref), len(hyp)
    # cost, s, d, i per cell; row by row
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for a in range(1, n + 1):
        cur = [(a, 0, a, 0)]
        for b in range(1, m + 1):
            same = ref[a - 1] == hyp[b - 1]
            c_sub = prev[b - 1]
            sub = (c_sub[0] + (0 if same else 1), c_sub[1] + (0 if same else 1), c_sub[2], c_sub[3])
            dl = prev[b]
            dele = (dl[0] + 1, dl[1], dl[2] + 1, dl[3])
            il = cur[b - 1]
            ins = (il[0] + 1, il[1], il[2], il[3] + 1)
            cur.append(min(sub, dele, ins))
        prev = cur
    _, s, d, i = prev[m]
    return s, d, i


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    S = D = I = N = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalise(r), normalise(h)
        s, d, i = edit_counts(rw, hw)
        S, D, I, N = S + s, D + d, I + i, N + len(rw)
    return {"wer": round(100.0 * (S + D + I) / max(N, 1), 3), "sub": S, "del": D, "ins": I, "ref_words": N}
