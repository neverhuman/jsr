"""Diagnostic transducer beam search (H12): how much WER is search error?

`modified_beam_search` is the frame-synchronous beam of icefall's `modified_beam_search`: at most
one non-blank token per encoder frame, hypotheses with equal token sequences merged by log-sum-exp,
the `beam` best kept after every frame. With beam=1 it equals `greedy_decode(..., max_symbols=1)`.
Python and per-utterance: an evaluation tool, not a runtime path.
"""
from __future__ import annotations

import torch

from train.transducer import BLANK, Joiner, StatelessPredictor


@torch.no_grad()
def modified_beam_search(predictor: StatelessPredictor, joiner: Joiner, enc_p: torch.Tensor,
                         lens: torch.Tensor, beam: int = 4, blank_penalty: float = 0.0,
                         length_norm: bool = False, nbest: list | None = None) -> list[list[int]]:
    """enc_p (B, T, J) already projected; returns the best token list per utterance.

    blank_penalty is subtracted from the blank log-probability at every frame (> 0 favours
    emitting tokens, countering deletions). length_norm picks the final hypothesis by score per
    token instead of total score (as icefall's get_best(length_norm=True)). If `nbest` is a list,
    each utterance's final beam is appended to it as [(tokens, score), ...], best first."""
    if beam < 1:
        raise ValueError("beam must be >= 1")
    if blank_penalty < 0:
        raise ValueError("blank_penalty must be >= 0")
    out = []
    for b in range(enc_p.shape[0]):
        hyps: dict[tuple[int, ...], float] = {(): 0.0}
        for t in range(int(lens[b])):
            seqs = list(hyps)
            ctx = torch.tensor([[(s[-2] if len(s) > 1 else BLANK), (s[-1] if s else BLANK)]
                                for s in seqs], dtype=torch.long, device=enc_p.device)
            logp = torch.log_softmax(
                joiner.logits(enc_p[b, t].expand(len(seqs), -1), predictor(ctx)).float(), dim=-1)
            if blank_penalty:
                logp[:, BLANK] -= blank_penalty
            scores = torch.tensor([hyps[s] for s in seqs], device=enc_p.device)[:, None] + logp
            top = torch.topk(scores.flatten(), min(beam, scores.numel()))
            new: dict[tuple[int, ...], float] = {}
            v = logp.shape[1]
            for sc, flat in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                h, k = divmod(flat, v)
                s = seqs[h] if k == BLANK else seqs[h] + (k,)
                new[s] = float(torch.logaddexp(torch.tensor(new[s]), torch.tensor(sc))) \
                    if s in new else sc
            hyps = new
        key = (lambda kv: kv[1] / max(1, len(kv[0]))) if length_norm else (lambda kv: kv[1])
        ranked = sorted(hyps.items(), key=key, reverse=True)
        if nbest is not None:
            nbest.append([(list(t), float(sc)) for t, sc in ranked])
        best = ranked[0]
        out.append(list(best[0]))
    return out
