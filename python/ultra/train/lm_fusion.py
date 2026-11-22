"""Token-level n-gram shallow fusion for the transducer beam (H42, owner ruling O17).

The streaming-compatible "+LM" decode. Every non-blank token k that extends a hypothesis gets

    lm_weight * ln P_lm(k | last n-1 tokens)  -  lodr_weight * ln P_lodr(k | last m-1 tokens)

added to its transducer log-probability; blank gets nothing, and no end-of-sentence term is ever
added (a streaming decoder does not know where the utterance ends). P_lm is a KenLM n-gram over
the model's tokens estimated on the LibriSpeech LM text; P_lodr is a low-order n-gram over the same
tokens from the 960 h training transcripts (LODR: it stands in for the language model the
transducer absorbed from those transcripts, which is subtracted).

Tokens are written as decimal output indices ("17"; index 0 is blank) both in the LM training text
(`python -m train.lm_decode tokens`) and here, so piece strings such as "<unk>" never collide
with KenLM's own symbols. A token the LM never saw scores as KenLM's <unk>.

Cost model for an engine: one n-gram lookup per (hypothesis, candidate token). Here a whole row
(all tokens after one context) is computed once per distinct context and cached, because the
Python beam adds the row to every frame of that hypothesis.
"""
from __future__ import annotations

import math

import torch

LN10 = math.log(10.0)
BOS = -1  # marks "sentence start" at the front of a context shorter than the n-gram order needs


def context(seq, n: int) -> tuple[int, ...]:
    """The n-gram context of a token history: its last n tokens, or (BOS, *history) when the
    history is shorter than n. n = 0 (unigram) has the empty context."""
    if n <= 0:
        return ()
    if len(seq) >= n:
        return tuple(seq[len(seq) - n:])
    return (BOS, *seq)


def narrow(ctx: tuple[int, ...], n: int) -> tuple[int, ...]:
    """The length-n context implied by a longer (or equal) context `ctx` of the same history."""
    hist = ctx[1:] if ctx[:1] == (BOS,) else ctx
    if n <= 0:
        return ()
    if len(hist) >= n:
        return hist[len(hist) - n:]
    if ctx[:1] != (BOS,):
        raise ValueError(f"context {ctx} is shorter than {n} tokens and does not start a sentence")
    return (BOS, *hist)


class TokenNgram:
    """ln P(token | context) rows of a KenLM model over decimal token indices, cached by context."""

    def __init__(self, path: str, vocab: int, max_rows: int = 1 << 15):
        import kenlm  # optional dependency: only the "+LM" decode needs it

        self.kenlm = kenlm
        self.path = path
        self.model = kenlm.Model(path)
        self.order = int(self.model.order)
        self.vocab = vocab
        self.words = [str(k) for k in range(vocab)]
        self.max_rows = max_rows
        self.rows: dict[tuple[int, ...], torch.Tensor] = {}
        self.lookups = 0  # n-gram queries issued (rows * (vocab - 1) + context walks)

    def state(self, ctx: tuple[int, ...]):
        """KenLM state after `ctx` (a value of `context(seq, order - 1)`)."""
        st = self.kenlm.State()
        if ctx[:1] == (BOS,):
            self.model.BeginSentenceWrite(st)
            ctx = ctx[1:]
        else:
            self.model.NullContextWrite(st)
        for k in ctx:
            nxt = self.kenlm.State()
            self.model.BaseScore(st, self.words[k], nxt)
            st = nxt
        self.lookups += len(ctx)
        return st

    def row(self, ctx: tuple[int, ...]) -> torch.Tensor:
        """(vocab,) natural-log probabilities of every token after `ctx`; index 0 (blank) is 0."""
        got = self.rows.get(ctx)
        if got is None:
            st, out, base = self.state(ctx), self.kenlm.State(), self.model.BaseScore
            vals = [0.0]
            vals.extend(base(st, w, out) for w in self.words[1:])
            got = torch.tensor(vals, dtype=torch.float32) * LN10
            self.lookups += self.vocab - 1
            if len(self.rows) >= self.max_rows:
                self.rows.pop(next(iter(self.rows)))
            self.rows[ctx] = got
        return got


class NgramFusion:
    """The `fusion` argument of `train.beam.modified_beam_search`: called with the live
    hypotheses (token tuples), returns their (len, vocab) additive scores (0 in the blank column).
    """

    def __init__(self, lm: TokenNgram, lm_weight: float, lodr: TokenNgram | None = None,
                 lodr_weight: float = 0.0, max_rows: int = 1 << 15):
        if lm_weight < 0 or lodr_weight < 0:
            raise ValueError("lm_weight and lodr_weight must be >= 0")
        if lodr_weight and lodr is None:
            raise ValueError("lodr_weight needs a LODR model")
        self.lm, self.lm_weight = lm, float(lm_weight)
        self.lodr, self.lodr_weight = (lodr, float(lodr_weight)) if lodr_weight else (None, 0.0)
        self.n = max(lm.order, self.lodr.order if self.lodr else 1) - 1
        self.max_rows = max_rows
        self.rows: dict[tuple[int, ...], torch.Tensor] = {}

    def row(self, seq) -> torch.Tensor:
        key = context(seq, self.n)
        got = self.rows.get(key)
        if got is None:
            got = self.lm_weight * self.lm.row(narrow(key, self.lm.order - 1))
            if self.lodr is not None:
                got = got - self.lodr_weight * self.lodr.row(narrow(key, self.lodr.order - 1))
            if len(self.rows) >= self.max_rows:
                self.rows.pop(next(iter(self.rows)))
            self.rows[key] = got
        return got

    def __call__(self, seqs) -> torch.Tensor:
        return torch.stack([self.row(s) for s in seqs])


class FastBeam:
    """NumPy re-implementation of `train.beam.modified_beam_search` (no length normalisation) for
    weight sweeps: the same frame-synchronous search (one token per frame at most, equal token
    sequences merged by log-sum-exp, `beam` best kept, float32 running scores), about 20x faster
    because it avoids per-frame torch call overhead. Scores agree with the torch beam up to float32
    rounding, so a near-tie can resolve differently on rare utterances; `train.beam` stays the
    reference and reported rows are decoded with it."""

    def __init__(self, predictor, joiner):
        import numpy as np

        self.np = np
        emb = predictor.emb.weight.detach().numpy().astype(np.float32)
        w = predictor.proj.weight.detach().numpy().astype(np.float32)
        e = emb.shape[1]
        self.older = emb @ w[:, :e].T  # (vocab, joiner): the older context token's share
        self.newer = emb @ w[:, e:].T + predictor.proj.bias.detach().numpy()
        self.out_w = joiner.out.weight.detach().numpy().T.astype(np.float32).copy()  # (joiner, V)
        self.out_b = joiner.out.bias.detach().numpy().astype(np.float32)
        self.act = joiner.act

    def search(self, enc_p, beam: int = 4, blank_penalty: float = 0.0, fusion=None):
        """enc_p (T, joiner) float32 array -> [(tokens, score), ...] of the final beam, best first."""
        np = self.np
        if beam < 1 or blank_penalty < 0:
            raise ValueError("beam must be >= 1 and blank_penalty >= 0")
        v = self.out_b.shape[0]
        hyps: dict[tuple[int, ...], float] = {(): 0.0}
        for t in range(enc_p.shape[0]):
            seqs = list(hyps)
            h = enc_p[t] + self.older[[s[-2] if len(s) > 1 else 0 for s in seqs]] \
                + self.newer[[s[-1] if s else 0 for s in seqs]]
            h = np.tanh(h) if self.act == "tanh" else np.clip(h, 0.0, 6.0)
            z = h @ self.out_w + self.out_b
            z -= z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            if blank_penalty:
                logp[:, 0] -= np.float32(blank_penalty)
            if fusion is not None:
                logp += np.stack([fusion.row(s).numpy() for s in seqs])
            scores = (np.array([hyps[s] for s in seqs], dtype=np.float32)[:, None] + logp).ravel()
            k = min(beam, scores.size)
            top = np.argpartition(scores, -k)[-k:]
            top = top[np.argsort(-scores[top], kind="stable")]
            new: dict[tuple[int, ...], float] = {}
            for flat in top.tolist():
                hi, tok = divmod(flat, v)
                s = seqs[hi] if tok == 0 else (*seqs[hi], tok)
                sc = float(scores[flat])
                new[s] = float(np.logaddexp(np.float32(new[s]), np.float32(sc))) if s in new else sc
            hyps = new
        return [(list(s), sc) for s, sc in sorted(hyps.items(), key=lambda kv: kv[1], reverse=True)]
