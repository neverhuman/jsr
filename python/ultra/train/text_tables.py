"""H46 item 4 ("+text"): hashed 2-4-token context tables for the stateless predictor, trained
with a text-only loss on the LibriSpeech LM text (research/H46_lm_and_text/README.md,
amendment 3). Results that use it are labelled "+text".

`TextPredictor` is a `StatelessPredictor` whose output gains

    W_t · [T2[g2 mod P2]; T3[g3 mod P3]; T4[g4 mod P4]]

where g_n is the exact id of the last n emitted tokens (oldest first, 0-padded before the
utterance start), g_n = sum_i c_i * 501^(n-1-i) < 501^4 < 2^63, P_n are primes and W_t starts at
zero, so at initialisation the predictor output equals the plain predictor's bit for bit. Its
`context_len` is the table width (4); the dense path reads the last `dense_len` (2) tokens, so
every caller that builds `predictor.context_len`-wide contexts (greedy decode, the losses, the
Python beam once it reads `context_len`) works unchanged.

`text_loss` is the text-only objective: next-token cross-entropy over the 500 non-blank tokens
through the frozen joiner at a fixed "no audio" encoder vector `e_null`, with the gradient
reaching only the tables and W_t (the dense predictor and the joiner see it through a
stop-gradient). `TextStream` draws (context, next token) pairs from a `train.nlm tokens` stream.

Engine note (Codex): per new hypothesis context, three exact ids, three `mod`s, three 64-wide
row loads and one (192 -> joiner) product added to pred_p; the beam state carries 4 tokens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from train.errors import TextTablesError
from train.transducer import BLANK, Joiner, StatelessPredictor, _act

V_TOKENS = 501
PRIMES = {2: 16381, 3: 16369, 4: 16363}


def ngram_ids(c: torch.Tensor) -> torch.Tensor:
    """(..., n) int64 tokens, oldest first -> (...) exact n-gram ids sum_i c_i * 501^(n-1-i)."""
    g = torch.zeros(c.shape[:-1], dtype=torch.long, device=c.device)
    for i in range(c.shape[-1]):
        g = g * V_TOKENS + c[..., i].long()
    return g


class TextTables(nn.Module):
    def __init__(self, joiner: int, dim: int = 64, orders: tuple[int, ...] = (2, 3, 4)):
        super().__init__()
        self.orders = tuple(orders)
        self.width = max(self.orders)
        self.tables = nn.ModuleList(nn.Embedding(PRIMES[n], dim) for n in self.orders)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        self.out = nn.Linear(dim * len(self.orders), joiner, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx (..., W >= width) tokens oldest .. newest -> (..., joiner)."""
        w = ctx.shape[-1]
        rows = [t(ngram_ids(ctx[..., w - n:]) % PRIMES[n])
                for n, t in zip(self.orders, self.tables, strict=True)]
        return self.out(torch.cat(rows, dim=-1))


class TextPredictor(StatelessPredictor):
    """StatelessPredictor + TextTables. `context_len` is the context width callers must build
    (the table width); the dense embedding / projection read the last `dense_len` tokens."""

    def __init__(self, vocab_out: int, emb: int = 256, joiner: int = 320, dense_len: int = 2,
                 dim: int = 64, orders: tuple[int, ...] = (2, 3, 4)):
        super().__init__(vocab_out, emb, joiner, context_len=dense_len)
        self.dense_len = dense_len
        self.text = TextTables(joiner, dim, orders)
        self.context_len = max(dense_len, self.text.width)

    def dense(self, ctx: torch.Tensor) -> torch.Tensor:
        e = self.emb(ctx[..., ctx.shape[-1] - self.dense_len:])
        return self.proj(e.flatten(-2))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.dense(ctx) + self.text(ctx)

    def load_plain(self, state: dict) -> None:
        """Initialise from a plain StatelessPredictor state dict (B0): emb / proj copied, the
        tables keep their init and W_t stays zero."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("text.") for k in missing):
            raise TextTablesError(f"not a plain predictor state: missing {missing}, "
                                  f"unexpected {unexpected}")


def text_loss(predictor: TextPredictor, joiner: Joiner, ctx: torch.Tensor, tgt: torch.Tensor,
              e_null: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL over the non-blank tokens of (ctx (N, W), tgt (N,) in 1..500):
    logits = joiner.out(act(e_null + dense(ctx)^sg + text(ctx))) with the joiner frozen."""
    pred = predictor.dense(ctx).detach() + predictor.text(ctx)
    h = _act(e_null.detach() + pred, joiner.act)
    z = nn.functional.linear(h, joiner.out.weight.detach(), joiner.out.bias.detach()).float()
    lp = torch.log_softmax(z[:, BLANK + 1:], dim=-1)  # renormalised without blank
    return -lp.gather(1, (tgt.long() - 1)[:, None]).mean()


@torch.no_grad()
def mean_enc_proj(model, batches) -> torch.Tensor:
    """e_null: the mean joiner.enc_proj output over the valid frames of `batches` (an iterable
    of train.data Batch objects), computed with the model in eval mode."""
    was = model.training
    model.eval()
    tot, n = None, 0
    for b in batches:
        enc, lens = model.encode(b.feats, b.feat_lens, None)
        e = model.joiner.enc_proj(enc.float())
        mask = torch.arange(e.shape[1], device=e.device)[None] < lens[:, None]
        s = e[mask].sum(0)
        tot = s if tot is None else tot + s
        n += int(mask.sum())
    model.train(was)
    return tot / max(n, 1)


class TextStream:
    """(context, next token) pairs drawn uniformly from a `train.nlm tokens` stream (uint16,
    0 after every sentence); contexts reset at sentence breaks (train.nlm.contexts)."""

    def __init__(self, path: str, width: int, seed: int = 1234):
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if data.shape[0] == 0 or int(data[-1]) != 0 or int(data.max()) >= V_TOKENS:
            raise TextTablesError(f"{path}: not a train.nlm token stream (tokens < {V_TOKENS}, "
                                  f"0 after every sentence)")
        self.stream = torch.from_numpy(np.asarray(data, dtype=np.int16))
        self.width = width
        self.gen = torch.Generator().manual_seed(seed)

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        from train.nlm import contexts

        pos = torch.randint(1, self.stream.shape[0], (n,), generator=self.gen)
        tgt = self.stream[pos].long()
        keep = tgt > 0
        pos, tgt = pos[keep], tgt[keep]
        return contexts(self.stream, pos, self.width), tgt
