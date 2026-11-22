"""Stateless predictor, joiner, memory-bounded RNN-T loss, and greedy decoding.

Predictor (spec v1.0-B §17.7): the last two non-blank tokens, 256-d embeddings each, concatenated
(512) and projected to the joiner width. Token order is (older, newer); beginning-of-sequence
padding is the blank index 0.

Joiner: logits = W_out . act(P_enc h_t + P_pred g_u) + b. act is tanh (the reproduced RNN-T
default) or a clipped ReLU (master spec §4.6), selected in the config.

RNN-T loss. The full (B, T, U+1, V+1) logit lattice is never materialised for the whole batch:
`joiner_logprobs` computes it in time chunks and keeps only the blank and target log-probs
(B, T, U+1). Its backward recomputes each chunk and applies the analytic softmax gradient. The
forward-variable recursion over those log-probs is plain torch with autograd. Both pieces are
checked against a brute-force alignment enumeration and against full-lattice autograd in
tests/test_train_loss.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from train.errors import ShapeError

BLANK = 0
CLIP = 6.0


def _act(x: torch.Tensor, kind: str) -> torch.Tensor:
    return torch.tanh(x) if kind == "tanh" else torch.clamp(x, 0.0, CLIP)


def _act_grad_from_out(h: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "tanh":
        return 1.0 - h * h
    return ((h > 0) & (h < CLIP)).to(h.dtype)


class StatelessPredictor(nn.Module):
    def __init__(self, vocab_out: int, emb: int = 256, joiner: int = 320, context_len: int = 2):
        super().__init__()
        if context_len < 1:
            raise ShapeError(f"context_len must be >= 1, got {context_len}")
        self.context_len = context_len
        self.emb = nn.Embedding(vocab_out, emb)
        self.proj = nn.Linear(context_len * emb, joiner)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx (..., context_len) token indices (oldest .. newest) -> (..., joiner)."""
        e = self.emb(ctx)
        return self.proj(e.flatten(-2))

    @staticmethod
    def contexts(targets: torch.Tensor, context_len: int = 2) -> torch.Tensor:
        """targets (B, U) -> contexts (B, U+1, context_len): context before emitting target u."""
        b, u = targets.shape[0], targets.shape[1]
        z = targets.new_zeros(b, context_len)
        padded = torch.cat([z, targets], dim=1)  # (B, U + context_len)
        return torch.stack([padded[:, i:i + u + 1] for i in range(context_len)], dim=-1)


class Joiner(nn.Module):
    def __init__(self, d_enc: int, joiner: int, vocab_out: int, act: str = "tanh"):
        super().__init__()
        if act not in ("tanh", "clipped_relu"):
            raise ShapeError(f"unknown joiner activation {act}")
        self.enc_proj = nn.Linear(d_enc, joiner)
        self.out = nn.Linear(joiner, vocab_out)
        self.act = act

    def logits(self, enc_p: torch.Tensor, pred_p: torch.Tensor) -> torch.Tensor:
        return self.out(_act(enc_p + pred_p, self.act))


class _JoinerLogProbs(torch.autograd.Function):
    """(enc_p (B,T,J), pred_p (B,U1,J), W (V,J), b (V,), targets (B,U1-1)) ->
    (lp_blank (B,T,U1), lp_label (B,T,U1-1)), computed in time chunks of `tc` frames."""

    @staticmethod
    def forward(ctx, enc_p, pred_p, w, bias, targets, act, tc, act_q=False):
        B, T, J = enc_p.shape
        U1 = pred_p.shape[1]
        lpb = enc_p.new_empty(B, T, U1)
        lpl = enc_p.new_empty(B, T, U1 - 1)
        tgt = targets[:, None, :, None]
        for t0 in range(0, T, tc):
            t1 = min(T, t0 + tc)
            h = _act(enc_p[:, t0:t1, None, :] + pred_p[:, None, :, :], act)
            z = torch.matmul(_fq(h, act_q), w.t()) + bias  # (B, tc, U1, V)
            lse = torch.logsumexp(z, dim=-1)
            lpb[:, t0:t1] = z[..., BLANK] - lse
            if U1 > 1:
                zl = torch.gather(z[:, :, :-1], 3, tgt.expand(B, t1 - t0, U1 - 1, 1)).squeeze(-1)
                lpl[:, t0:t1] = zl - lse[:, :, :-1]
        ctx.save_for_backward(enc_p, pred_p, w, bias, targets)
        ctx.act, ctx.tc, ctx.act_q = act, tc, act_q
        return lpb, lpl

    @staticmethod
    def backward(ctx, g_lpb, g_lpl):
        enc_p, pred_p, w, bias, targets = ctx.saved_tensors
        act, tc, act_q = ctx.act, ctx.tc, ctx.act_q
        B, T, J = enc_p.shape
        U1 = pred_p.shape[1]
        g_enc = torch.zeros_like(enc_p)
        g_pred = torch.zeros_like(pred_p)
        g_w = torch.zeros_like(w)
        g_b = torch.zeros_like(bias)
        tgt = targets[:, None, :, None]
        for t0 in range(0, T, tc):
            t1 = min(T, t0 + tc)
            n = t1 - t0
            h = _act(enc_p[:, t0:t1, None, :] + pred_p[:, None, :, :], act)
            hq = _fq(h, act_q)
            z = torch.matmul(hq, w.t()) + bias
            p = torch.softmax(z, dim=-1)
            gb = g_lpb[:, t0:t1]  # (B, n, U1)
            gl = torch.zeros_like(gb)
            if U1 > 1:
                gl[:, :, :-1] = g_lpl[:, t0:t1]
            gz = p * (-(gb + gl))[..., None]
            gz[..., BLANK] += gb
            if U1 > 1:
                gz[:, :, :-1].scatter_add_(3, tgt.expand(B, n, U1 - 1, 1), gl[:, :, :-1, None])
            g_b += gz.sum(dim=(0, 1, 2))
            g_w += torch.einsum("btuv,btuj->vj", gz, hq)
            gpre = torch.matmul(gz, w) * _act_grad_from_out(h, act)  # STE through the act quant
            g_enc[:, t0:t1] = gpre.sum(dim=2)
            g_pred += gpre.sum(dim=1)
        return g_enc, g_pred, g_w, g_b, None, None, None, None


def _fq(h: torch.Tensor, on: bool) -> torch.Tensor:
    if not on:
        return h
    from train.qat import fake_quant_rows

    return fake_quant_rows(h).detach()


def joiner_logprobs(joiner: Joiner, enc_p, pred_p, targets, budget_elems: int = 1 << 25):
    """Blank and target log-probs over the lattice without keeping it whole. A ternary joiner
    output layer (train.qat.TernaryLinear) contributes its STE weight and per-row int8 inputs."""
    B, T, _ = enc_p.shape
    U1 = pred_p.shape[1]
    V = joiner.out.weight.shape[0]
    tc = max(1, min(T, budget_elems // max(1, B * U1 * V)))
    ternary = hasattr(joiner.out, "effective_weight")
    w = joiner.out.effective_weight().float() if ternary else joiner.out.weight
    return _JoinerLogProbs.apply(enc_p, pred_p, w, joiner.out.bias, targets, joiner.act, tc,
                                 ternary)


def full_lattice_logprobs(joiner: Joiner, enc_p, pred_p, targets):
    """Reference: the whole lattice through autograd (small fixtures only)."""
    z = joiner.logits(enc_p[:, :, None, :], pred_p[:, None, :, :])
    lp = torch.log_softmax(z, dim=-1)
    lpb = lp[..., BLANK]
    tgt = targets[:, None, :, None].expand(-1, z.shape[1], -1, 1)
    lpl = torch.gather(lp[:, :, :-1], 3, tgt).squeeze(-1)
    return lpb, lpl


def rnnt_nll(lpb: torch.Tensor, lpl: torch.Tensor, t_lens: torch.Tensor, u_lens: torch.Tensor):
    """Per-sequence RNN-T negative log-likelihood from blank/label log-probs.

    alpha(t, u) = logaddexp(alpha(t-1, u) + lpb(t-1, u), alpha(t, u-1) + lpl(t, u-1)).
    Within one t the u-recursion is a log-cumulative-sum, so the loop runs over T only.
    Padding beyond (t_len, u_len) never feeds a used cell. Computed in float64.
    """
    B, T, U1 = lpb.shape
    if lpb.device.type == "mps":
        # No float64 on Apple MPS: run the (small, B x T x U) recursion on the CPU in float64;
        # autograd carries the gradient back across the device copy.
        return rnnt_nll(lpb.cpu(), lpl.cpu(), t_lens.cpu(), u_lens.cpu()).float().to(lpb.device)
    lpb, lpl = lpb.double(), lpl.double()
    neg = torch.full((B, 1), float("-inf"), dtype=torch.float64, device=lpb.device)
    zero_col = torch.zeros(B, 1, dtype=torch.float64, device=lpb.device)
    rows = []
    a_in = torch.cat([zero_col, neg.expand(B, U1 - 1)], dim=1)
    for t in range(T):
        s = torch.cat([zero_col, torch.cumsum(lpl[:, t], dim=1)], dim=1)  # (B, U1)
        alpha_t = s + torch.logcumsumexp(a_in - s, dim=1)
        rows.append(alpha_t)
        a_in = alpha_t + lpb[:, t]
    alpha = torch.stack(rows, dim=1)  # (B, T, U1)
    bi = torch.arange(B, device=lpb.device)
    tl, ul = t_lens - 1, u_lens
    return -(alpha[bi, tl, ul] + lpb[bi, tl, ul])


def rnnt_bruteforce(lpb: torch.Tensor, lpl: torch.Tensor, t_len: int, u_len: int) -> torch.Tensor:
    """Enumerate every alignment of one sequence (tiny fixtures): -log sum_paths exp(score)."""
    scores = []

    def walk(t, u, acc):
        if t == t_len - 1 and u == u_len:
            scores.append(acc + lpb[t, u])
            return
        if u < u_len:
            walk(t, u + 1, acc + lpl[t, u])
        if t < t_len - 1:
            walk(t + 1, u, acc + lpb[t, u])

    walk(0, 0, torch.zeros((), dtype=lpb.dtype))
    return -torch.logsumexp(torch.stack(scores), dim=0)


@torch.no_grad()
def greedy_decode(predictor: StatelessPredictor, joiner: Joiner, enc_p: torch.Tensor,
                  lens: torch.Tensor, max_symbols: int = 8) -> tuple[list[list[int]], int]:
    """Frame-synchronous greedy decoding, batched. enc_p (B, T, J) already projected.

    Blank advances time; a non-blank token is emitted and updates the predictor context. After
    `max_symbols` emissions at one frame, time advances anyway and the forced-advance counter
    increments (spec v1.0-B §17.7). Returns (token lists, forced advances).
    """
    B, T, _ = enc_p.shape
    k_ctx = getattr(predictor, "context_len", 2)
    ctx = torch.zeros(B, k_ctx, dtype=torch.long, device=enc_p.device)
    out: list[list[int]] = [[] for _ in range(B)]
    forced = 0
    for t in range(T):
        active = lens > t
        for _ in range(max_symbols):
            if not bool(active.any()):
                break
            idx = active.nonzero().squeeze(1)
            z = joiner.logits(enc_p[idx, t], predictor(ctx[idx]))
            k = z.argmax(dim=-1)
            emit = k != BLANK
            for i, kk in zip(idx[emit].tolist(), k[emit].tolist(), strict=True):
                out[i].append(kk)
            ei = idx[emit]
            if k_ctx == 1:
                ctx[ei] = k[emit].unsqueeze(1)
            else:
                ctx[ei] = torch.cat([ctx[ei, 1:], k[emit].unsqueeze(1)], dim=1)
            active = torch.zeros_like(active)
            active[ei] = True
        else:
            forced += int(active.sum())
    return out, forced
