"""Joiner z-loss (H23 stability option).

z-loss = mean over lattice cells of logsumexp(z)^2, where z are the joiner's output logits. It keeps
the log-partition near 0, so logits cannot drift to large magnitudes at high learning rates (PaLM,
Chowdhery et al. 2022, sec. 5; ST-MoE, Zoph et al. 2022, for the router variant). Weight ~1e-4.

- Pruned loss: averaged over the band logits k2 already built, on valid frames only.
- Exact loss: the full lattice is never materialised (train.transducer streams it in time
  chunks), so the z-loss is taken on one cell per valid frame, along the straight alignment
  u = floor(t U / T). That cheap proxy covers the region the RNN-T posterior concentrates on.
"""
from __future__ import annotations

import torch


def frame_mask(enc_lens: torch.Tensor, t: int) -> torch.Tensor:
    return torch.arange(t, device=enc_lens.device)[None, :] < enc_lens[:, None]


def band_zloss(logits: torch.Tensor, enc_lens: torch.Tensor) -> torch.Tensor:
    """logits (B, T, S, V) from k2.do_rnnt_pruning + joiner."""
    lse = torch.logsumexp(logits.float(), dim=-1)  # (B, T, S)
    m = frame_mask(enc_lens, logits.shape[1]).to(lse.dtype)[..., None].expand_as(lse)
    return (lse.pow(2) * m).sum() / m.sum().clamp(min=1.0)


def diagonal_zloss(joiner, enc_p: torch.Tensor, pred_p: torch.Tensor, enc_lens: torch.Tensor,
                   tgt_lens: torch.Tensor) -> torch.Tensor:
    """enc_p (B, T, J), pred_p (B, U+1, J): one joiner cell per frame on the straight alignment."""
    B, T, _ = enc_p.shape
    t = torch.arange(T, device=enc_p.device)[None, :].expand(B, T)
    u = (t * (tgt_lens[:, None] + 1)) // enc_lens.clamp(min=1)[:, None]
    u = torch.minimum(u, tgt_lens[:, None]).clamp(min=0).long()
    pp = torch.gather(pred_p, 1, u[..., None].expand(B, T, pred_p.shape[-1]))
    lse = torch.logsumexp(joiner.logits(enc_p, pp).float(), dim=-1)  # (B, T)
    m = frame_mask(enc_lens, T).to(lse.dtype)
    return (lse.pow(2) * m).sum() / m.sum().clamp(min=1.0)
