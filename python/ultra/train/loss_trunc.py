"""Wave 2A (WER-7a): Per-utterance loss truncation and Tukey/Cauchy outlier capping.

Guards semi-supervised pseudo-label training against corrupted transcript outliers (top 1 - q
fraction of per-token losses, e.g. q = 0.995 drops top 0.5% after warmup). When q <= 0.0, this
module is a bit-exact no-op.
"""
from __future__ import annotations

import collections
import math

import torch


def truncate_utterance_losses(
    nll: torch.Tensor,
    lens: torch.Tensor,
    q: float,
    *,
    mode: str = "drop",
    threshold: float | None = None,
) -> tuple[torch.Tensor, int, dict[str, float]]:
    """Apply WER-7a per-utterance loss truncation or Tukey/Cauchy capping.

    Args:
      nll: (B,) per-utterance NLL (differentiable).
      lens: (B,) target token lengths per utterance.
      q: keep quantile in (0, 1); <= 0.0 disables truncation (bit-exact identity).
      mode: 'drop' (hard truncation), 'cauchy' (smooth log1p cap), or 'tukey' (biweight cap).
      threshold: optional pre-computed per-token loss threshold; if None, uses batch q-quantile.

    Returns ``(adjusted_nll, effective_tokens, stats_dict)``.
    """
    if q <= 0.0 or nll.numel() == 0:
        return nll, int(lens.sum()), {"trunc_dropped": 0.0, "trunc_thr": 0.0}
    if not (0.0 < q < 1.0):
        raise ValueError(f"loss_trunc_q must be 0.0 (off) or in (0, 1), got {q}")
    if mode not in ("drop", "cauchy", "tukey"):
        raise ValueError(f"unknown loss_trunc mode {mode!r}; expected 'drop', 'cauchy', or 'tukey'")

    lens_f = lens.to(nll.dtype).clamp(min=1.0)
    per_tok = nll / lens_f
    per_tok_d = per_tok.detach().float()
    finite = torch.isfinite(per_tok_d)
    if not bool(finite.any()):
        return nll, int(lens.sum()), {"trunc_dropped": 0.0, "trunc_thr": float("nan")}
    thr_val = float(threshold) if (threshold is not None and math.isfinite(threshold) and threshold > 0.0) else float(
        torch.quantile(per_tok_d[finite], q)
    )
    thr = nll.new_tensor(max(thr_val, 1e-6))

    if mode == "drop":
        keep = per_tok.detach() <= thr
        if not bool(keep.any()):
            keep = torch.zeros_like(keep)
            keep[int(torch.argmin(per_tok_d))] = True
        adj = torch.where(keep, nll, nll * 0.0)
        kept_tokens = int((lens * keep.to(lens.dtype)).sum())
        dropped = int((~keep).sum())
        return adj, max(1, kept_tokens), {"trunc_dropped": float(dropped), "trunc_thr": thr_val}

    if mode == "cauchy":
        # Smooth C^1 Cauchy cap above thr: per_tok <= thr unchanged (grad=1), > thr -> thr * (1 + log1p(e/thr)).
        over = per_tok > thr
        excess = torch.clamp(per_tok - thr, min=0.0)
        capped = torch.where(over, thr + thr * torch.log1p(excess / thr), per_tok)
        dropped = int(over.detach().sum())
        return capped * lens_f, int(lens.sum()), {"trunc_dropped": float(dropped), "trunc_thr": thr_val}

    # mode == "tukey": Tukey biweight attenuation above thr, zero gradient beyond 3 * thr.
    # Integral of (1 - u^2)^2 for u = (x - thr) / (2 * thr) in [0, 1] is 2*thr * (u - 2/3 u^3 + 1/5 u^5).
    over = per_tok > thr
    u = torch.clamp((per_tok - thr) / (2.0 * thr), min=0.0, max=1.0)
    capped = torch.where(over, thr + (2.0 * thr) * (u - (2.0 / 3.0) * u.pow(3) + 0.2 * u.pow(5)), per_tok)
    dropped = int((per_tok.detach() >= 3.0 * thr).sum())
    return capped * lens_f, int(lens.sum()), {"trunc_dropped": float(dropped), "trunc_thr": thr_val}


class LossTruncator:
    """Stateful rolling-quantile WER-7a loss truncator with warmup."""

    def __init__(self, q: float = 0.0, warmup_steps: int = 0, mode: str = "drop",
                 window: int = 4096, min_history: int = 16) -> None:
        if q != 0.0 and not (0.0 < q < 1.0):
            raise ValueError(f"loss_trunc_q must be 0.0 (off) or in (0, 1), got {q}")
        if mode not in ("drop", "cauchy", "tukey"):
            raise ValueError(f"unknown loss_trunc mode {mode!r}")
        self.q = float(q)
        self.warmup_steps = max(0, int(warmup_steps))
        self.mode = mode
        self.min_history = max(1, int(min_history))
        self.history: collections.deque[float] = collections.deque(maxlen=max(16, int(window)))

    @property
    def active(self) -> bool:
        return self.q > 0.0

    def __call__(self, nll: torch.Tensor, lens: torch.Tensor, step: int = 0) -> tuple[torch.Tensor, int, dict[str, float]]:
        if not self.active:
            return nll, int(lens.sum()), {"trunc_dropped": 0.0, "trunc_thr": 0.0}
        per_tok = (nll.detach().float() / lens.to(torch.float32).clamp(min=1.0)).cpu().tolist()
        for v in per_tok:
            if math.isfinite(v):
                self.history.append(v)
        if step < self.warmup_steps:
            return nll, int(lens.sum()), {"trunc_dropped": 0.0, "trunc_thr": 0.0}
        thr = None
        if len(self.history) >= self.min_history:
            thr = float(torch.quantile(torch.tensor(list(self.history), dtype=torch.float32), self.q))
        return truncate_utterance_losses(nll, lens, self.q, mode=self.mode, threshold=thr)
