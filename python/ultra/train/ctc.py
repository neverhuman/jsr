"""Dense auxiliary CTC with an explicit feasibility check (spec v1.0-B §17.6, master spec §4.2).

A CTC target with U tokens and R adjacent repeats needs T >= U + R frames. Infeasible sequences
are masked out of the CTC term and counted (overall and by speaking-rate decile); their
transducer supervision is kept. An infinite CTC loss is never zeroed silently: `ctc_loss` runs with
zero_infinity=False on feasible sequences only and raises if one still comes back infinite.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from train.errors import ShapeError


def adjacent_repeats(targets: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Number of positions i < len-1 with targets[i] == targets[i+1], per sequence."""
    if targets.shape[1] < 2:
        return torch.zeros_like(lens)
    same = targets[:, 1:] == targets[:, :-1]
    valid = torch.arange(targets.shape[1] - 1, device=targets.device)[None] < (lens[:, None] - 1)
    return (same & valid).sum(dim=1)


def feasible(t_lens: torch.Tensor, targets: torch.Tensor, u_lens: torch.Tensor) -> torch.Tensor:
    return t_lens >= u_lens + adjacent_repeats(targets, u_lens)


@dataclass
class FeasibilityStats:
    """Counts of infeasible CTC targets; rate = tokens per encoder frame (speaking-rate proxy)."""

    total: int = 0
    infeasible: int = 0
    rates: list[float] = field(default_factory=list)
    flags: list[bool] = field(default_factory=list)

    def add(self, ok: torch.Tensor, t_lens: torch.Tensor, u_lens: torch.Tensor) -> None:
        self.total += int(ok.numel())
        self.infeasible += int((~ok).sum())
        self.rates.extend((u_lens.float() / t_lens.clamp(min=1).float()).tolist())
        self.flags.extend((~ok).tolist())

    def by_decile(self) -> list[dict]:
        if not self.rates:
            return []
        order = sorted(range(len(self.rates)), key=lambda i: self.rates[i])
        n = len(order)
        out = []
        for d in range(10):
            sel = order[d * n // 10:(d + 1) * n // 10]
            if sel:
                bad = sum(self.flags[i] for i in sel)
                out.append({"decile": d, "n": len(sel), "infeasible": bad,
                            "max_rate": round(self.rates[sel[-1]], 4)})
        return out

    def summary(self) -> dict:
        dec = self.by_decile()
        fastest = dec[-1] if dec else {"n": 0, "infeasible": 0}
        return {"ctc_total": self.total, "ctc_infeasible": self.infeasible,
                "ctc_fastest_decile_infeasible_frac":
                    round(fastest["infeasible"] / max(1, fastest["n"]), 5)}


def ctc_loss(log_probs: torch.Tensor, t_lens: torch.Tensor, targets: torch.Tensor,
             u_lens: torch.Tensor, stats: FeasibilityStats | None = None):
    """log_probs (B, T, V) log-softmax. Returns (sum of per-sequence NLL over feasible sequences,
    number of target tokens in them, feasibility mask)."""
    ok = feasible(t_lens, targets, u_lens)
    if stats is not None:
        stats.add(ok, t_lens, u_lens)
    if not bool(ok.any()):
        return log_probs.sum() * 0.0, 0, ok
    lp = log_probs[ok].transpose(0, 1).float()
    nll = F.ctc_loss(lp, targets[ok], t_lens[ok], u_lens[ok], blank=0, reduction="none",
                     zero_infinity=False)
    if not bool(torch.isfinite(nll).all()):
        raise ShapeError("CTC loss infinite on a sequence that passed the feasibility check")
    return nll.sum(), int(u_lens[ok].sum()), ok
