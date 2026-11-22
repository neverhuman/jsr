"""Muon (MomentUm Orthogonalized by Newton-schulz) with AdamW for the rest (H23 arm d).

Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks" (2024 blog post
and reference code). Update RMS matching, so Muon can share AdamW's lr and weight decay, follows
Jingyuan Liu et al., "Muon is Scalable for LLM Training" (Moonshot AI, arXiv:2502.16982, 2025).
Reimplemented from those descriptions.

Muon groups (2-D hidden weight matrices of the encoder blocks):
  B = mu B + g;  G = g + mu B (Nesterov);  O = NewtonSchulz5(G)  (approximately U V^T of G's SVD)
  W = W (1 - lr wd) - lr * 0.2 * sqrt(max(rows, cols)) * O
The 0.2 * sqrt(max dim) factor gives O an update RMS close to AdamW's (Moonshot), so the AdamW lr
and schedule carry over unchanged. Everything else (embeddings, norms, biases, the conv stem, the
depthwise convs, and the output layers: CTC head, predictor, joiner, simple joiner) uses AdamW with
the trainer's lr, betas and weight-decay split, in the same optimiser so one LR scheduler drives
both.
"""
from __future__ import annotations

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Jordan's quintic iteration (tuned for 5 steps)


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a 2-D matrix: its singular values are pushed into ~[0.7, 1.2]."""
    a, b, c = NS_COEFFS
    dtype = torch.bfloat16 if g.is_cuda else torch.float32
    x = g.to(dtype)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    if tall:
        x = x.T
    return x.to(g.dtype)


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    """2-D hidden matrices inside the encoder (stem, heads, predictor and joiner excluded)."""
    return p.ndim == 2 and name.startswith("encoder.") and not name.startswith("encoder.stem")


class Muon(torch.optim.Optimizer):
    """Param groups carry ``use_muon``; non-Muon groups get a standard decoupled AdamW step."""

    def __init__(self, param_groups, lr: float = 5e-4, momentum: float = 0.95,
                 betas: tuple[float, float] = (0.9, 0.98), eps: float = 1e-8,
                 weight_decay: float = 0.0, ns_steps: int = 5, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if group["use_muon"]:
                    g = p.grad
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf = st["momentum_buffer"]
                    mu = group["momentum"]
                    buf.mul_(mu).add_(g)
                    u = g.add(buf, alpha=mu) if group["nesterov"] else buf
                    o = newton_schulz(u, group["ns_steps"])
                    scale = 0.2 * max(p.shape[0], p.shape[1]) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
                    continue
                b1, b2 = group["betas"]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                if wd:
                    p.mul_(1 - lr * wd)
                denom = (v / (1 - b2 ** t)).sqrt_().add_(group["eps"])
                p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
        return loss
