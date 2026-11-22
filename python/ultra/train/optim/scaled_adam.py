"""ScaledAdam optimiser and Eden LR schedule (H23 arm b).

Credit: Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Yang, Zengrui Jin,
Long Lin and Daniel Povey, "Zipformer: A faster and better encoder for automatic speech
recognition", ICLR 2024, Section 4 (ScaledAdam, Eden). The reference implementation is icefall's
``egs/librispeech/ASR/zipformer/optim.py`` (k2-fsa/icefall, Apache-2.0). This module is a fresh
reimplementation from the paper's equations, not a copy of that code. Simplifications versus
icefall: no batched multi-tensor kernels, the parameter RMS is recomputed every step (icefall:
every ``size_update_period`` = 4 steps), and the built-in median-norm gradient clipping is left to
the trainer's ``grad_clip_mode=adaptive`` (train/optim/clip.py).

Update for a non-scalar tensor theta with gradient g (step t, lr alpha_t):

  r      = clamp(RMS(theta), param_min_rms, param_max_rms)       # the parameter's own scale
  m, v   = Adam moments of g (v bias-corrected)
  d_dir  = -alpha_t * r * m / (sqrt(v) + eps)                     # scale-relative Adam step
  h      = sum(g * theta)                                         # gradient w.r.t. log scale
  m', v' = Adam moments of h
  d_scl  = -alpha_t * scalar_lr_scale * m' / (sqrt(v') + eps) * theta   # learn the scale
  theta += d_dir + d_scl     (scale growth is suppressed once RMS(theta) > param_max_rms)

Scalars (numel == 1) get a plain Adam step at lr * scalar_lr_scale, clamped to
[-scalar_max, scalar_max]. Because updates are relative to the parameter's RMS, the base LR is about
two orders of magnitude larger than AdamW's (icefall's default is 0.045). There is no weight decay:
the RMS ceiling plays that role.
"""
from __future__ import annotations

import math

import torch

from train.errors import OptimConfigError


class ScaledAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.045, betas: tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-8, scalar_lr_scale: float = 0.1, param_min_rms: float = 1e-5,
                 param_max_rms: float = 3.0, scalar_max: float = 10.0):
        if lr <= 0:
            raise OptimConfigError("lr must be > 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, scalar_lr_scale=scalar_lr_scale,
                        param_min_rms=param_min_rms, param_max_rms=param_max_rms,
                        scalar_max=scalar_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if p.numel() > 1:
                        st["scale_avg"] = torch.zeros((), device=p.device)
                        st["scale_avg_sq"] = torch.zeros((), device=p.device)
                st["step"] += 1
                t = st["step"]
                bc2 = 1.0 - b2 ** t
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v / bc2).sqrt_().add_(eps)
                if p.numel() == 1:
                    p.add_((m / denom).to(p.dtype), alpha=-lr * group["scalar_lr_scale"])
                    p.clamp_(-group["scalar_max"], group["scalar_max"])
                    continue
                pf = p.float()
                # Tensor ops only (no host sync per parameter on GPU).
                rms = pf.pow(2).mean().sqrt()
                r = rms.clamp(group["param_min_rms"], group["param_max_rms"])
                # Scale (log-magnitude) learning.
                h = (g * pf).sum()
                ms, vs = st["scale_avg"], st["scale_avg_sq"]
                ms.mul_(b1).add_(h, alpha=1 - b1)
                vs.mul_(b2).add_(h * h, alpha=1 - b2)
                s_step = -lr * group["scalar_lr_scale"] * ms / ((vs / bc2).sqrt() + eps)
                # Above the RMS ceiling only shrinking is allowed.
                s_step = torch.where(rms > group["param_max_rms"], s_step.clamp(max=0.0), s_step)
                delta = (m / denom).mul_(r * -lr).add_(pf * s_step)
                p.add_(delta.to(p.dtype))
        return loss


def eden_lambda(base_warmup: int = 500, lr_batches: float = 7500.0, lr_epochs: float = 3.5,
                steps_per_epoch: float = 0.0, warmup_start: float = 0.5):
    """Eden LR factor (Zipformer paper, eq. for Eden):

      factor(t, e) = ((t^2 + B^2) / B^2) ** -0.25 * ((e^2 + E^2) / E^2) ** -0.25 * w(t)
      w(t) = warmup_start + (1 - warmup_start) * t / base_warmup   for t < base_warmup, else 1

    with t the optimiser step, B = lr_batches and e the (fractional) epoch. The trainer has no
    per-step epoch in its LR scheduler, so e = t / steps_per_epoch (0 disables the epoch term).
    Returns a LambdaLR-compatible callable (factor relative to the base lr).
    """
    def f(step: int) -> float:
        t = float(step)
        x = ((t * t + lr_batches ** 2) / lr_batches ** 2) ** -0.25
        if steps_per_epoch > 0:
            e = t / steps_per_epoch
            x *= ((e * e + lr_epochs ** 2) / lr_epochs ** 2) ** -0.25
        if step < base_warmup:
            x *= warmup_start + (1.0 - warmup_start) * t / base_warmup
        return x if math.isfinite(x) else 0.0
    return f
