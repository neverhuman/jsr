"""Schedule-Free AdamW (H23 arm c).

Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled and Ashok
Cutkosky, "The Road Less Scheduled", NeurIPS 2024 (arXiv:2405.15682), Algorithm 1 (AdamW form).
Reimplemented from the paper.

Three sequences per parameter: z (the base SGD/Adam iterate), x (a weighted running average of
z, the weights to evaluate) and y = (1 - beta1) z + beta1 x (where gradients are taken). In
train mode the model's parameters hold y; ``eval()`` swaps in x and ``train()`` swaps back.
Per step t:

  v      = beta2 v + (1 - beta2) g^2
  gamma  = lr_t * sqrt(1 - beta2^t)        # lr_t: the group lr (trainer warm-up, then flat)
  z      = z - gamma * g / (sqrt(v) + eps) - gamma * wd * y
  c      = gamma^2 / sum_{i<=t} gamma_i^2
  x      = (1 - c) x + c z
  y      = (1 - beta1) z + beta1 x

There is no decay schedule: the averaging replaces it, so any checkpoint is usable. The trainer must
evaluate and export under ``eval()`` (``train.optim.eval_weights``). The model has no BatchNorm,
so no statistics need recomputing after the swap.
"""
from __future__ import annotations

import torch

from train.errors import OptimConfigError


class ScheduleFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 2.5e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, train_mode=True,
                        weight_sum=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        """Parameters := x (the averaged weights). Idempotent."""
        for group in self.param_groups:
            if group["train_mode"]:
                for p in group["params"]:
                    st = self.state[p]
                    if "x" in st:
                        p.copy_(st["x"].to(p.dtype))
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Parameters := y = (1 - beta1) z + beta1 x. Idempotent."""
        for group in self.param_groups:
            if not group["train_mode"]:
                b1 = group["betas"][0]
                for p in group["params"]:
                    st = self.state[p]
                    if "x" in st:
                        p.copy_(torch.lerp(st["z"], st["x"], b1).to(p.dtype))
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group["train_mode"]:
                raise OptimConfigError("ScheduleFreeAdamW.step() in eval mode; "
                                       "call train() first")
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            gamma = None
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["z"] = p.detach().float().clone()
                    st["x"] = p.detach().float().clone()
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                g = p.grad.float()
                y = p.detach().float()
                v = st["exp_avg_sq"]
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                gamma = lr * (1.0 - b2 ** t) ** 0.5
                z = st["z"]
                z.addcdiv_(g, v.sqrt().add_(eps), value=-gamma)
                if wd:
                    z.add_(y, alpha=-gamma * wd)
                ws = group["weight_sum"] + gamma * gamma
                c = gamma * gamma / ws if ws > 0 else 1.0
                st["x"].lerp_(z, c)
                p.copy_(torch.lerp(z, st["x"], b1).to(p.dtype))
            if gamma is not None:
                group["weight_sum"] += gamma * gamma
        return loss
