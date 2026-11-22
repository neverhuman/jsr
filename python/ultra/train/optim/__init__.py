"""Optimiser, LR-schedule and gradient-clipping choices for train.trainer (H23 bake-off).

``build_optimizer(model, cfg, lr_lambda)`` returns ``(optimizer, scheduler)``:

- ``adamw`` (default): the trainer's original AdamW (decay on >= 2-D weights) + LambdaLR(warm-up,
  cosine). It is built exactly as before, so default runs are bit-for-bit unchanged (test).
- ``scaled_adam``: ScaledAdam + Eden (Zipformer, icefall; train/optim/scaled_adam.py).
- ``schedulefree_adamw``: Schedule-Free AdamW (Defazio et al. 2024) with a linear warm-up then
  a flat lr; evaluate under ``eval_weights``.
- ``muon``: Muon on the encoder blocks' 2-D matrices + AdamW on the rest, same lr schedule.

``lookahead_k > 0`` wraps any of them except Schedule-Free in Lookahead(k, alpha).
"""
from __future__ import annotations

import contextlib

import torch

from train.errors import OptimConfigError
from train.optim.clip import GradClipper
from train.optim.lookahead import Lookahead
from train.optim.muon import Muon, is_muon_param
from train.optim.scaled_adam import ScaledAdam, eden_lambda
from train.optim.schedulefree import ScheduleFreeAdamW

OPTIMIZERS = ("adamw", "scaled_adam", "schedulefree_adamw", "muon")

__all__ = ["OPTIMIZERS", "GradClipper", "Lookahead", "Muon", "ScaledAdam", "ScheduleFreeAdamW",
           "build_optimizer", "eval_weights", "make_clipper"]


def _decay_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model: torch.nn.Module, cfg, lr_lambda):
    name = getattr(cfg, "optimizer", "adamw")
    if name == "adamw":
        opt = torch.optim.AdamW(_decay_groups(model, cfg.weight_decay), lr=cfg.lr,
                                betas=cfg.betas)
        lam = lr_lambda(cfg)
    elif name == "scaled_adam":
        opt = ScaledAdam(model.parameters(), lr=cfg.lr, betas=cfg.betas)
        lam = eden_lambda(cfg.warmup, cfg.eden_lr_batches, cfg.eden_lr_epochs,
                          cfg.eden_steps_per_epoch)
    elif name == "schedulefree_adamw":
        opt = ScheduleFreeAdamW(_decay_groups(model, cfg.weight_decay), lr=cfg.lr,
                                betas=(cfg.betas[0], cfg.sf_beta2))
        w = max(1, cfg.warmup)
        lam = lambda step: min(1.0, (step + 1) / w)  # noqa: E731
    elif name == "muon":
        named = list(model.named_parameters())
        mu = [p for n, p in named if is_muon_param(n, p)]
        decay = [p for n, p in named if not is_muon_param(n, p) and p.ndim >= 2]
        no_decay = [p for n, p in named if p.ndim < 2]
        opt = Muon([{"params": mu, "use_muon": True, "weight_decay": cfg.weight_decay},
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0}],
                   lr=cfg.lr, momentum=cfg.muon_momentum, betas=cfg.betas)
        lam = lr_lambda(cfg)
    else:
        raise OptimConfigError(f"unknown optimizer {name!r}; expected one of {OPTIMIZERS}")
    if getattr(cfg, "lookahead_k", 0):
        if name == "schedulefree_adamw":
            raise OptimConfigError("lookahead is not combined with schedulefree_adamw "
                                   "(both average)")
        opt = Lookahead(opt, cfg.lookahead_k, cfg.lookahead_alpha)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)


def make_clipper(cfg) -> GradClipper:
    return GradClipper(getattr(cfg, "grad_clip_mode", "fixed"), cfg.clip,
                       getattr(cfg, "clip_k", 2.0), getattr(cfg, "clip_window", 200),
                       getattr(cfg, "agc_lambda", 0.1))


@contextlib.contextmanager
def eval_weights(opt: torch.optim.Optimizer):
    """Within the block the model holds the weights to evaluate/export (Schedule-Free's x);
    a no-op for every other optimiser."""
    sf = opt.inner if isinstance(opt, Lookahead) else opt
    if not isinstance(sf, ScheduleFreeAdamW):
        yield
        return
    sf.eval()
    try:
        yield
    finally:
        sf.train()
