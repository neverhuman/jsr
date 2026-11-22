"""Lookahead wrapper (Michael R. Zhang, James Lucas, Geoffrey Hinton and Jimmy Ba, "Lookahead
Optimizer: k steps forward, 1 step back", NeurIPS 2019). Reimplemented from the paper.

The inner ("fast") optimiser steps as usual. Every k steps the slow weights move toward the fast
ones, slow += alpha (fast - slow), and the fast weights are reset to the slow weights. param_groups
are shared with the inner optimiser, so an LR scheduler on the wrapper drives the inner lr.
"""
from __future__ import annotations

import torch

from train.errors import OptimConfigError


class Lookahead(torch.optim.Optimizer):
    def __init__(self, inner: torch.optim.Optimizer, k: int = 6, alpha: float = 0.5):
        if k < 1 or not 0.0 < alpha <= 1.0:
            raise OptimConfigError("lookahead needs k >= 1 and 0 < alpha <= 1")
        self.inner, self.k, self.alpha = inner, k, alpha
        self.defaults = inner.defaults
        self.param_groups = inner.param_groups
        self.state = inner.state
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        self._optimizer_state_dict_pre_hooks = {}
        self._optimizer_state_dict_post_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}
        self.counter = 0
        self.slow = [p.detach().clone() for g in self.param_groups for p in g["params"]]

    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.inner.step(closure)
        self.counter += 1
        if self.counter % self.k == 0:
            for p, s in zip(self._params(), self.slow, strict=True):
                s.add_(p.detach() - s, alpha=self.alpha)
                p.copy_(s)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "slow": [s.clone() for s in self.slow],
                "counter": self.counter, "k": self.k, "alpha": self.alpha}

    def load_state_dict(self, sd: dict) -> None:
        self.inner.load_state_dict(sd["inner"])
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state
        for s, v in zip(self.slow, sd["slow"], strict=True):
            s.copy_(v)
        self.counter = int(sd["counter"])
