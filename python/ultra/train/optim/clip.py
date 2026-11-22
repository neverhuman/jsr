"""Gradient clipping modes for the trainer (H23 gradient smoothing).

- ``fixed``: ``torch.nn.utils.clip_grad_norm_(params, clip)``, exactly the trainer's original call.
- ``adaptive``: clip the global norm to ``clip_k`` x the running median of the last
  ``clip_window`` finite pre-clip norms (fixed ``clip`` until ``clip_window // 4`` norms are seen).
  A single spike is cut to a few times the typical norm instead of the fixed 5.0.
- ``agc``: unit-wise adaptive gradient clipping (Brock et al. 2021, "High-Performance Large-Scale
  Image Recognition Without Normalization", NFNets). Each output unit's gradient (a row of a weight
  matrix; a whole 1-D tensor) is held at ``agc_lambda`` x max(||W_unit||, 1e-3). As in the paper,
  the final classifier layers (``joiner.out``, ``ctc_head``, ``simple_*``) are exempt.

Each mode returns the global pre-clip L2 norm (as ``clip_grad_norm_`` does), which the trainer logs
and checks for finiteness.
"""
from __future__ import annotations

import math
from collections import deque

import torch

from train.errors import OptimConfigError

MODES = ("fixed", "adaptive", "agc")
AGC_EXEMPT = ("joiner.out.", "ctc_head.", "simple_am.", "simple_lm.")


class GradClipper:
    def __init__(self, mode: str = "fixed", clip: float = 5.0, k: float = 2.0, window: int = 200,
                 agc_lambda: float = 0.1, agc_eps: float = 1e-3):
        if mode not in MODES:
            raise OptimConfigError(f"unknown grad_clip_mode {mode!r}; "
                                   f"expected one of {MODES}")
        if window < 4 and mode == "adaptive":
            raise OptimConfigError("clip_window must be >= 4")
        self.mode, self.clip, self.k, self.window = mode, clip, k, window
        self.agc_lambda, self.agc_eps = agc_lambda, agc_eps
        self.history: deque[float] = deque(maxlen=max(1, window))
        self.last_threshold = clip
        self.clipped = 0  # steps on which the adaptive threshold or AGC changed a gradient

    # ---- state (adaptive history survives a resume) -------------------------------------------
    def state_dict(self) -> dict:
        return {"mode": self.mode, "history": list(self.history), "clipped": self.clipped}

    def load_state_dict(self, s: dict | None) -> None:
        if not s:
            return
        self.history = deque(s.get("history", []), maxlen=max(1, self.window))
        self.clipped = int(s.get("clipped", 0))

    def threshold(self) -> float:
        if self.mode != "adaptive" or len(self.history) < max(1, self.window // 4):
            return self.clip
        h = sorted(self.history)
        n = len(h)
        med = h[n // 2] if n % 2 else 0.5 * (h[n // 2 - 1] + h[n // 2])
        return self.k * med

    def __call__(self, model: torch.nn.Module) -> float:
        if self.mode == "fixed":
            return float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip))
        if self.mode == "adaptive":
            thr = self.threshold()
            self.last_threshold = thr
            g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), thr))
            if math.isfinite(g):
                self.clipped += int(g > thr)
                self.history.append(g)
            return g
        return self._agc(model)

    @torch.no_grad()
    def _agc(self, model: torch.nn.Module) -> float:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        total = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])))
        if not math.isfinite(total):
            return total  # the trainer pauses on a non-finite norm; leave the grads untouched
        any_clipped = False
        for name, p in model.named_parameters():
            if p.grad is None or name.startswith(AGC_EXEMPT):
                continue
            w, g = p.detach().float(), p.grad.float()
            if p.ndim >= 2:
                wn = w.flatten(1).norm(dim=1).clamp(min=self.agc_eps)
                gn = g.flatten(1).norm(dim=1)
            else:
                wn = w.norm().clamp(min=self.agc_eps).reshape(1)
                gn = g.norm().reshape(1)
            lim = self.agc_lambda * wn
            scale = torch.where(gn > lim, lim / gn.clamp(min=1e-12), torch.ones_like(gn))
            if bool((scale < 1).any()):
                any_clipped = True
                shape = (-1,) + (1,) * (p.ndim - 1) if p.ndim >= 2 else ()
                p.grad.mul_(scale.reshape(shape).to(p.grad.dtype) if shape else scale[0])
        self.clipped += int(any_clipped)
        # Global fixed clip as a backstop after the unit-wise clip.
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        return total
