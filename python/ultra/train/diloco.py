"""Multi-node training coordination for heterogeneous GPU fleets (`Wave H23`).

Provides three drop-in primitives designed for ULTRA's 0.20 ms 6x A100 fabric
plus LAN RTX 4090/3090 nodes under `fleet_guard` duty-cycling:
  1. `StepBoundaryThrottle`: DDP-safe rank-0 inter-step sleep throttle driven by
     `guard_ceiling`, eliminating the `0.85^6 = 37.7%` compound-stall and NCCL
     timeout trap caused by unsynchronized per-host `SIGSTOP` pauses.
  2. `DiLoCoOuterOptimizer` + BF16 delta pack/unpack: DeepMind DiLoCo (`H` local
     steps + Outer Nesterov Momentum) supporting weighted heterogeneous workers
     and `k-of-n` quorum fault tolerance (`116 MB` BF16 delta for 58M `S` model).
  3. `greedy_model_soup`: Wortsman et al. greedy checkpoint averaging across
     parallel cooldown branches or seeds (zero inference overhead).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch


@dataclass
class ThrottleState:
    target_duty: float = 1.0
    sleep_s: float = 0.0
    updated_at: float = 0.0


class StepBoundaryThrottle:
    """DDP-safe rate limiter: sleeps *between* steps instead of sending `SIGSTOP` mid-AllReduce."""

    def __init__(self, control_path: str | None = None, max_sleep_s: float = 2.0):
        self.control_path = control_path
        self.max_sleep_s = float(max_sleep_s)
        self.last = ThrottleState()

    def read_control(self) -> ThrottleState:
        if not self.control_path or not os.path.exists(self.control_path):
            return ThrottleState()
        try:
            with open(self.control_path) as fh:
                raw = json.load(fh)
            duty = float(raw.get("target_duty", 1.0))
            duty = max(0.1, min(1.0, duty))
            explicit_sleep = float(raw.get("sleep_s", 0.0))
            self.last = ThrottleState(
                target_duty=duty,
                sleep_s=max(0.0, min(self.max_sleep_s, explicit_sleep)),
                updated_at=float(raw.get("t", time.time())),
            )
            return self.last
        except (OSError, ValueError, TypeError):
            return self.last

    def compute_sleep_s(self, step_wall_s: float) -> float:
        st = self.read_control()
        if st.sleep_s > 0.0:
            return st.sleep_s
        if st.target_duty >= 0.999 or step_wall_s <= 0.0:
            return 0.0
        needed = step_wall_s * (1.0 / st.target_duty - 1.0)
        return max(0.0, min(self.max_sleep_s, needed))

    def maybe_sleep(self, step_wall_s: float, broadcast_fn: Callable[[float], float] | None = None) -> float:
        """Compute sleep on rank 0 (optional broadcast across DDP ranks), then sleep cleanly."""
        s = self.compute_sleep_s(step_wall_s)
        if broadcast_fn is not None:
            s = float(broadcast_fn(s))
        if s > 0.0:
            time.sleep(s)
        return s


def pack_bf16_delta(
    anchor: Mapping[str, torch.Tensor],
    worker: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute compact CPU BF16 pseudo-gradient `anchor - worker` (~116 MB for 58M S model)."""
    delta: dict[str, torch.Tensor] = {}
    for k, a in anchor.items():
        w = worker[k]
        if a.is_floating_point():
            delta[k] = (a.detach().float().cpu() - w.detach().float().cpu()).to(torch.bfloat16)
        else:
            delta[k] = w.detach().cpu().clone()
    return delta


def unpack_bf16_delta(delta: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Promote packed BF16 delta tensors back to FP32 for outer accumulation."""
    return {
        k: (v.float() if v.is_floating_point() else v.clone())
        for k, v in delta.items()
    }


class DiLoCoOuterOptimizer:
    """Outer Nesterov SGD over worker pseudo-gradients (Douillard et al., DiLoCo 2023).

    Update per outer round:
      g_t = weighted_mean(anchor - worker_k)
      v_{t+1} = momentum * v_t + g_t
      theta_{t+1} = theta_t - lr * (momentum * v_{t+1} + g_t)   [if nesterov else lr * v_{t+1}]
    """

    def __init__(
        self,
        initial_state: Mapping[str, torch.Tensor],
        lr: float = 0.7,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.round = 0
        self.anchor: dict[str, torch.Tensor] = {
            k: (v.detach().float().cpu().clone() if v.is_floating_point() else v.detach().cpu().clone())
            for k, v in initial_state.items()
        }
        self.velocity: dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in self.anchor.items() if v.is_floating_point()
        }

    @torch.no_grad()
    def step_from_deltas(
        self,
        worker_deltas: list[Mapping[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate `K` worker pseudo-gradients (optionally weighted by audio-seconds)."""
        if not worker_deltas:
            raise ValueError("worker_deltas must be non-empty")
        n = len(worker_deltas)
        if weights is None:
            norm_w = [1.0 / n] * n
        else:
            if len(weights) != n or any(w <= 0.0 for w in weights):
                raise ValueError("weights must match worker_deltas length and be > 0")
            s = float(sum(weights))
            norm_w = [float(w) / s for w in weights]

        for k, p in self.anchor.items():
            if not p.is_floating_point():
                p.copy_(worker_deltas[0][k].cpu())
                continue
            g = torch.zeros_like(p)
            for wd, w in zip(worker_deltas, norm_w, strict=True):
                g.add_(wd[k].detach().float().cpu(), alpha=w)
            v = self.velocity[k]
            v.mul_(self.momentum).add_(g)
            upd = g.add(v, alpha=self.momentum) if self.nesterov else v
            p.sub_(upd, alpha=self.lr)
        self.round += 1
        return {k: v.clone() for k, v in self.anchor.items()}

    @torch.no_grad()
    def step_from_states(
        self,
        worker_states: list[Mapping[str, torch.Tensor]],
        weights: list[float] | None = None,
        bf16_compress: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper taking full worker `state_dict`s directly."""
        if bf16_compress:
            deltas = [unpack_bf16_delta(pack_bf16_delta(self.anchor, ws)) for ws in worker_states]
        else:
            deltas = [
                {
                    k: (
                        self.anchor[k] - ws[k].detach().float().cpu()
                        if self.anchor[k].is_floating_point()
                        else ws[k].detach().cpu()
                    )
                    for k in self.anchor
                }
                for ws in worker_states
            ]
        return self.step_from_deltas(deltas, weights=weights)

    def state_dict(self) -> dict:
        return {
            "round": self.round,
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "anchor": {k: v.clone() for k, v in self.anchor.items()},
            "velocity": {k: v.clone() for k, v in self.velocity.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.round = int(sd["round"])
        self.lr = float(sd["lr"])
        self.momentum = float(sd["momentum"])
        self.nesterov = bool(sd["nesterov"])
        self.anchor = {k: v.clone().cpu() for k, v in sd["anchor"].items()}
        self.velocity = {k: v.clone().cpu() for k, v in sd["velocity"].items()}


@torch.no_grad()
def greedy_model_soup(
    candidates: list[tuple[str, Mapping[str, torch.Tensor], float]],
    eval_fn: Callable[[dict[str, torch.Tensor]], float],
) -> tuple[dict[str, torch.Tensor], float, list[str]]:
    """Greedy Model Soup (Wortsman et al.): sort best-first (lower score = better WER),
    add a candidate to the running average iff it strictly improves `eval_fn`."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    ranked = sorted(candidates, key=lambda x: x[2])
    best_name, first_sd, best_score = ranked[0]
    selected: list[Mapping[str, torch.Tensor]] = [first_sd]
    names = [best_name]
    cur_avg = {k: v.detach().float().cpu().clone() if v.is_floating_point() else v.detach().cpu().clone()
               for k, v in first_sd.items()}
    for name, sd, _ in ranked[1:]:
        n_new = len(selected) + 1
        trial = {}
        for k, v in cur_avg.items():
            if v.is_floating_point():
                trial[k] = (v * len(selected) + sd[k].detach().float().cpu()) / n_new
            else:
                trial[k] = v.clone()
        score = float(eval_fn(trial))
        if score < best_score:
            best_score = score
            selected.append(sd)
            names.append(name)
            cur_avg = trial
    return cur_avg, best_score, names
