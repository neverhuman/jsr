"""Smooth world-1 step-boundary GPU throttle and coalesced per-step DDP scalar sync.
Torch-free at module import (lazy torch import inside live DDP collective helper).
"""
from __future__ import annotations

import collections
import math
import os
import subprocess
import time
from typing import Callable, NamedTuple, Sequence

MIN_SAMPLE_INTERVAL_S, SUM_SLICE, MAX_SLICE = 2.0, slice(0, 3), slice(3, 5)


class StepScalars(NamedTuple):
    """Per-step scalars reduced across DDP ranks (SUM: first 3, MAX: last 2)."""
    audio_s: float
    loss_num: float
    loss_den: float
    throttle_nap_s: float
    stop_flag: bool

    @property
    def mean_loss(self) -> float:
        return self.loss_num / self.loss_den if self.loss_den > 0.0 else float("nan")


def _physical_gpu(index: int | str) -> str:
    if isinstance(index, str):
        return index
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return vis[index] if 0 <= index < len(vis) else str(index)


def read_nvidia_smi_util(gpu_index: int | str = 0) -> float | None:
    """Return current GPU utilisation in [0.0, 1.0] via nvidia-smi, or None on failure."""
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
           "-i", _physical_gpu(gpu_index)]
    try:
        val = float(subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.split()[0]) / 100.0
        return max(0.0, min(1.0, val)) if math.isfinite(val) else None
    except (IndexError, ValueError, OSError, subprocess.SubprocessError):
        return None


class SoloStepThrottle:
    """Smooth step-boundary GPU-util controller for world == 1 trainers and DiLoCo islands."""

    def __init__(self, target: float, gpu_index: int | str = 0, *, sample_interval_s: float = 2.0,
                 max_sleep_s: float = 2.0, ema_alpha: float = 0.4, window_s: float = 300.0,
                 live_dir: str | None = None, util_reader: Callable[[], float | None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        self.target, self.gpu = float(target), _physical_gpu(gpu_index)
        self.sample_interval_s = max(MIN_SAMPLE_INTERVAL_S, float(sample_interval_s))
        self.max_sleep_s = max(0.0, float(max_sleep_s))
        self.ema_alpha = max(0.05, min(1.0, float(ema_alpha)))
        self.live = os.path.join(live_dir, "UTIL_TARGET") if live_dir else None
        self._util_reader = util_reader or (lambda: read_nvidia_smi_util(self.gpu))
        self._clock, self._sleep = clock, sleep_fn
        self.samples: collections.deque[float] = collections.deque(maxlen=max(1, int(round(window_s / self.sample_interval_s))))
        self.duty, self.step_s, self.slept_s, self._slept_mark = 1.0, 0.0, 0.0, 0.0
        self._active_util = self._last_sample_t = None
        self._win_compute_s, self._win_slept_s = 0.0, 0.0

    def _live_target(self) -> None:
        if self.live and os.path.exists(self.live):
            try:
                with open(self.live) as fh:
                    self.target = float(fh.read().strip())
            except (OSError, ValueError):
                pass

    def util_mean(self, last: int | None = None) -> float | None:
        vals = list(self.samples)[-last:] if last else list(self.samples)
        return sum(vals) / len(vals) if vals else None

    def _sample(self, now: float) -> None:
        self._live_target()
        self._last_sample_t = now
        u = self._util_reader()
        if u is None or not math.isfinite(u):
            return
        u = max(0.0, min(1.0, float(u)))
        self.samples.append(u)
        total = self._win_compute_s + self._win_slept_s
        duty = self._win_compute_s / total if total > 0.0 else 1.0
        self._win_compute_s, self._win_slept_s = 0.0, 0.0
        active_u = u / max(duty, 0.1)
        a = self.ema_alpha
        self._active_util = active_u if self._active_util is None else (1.0 - a) * self._active_util + a * active_u

    def sleep_for(self, step: int, step_seconds: float) -> float:
        del step  # signature parity with train.dist.StepThrottle.sleep_for
        if self.target <= 0.0 or step_seconds <= 0.0:
            self.duty = 1.0
            return 0.0
        self.step_s = 0.9 * self.step_s + 0.1 * step_seconds if self.step_s else step_seconds
        self._win_compute_s += step_seconds
        now = self._clock()
        if self._last_sample_t is None or (now - self._last_sample_t) >= self.sample_interval_s:
            self._sample(now)
        if self.target <= 0.0 or self._active_util is None or self._active_util <= self.target:
            self.duty = 1.0
            return 0.0
        nap = min(self.max_sleep_s, max(0.0, self.step_s * (self._active_util - self.target) / self.target))
        self.duty = self.step_s / (self.step_s + nap) if (self.step_s + nap) > 0.0 else 1.0
        return nap

    def pause(self, seconds: float) -> None:
        if seconds > 0.0:
            self.slept_s, self._win_slept_s = self.slept_s + seconds, self._win_slept_s + seconds
            self._sleep(seconds)

    def stats(self) -> dict:
        m = self.util_mean()
        return {"duty": round(self.duty, 3), "util_5min": round(m, 3) if m is not None else -1.0,
                "target": self.target, "slept_s": round(self.slept_s, 2)}

    def telemetry(self, *, reset_interval: bool = True) -> dict:
        """Emit trainer.py-compatible interval fields for world == 1 logging."""
        ts, slept = self.stats(), self.slept_s - self._slept_mark
        self._slept_mark = self.slept_s if reset_interval else self._slept_mark
        return {"rank_throttle_sleep_s": [round(slept, 2)], "rank_duty": [ts["duty"]],
                "rank_gpu_util_5min": [ts["util_5min"]], "util_target": ts["target"]}


def pack_step_scalars(audio_s: float, loss_num: float, loss_den: float,
                      throttle_nap_s: float, stop_flag: bool | float) -> tuple[float, ...]:
    """Pack per-step scalars into one 5-element vector: [0:3] SUM, [3:5] MAX."""
    return (float(audio_s), float(loss_num), float(loss_den),
            float(throttle_nap_s), 1.0 if float(stop_flag) > 0.0 else 0.0)


def unpack_step_scalars(vec: Sequence[float]) -> StepScalars:
    """Unpack a 5-element scalar vector into a typed ``StepScalars`` tuple."""
    if len(vec) != 5:
        raise ValueError(f"expected 5 packed step scalars, got {len(vec)}")
    return StepScalars(float(vec[0]), float(vec[1]), float(vec[2]),
                       float(vec[3]), bool(float(vec[4]) > 0.0))


def coalesce_all_reduce_scalars(
    audio_s: float, loss_num: float, loss_den: float, throttle_nap_s: float, stop_flag: bool | float,
    *, device=None, reduce_sum: Callable[[list[float]], Sequence[float]] | None = None,
    reduce_max: Callable[[list[float]], Sequence[float]] | None = None,
) -> StepScalars:
    """Coalesce 5 per-step scalar reductions into 2 batched FP64 collectives (1 SUM + 1 MAX)."""
    vec = pack_step_scalars(audio_s, loss_num, loss_den, throttle_nap_s, stop_flag)
    if (reduce_sum is None) != (reduce_max is None):
        raise ValueError("reduce_sum and reduce_max must both be set or both omitted")
    if reduce_sum is not None and reduce_max is not None:
        return unpack_step_scalars(list(reduce_sum(list(vec[SUM_SLICE]))) + list(reduce_max(list(vec[MAX_SLICE]))))
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return unpack_step_scalars(vec)
    t_sum = torch.tensor(vec[SUM_SLICE], dtype=torch.float64, device=device)
    t_max = torch.tensor(vec[MAX_SLICE], dtype=torch.float64, device=device)
    for work in (dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, async_op=True),
                 dist.all_reduce(t_max, op=dist.ReduceOp.MAX, async_op=True)):
        if work is not None:
            work.wait()
    return unpack_step_scalars(t_sum.tolist() + t_max.tolist())
