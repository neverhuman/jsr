"""Smooth world-1 step-boundary GPU throttle and coalesced per-step DDP scalar sync.
Torch-free at module import (lazy torch import inside live DDP collective helper).
"""
from __future__ import annotations

import collections
import math
import os
import subprocess
import threading
import time
from typing import Callable, NamedTuple, Sequence

DEFAULT_PERIOD_S = 1.13
MIN_PERIOD_S = 0.01
SUM_SLICE, MAX_SLICE = slice(0, 3), slice(3, 5)


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
    """Smooth step-boundary GPU-util controller for world == 1 trainers and DiLoCo islands.

    Uses a time-uniform background sampler (long-lived ``nvidia-smi -lms`` stream in production,
    matching ``train.dist.StepThrottle``) decoupled from step boundaries so rolling means cover
    active compute and sleeps alike without step-boundary bias or duty ratcheting.
    """

    def __init__(
        self,
        target: float,
        gpu_index: int | str = 0,
        *,
        period_s: float = DEFAULT_PERIOD_S,
        window_s: float = 300.0,
        control_window_s: float = 30.0,
        update_interval_s: float = 20.0,
        floor: float = 0.5,
        max_sleep_s: float = 2.0,
        live_dir: str | None = None,
        util_reader: Callable[[], float | None] | None = None,
        start_sampler: bool | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.target, self.gpu = float(target), _physical_gpu(gpu_index)
        self.period_s = max(MIN_PERIOD_S, float(period_s))
        self.floor = max(0.05, min(1.0, float(floor)))
        self.max_sleep_s = max(0.0, float(max_sleep_s))
        self.live = os.path.join(live_dir, "UTIL_TARGET") if live_dir else None
        self._util_reader = util_reader
        self._sleep = sleep_fn
        self._ctrl_len = max(1, int(round(control_window_s / self.period_s)))
        self._upd_every = max(1, int(round(update_interval_s / self.period_s)))
        self._min_samples = max(1, int(round(min(20.0, control_window_s) / self.period_s)))
        maxlen = max(self._ctrl_len, int(round(window_s / self.period_s)))
        self.samples: collections.deque[float] = collections.deque(maxlen=maxlen)
        self.duty, self.step_s, self.slept_s, self._slept_mark = 1.0, 0.0, 0.0, 0.0
        self._sample_count, self._last_ctrl_sample = 0, 0
        self._stop = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        auto_start = (self.target > 0.0) if start_sampler is None else bool(start_sampler)
        if auto_start and self.target > 0.0:
            self._thread = threading.Thread(target=self._sample_loop, name="solo-gpu-util", daemon=True)
            self._thread.start()

    def __enter__(self) -> SoloStepThrottle:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Stop the background time-uniform sampler thread and any nvidia-smi child cleanly."""
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def record_sample(self, util: float | None) -> bool:
        """Ingest one time-uniform GPU utilisation sample in [0.0, 1.0]."""
        if util is None or not math.isfinite(util):
            return False
        self.samples.append(max(0.0, min(1.0, float(util))))
        self._sample_count += 1
        return True

    def _sample_loop(self) -> None:
        if self._util_reader is not None:
            while not self._stop.is_set():
                self.record_sample(self._util_reader())
                self._stop.wait(self.period_s)
            return
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
               "-i", self.gpu, "-lms", str(max(100, int(round(self.period_s * 1000))))]
        while not self._stop.is_set():
            try:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True) as proc:
                    self._proc = proc
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        if self._stop.is_set():
                            break
                        try:
                            self.record_sample(float(line.split()[0]) / 100.0)
                        except (IndexError, ValueError):
                            continue
            except OSError:
                pass
            finally:
                self._proc = None
            if not self._stop.is_set():
                self._stop.wait(5.0)

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

    def sleep_for(self, step: int, step_seconds: float) -> float:
        del step  # signature parity with train.dist.StepThrottle.sleep_for
        if self.target <= 0.0 or step_seconds <= 0.0:
            self.duty = 1.0
            return 0.0
        self.step_s = 0.9 * self.step_s + 0.1 * step_seconds if self.step_s else step_seconds
        if (self._sample_count - self._last_ctrl_sample >= self._upd_every
                and len(self.samples) >= self._min_samples):
            self._live_target()
            self._last_ctrl_sample = self._sample_count
            if self.target <= 0.0:
                self.duty = 1.0
                return 0.0
            m = self.util_mean(self._ctrl_len)
            if m is not None:
                if m > self.target:
                    delta = min(0.03, max(0.005, 0.5 * (m - self.target)))
                    self.duty = max(self.floor, self.duty - delta)
                elif m < self.target - 0.01:
                    delta = min(0.03, max(0.005, 0.5 * (self.target - 0.01 - m)))
                    self.duty = min(1.0, self.duty + delta)
        return min(self.max_sleep_s, max(0.0, self.step_s * (1.0 / self.duty - 1.0)))

    def pause(self, seconds: float) -> None:
        if seconds > 0.0:
            self.slept_s += seconds
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
