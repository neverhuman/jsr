"""Data-parallel training helpers (torchrun; NCCL on GPUs, gloo on CPU).

Design (H23/D2c DDP):
  * every rank holds identical weights (broadcast from rank 0 after set-up) and takes identical
    optimiser steps: gradients are averaged with an fp32 bucketed all-reduce after backward, and
    clipping happens after the sync;
  * batches are disjoint per rank (ParquetBatches assigns packed groups g -> rank g % world;
    other sources are sharded by batch index), and each loss component is rescaled by
    world * N_rank / N_total with all-reduced token counts, so the update equals one process on
    the concatenated batch;
  * scalars that drive decisions (loss for the divergence watch, stop flags, dev WER) are
    all-reduced or broadcast, so every rank takes the same branch;
  * GPU-utilisation throttling is a synchronised sleep at the step boundary (max over ranks),
    never a per-rank SIGSTOP, which would stall the collective.
"""
from __future__ import annotations

import os
import subprocess
import time

import torch
import torch.distributed as dist

BUCKET_BYTES = 32 << 20


def init_from_env(device_type: str) -> tuple[int, int, int]:
    """(rank, world, local_rank); initialises the process group when WORLD_SIZE > 1."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, 0
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    return rank, world, local


def active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def broadcast_state(module: torch.nn.Module) -> None:
    for t in module.state_dict().values():
        dist.broadcast(t, 0)


def all_reduce_grads(params: list[torch.nn.Parameter]) -> None:
    """Average gradients over ranks (fp32 buckets). Parameters without a gradient are skipped, as
    the optimiser skips them; the graph is the same on every rank, so the set is too."""
    world = dist.get_world_size()
    grads = [p.grad for p in params if p.grad is not None]
    bucket, size = [], 0
    for g in grads + [None]:
        if g is not None:
            bucket.append(g)
            size += g.numel() * 4
        if bucket and (g is None or size >= BUCKET_BYTES):
            flat = torch.cat([b.reshape(-1).float() for b in bucket])
            dist.all_reduce(flat)
            flat /= world
            off = 0
            for b in bucket:
                n = b.numel()
                b.copy_(flat[off:off + n].view_as(b).to(b.dtype))
                off += n
            bucket, size = [], 0


def all_reduce_sum(values: list[float], device) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def all_reduce_max(value: float, device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t[0])


def all_reduce_max_vec(values: list[float], device) -> list[float]:
    """Single batched element-wise MAX collective across ranks."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def broadcast_value(value: float | None, device) -> float | None:
    t = torch.tensor([float("nan") if value is None else value], dtype=torch.float64, device=device)
    dist.broadcast(t, 0)
    v = float(t[0])
    return None if v != v else v


def _host_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as fh:
            parts = list(map(int, fh.readline().split()[1:]))
        return sum(parts), parts[3] + parts[4]
    except (OSError, ValueError, IndexError):
        return None


class StepThrottle:
    """Keeps a GPU's (and optionally host CPU's) 5-minute mean utilisation near a target by
    sleeping cleanly at step boundaries (never mid-step SIGSTOP).

    Works identically for single-GPU (`world == 1`) and DDP (`world > 1`) runs. In DDP, pass
    `cluster_gpu_mean` (the hottest rank's 5-min GPU mean via `all_reduce_max_vec`) so every rank
    computes the exact same sleep without starving faster peers or letting the hottest GPU exceed
    the 5-minute ceiling. A `UTIL_TARGET` file in `live_dir` overrides the GPU target at run time.
    """

    def __init__(self, target: float, gpu_index: int | str, every: int = 25, period_s: float = 2.0,
                 window: int = 150, live_dir: str | None = None, floor: float = 0.5,
                 cpu_target: float = 0.0):
        import collections
        import threading

        self.target, self.gpu, self.every, self.floor = float(target), gpu_index, every, floor
        self.cpu_target = float(cpu_target)
        self.live = os.path.join(live_dir, "UTIL_TARGET") if live_dir else None
        self.duty, self.step_s, self.slept_s = 1.0, 0.0, 0.0
        self.samples: collections.deque = collections.deque(maxlen=window)
        self.cpu_samples: collections.deque = collections.deque(maxlen=window)
        self._period = period_s
        self._cpu_prev = _host_cpu_times()
        if self.target > 0 or self.cpu_target > 0:
            threading.Thread(target=self._sample_loop, name="gpu-util", daemon=True).start()

    def _util(self) -> float | None:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                "-i", str(self.gpu)], capture_output=True, text=True, timeout=10)
            return float(r.stdout.split()[0]) / 100.0
        except (IndexError, ValueError, OSError, subprocess.SubprocessError):
            return None

    def _sample_cpu(self) -> float | None:
        cur = _host_cpu_times()
        if cur is None or self._cpu_prev is None:
            self._cpu_prev = cur
            return None
        dt = max(1, cur[0] - self._cpu_prev[0])
        di = max(0, cur[1] - self._cpu_prev[1])
        self._cpu_prev = cur
        return max(0.0, min(1.0, 1.0 - di / dt))

    def _sample_loop(self) -> None:
        while True:
            if self.target > 0:
                u = self._util()
                if u is not None:
                    self.samples.append(u)
            if self.cpu_target > 0:
                c = self._sample_cpu()
                if c is not None:
                    self.cpu_samples.append(c)
            time.sleep(self._period)

    def util_mean(self) -> float | None:
        s = list(self.samples)
        return sum(s) / len(s) if s else None

    def cpu_mean(self) -> float | None:
        s = list(self.cpu_samples)
        return sum(s) / len(s) if s else None

    def _live_target(self) -> None:
        if self.live and os.path.exists(self.live):
            try:
                with open(self.live) as fh:
                    self.target = float(fh.read().strip())
            except (OSError, ValueError):
                pass

    def sleep_for(self, step: int, step_seconds: float,
                  cluster_gpu_mean: float | None = None,
                  cluster_cpu_mean: float | None = None) -> float:
        if self.target <= 0 and self.cpu_target <= 0:
            return 0.0
        self.step_s = 0.9 * self.step_s + 0.1 * step_seconds if self.step_s else step_seconds
        if step % self.every == 0:
            self._live_target()
            m_gpu = cluster_gpu_mean if cluster_gpu_mean is not None else self.util_mean()
            m_cpu = cluster_cpu_mean if cluster_cpu_mean is not None else self.cpu_mean()
            ready_gpu = self.target > 0 and m_gpu is not None and (
                cluster_gpu_mean is not None or len(self.samples) >= 15
            )
            ready_cpu = self.cpu_target > 0 and m_cpu is not None and (
                cluster_cpu_mean is not None or len(self.cpu_samples) >= 15
            )
            if ready_gpu or ready_cpu:
                over = (ready_gpu and m_gpu > self.target) or (ready_cpu and m_cpu > self.cpu_target)
                roomy = (not ready_gpu or m_gpu < self.target - 0.03) and (
                    not ready_cpu or m_cpu < self.cpu_target - 0.03
                )
                if over:
                    self.duty = max(self.floor, self.duty - 0.02)
                elif roomy:
                    self.duty = min(1.0, self.duty + 0.02)
        return self.step_s * (1.0 / self.duty - 1.0)

    def pause(self, seconds: float) -> None:
        if seconds > 0:
            self.slept_s += seconds
            time.sleep(seconds)

    def stats(self) -> dict:
        m = self.util_mean()
        c = self.cpu_mean()
        out = {"duty": round(self.duty, 3), "util_5min": round(m, 3) if m is not None else -1.0,
               "target": self.target, "slept_s": round(self.slept_s, 2)}
        if self.cpu_target > 0:
            out["cpu_5min"] = round(c, 3) if c is not None else -1.0
            out["cpu_target"] = self.cpu_target
        return out


def physical_gpu(index: int) -> str:
    """nvidia-smi id of CUDA device `index`: CUDA_VISIBLE_DEVICES renumbers devices (a rank
    pinned to physical GPU 1 sees it as cuda:0), while nvidia-smi uses physical ids or UUIDs."""
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return vis[index] if index < len(vis) else str(index)


def all_gather_floats(values: list[float], device) -> list[list[float]]:
    """Every rank's list of floats, gathered on every rank (diagnostics)."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [o.tolist() for o in out]


def state_checksum(module: torch.nn.Module) -> float:
    """Order-sensitive float64 checksum of all parameters and buffers (rank agreement check)."""
    total = 0.0
    for i, t in enumerate(module.state_dict().values()):
        if t.is_floating_point():
            total += float(t.double().sum()) * (1.0 + 1e-6 * i)
    return total
