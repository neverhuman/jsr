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


def broadcast_value(value: float | None, device) -> float | None:
    t = torch.tensor([float("nan") if value is None else value], dtype=torch.float64, device=device)
    dist.broadcast(t, 0)
    v = float(t[0])
    return None if v != v else v


class StepThrottle:
    """Keeps a GPU's utilisation near a target by sleeping at step boundaries. Each rank samples
    its own GPU (nvidia-smi) every `every` steps; the sleep is the max over ranks, so all ranks
    pause together. target <= 0 disables it."""

    def __init__(self, target: float, gpu_index: int, every: int = 50):
        self.target, self.gpu, self.every = target, gpu_index, every
        self.duty, self.samples, self.step_s = 1.0, [], 0.0

    def _util(self) -> float | None:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                            "-i", str(self.gpu)], capture_output=True, text=True, timeout=10)
        try:
            return float(r.stdout.split()[0]) / 100.0
        except (IndexError, ValueError):
            return None

    def sleep_for(self, step: int, step_seconds: float) -> float:
        if self.target <= 0:
            return 0.0
        self.step_s = 0.9 * self.step_s + 0.1 * step_seconds if self.step_s else step_seconds
        if step % self.every == 0:
            u = self._util()
            if u is not None:
                self.samples = (self.samples + [u])[-6:]
                m = sum(self.samples) / len(self.samples)
                if m > self.target:
                    self.duty = max(0.5, self.duty - 0.02)
                elif m < self.target - 0.05:
                    self.duty = min(1.0, self.duty + 0.01)
        return self.step_s * (1.0 / self.duty - 1.0)

    @staticmethod
    def pause(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


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
