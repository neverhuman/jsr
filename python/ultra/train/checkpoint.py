"""Atomic checkpoints (spec v1.0-B §26.8).

A checkpoint holds model, optimizer, scheduler, every RNG state (python, numpy, torch CPU, CUDA),
the sampler position, the step, the precision phase, the config and the data/tokenizer manifests.
Writing: torch.save to `<name>.tmp`, fsync, rename to `<name>.pt`, then write `<name>.sha256` as the
completion marker, then atomically replace `LATEST`. A checkpoint without its marker, or whose
hash does not match, is refused.

CPU training with a fixed thread count resumes bit-exactly (tested). GPU resumes are recorded as
statistically reproducible only.
"""
from __future__ import annotations

import glob
import hashlib
import os
import random

import numpy as np
import torch

from train.errors import CheckpointError


def _fsync_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save(run_dir: str, step: int, payload: dict, keep: int = 3) -> str:
    os.makedirs(run_dir, exist_ok=True)
    name = f"ckpt-{step:08d}"
    path = os.path.join(run_dir, name + ".pt")
    tmp = path + ".tmp"
    payload = dict(payload, step=step, rng=rng_state())
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_write(os.path.join(run_dir, name + ".sha256"), (_sha256(path) + "\n").encode())
    _fsync_write(os.path.join(run_dir, "LATEST"), (name + "\n").encode())
    done = sorted(glob.glob(os.path.join(run_dir, "ckpt-*.pt")))
    for old in done[:-keep]:
        os.remove(old)
        marker = old[:-3] + ".sha256"
        if os.path.exists(marker):
            os.remove(marker)
    return path


def latest(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "LATEST")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return os.path.join(run_dir, fh.read().strip() + ".pt")


def load(path: str, restore_rng: bool = True) -> dict:
    marker = path[:-3] + ".sha256"
    if not os.path.exists(path) or not os.path.exists(marker):
        raise CheckpointError(f"incomplete checkpoint (no file or completion marker): {path}")
    with open(marker) as fh:
        want = fh.read().strip()
    got = _sha256(path)
    if got != want:
        raise CheckpointError(f"checksum mismatch for {path}: {got} != {want}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if restore_rng:
        set_rng_state(payload["rng"])
    return payload
