"""Opt-in CUDA witness for the sealed k2/DDP restart (C2).

Lives outside `python/ultra/tests` so the default suite (`just pytest`, CI pytest-light) never collects it, and it
carries no skip: run it only on reserved CUDA devices, where it fails loudly if CUDA or k2 is missing.

    just cuda-witness      # = cd python/ultra && python -m pytest -q tests_cuda
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from test_train_sealed_fork import _restart_worker, sources  # noqa: E402,F401  (pytest fixture)
from train import checkpoint  # noqa: E402


def test_pruned_cuda_active_helper_resume(sources):  # noqa: F811
    import k2  # noqa: F401  (the allocated proof must exercise real k2)

    world = min(2, torch.cuda.device_count())
    assert world, "CUDA restart witness requires an allocated CUDA device"
    if world == 1:
        _restart_worker(0, sources, True, 1)
    else:
        mp.spawn(_restart_worker, args=(sources, True, world), nprocs=world, join=True)
    payload = checkpoint.load(checkpoint.latest(str(sources / "interrupted")), restore_rng=False)
    assert len(payload["objective_state"]["ranks"]) == world
    assert all("device_rng" in state["rng"] for state in payload["objective_state"]["ranks"])
