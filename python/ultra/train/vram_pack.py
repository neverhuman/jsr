"""Opt-in per-process CUDA memory-fraction guard and VRAM co-packing planner (H50).

Torch-free at module import: ``torch`` is imported lazily only inside
``apply_gpu_mem_fraction`` so light CI lanes and pure-Python schedulers can import
this module without CUDA or PyTorch installed.
"""
from __future__ import annotations

import math
import os

MAX_SAFE_FRACTION = 0.90
DEFAULT_TARGET_PCT = 0.85
TRAINER_DEFAULT_CAP = 0.88
COPACK_COMBINED_CAP = 0.88
VALID_WORKLOADS = ("trainer", "decode_or_label")
RECOMMENDED_ALLOC_CONF = "expandable_segments:True"


def _has_expandable_segments(alloc_conf: str) -> bool:
    for item in alloc_conf.split(","):
        key, sep, val = item.partition(":")
        if sep and key.strip().lower() == "expandable_segments" and val.strip().lower() == "true":
            return True
    return False


def _validate_fraction(fraction: float, upper: float = MAX_SAFE_FRACTION, name: str = "fraction") -> float:
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError(f"{name} must be a finite number in (0.0, {upper:.2f}]")
    val = float(fraction)
    if not math.isfinite(val) or val <= 0.0 or val > upper:
        raise ValueError(f"{name}={fraction!r} outside safe range (0.0, {upper:.2f}] (OOM cliff guard)")
    return val


def apply_gpu_mem_fraction(fraction: float, device_index: int | None = None) -> dict:
    """Cap PyTorch's CUDA caching allocator for this process to ``fraction`` (<= 0.90).

    Must be called before allocating tensors on the target CUDA device. Returns a
    structured telemetry dict whether or not CUDA is available on the current host.
    """
    frac = _validate_fraction(fraction, MAX_SAFE_FRACTION, "gpu_mem_fraction")
    if device_index is not None and (isinstance(device_index, bool) or int(device_index) < 0):
        raise ValueError(f"device_index must be None or a non-negative int, got {device_index!r}")
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    expandable = _has_expandable_segments(alloc_conf)
    base = {
        "applied": False,
        "fraction": round(frac, 4),
        "device_index": device_index,
        "expandable_segments": expandable,
        "alloc_conf": alloc_conf,
        "recommended_alloc_conf": RECOMMENDED_ALLOC_CONF,
    }
    try:
        import torch
    except ImportError:
        return base
    if not bool(torch.cuda.is_available()):
        return base
    resolved = int(torch.cuda.current_device()) if device_index is None else int(device_index)
    torch.cuda.set_per_process_memory_fraction(frac, resolved)
    return {**base, "applied": True, "device_index": resolved}


def recommend_pack_plan(
    free_mb: int,
    total_mb: int,
    job_vram_mb: int,
    *,
    target_pct: float = DEFAULT_TARGET_PCT,
    max_pct: float = MAX_SAFE_FRACTION,
    exclusive: bool = False,
    workload: str = "trainer",
) -> dict:
    """Plan safe job count, per-process CUDA memory cap, and batch scale factor.

    ``workload="trainer"`` (default) keeps ``max_co_packed_jobs=1`` (solo GPU, avoiding
    non-MPS CUDA time-slicing throughput loss) and recommends ``batch_scale_factor``
    targeting 75-88% VRAM with ``gpu_mem_fraction <= 0.88``.
    ``workload="decode_or_label"`` permits multi-job co-packing (combined <= 0.88) unless
    ``exclusive=True`` (timed scaling / metrology runs always stay at 1 job).
    """
    if total_mb <= 0 or job_vram_mb <= 0 or free_mb < 0 or free_mb > total_mb:
        raise ValueError(
            f"invalid VRAM inputs: free_mb={free_mb}, total_mb={total_mb}, job_vram_mb={job_vram_mb}"
        )
    if workload not in VALID_WORKLOADS:
        raise ValueError(f"workload must be one of {VALID_WORKLOADS}, got {workload!r}")
    max_cap = _validate_fraction(max_pct, MAX_SAFE_FRACTION, "max_pct")
    target_cap = _validate_fraction(target_pct, max_cap, "target_pct")

    used_mb = total_mb - free_mb
    solo_only = exclusive or workload == "trainer"
    eff_cap = min(max_cap, TRAINER_DEFAULT_CAP if solo_only else COPACK_COMBINED_CAP)
    max_total_mb = total_mb * max_cap
    headroom_max_mb = max(0.0, max_total_mb - used_mb)
    headroom_eff_mb = max(0.0, total_mb * eff_cap - used_mb)
    headroom_target_mb = max(0.0, total_mb * target_cap - used_mb)

    raw_jobs = int((headroom_max_mb if solo_only else headroom_eff_mb) // job_vram_mb)
    max_jobs = min(1, raw_jobs) if solo_only else raw_jobs
    if max_jobs <= 0:
        used_pct = round(100.0 * used_mb / total_mb, 1)
        return {
            "fits": False,
            "workload": workload,
            "exclusive": exclusive,
            "mode": "insufficient_vram",
            "max_co_packed_jobs": 0,
            "gpu_mem_fraction": 0.0,
            "vram_pct": used_pct,
            "cap_vram_pct": used_pct,
            "batch_scale_factor": 0.0,
            "co_pack_batch_scale": 0.0,
            "recommended_alloc_conf": RECOMMENDED_ALLOC_CONF,
        }

    if max_jobs == 1:
        cap_share = min(eff_cap, headroom_max_mb / total_mb)
        raw_share = max(cap_share, job_vram_mb / total_mb)
        max_share = headroom_max_mb / total_mb
        mode = "exclusive_single" if exclusive else "solo_scale"
    else:
        raw_share = max(headroom_target_mb / (max_jobs * total_mb), job_vram_mb / total_mb)
        max_share = headroom_eff_mb / (max_jobs * total_mb)
        mode = "co_pack"

    ceil_share = math.ceil(raw_share * 100.0 - 1e-9) / 100.0
    floor_max = math.floor(max_share * 100.0 + 1e-9) / 100.0
    gpu_mem_fraction = round(min(ceil_share, floor_max, MAX_SAFE_FRACTION), 4)

    batch_scale = round(max(1.0, headroom_target_mb / job_vram_mb), 2)
    co_pack_scale = round(max(1.0, (gpu_mem_fraction * total_mb) / job_vram_mb), 2)
    projected_mb = (
        min(max_total_mb, used_mb + job_vram_mb * batch_scale)
        if max_jobs == 1
        else used_mb + max_jobs * job_vram_mb
    )
    return {
        "fits": True,
        "workload": workload,
        "exclusive": exclusive,
        "mode": mode,
        "max_co_packed_jobs": max_jobs,
        "gpu_mem_fraction": gpu_mem_fraction,
        "vram_pct": round(100.0 * projected_mb / total_mb, 1),
        "cap_vram_pct": round(100.0 * (used_mb / total_mb + max_jobs * gpu_mem_fraction), 1),
        "batch_scale_factor": batch_scale,
        "co_pack_batch_scale": co_pack_scale,
        "recommended_alloc_conf": RECOMMENDED_ALLOC_CONF,
    }
