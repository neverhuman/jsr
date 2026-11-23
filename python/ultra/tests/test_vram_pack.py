"""Torch-free tests for H50 per-process CUDA memory cap and VRAM co-packing planner."""
from __future__ import annotations

import ast
import os
import sys
import types

import pytest

ULTRA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ULTRA_ROOT not in sys.path:
    sys.path.insert(0, ULTRA_ROOT)

from train.vram_pack import MAX_SAFE_FRACTION, apply_gpu_mem_fraction, recommend_pack_plan  # noqa: E402


def test_vram_pack_import_is_torch_free_and_narrow_exceptions():
    src_path = os.path.join(ULTRA_ROOT, "train", "vram_pack.py")
    tree = ast.parse(open(src_path, encoding="utf-8").read(), filename=src_path)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "torch" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "torch"
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None
            assert not (isinstance(node.type, ast.Name) and node.type.id == "Exception")


def test_trainer_default_is_solo_gpu_zero_timeslicing():
    a100 = recommend_pack_plan(free_mb=40960, total_mb=40960, job_vram_mb=16000)
    assert a100["fits"] is True
    assert a100["workload"] == "trainer"
    assert a100["max_co_packed_jobs"] == 1
    assert a100["gpu_mem_fraction"] == 0.88
    assert a100["batch_scale_factor"] >= 2.0
    assert 75.0 <= a100["vram_pct"] <= 88.0


def test_decode_or_label_a100_two_way_copacking():
    plan = recommend_pack_plan(
        free_mb=40960, total_mb=40960, job_vram_mb=16000, workload="decode_or_label"
    )
    assert plan["fits"] is True
    assert plan["mode"] == "co_pack"
    assert plan["max_co_packed_jobs"] == 2
    assert plan["gpu_mem_fraction"] == 0.43
    assert plan["gpu_mem_fraction"] <= 0.43
    assert 78.0 <= plan["vram_pct"] <= 88.0
    assert 78.0 <= plan["cap_vram_pct"] <= 88.0


def test_24gb_single_job_high_batch_scaling():
    plan = recommend_pack_plan(free_mb=24576, total_mb=24576, job_vram_mb=8000)
    assert plan["fits"] is True
    assert plan["max_co_packed_jobs"] == 1
    assert plan["gpu_mem_fraction"] == 0.88
    assert plan["batch_scale_factor"] >= 2.0
    assert 80.0 <= plan["vram_pct"] <= 85.0


def test_exclusive_always_caps_to_one_job():
    for workload in ("trainer", "decode_or_label"):
        for total_mb, job_mb in ((40960, 16000), (40960, 8000), (24576, 6000)):
            plan = recommend_pack_plan(total_mb, total_mb, job_mb, exclusive=True, workload=workload)
            assert plan["exclusive"] is True
            assert plan["max_co_packed_jobs"] == 1
            assert plan["gpu_mem_fraction"] <= MAX_SAFE_FRACTION


def test_cotenant_headroom_and_insufficient_vram():
    cotenant = recommend_pack_plan(
        free_mb=24576, total_mb=40960, job_vram_mb=16000, workload="decode_or_label"
    )
    assert cotenant["fits"] is True
    assert cotenant["max_co_packed_jobs"] == 1
    assert cotenant["cap_vram_pct"] <= 90.0

    tight = recommend_pack_plan(free_mb=8000, total_mb=24576, job_vram_mb=16000)
    assert tight["fits"] is False
    assert tight["max_co_packed_jobs"] == 0
    assert tight["gpu_mem_fraction"] == 0.0


def test_oom_safety_guard_and_invalid_workload():
    for bad in (0.95, 0.9001, 1.0, 0.0, -0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            apply_gpu_mem_fraction(bad)
    with pytest.raises(ValueError):
        recommend_pack_plan(40960, 40960, 16000, max_pct=0.95)
    with pytest.raises(ValueError):
        recommend_pack_plan(40960, 40960, 16000, workload="unknown")


def test_apply_gpu_mem_fraction_with_fake_cuda(monkeypatch):
    calls = []
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        set_per_process_memory_fraction=lambda f, idx: calls.append((f, idx)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    res = apply_gpu_mem_fraction(0.43, device_index=1)
    assert calls == [(0.43, 1)]
    assert res["applied"] is True
    assert res["fraction"] == 0.43
    assert res["device_index"] == 1
    assert res["expandable_segments"] is True

    fake_cuda.is_available = lambda: False
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    off = apply_gpu_mem_fraction(0.88)
    assert off["applied"] is False
    assert off["expandable_segments"] is False
    assert len(calls) == 1
