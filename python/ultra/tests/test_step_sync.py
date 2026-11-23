"""Torch-free unit and closed-loop tests for SoloStepThrottle and coalesced step scalars."""
from __future__ import annotations

import ast
import math
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.step_sync import (  # noqa: E402
    MIN_SAMPLE_INTERVAL_S,
    SoloStepThrottle,
    StepScalars,
    coalesce_all_reduce_scalars,
    pack_step_scalars,
    unpack_step_scalars,
)


def test_import_is_torch_free():
    src = (pathlib.Path(__file__).resolve().parents[1] / "train" / "step_sync.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")
    assert MIN_SAMPLE_INTERVAL_S >= 2.0


def test_solo_throttle_zero_sleep_below_or_at_target():
    now, reads = [0.0], [0]

    def reader():
        reads[0] += 1
        return 0.85

    th = SoloStepThrottle(0.86, util_reader=reader, clock=lambda: now[0], sleep_fn=lambda _: None)
    for step in range(1, 20):
        now[0] += 0.4
        assert th.sleep_for(step, 0.4) == 0.0
    assert reads[0] <= 5 and th.stats()["duty"] == 1.0


def test_solo_throttle_proportional_sleep_and_max_clamp():
    th = SoloStepThrottle(0.85, max_sleep_s=0.10, util_reader=lambda: 0.935,
                          clock=lambda: 0.0, sleep_fn=lambda _: None)
    assert math.isclose(th.sleep_for(1, 0.5), 0.05, rel_tol=1e-9)
    clamped = SoloStepThrottle(0.85, max_sleep_s=0.04, util_reader=lambda: 0.99,
                               clock=lambda: 0.0, sleep_fn=lambda _: None)
    assert math.isclose(clamped.sleep_for(1, 1.0), 0.04, rel_tol=1e-9)


def test_solo_throttle_bounded_sampling_and_telemetry(tmp_path):
    now, reads = [0.0], [0]

    def reader():
        reads[0] += 1
        return 0.95

    th = SoloStepThrottle(0.86, sample_interval_s=0.1, live_dir=str(tmp_path),
                          util_reader=reader, clock=lambda: now[0], sleep_fn=lambda _: None)
    assert th.sample_interval_s >= 2.0
    for step in range(1, 16):
        now[0] += 0.25  # 3.75 s total -> at most 2 reads
        th.pause(th.sleep_for(step, 0.25))
    assert reads[0] == 2
    tel = th.telemetry()
    assert set(tel) >= {"rank_throttle_sleep_s", "rank_gpu_util_5min", "rank_duty", "util_target"}
    assert len(tel["rank_throttle_sleep_s"]) == 1 and tel["rank_throttle_sleep_s"][0] > 0.0
    assert len(tel["rank_gpu_util_5min"]) == 1 and 0.0 < tel["rank_gpu_util_5min"][0] <= 1.0
    assert th.telemetry()["rank_throttle_sleep_s"] == [0.0]


def test_solo_throttle_closed_loop_5min_mean_tracks_target_without_freeze():
    target, step_s = 0.86, 0.45
    now, win_active, win_total = [0.0], [0.0], [0.0]
    observed: list[float] = []

    def sim_util():
        u = 0.96 if win_total[0] <= 0.0 else win_active[0] / win_total[0]
        win_active[0] = win_total[0] = 0.0
        observed.append(u)
        return u

    def sim_sleep(s: float):
        now[0] += s
        win_total[0] += s

    th = SoloStepThrottle(target, sample_interval_s=2.0, max_sleep_s=0.5,
                          util_reader=sim_util, clock=lambda: now[0], sleep_fn=sim_sleep)
    for step in range(1, 900):  # > 400 s simulated (> full 5-min window)
        active_u = 0.96 + 0.02 * math.sin(step * 0.13)
        now[0] += step_s
        win_active[0] += active_u * step_s
        win_total[0] += step_s
        th.pause(th.sleep_for(step, step_s))

    mean_5min = th.util_mean()
    assert mean_5min is not None and target - 0.03 <= mean_5min <= target + 0.02
    assert min(observed) > 0.75  # never oscillates to 0% like SIGSTOP


def test_pack_unpack_and_coalesced_multi_rank_parity():
    vec = pack_step_scalars(128.5, 412.25, 100.0, 0.035, True)
    assert vec == (128.5, 412.25, 100.0, 0.035, 1.0)
    assert unpack_step_scalars(vec) == StepScalars(128.5, 412.25, 100.0, 0.035, True)
    assert math.isclose(unpack_step_scalars(vec).mean_loss, 4.1225, rel_tol=1e-12)
    assert unpack_step_scalars(pack_step_scalars(1.0, 2.0, 4.0, 0.0, False)).stop_flag is False
    with pytest.raises(ValueError):
        unpack_step_scalars([1.0, 2.0, 3.0])

    ranks = [(42.5, 180.0, 50.0, 0.012, False), (38.0, 145.5, 40.0, 0.048, False),
             (51.25, 220.75, 65.0, 0.025, True), (29.75, 110.25, 35.0, 0.031, False)]
    packed = [pack_step_scalars(*r) for r in ranks]
    calls = [0, 0]

    def fake_sum(local: list[float]) -> list[float]:
        calls[0] += 1
        assert len(local) == 3
        return [sum(p[i] for p in packed) for i in range(3)]

    def fake_max(local: list[float]) -> list[float]:
        calls[1] += 1
        assert len(local) == 2
        return [max(p[3 + i] for p in packed) for i in range(2)]

    out = coalesce_all_reduce_scalars(*ranks[0], reduce_sum=fake_sum, reduce_max=fake_max)
    assert calls == [1, 1]  # 2 batched collectives instead of 5
    assert math.isclose(out.audio_s, sum(r[0] for r in ranks), rel_tol=1e-12)
    assert math.isclose(out.loss_num, sum(r[1] for r in ranks), rel_tol=1e-12)
    assert math.isclose(out.loss_den, sum(r[2] for r in ranks), rel_tol=1e-12)
    assert math.isclose(out.mean_loss, sum(r[1] for r in ranks) / sum(r[2] for r in ranks), rel_tol=1e-12)
    assert math.isclose(out.throttle_nap_s, max(r[3] for r in ranks), rel_tol=1e-12)
    assert out.stop_flag is any(r[4] for r in ranks)
