"""Torch-free unit and realistic-reader witness tests for SoloStepThrottle & step_sync."""
from __future__ import annotations

import ast
import math
import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.step_sync import (  # noqa: E402
    DEFAULT_PERIOD_S,
    SoloStepThrottle,
    StepScalars,
    coalesce_all_reduce_scalars,
    pack_step_scalars,
    unpack_step_scalars,
)


def _run_realistic_1s_reader_sim(
    *, step_s: float, target: float, active_u: float, total_s: float = 1200.0,
    max_sleep_s: float = 2.0,
) -> dict[str, float]:
    """Simulate a real training loop with an instantaneous last-1s GPU-util reader (like NVML).

    During each step the GPU is busy for ``step_s`` at ``active_u``, then idle (0.0) for ``nap``.
    The time-uniform sampler polls ``last_1s_util(t)`` at uniform wall-clock ticks ``k * period_s``
    across BOTH busy and nap intervals, while we also record what a step-boundary read would see.
    """
    th = SoloStepThrottle(target, max_sleep_s=max_sleep_s, start_sampler=False, sleep_fn=lambda _: None)
    busy_intervals: list[tuple[float, float]] = []
    naps: list[float] = []
    step_boundary_reads: list[float] = []
    t, next_tick, step = 0.0, th.period_s, 0

    def last_1s_util(now: float) -> float:
        lo, hi = max(0.0, now - 1.0), now
        if hi <= lo:
            return 0.0
        busy = 0.0
        for a, b in reversed(busy_intervals):
            if b <= lo:
                break
            if a < hi:
                busy += min(b, hi) - max(a, lo)
        return active_u * busy / (hi - lo)

    while t < total_s:
        step += 1
        a, b = t, t + step_s
        busy_intervals.append((a, b))
        while next_tick <= b:
            th.record_sample(last_1s_util(next_tick))
            next_tick += th.period_s
        t = b
        step_boundary_reads.append(last_1s_util(t))
        nap = th.sleep_for(step, step_s)
        th.pause(nap)
        naps.append(nap)
        t_end = t + nap
        while next_tick <= t_end:
            th.record_sample(last_1s_util(next_tick))
            next_tick += th.period_s
        t = t_end

    lo, hi = total_s - 300.0, total_s
    true_busy = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in busy_intervals if b > lo and a < hi)
    return {
        "true_5min": active_u * true_busy / 300.0,
        "est_5min": th.util_mean() or 0.0,
        "duty": th.duty,
        "max_nap": max(naps),
        "last_nap": naps[-1],
        "step_boundary_mean": sum(step_boundary_reads[-100:]) / 100.0,
    }


def test_import_is_torch_free():
    src = (pathlib.Path(__file__).resolve().parents[1] / "train" / "step_sync.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")
    assert DEFAULT_PERIOD_S > 1.0


def test_realistic_1s_reader_witness_no_ratchet_at_1s_steps():
    """Witness for D2c bug: last-1s reader reads 1.0 at every 1 s step boundary, yet true 5-min
    util tracks target=0.86 cleanly (never ratchets to max_sleep_s / 0.32 util)."""
    r1 = _run_realistic_1s_reader_sim(step_s=1.0, target=0.86, active_u=1.0)
    assert r1["step_boundary_mean"] > 0.99  # confirms step-boundary polling would see ~100%
    assert 0.83 <= r1["true_5min"] <= 0.88
    assert abs(r1["est_5min"] - r1["true_5min"]) < 0.01
    assert r1["max_nap"] < 0.40  # never ratchets toward max_sleep_s=2.0

    r_fast = _run_realistic_1s_reader_sim(step_s=0.45, target=0.86, active_u=0.96)
    assert 0.83 <= r_fast["true_5min"] <= 0.88

    r_below = _run_realistic_1s_reader_sim(step_s=1.0, target=0.86, active_u=0.75)
    assert r_below["max_nap"] == 0.0 and r_below["duty"] == 1.0


def test_solo_throttle_max_sleep_clamp_telemetry_and_live_thread(tmp_path):
    # Max sleep clamp & trainer.py telemetry schema
    th = SoloStepThrottle(0.85, max_sleep_s=0.04, live_dir=str(tmp_path),
                          start_sampler=False, sleep_fn=lambda _: None)
    for _ in range(30):
        th.record_sample(1.0)
    nap = th.sleep_for(1, 2.0)
    assert 0.0 < nap <= 0.04
    th.pause(nap)
    tel = th.telemetry()
    assert set(tel) >= {"rank_throttle_sleep_s", "rank_gpu_util_5min", "rank_duty", "util_target"}
    assert tel["rank_throttle_sleep_s"] == [round(nap, 2)]
    assert tel["rank_gpu_util_5min"] == [1.0]
    assert th.telemetry()["rank_throttle_sleep_s"] == [0.0]

    # Live background daemon thread samples uniformly and shuts down cleanly on close()
    busy = [True]
    with SoloStepThrottle(0.86, period_s=0.01, util_reader=lambda: 1.0 if busy[0] else 0.0,
                          start_sampler=True) as live:
        time.sleep(0.05)
        busy[0] = False
        time.sleep(0.05)
        samples = list(live.samples)
    assert len(samples) >= 4 and 1.0 in samples and 0.0 in samples
    assert live._thread is not None and not live._thread.is_alive()


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
