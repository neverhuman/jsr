"""Tests for the closed-loop 80% elastic CPU worker pool (`ops/cpu_pool.py`)."""
from __future__ import annotations

import pytest

from train.ops.cpu_pool import PoolPolicy, decide_target_active, parse_cores


def test_parse_cores_handles_ranges_and_deduplicates():
    assert parse_cores("0-3,8,2-4") == [0, 1, 2, 3, 4, 8]
    with pytest.raises(ValueError):
        parse_cores("8-4")


def test_closed_loop_controller_holds_80_pct_band_and_ram_guard():
    p = PoolPolicy(cpu_low=0.75, cpu_high=0.83, min_free_ram_gb=12.0, max_workers=16)
    # Idle host ramps up cleanly by +2 when far below target
    assert decide_target_active(4, 16, cpu=0.30, free_ram_gb=64.0, p=p) == 6
    # Slightly below band ramps up by +1
    assert decide_target_active(10, 16, cpu=0.70, free_ram_gb=64.0, p=p) == 11
    # Inside [0.75, 0.83] holds steady
    assert decide_target_active(12, 16, cpu=0.80, free_ram_gb=64.0, p=p) == 12
    # Slightly above band backs off by -1
    assert decide_target_active(12, 16, cpu=0.85, free_ram_gb=64.0, p=p) == 11
    # Co-tenant spike > 0.88 backs off faster by -2
    assert decide_target_active(12, 16, cpu=0.91, free_ram_gb=64.0, p=p) == 10
    # Low RAM forces back-off even when CPU is low
    assert decide_target_active(8, 16, cpu=0.40, free_ram_gb=8.0, p=p) == 6
    # Never exceeds available alive slots or drops below 0
    assert decide_target_active(0, 16, cpu=0.95, free_ram_gb=2.0, p=p) == 0
    assert decide_target_active(10, 8, cpu=0.50, free_ram_gb=64.0, p=p) == 8
