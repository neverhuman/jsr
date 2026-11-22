"""Unit tests for hardware CPU power and energy parsing & aggregation."""

import pytest

from tools.power_bench import (
    parse_turbostat_rows,
    rapl_delta_joules,
    summarize_core_power,
    summarize_pkg8_power,
)


SAMPLE_TURBOSTAT = """
Core\tCPU\tBusy%\tBzy_MHz\tCorWatt\tPkgWatt
-\t-\t45.15\t4109\t67.37\t187.55
46\t46\t99.99\t4120\t6.08\t187.48
47\t47\t1.23\t2055\t0.66
52\t52\t100.00\t4119\t5.83
"""


def test_parse_turbostat_rows_filters_target_cpu():
    rows = parse_turbostat_rows(SAMPLE_TURBOSTAT, target_cpus={46})
    per_core = [r for r in rows if not r["summary"]]
    assert len(per_core) == 1
    assert per_core[0]["CPU"] == 46
    assert per_core[0]["CorWatt"] == pytest.approx(6.08)
    assert per_core[0]["PkgWatt"] == pytest.approx(187.48)


def test_rapl_delta_handles_wraparound():
    assert rapl_delta_joules(1_000_000, 3_500_000, 10_000_000) == pytest.approx(2.5)
    assert rapl_delta_joules(9_000_000, 1_000_000, 10_000_000) == pytest.approx(2.0)


def test_summarize_core_power_energy_and_mw_per_stream():
    row = summarize_core_power(
        name="jsr-d2e",
        mode="streaming",
        audio_seconds=720.0,   # 0.2 audio-hours
        wall_seconds=28.8,     # 25.0x real-time -> 0.04 cpu-s/s
        pre_idle_watts=[0.85, 0.85],
        active_watts=[5.85, 5.85, 1.2],
        post_idle_watts=[0.85, 0.85],
        active_mhz=[4120.0, 4120.0, 2100.0],
        busy_pcts=[100.0, 99.8, 5.0],
    )
    assert row["active_core_w"] == pytest.approx(5.85)
    assert row["dynamic_core_w"] == pytest.approx(5.00)
    assert row["rtfx_1core"] == pytest.approx(25.0)
    # 5.85 W * 0.04 = 0.234 W = 234 mW/stream = 0.234 Wh/audio-h
    assert row["mw_per_live_stream"] == pytest.approx(234.0)
    assert row["dynamic_mw_per_live_stream"] == pytest.approx(200.0)
    assert row["core_wh_per_audio_h"] == pytest.approx(0.234)


def test_summarize_pkg8_power():
    row = summarize_pkg8_power(
        name="jsr-8w",
        mode="streaming",
        audio_seconds=3600.0,
        wall_seconds=18.0,   # 200x wall RTFx across 8 cores
        pre_pkg_w=100.0,
        post_pkg_w=100.0,
        run_pkg_j=160.0 * 18.0,  # +60W dynamic package delta
        n_cores=8,
    )
    assert row["wall_rtfx_8core"] == pytest.approx(200.0)
    assert row["delta_pkg_w"] == pytest.approx(60.0)
    # 60 W / 200x = 0.30 W = 300 mW/stream = 0.30 Wh/audio-h
    assert row["delta_pkg_mw_per_stream"] == pytest.approx(300.0)
    assert row["delta_pkg_wh_per_audio_h"] == pytest.approx(0.30)
