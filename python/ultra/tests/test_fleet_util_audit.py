"""Torch-free unit and regression tests for tools.fleet_util_audit."""

from __future__ import annotations

import json
from pathlib import Path

from tools.fleet_util_audit import (
    audit_fleet,
    audit_host,
    classify_gpu,
    format_report,
    parse_linux_probe,
    parse_mac_probe,
    read_xbabe0_snapshot,
)


def test_codex_fixture_high_vram_low_util_no_cpu_pct() -> None:
    """Exact Codex 07:05:45Z / v3-reviewer 09:17:02Z regression fixture."""
    uid = "GPU-11111111-2222-3333-4444-555555555555"
    probe = (
        f"GPU 0, {uid}, NVIDIA GeForce RTX 3090, 27, 22942, 24576\n"
        f"APP 9001, 22942, {uid}\n"
        "PS 9001 /home/ubuntu/jsr/.venv/bin/python -m train.trainer --run-dir /srv/jsr/runs/R1\n"
        "LOAD 100.00 NCPU 128\n"
    )
    parsed = parse_linux_probe("xbabe2", probe)
    assert parsed["cpu_pct"] is None
    assert parsed["gpus"][0]["uuid"] == uid

    host_res = audit_host(parsed)
    gpu = host_res["gpus"][0]
    assert gpu["uuid"] == uid
    assert gpu["owner_class"] == "jsr_active"
    assert gpu["vram_band"] == "over_cap"
    assert gpu["vram_pct"] == 93.4
    assert host_res["idle_jsr_gpus"] == []
    assert host_res["cpu_pct"] is None
    assert host_res["cpu_band"] == "unknown"
    assert host_res["idle_cpu_cores_to_85pct"] is None
    assert host_res["load1"] == 100.0

    fleet = audit_fleet([parsed])
    expected_id = f"xbabe2:0:{uid}"
    assert expected_id in fleet["over_cap_gpus"]
    assert fleet["idle_jsr_gpus"] == []
    assert any("OVER_CAP_VRAM" in r and uid in r for r in fleet["recommendations"])
    assert not any("IDLE_JSR_GPU" in r for r in fleet["recommendations"])


def test_vram_band_80_90_and_cotenant_85_ceiling() -> None:
    g76 = classify_gpu(0, "NVIDIA A100", 82.0, 31130.0, 40960.0, jsr_pids=[1], jsr_vram_mb=31130.0)
    g79 = classify_gpu(0, "NVIDIA A100", 82.0, 32358.0, 40960.0, jsr_pids=[1], jsr_vram_mb=32358.0)
    g80 = classify_gpu(0, "NVIDIA A100", 82.0, 32768.0, 40960.0, jsr_pids=[1], jsr_vram_mb=32768.0)
    g87_jsr = classify_gpu(0, "NVIDIA A100", 82.0, 35635.0, 40960.0, jsr_pids=[1], jsr_vram_mb=35635.0)
    g87_co = classify_gpu(
        0, "NVIDIA A10", 50.0, 20034.0, 23028.0, other_project_vram_mb=10000.0,
        jsr_vram_mb=10034.0, jsr_pids=[10], other_pids=[20], uuid="GPU-CO87",
    )
    assert g76["vram_band"] == "under_target" and g79["vram_band"] == "under_target"
    assert g80["vram_band"] == "target_80_90" and g87_jsr["vram_band"] == "target_80_90"
    assert g87_jsr["vram_ceiling_pct"] == 90.0
    assert g87_co["vram_ceiling_pct"] == 85.0 and g87_co["vram_band"] == "over_cap" and g87_co["co_tenant"] is True

    fleet = audit_fleet([{"host": "xbabe6", "ncpu": 30, "load1": 20.0, "cpu_pct": 80.0, "gpus": [g87_co]}])
    assert fleet["over_cap_gpus"] == ["xbabe6:0:GPU-CO87"]
    assert any("OVER_CAP_VRAM xbabe6:0:GPU-CO87" in r for r in fleet["recommendations"])


def test_owner_evidence_lease_and_bounded_cap_window() -> None:
    leased = classify_gpu(0, "A100", 0.0, 0.0, 40960.0, uuid="GPU-L0", lease_active=True)
    assert leased["owner_class"] == "jsr_active"
    idle = classify_gpu(1, "A100", 0.0, 4.0, 40960.0, uuid="GPU-L1", jsr_pids=[], other_pids=[], lease_active=False)
    assert idle["owner_class"] == "jsr_idle"

    inst_spike = classify_gpu(0, "RTX 4090", 97.0, 20000.0, 24564.0, uuid="GPU-INST", jsr_pids=[7], jsr_vram_mb=20000.0)
    assert inst_spike["cap_window_valid"] is False
    f_inst = audit_fleet([{"host": "xbabe1", "ncpu": 102, "load1": 10.0, "cpu_pct": 20.0, "gpus": [inst_spike]}])
    assert f_inst["over_cap_gpus"] == []

    win_cap = classify_gpu(
        0, "RTX 4090", 95.0, 20000.0, 24564.0, uuid="GPU-WIN", jsr_pids=[7], jsr_vram_mb=20000.0,
        util_5min_mean=93.5,
    )
    assert win_cap["cap_window_valid"] is True and win_cap["band"] == "over_cap"
    f_win = audit_fleet([{"host": "xbabe1", "ncpu": 102, "load1": 10.0, "cpu_pct": 20.0, "gpus": [win_cap]}])
    assert f_win["over_cap_gpus"] == ["xbabe1:0:GPU-WIN"]


def test_dual_gpu_imbalance_preserves_uuid() -> None:
    rec = {
        "host": "xbabe5", "ncpu": 60, "load1": 5.5, "cpu_pct": 9.2,
        "gpus": [
            {"index": 0, "uuid": "GPU-X5-0", "name": "A100", "util_pct": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 40960.0, "jsr_pids": []},
            {"index": 1, "uuid": "GPU-X5-1", "name": "A100", "util_pct": 84.0, "vram_used_mb": 15865.0, "vram_total_mb": 40960.0, "jsr_pids": [55], "jsr_vram_mb": 15865.0},
        ],
    }
    res = audit_host(rec)
    assert res["dual_gpu_imbalance"] is True
    assert res["idle_jsr_gpus"] == ["xbabe5:0:GPU-X5-0"]
    assert res["status"] == "imbalanced"
    fleet = audit_fleet([rec])
    assert fleet["dual_gpu_imbalanced_hosts"] == ["xbabe5"]
    assert fleet["under_vram_jsr_gpus"] == ["xbabe5:0:GPU-X5-0", "xbabe5:1:GPU-X5-1"]


def test_busy_cpu_not_load_and_xbabe0_exempt() -> None:
    h_busy = audit_host({"host": "xbabe3", "ncpu": 128, "load1": 115.0, "cpu_pct": 12.0, "gpus": []})
    assert h_busy["cpu_pct"] == 12.0 and h_busy["load1"] == 115.0
    assert h_busy["idle_cpu_cores_to_85pct"] == 93

    h_x5 = audit_host({"host": "xbabe5", "ncpu": 60, "load1": 5.5, "cpu_pct": 9.2, "gpus": []})
    h_x1 = audit_host({"host": "xbabe1", "ncpu": 102, "load1": 20.4, "cpu_pct": 20.0, "gpus": []})
    h_at80 = audit_host({"host": "xbabe3", "ncpu": 128, "load1": 102.0, "cpu_pct": 80.0, "gpus": []})
    assert 44 <= h_x5["idle_cpu_cores_to_85pct"] <= 46
    assert 65 <= h_x1["idle_cpu_cores_to_85pct"] <= 67
    assert h_at80["idle_cpu_cores_to_85pct"] == 0

    locked = audit_host({"host": "xbabe0", "ncpu": 128, "load1": 1.0, "cpu_pct": 1.0, "gpus": [], "timing_lock": True})
    assert locked["timing_quiet_exempt"] is True and locked["idle_cpu_cores_to_85pct"] == 0
    assert audit_fleet([{"host": "xbabe0", "ncpu": 128, "load1": 1.0, "cpu_pct": 1.0, "gpus": [], "timing_lock": True}])["recommendations"] == []


def test_per_pid_cotenant_mac_shared_and_snapshot(tmp_path: Path) -> None:
    linux_text = (
        "CPU_BUSY 425\nWINDOW_S 30\n"
        "USAMP GPU-AAA, 10\nUSAMP GPU-AAA, 12\n"
        "GPU 0, GPU-AAA, NVIDIA A10, 0, 16501, 23028\n"
        "APP 1234, 16492, GPU-AAA\n"
        "PS 1234 python3 -m cotenant_worker --run-dir /srv/other-project/runs/r1\n"
        "LOAD 33.94 NCPU 30\n"
    )
    parsed = parse_linux_probe("xbabe9", linux_text)
    assert parsed["cpu_pct"] == 42.5
    audited = audit_host(parsed)
    assert audited["other_project_gpus"] == ["xbabe9:0:GPU-AAA"]
    assert audited["gpus"][0]["cap_window_valid"] is True

    mac = parse_mac_probe("xbabe4", 'CPU_BUSY 185\nNCPU 32\nLOAD { 7.19 7.20 6.65 }\nGPU Device Utilization %"=3\n')
    mac_audit = audit_host(mac)
    assert mac_audit["cpu_pct"] == 18.5
    assert mac_audit["gpus"][0]["memory_kind"] == "shared"
    assert mac_audit["gpus"][0]["vram_band"] == "shared"
    assert mac_audit["gpus"][0]["vram_pct"] is None
    assert "shared" in format_report(audit_fleet([mac]))

    snap = tmp_path / "dash_snapshot.json"
    snap.write_text(json.dumps({"nodes": {"xbabe0": {"up": True, "ncpu": 128, "load": 1.0, "cpu_pct": 1.2, "timing_lock": True, "gpus": []}}}), encoding="utf-8")
    x0 = read_xbabe0_snapshot(snapshot_path=snap)
    assert x0["host"] == "xbabe0" and x0["timing_lock"] is True and x0["cpu_pct"] == 1.2
