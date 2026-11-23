"""Unit tests for hardened H50 island-side DiLoCo fleet launchers (D1 & D2)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ast
import json
from pathlib import Path
import shlex
import subprocess
from unittest import mock

import pytest

from train.ops import launch_v3_diloco as launcher

REPO_ROOT = Path(__file__).resolve().parents[3]
PR73_DILOCO_SET_KEYS = frozenset({"diloco_hub", "diloco_worker", "diloco_every", "diloco_dtype"})
PR73_HUB_CLI_FLAGS = frozenset({
    "--cohort", "--init-anchor", "--work-dir", "--host", "--port", "--roster", "--allow-async-experimental",
    "--auth-token-file", "--quorum-k", "--quorum-timeout-s", "--outer-lr", "--outer-momentum",
    "--staleness-alpha", "--min-cosine", "--hub-ema-decay", "--keep-last-k", "--keep-every-m", "--ema", "--resume",
})


def _main_train_config_fields() -> set[str]:
    tree = ast.parse((REPO_ROOT / "python/ultra/train/trainer.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig":
            return {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    raise AssertionError("TrainConfig class not found in trainer.py")


def _extract_set_keys(cmd: str) -> set[str]:
    after = cmd.split("--set ", 1)[1].split(" >> ", 1)[0]
    return {tok.split("=", 1)[0] for tok in shlex.split(after) if "=" in tok}


def _extract_hub_flags(cmd: str) -> set[str]:
    seg = cmd.split("-m train.diloco_hub ", 1)[1].split(" >> ", 1)[0]
    return {tok for tok in shlex.split(seg) if tok.startswith("--")}


def _live_inventory(
    *, age_s: int = 15, x6_free: int = 6100, x6_cpu: float = 56.0, x6_other: bool = True,
    x7_free: int = 6913, x7_other: bool = True, x8_free: int = 6900, x8_other: bool = True,
) -> dict:
    probed = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        ("xbabe5", 0, "NVIDIA A100-PCIE-40GB", 40960, 40400, "GPU-aaaa1111-0000-0000-0000-000000000005", 4.0, False, "198.51.100.15"),
        ("xbabe5", 1, "NVIDIA A100-PCIE-40GB", 40960, 40400, "GPU-aaaa1111-0000-0000-0000-000000000015", 4.0, False, "198.51.100.15"),
        ("xbabe7", 0, "NVIDIA A100-PCIE-40GB", 40960, x7_free, "GPU-aaaa1111-0000-0000-0000-000000000007", 8.0, x7_other, "198.51.100.17"),
        ("xbabe7", 1, "NVIDIA A100-PCIE-40GB", 40960, x7_free, "GPU-aaaa1111-0000-0000-0000-000000000017", 8.0, x7_other, "198.51.100.17"),
        ("xbabe8", 0, "NVIDIA A100-PCIE-40GB", 40960, x8_free, "GPU-aaaa1111-0000-0000-0000-000000000008", 6.0, x8_other, "198.51.100.18"),
        ("xbabe9", 0, "NVIDIA A100-PCIE-40GB", 40960, 40400, "GPU-aaaa1111-0000-0000-0000-000000000009", 5.0, False, "198.51.100.19"),
        ("xbabe1", 0, "NVIDIA GeForce RTX 4090", 24564, 23900, "GPU-aaaa1111-0000-0000-0000-000000000001", 12.0, False, "203.0.113.11"),
        ("xbabe2", 0, "NVIDIA GeForce RTX 3090", 24576, 24000, "GPU-aaaa1111-0000-0000-0000-000000000002", 18.0, False, "203.0.113.12"),
        ("xbabe3", 0, "NVIDIA GeForce RTX 3090", 24576, 24124, "GPU-aaaa1111-0000-0000-0000-000000000003", 25.0, False, "203.0.113.13"),
        ("xbabe6", 0, "NVIDIA A10", 23028, x6_free, "GPU-aaaa1111-0000-0000-0000-000000000006", x6_cpu, x6_other, "198.51.100.16"),
    ]
    return {"probed_at_utc": probed, "gpus": [
        {"host": h, "gpu": g, "name": n, "total_mib": t, "free_mib": f, "gpu_uuid": u,
         "cpu_pct": c, "other_project": o, "ssh_ip": ip}
        for h, g, n, t, f, u, c, o, ip in rows
    ]}


def _write_inv(tmp_path: Path, **kw) -> Path:
    p = tmp_path / "live_inv.json"
    p.write_text(json.dumps(_live_inventory(**kw)))
    return p


def test_content_hash_matches_lc_all_c_shell_pipeline_and_deconflicts_d0():
    py_hash, commit, _ = launcher.compute_content_tree_hash(REPO_ROOT)
    shell_hash = subprocess.check_output(
        ["bash", "-lc", f"cd {REPO_ROOT} && {launcher.CONTENT_HASH_PIPE}"], text=True,
    ).split()[0]
    assert py_hash == shell_hash
    assert len(py_hash) == 64 and len(commit) == 40
    assert launcher.verify_content_hash(py_hash, py_hash) == py_hash
    with pytest.raises(ValueError, match="content tree-hash mismatch"):
        launcher.verify_content_hash(py_hash, "0" * 64)
    # D0 sync DDP is owned solely by @trainer-lane's h50_topo.sh and rejected here.
    slots, probed = launcher.load_live_inventory(_live_inventory())
    with pytest.raises(ValueError, match="h50_topo.sh"):
        launcher.build_launch_plan("d0-sync", content_hash=py_hash, commit_sha=commit, clean_tree=True,
                                   inventory=slots, probed_at_utc=probed, run_name="dup_d0")


def test_live_inventory_rejects_stale_missing_uuid_duplicate_and_xbabe0():
    assert not hasattr(launcher, "DEFAULT_FLEET_SNAPSHOT")
    with pytest.raises(ValueError, match="stale"):
        launcher.load_live_inventory(_live_inventory(age_s=900), max_age_s=600)
    bad_uuid = _live_inventory()
    bad_uuid["gpus"][0]["gpu_uuid"] = ""
    with pytest.raises(ValueError, match="gpu_uuid"):
        launcher.load_live_inventory(bad_uuid)
    dup_inv = _live_inventory()
    dup_inv["gpus"].append(dict(dup_inv["gpus"][0]))
    with pytest.raises(ValueError, match="duplicate GPU slot"):
        launcher.load_live_inventory(dup_inv)
    with pytest.raises(ValueError, match="forbidden"):
        launcher.normalize_slot_id("xbabe0:0")
    with pytest.raises(ValueError, match="duplicate GPU slot"):
        launcher.parse_roster("x5-g0,xbabe5:0")
    with pytest.raises(ValueError, match="valid IPv4"):
        launcher.validate_ipv4("xbabe5", "ssh_ip")


def test_vram_and_xbabe6_cpu_guards_skip_occupied_gpus():
    slots, _ = launcher.load_live_inventory(_live_inventory())
    admitted, skipped = launcher.filter_available_gpus(slots, min_free_vram_gb=18.0)
    assert {s.slot_id for s in admitted} == {"x5-g0", "x5-g1", "x9-g0", "x1-g0", "x2-g0", "x3-g0"}
    assert {"x7-g0", "x7-g1", "x8-g0", "x6-g0"} == {s["slot"] for s in skipped}
    hot_slots, _ = launcher.load_live_inventory(_live_inventory(x6_free=21500, x6_cpu=74.5, x6_other=False))
    adm_hot, skip_hot = launcher.filter_available_gpus(hot_slots)
    assert "x6-g0" not in {s.slot_id for s in adm_hot}
    assert any("cpu=74.5% >= 70% cap" in r for r in next(s for s in skip_hot if s["slot"] == "x6-g0")["reasons"])


def test_d1_whole_round_h_stages_data_rank_port_8768_and_tunnels(tmp_path, capsys):
    inv = _write_inv(tmp_path)
    slots, probed = launcher.load_live_inventory(_live_inventory())
    h, c, _ = launcher.compute_content_tree_hash(REPO_ROOT)
    ckpt_sha, data_sha = "a" * 64, "b" * 64

    # Reject non-multiple-of-H total_steps (2659 vs H=500), forbidden port 8766, duplicate slots, cross-site H<500.
    with pytest.raises(ValueError, match="exact positive whole multiple"):
        launcher.build_launch_plan("d1-diloco-fp32", content_hash=h, commit_sha=c, clean_tree=True, inventory=slots,
                                   probed_at_utc=probed, run_name="bad_steps", diloco_every=500, total_steps=2659)
    with pytest.raises(ValueError, match="forbidden"):
        launcher.build_launch_plan("d1-diloco-fp32", content_hash=h, commit_sha=c, clean_tree=True, inventory=slots,
                                   probed_at_utc=probed, run_name="bad_port", hub_port=8766)
    with pytest.raises(ValueError, match="duplicate GPU slot"):
        launcher.build_launch_plan("d1-diloco-fp32", content_hash=h, commit_sha=c, clean_tree=True, inventory=slots,
                                   probed_at_utc=probed, run_name="dup", roster_spec="x5-g0,x5-g0")
    with pytest.raises(ValueError, match="cross-site DiLoCo requires H >= 500"):
        launcher.build_launch_plan("d1-diloco-fp32", content_hash=h, commit_sha=c, clean_tree=True, inventory=slots,
                                   probed_at_utc=probed, run_name="bad_wan", roster_spec="x5-g0,x9-g0", d1_stage="smoke")
    with pytest.raises(RuntimeError, match="blocked by pre-flight guard"):
        launcher.build_launch_plan("d1-diloco-fp32", content_hash=h, commit_sha=c, clean_tree=True, inventory=slots,
                                   probed_at_utc=probed, run_name="occ", roster_spec="x5-g0,x7-g0")

    # Local stages protocol (H=2) and smoke (H=50) work on x5 local.
    for stage, exp_h, exp_steps in (("protocol", 2, 6), ("smoke", 50, 200)):
        rc = launcher.main(["--mode", "d1-diloco-fp32", "--d1-stage", stage, "--inventory-json", str(inv),
                            "--repo-root", str(REPO_ROOT), "--dry-run"])
        assert rc == 0
        p = json.loads(capsys.readouterr().out)
        assert p["diloco_every"] == exp_h and p["total_steps"] == exp_steps and p["total_steps"] % exp_h == 0
        assert p["coordinator"]["bind"] == "127.0.0.1" and p["coordinator"]["port"] == 8768

    # Cross-site matched (H=500): H0 hub on 127.0.0.1:8768, background + /status probe, ssh -N -R to IPs, data_rank/world.
    with mock.patch("subprocess.check_call") as m_call:
        rc2 = launcher.main([
            "--mode", "d1-diloco-fp32", "--d1-stage", "matched", "--roster", "x5-g0,x9-g0",
            "--inventory-json", str(inv), "--repo-root", str(REPO_ROOT),
            "--ckpt-sha256", ckpt_sha, "--data-manifest", "/data/manifest.json", "--data-sha256", data_sha, "--dry-run",
        ])
    assert rc2 == 0
    m_call.assert_not_called()
    cross = json.loads(capsys.readouterr().out)
    assert cross["coordinator"]["host"] == "xbabe3" and cross["coordinator"]["url"] == "http://127.0.0.1:8768"
    hub_cmd = next(cm["command"] for cm in cross["commands"] if cm["role"] == "diloco_hub")
    assert "--host 127.0.0.1 --port 8768" in hub_cmd and "--auth-token-file" in hub_cmd
    assert "--weighting" not in hub_cmd
    assert _extract_hub_flags(hub_cmd).issubset(PR73_HUB_CLI_FLAGS)
    assert "setsid nohup" in hub_cmd and "curl -fsS http://127.0.0.1:8768/status" in hub_cmd and "mkdir -p" in hub_cmd
    assert "LC_ALL=C sort" in hub_cmd and h in hub_cmd and "DIRTY_TREE" in hub_cmd
    assert ckpt_sha in hub_cmd and data_sha in hub_cmd
    tunnels = [cm for cm in cross["commands"] if cm["role"].startswith("tunnel:")]
    assert len(tunnels) == 2
    for t in tunnels:
        assert "-R 127.0.0.1:8768:127.0.0.1:8768" in t["command"]
        assert f"ubuntu@{t['to_ip']}" in t["command"] and "xbabe" not in t["command"]
    wcmds = [cm["command"] for cm in cross["commands"] if cm["role"].startswith("worker:")]
    assert "DILOCO_DATA_RANK=0 DILOCO_DATA_WORLD=2" in wcmds[0]
    assert "DILOCO_DATA_RANK=1 DILOCO_DATA_WORLD=2" in wcmds[1]
    valid_set_keys = _main_train_config_fields() | PR73_DILOCO_SET_KEYS
    for wc in wcmds:
        assert wc.startswith("mkdir -p ") and "diloco_dtype=float32" in wc and "total_steps=2500" in wc
        assert "launch_guarded.sh" in wc and "GPU_TARGET=0.87" in wc
        assert "DILOCO_AUTH_TOKEN_FILE=" in wc and 'DILOCO_AUTH_TOKEN="$(cat ' in wc
        assert "budget_s=750.0" in wc and "max_batch=160" in wc and "ddp_global_batches=true" in wc
        assert "ddp_bf16_comm" not in wc and "stratify_chunks" not in wc and "GPU_UUID_MISMATCH" in wc
        emitted_keys = _extract_set_keys(wc)
        assert emitted_keys.issubset(valid_set_keys), f"unknown --set keys: {emitted_keys - valid_set_keys}"
        for bad_k in ("data_rank", "data_world", "diloco_auth_token_file", "diloco_weighting"):
            assert bad_k not in emitted_keys


def test_d2_hetero_single_x5_ddp2_island_safe_a1env_default_and_nccl_profiles(tmp_path, capsys):
    inv = _write_inv(tmp_path, x6_free=21500, x6_cpu=45.0, x6_other=False)
    # Default (no --nccl-profile): safe a1env, NEVER sets SHM/CE/CE (NCCL_SHM_MEMCPY_MODE=3).
    for prof_args, exp_prof in (([], "a1env"), (["--nccl-profile", "tuned-gdr"], "tuned-gdr"), (["--nccl-profile", "tuned-ce"], "tuned-ce")):
        with mock.patch("subprocess.check_call") as m_call:
            rc = launcher.main(["--mode", "d2-hetero", *prof_args, "--inventory-json", str(inv),
                                "--repo-root", str(REPO_ROOT), "--dry-run"])
        assert rc == 0
        m_call.assert_not_called()
        plan = json.loads(capsys.readouterr().out)
        assert plan["nccl_profile"] == exp_prof
        assert plan["nccl_production_qualified"] is (exp_prof != "tuned-ce")
        islands = {w["island"]: w for w in plan["workers"]}
        assert set(islands) == {"x5-ddp2", "x1-g0", "x2-g0", "x3-g0", "x6-g0"}
        assert "x5-g0" not in islands and "x5-g1" not in islands
        assert islands["x5-ddp2"]["nproc"] == 2 and islands["x5-ddp2"]["gpus"] == [0, 1]
        assert [w["data_rank"] for w in plan["workers"]] == list(range(len(plan["workers"])))
        assert all(w["data_world"] == 5 and w["weighting"] == "rnnt_tokens" for w in plan["workers"])
        assert plan["coordinator"]["port"] == 8768 and plan["coordinator"]["weighting"] == "rnnt_tokens"
        assert plan["d2_gates"] == {"min_time_to_wer_reduction_pct": 10.0, "weighting": "rnnt_tokens",
                                    "equal_exposure": True, "require_energy_check": True, "require_sdi_check": True}
        x5_cmd = next(cm["command"] for cm in plan["commands"] if cm["role"] == "worker:x5-ddp2")
        assert "torch.distributed.run" in x5_cmd and "--nproc_per_node 2" in x5_cmd and "mkdir -p " in x5_cmd
        assert "ddp_bf16_comm" not in x5_cmd and "stratify_chunks" not in x5_cmd
        assert _extract_set_keys(x5_cmd).issubset(_main_train_config_fields() | PR73_DILOCO_SET_KEYS)
        x6_cmd = next(cm["command"] for cm in plan["commands"] if cm["role"] == "worker:x6-g0")
        assert "launch_guarded.sh" in x6_cmd and "GPU_TARGET=0.87" in x6_cmd and "CPU_TARGET=0.68" in x6_cmd
        assert _extract_hub_flags(next(cm["command"] for cm in plan["commands"] if cm["role"] == "diloco_hub")).issubset(PR73_HUB_CLI_FLAGS)
        if exp_prof == "a1env":
            assert "NCCL_IB_DISABLE=1" in x5_cmd
            assert "NCCL_SHM_USE_CUDA_MEMCPY" not in x5_cmd and "NCCL_SHM_MEMCPY_MODE=3" not in x5_cmd
        elif exp_prof == "tuned-gdr":
            assert "NCCL_NET_GDR_LEVEL=SYS" in x5_cmd and "NCCL_SOCKET_IFNAME=eno1" in x5_cmd
            assert "NCCL_SHM_USE_CUDA_MEMCPY" not in x5_cmd and "NCCL_SHM_MEMCPY_MODE=3" not in x5_cmd
        else:
            assert "CUDA_MODULE_LOADING=EAGER" in x5_cmd
            assert "NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3" in x5_cmd
            assert "NCCL_NET_GDR_LEVEL=SYS" in x5_cmd and "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET" in x5_cmd
            for inert in ("NCCL_IB_DISABLE", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_NTHREADS", "NCCL_NSOCKS_PERTHREAD"):
                assert inert not in x5_cmd and inert not in plan["nccl_env"]
        assert {"x7-g0", "x7-g1", "x8-g0"} == {s["slot"] for s in plan["skipped_gpus"]}


def test_host_ips_file_and_no_hardcoded_ips(tmp_path, capsys):
    for profile in launcher.HOST_PROFILES.values():
        assert "ssh_ip" not in profile and "private_ip" not in profile
    raw = _live_inventory()
    for g in raw["gpus"]:
        g["ssh_ip"] = ""
    inv = tmp_path / "no_ip_inv.json"
    inv.write_text(json.dumps(raw))
    hosts_json = tmp_path / "hosts.json"
    hosts_json.write_text(json.dumps({"x5": "198.51.100.25", "x9": "198.51.100.29", "h0": "203.0.113.23"}))
    rc = launcher.main([
        "--mode", "d1-diloco-fp32", "--d1-stage", "matched", "--roster", "x5-g0,x9-g0",
        "--inventory-json", str(inv), "--host-ips-file", str(hosts_json),
        "--repo-root", str(REPO_ROOT), "--dry-run",
    ])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    tunnels = {t["to_host"]: t["to_ip"] for t in plan["coordinator"]["tunnels"]}
    assert tunnels == {"xbabe5": "198.51.100.25", "xbabe9": "198.51.100.29"}
    hosts_txt = tmp_path / "hosts.txt"
    hosts_txt.write_text("# comment\n198.51.100.55 xbabe5 x5\n")
    assert launcher.load_host_ips(hosts_txt) == {"xbabe5": "198.51.100.55"}
