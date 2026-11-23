"""4 Mandatory Fault-Tolerance Proofs for Hardened Safetensors DiLoCo Hub (PR #73)."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import pickle
import threading
import time
import urllib.error
import urllib.request

import pytest
from safetensors.torch import load as st_load
import torch

import train.diloco_hub as diloco_hub_mod
from train.diloco_client import DiLoCoBarrierError, DiLoCoClient, serialize_tensors, sign_request
from train.diloco_hub import DiLoCoHub, make_hub_app

FAULT_TOKEN = "h50-fault-proof-token"


class _TinyModel(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc1, self.fc2 = torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
        self.register_buffer("cmvn_mean", torch.linspace(-0.5, 0.5, 8))
        self.register_buffer("cmvn_inv_std", torch.full((8,), 1.25))


def _payload(m: torch.nn.Module) -> dict:
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return {"model": sd, "ema": sd, "trainable_keys": [n for n, _ in m.named_parameters()],
            "model_config": {"family": "dense", "d_model": 8}, "tokenizer_sha256": "tok999", "step": 0}


def _make_delta(anchor: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tkeys, sd = {n for n, _ in model.named_parameters()}, model.state_dict()
    return {k: ((anchor[k].float() - sd[k].detach().float().cpu()).to(torch.float32) if k in tkeys else sd[k].detach().cpu().clone()) for k in anchor}


def _serve_hub(hub: DiLoCoHub):
    from aiohttp import web

    app = make_hub_app(hub, auth_token=FAULT_TOKEN)
    port_holder, ready, stop_ev = {}, threading.Event(), {}

    def _serve():
        async def _run():
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port_holder["port"] = site._server.sockets[0].getsockname()[1]
            ev = asyncio.Event()
            stop_ev["loop"], stop_ev["ev"] = asyncio.get_running_loop(), ev
            ready.set()
            await ev.wait()
            await runner.cleanup()

        asyncio.run(_run())

    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    assert ready.wait(timeout=5.0)
    return f"http://127.0.0.1:{port_holder['port']}", lambda: (stop_ev["loop"].call_soon_threadsafe(stop_ev["ev"].set), th.join(timeout=5.0))


def test_interrupted_commit_recovery(tmp_path, monkeypatch):
    """Proof 1: OSError mid-atomic-save leaves hub cleanly at round r and reload recovers round r cleanly."""
    base = _TinyModel(seed=11)
    hub = DiLoCoHub("Fault1", _payload(base), tmp_path, expected_roster=("wA", "wB"), outer_lr=0.7, outer_momentum=0.9)
    tkeys = {n for n, _ in base.named_parameters()}
    d1 = {k: (torch.full_like(v, 0.05) if k in tkeys else v.clone()) for k, v in base.state_dict().items()}

    sha0 = hub.anchor_sha256
    hub.submit_delta(worker_id="wA", host="hA", base_round=0, local_steps=10, audio_hours=1.0, local_loss=0.5, delta_sd=d1, anchor_sha256=sha0, data_rank=0, data_world=2)
    hub.submit_delta(worker_id="wB", host="hB", base_round=0, local_steps=10, audio_hours=1.0, local_loss=0.5, delta_sd=d1, anchor_sha256=sha0, data_rank=1, data_world=2)
    assert hub.round == 1
    r1_sha, r1_anchor, r1_vel = hub.anchor_sha256, {k: v.clone() for k, v in hub.opt.anchor.items()}, {k: v.clone() for k, v in hub.opt.velocity.items()}

    hub.submit_delta(worker_id="wA", host="hA", base_round=1, local_steps=10, audio_hours=1.0, local_loss=0.4, delta_sd=d1, anchor_sha256=r1_sha, data_rank=0, data_world=2)
    real_write = diloco_hub_mod._atomic_write_bytes

    def _fail_on_meta(dst, data):
        if dst.name == "outer_meta_r2.json":
            raise OSError("simulated disk crash mid-commit")
        real_write(dst, data)

    monkeypatch.setattr(diloco_hub_mod, "_atomic_write_bytes", _fail_on_meta)
    with pytest.raises(OSError, match="simulated disk crash"):
        hub.submit_delta(worker_id="wB", host="hB", base_round=1, local_steps=10, audio_hours=1.0, local_loss=0.4, delta_sd=d1, anchor_sha256=r1_sha, data_rank=1, data_world=2)

    assert hub.round == 1 and hub.anchor_sha256 == r1_sha and list(tmp_path.glob("*.tmp")) == []
    for k in tkeys:
        assert torch.equal(hub.opt.anchor[k], r1_anchor[k]) and torch.equal(hub.opt.velocity[k], r1_vel[k])
    assert hashlib.sha256((tmp_path / "latest_anchor.safetensors").read_bytes()).hexdigest() == r1_sha

    monkeypatch.setattr(diloco_hub_mod, "_atomic_write_bytes", real_write)
    reloaded = DiLoCoHub.from_work_dir(tmp_path)
    assert reloaded.round == 1 and reloaded.anchor_sha256 == r1_sha
    for k in tkeys:
        assert torch.equal(reloaded.opt.anchor[k], r1_anchor[k]) and torch.equal(reloaded.opt.velocity[k], r1_vel[k])
    reloaded.submit_delta(worker_id="wA", host="hA", base_round=1, local_steps=10, audio_hours=1.0, local_loss=0.4, delta_sd=d1, anchor_sha256=r1_sha, data_rank=0, data_world=2)
    r2 = reloaded.submit_delta(worker_id="wB", host="hB", base_round=1, local_steps=10, audio_hours=1.0, local_loss=0.4, delta_sd=d1, anchor_sha256=r1_sha, data_rank=1, data_world=2)
    assert r2["triggered_outer"] and reloaded.round == 2


def test_corrupt_delta_rejected_preserves_peers(tmp_path):
    """Proof 2: Corrupt deltas (NaN, bad shape, wrong hash, pickle, bad buffer) rejected without dropping peer wA."""
    base = _TinyModel(seed=22)
    hub = DiLoCoHub("Fault2", _payload(base), tmp_path, expected_roster=("wA", "wB"), outer_lr=0.5, outer_momentum=0.0)
    url, stop = _serve_hub(hub)
    tkeys = {n for n, _ in base.named_parameters()}
    valid_d = {k: (torch.full_like(v, 0.1) if k in tkeys else v.clone()) for k, v in base.state_dict().items()}

    def _post(wid: str, rank: int, payload: bytes, anchor_sha: str | None = None):
        asha = hub.anchor_sha256 if anchor_sha is None else anchor_sha
        meta = {"worker_id": wid, "host": "h", "base_round": 0, "anchor_sha256": asha,
                "data_rank": rank, "data_world": 2, "local_steps": 5, "audio_hours": 1.0, "local_loss": 0.5}
        sig = sign_request(FAULT_TOKEN, wid, 0, asha, hashlib.sha256(payload).hexdigest())
        req = urllib.request.Request(f"{url}/delta", data=payload, headers={"X-DiLoCo-Meta": json.dumps(meta), "X-DiLoCo-Auth": sig}, method="POST")
        return urllib.request.urlopen(req, timeout=5.0)

    try:
        with _post("wA", 0, serialize_tensors(valid_d)) as resp:
            assert json.loads(resp.read())["accepted"]
        assert hub.status()["pending_workers"] == ["wA"] and hub.round == 0

        nan_d = {k: (torch.full_like(v, float("nan")) if k == "fc1.weight" else v.clone()) for k, v in valid_d.items()}
        bad_shape = {k: (torch.zeros(2, 2) if k == "fc1.weight" else v.clone()) for k, v in valid_d.items()}
        bad_buf = {k: (v + 1.0 if k == "cmvn_mean" else v.clone()) for k, v in valid_d.items()}
        pt_buf = io.BytesIO()
        torch.save(valid_d, pt_buf)
        for body, asha, expected_code in (
            (serialize_tensors(nan_d), None, 400), (serialize_tensors(bad_shape), None, 400),
            (serialize_tensors(valid_d), "f" * 64, 409), (pickle.dumps(valid_d), None, 400),
            (pt_buf.getvalue(), None, 400), (serialize_tensors(bad_buf), None, 409),
        ):
            with pytest.raises(urllib.error.HTTPError) as ei:
                _post("wB", 1, body, anchor_sha=asha)
            assert ei.value.code == expected_code and hub.status()["pending_workers"] == ["wA"] and hub.round == 0

        with _post("wB", 1, serialize_tensors(valid_d)) as resp:
            res = json.loads(resp.read())
        assert res["triggered_outer"] and res["current_round"] == 1 and res["round_record"]["workers"] == ["wA", "wB"]
    finally:
        stop()


def test_member_dropout_blocks_barrier_no_subset_commit(tmp_path):
    """Proof 3: Missing roster member never triggers subset commit; blocking client raises DiLoCoBarrierError."""
    base = _TinyModel(seed=33)
    hub = DiLoCoHub("Fault3", _payload(base), tmp_path, expected_roster=("wA", "wB"), quorum_timeout_s=0.1, outer_lr=0.5, outer_momentum=0.0)
    url, stop = _serve_hub(hub)
    try:
        m_a = _TinyModel(seed=33)
        c_a = DiLoCoClient(url, "wA", m_a.state_dict(), auth_token=FAULT_TOKEN, data_rank=0, data_world=2, barrier_timeout_s=0.6, poll_interval_s=0.05)
        c_a.pull_anchor(m_a, force=True)
        with torch.no_grad():
            for p in m_a.parameters():
                p.add_(0.25)
        t0 = time.monotonic()
        with pytest.raises(DiLoCoBarrierError, match="barrier timeout"):
            c_a.sync(m_a, local_steps=20, audio_hours=1.0, local_loss=0.6)
        assert (time.monotonic() - t0) >= 0.45
        time.sleep(0.15)
        assert hub.round == 0 and hub.status()["pending_workers"] == ["wA"] and c_a.base_round == 0
    finally:
        stop()


def test_last_good_restart_preserves_outer_momentum_and_adamw(tmp_path):
    """Proof 4: 2 rounds -> stop hub & workers -> reload outer_state_r2 + AdamW -> round 3 matches uninterrupted."""
    batches = {}
    for w_idx, wid in enumerate(("wA", "wB")):
        for rnd in range(3):
            g = torch.Generator().manual_seed(1000 + w_idx * 100 + rnd)
            batches[(wid, rnd)] = [(torch.randn(4, 8, generator=g), torch.randn(4, 4, generator=g)) for _ in range(3)]

    def _run_round(hub: DiLoCoHub, models: dict[str, _TinyModel], opts: dict[str, torch.optim.AdamW], rnd: int):
        _, sha, raw = hub.anchor_state()
        anchor_sd = st_load(raw)
        for rank, wid in enumerate(("wA", "wB")):
            models[wid].load_state_dict(anchor_sd, strict=True)
            for x, y in batches[(wid, rnd)]:
                opts[wid].zero_grad()
                torch.nn.functional.mse_loss(models[wid].fc2(torch.relu(models[wid].fc1(x))), y).backward()
                opts[wid].step()
            hub.submit_delta(worker_id=wid, host=f"h_{wid}", base_round=rnd, local_steps=3, audio_hours=1.0, local_loss=0.1,
                             delta_sd=_make_delta(anchor_sd, models[wid]), anchor_sha256=sha, data_rank=rank, data_world=2,
                             worker_pos={"step": (rnd + 1) * 3, "epoch": 0})

    init_m = _TinyModel(seed=44)
    ctrl_hub = DiLoCoHub("Ctrl", _payload(init_m), tmp_path / "ctrl", expected_roster=("wA", "wB"), outer_lr=0.7, outer_momentum=0.9, hub_ema_decay=0.85)
    ctrl_models = {w: _TinyModel(seed=44) for w in ("wA", "wB")}
    ctrl_opts = {w: torch.optim.AdamW(ctrl_models[w].parameters(), lr=1e-2) for w in ("wA", "wB")}
    for rnd in range(3):
        _run_round(ctrl_hub, ctrl_models, ctrl_opts, rnd)

    rst_dir = tmp_path / "rst"
    rst_hub = DiLoCoHub("Ctrl", _payload(init_m), rst_dir, expected_roster=("wA", "wB"), outer_lr=0.7, outer_momentum=0.9, hub_ema_decay=0.85)
    rst_models = {w: _TinyModel(seed=44) for w in ("wA", "wB")}
    rst_opts = {w: torch.optim.AdamW(rst_models[w].parameters(), lr=1e-2) for w in ("wA", "wB")}
    for rnd in range(2):
        _run_round(rst_hub, rst_models, rst_opts, rnd)
    assert rst_hub.round == 2 and (rst_dir / "outer_state_r2.safetensors").is_file()

    saved_workers = {}
    for w in ("wA", "wB"):
        wp = tmp_path / f"worker_{w}_r2.pt"
        torch.save({"model": rst_models[w].state_dict(), "opt": rst_opts[w].state_dict()}, wp)
        saved_workers[w] = wp
    del rst_hub, rst_models, rst_opts

    reloaded_hub = DiLoCoHub.from_work_dir(rst_dir)
    assert reloaded_hub.round == 2 and reloaded_hub.status()["worker_positions"]["wA"]["step"] == 6
    reloaded_models, reloaded_opts = {}, {}
    for w in ("wA", "wB"):
        ckpt = torch.load(saved_workers[w], map_location="cpu", weights_only=True)
        m = _TinyModel(seed=999)
        m.load_state_dict(ckpt["model"], strict=True)
        o = torch.optim.AdamW(m.parameters(), lr=1e-2)
        o.load_state_dict(ckpt["opt"])
        reloaded_models[w], reloaded_opts[w] = m, o

    _run_round(reloaded_hub, reloaded_models, reloaded_opts, 2)
    assert reloaded_hub.round == 3
    ctrl_sd, rst_sd = st_load(ctrl_hub.anchor_state()[2]), st_load(reloaded_hub.anchor_state()[2])
    for k in ctrl_sd:
        assert torch.allclose(rst_sd[k], ctrl_sd[k], atol=1e-6, rtol=0.0)
    for k in ctrl_hub.opt.velocity:
        assert torch.allclose(reloaded_hub.opt.velocity[k], ctrl_hub.opt.velocity[k], atol=1e-6, rtol=0.0)
        assert torch.allclose(reloaded_hub.hub_ema[k], ctrl_hub.hub_ema[k], atol=1e-6, rtol=0.0)
    assert torch.equal(rst_sd["cmvn_mean"], init_m.cmvn_mean) and torch.equal(rst_sd["cmvn_inv_std"], init_m.cmvn_inv_std)
    for w in ("wA", "wB"):
        for pid, c_st in ctrl_opts[w].state_dict()["state"].items():
            r_st = reloaded_opts[w].state_dict()["state"][pid]
            assert torch.allclose(r_st["exp_avg"], c_st["exp_avg"], atol=1e-6, rtol=0.0)
            assert torch.allclose(r_st["exp_avg_sq"], c_st["exp_avg_sq"], atol=1e-6, rtol=0.0)


def test_trainer_and_client_lazy_import_without_safetensors(monkeypatch):
    """Trainer.run() with DiLoCo disabled (and importing diloco_client) must NOT import safetensors."""
    import importlib
    import sys
    from types import SimpleNamespace
    from train.trainer import TrainConfig, Trainer

    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    monkeypatch.delitem(sys.modules, "train.diloco_client", raising=False)

    fake = SimpleNamespace(
        world=1, rank=0, local_rank=0, step=0, device=torch.device("cpu"),
        cfg=TrainConfig(diloco_hub="", diloco_every=0),
        _batches=lambda: iter(()),
    )
    Trainer.run(fake, max_steps=0)
    assert "train.diloco_client" not in sys.modules
    mod = importlib.import_module("train.diloco_client")
    assert mod.TrainerDiLoCoHook(None, 0).log_fields() == {}

