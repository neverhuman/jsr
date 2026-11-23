"""Unit and end-to-end HTTP tests for hardened safetensors DiLoCoHub, DiLoCoClient, and Promotion CLI."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import pickle
import threading
import urllib.error
import urllib.request

import pytest
import safetensors.torch as st_torch
import torch

from train.diloco import pack_bf16_delta
from train.diloco_client import DiLoCoClient, TrainerDiLoCoHook, sign_request
from train.diloco_hub import DiLoCoConflictError, DiLoCoHub, DiLoCoValidationError, load_init_anchor, make_hub_app
from train.diloco_promote import PromotionError, main as promote_main, promote_anchor

TEST_TOKEN = "h50-test-shared-secret-token"


class _TinyWithBuffer(torch.nn.Module):
    def __init__(self, fill: float = 1.0, cmvn_val: float = 0.25):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8, bias=True)
        self.fc2 = torch.nn.Linear(8, 4, bias=True)
        self.register_buffer("cmvn_mean", torch.full((8,), cmvn_val, dtype=torch.float32))
        self.register_buffer("step_count", torch.tensor(7, dtype=torch.int64))
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(fill)


def _payload(m: torch.nn.Module) -> dict:
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return {
        "model": sd, "ema": sd, "trainable_keys": [n for n, _ in m.named_parameters()],
        "model_config": {"family": "dense", "d_model": 8}, "tokenizer_sha256": "abc123", "step": 0,
    }


def _serve_hub(hub: DiLoCoHub, token: str = TEST_TOKEN):
    from aiohttp import web

    app = make_hub_app(hub, auth_token=token)
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

    def _stop():
        stop_ev["loop"].call_soon_threadsafe(stop_ev["ev"].set)
        th.join(timeout=5.0)

    return f"http://127.0.0.1:{port_holder['port']}", _stop


_CANARY_FIRED = False


def _trip_canary():
    global _CANARY_FIRED
    _CANARY_FIRED = True
    return {}


class _ExploitPickle:
    def __reduce__(self):
        return (_trip_canary, ())


def test_security_rejects_unauthenticated_and_raw_pickle_without_unpickling(tmp_path):
    """Unauthenticated routes return 401; raw pickle/torch.save POSTed to /delta returns 400 without unpickling."""
    global _CANARY_FIRED
    _CANARY_FIRED = False
    base = _TinyWithBuffer(1.0)
    hub = DiLoCoHub("V3_Sec", _payload(base), tmp_path, expected_roster=("wA", "wB"))
    url, stop = _serve_hub(hub)
    try:
        # 1. Missing/bad HMAC auth on GET /status, GET /anchor, POST /delta -> 401.
        for path, method in (("/status", "GET"), ("/anchor", "GET")):
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(urllib.request.Request(f"{url}{path}", method=method), timeout=5.0)
            assert ei.value.code == 401

        meta = {"worker_id": "wA", "base_round": 0, "anchor_sha256": hub.anchor_sha256, "data_rank": 0, "data_world": 2}
        bad_req = urllib.request.Request(
            f"{url}/delta", data=b"junk",
            headers={"X-DiLoCo-Meta": json.dumps(meta), "X-DiLoCo-Auth": "0" * 64}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad_req, timeout=5.0)
        assert ei.value.code == 401

        # 2. Signed raw pickle AND torch.save payloads MUST be rejected (400) without invoking __reduce__.
        buf = io.BytesIO()
        torch.save(_ExploitPickle(), buf)
        for bad_payload in (pickle.dumps(_ExploitPickle()), buf.getvalue()):
            psha = hashlib.sha256(bad_payload).hexdigest()
            sig = sign_request(TEST_TOKEN, "wA", 0, hub.anchor_sha256, psha)
            req = urllib.request.Request(
                f"{url}/delta", data=bad_payload,
                headers={"X-DiLoCo-Meta": json.dumps(meta), "X-DiLoCo-Auth": sig}, method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=5.0)
            assert ei.value.code == 400
        assert not _CANARY_FIRED, "pickle __reduce__ MUST NEVER execute on network route!"
    finally:
        stop()


def test_sync_exact_fp32_blocking_http_and_buffers_and_order_invariance(tmp_path):
    """Exact-sync HTTP barrier: concurrent blocking clients, uniform 1/N order-invariance, intact buffers."""
    base = _TinyWithBuffer(2.0, cmvn_val=0.375)
    hub = DiLoCoHub("V3_Sync", _payload(base), tmp_path, expected_roster=("wA", "wB"), outer_lr=0.5, outer_momentum=0.0)
    url, stop = _serve_hub(hub)
    try:
        m1, m2 = _TinyWithBuffer(2.0, cmvn_val=0.375), _TinyWithBuffer(2.0, cmvn_val=0.375)
        c1 = DiLoCoClient(url, "wA", m1.state_dict(), auth_token=TEST_TOKEN, data_rank=0, data_world=2, barrier_timeout_s=10.0)
        c2 = DiLoCoClient(url, "wB", m2.state_dict(), auth_token=TEST_TOKEN, data_rank=1, data_world=2, barrier_timeout_s=10.0)
        c1.pull_anchor(m1, force=True)
        c2.pull_anchor(m2, force=True)
        with torch.no_grad():
            for p in m1.parameters():
                p.fill_(1.0)  # delta = +1.0
            for p in m2.parameters():
                p.fill_(1.5)  # delta = +0.5

        out1: dict = {}

        def _run_w1():
            out1["res"] = c1.sync(m1, local_steps=100, audio_hours=10.0, local_loss=0.4)

        t1 = threading.Thread(target=_run_w1)
        t1.start()
        # Different audio_hours (10.0 vs 1.0) MUST NOT skew uniform 1/N sync average ((1.0+0.5)/2 = 0.75).
        res2 = c2.sync(m2, local_steps=100, audio_hours=1.0, local_loss=0.5)
        t1.join(timeout=10.0)
        assert not t1.is_alive()
        res1 = out1["res"]

        assert res1["diloco_round"] == 1 and res2["diloco_round"] == 1
        assert res1["diloco_updated"] and res2["diloco_updated"]
        assert res1["diloco_anchor_sha256"] == res2["diloco_anchor_sha256"] == hub.anchor_sha256
        expected = 2.0 - 0.5 * 0.75
        assert abs(float(m1.fc1.weight.mean().item()) - expected) < 1e-6
        assert abs(float(m2.fc1.weight.mean().item()) - expected) < 1e-6
        assert torch.equal(m1.cmvn_mean, torch.full((8,), 0.375))
        assert torch.equal(m1.step_count, torch.tensor(7, dtype=torch.int64))
        rnd, upd = c1.pull_anchor(m1)
        assert rnd == 1 and not upd  # 304 Not Modified
    finally:
        stop()


def test_sync_validation_idempotency_rank_conflict_and_retention(tmp_path):
    """Require roster by default, idempotent duplicate 200, conflicting duplicate/rank 409, retention pruning."""
    base = _TinyWithBuffer(3.0)
    with pytest.raises(ValueError, match="expected_roster"):
        DiLoCoHub("NoRoster", _payload(base), tmp_path / "fail")

    hub = DiLoCoHub(
        "V3_Val", _payload(base), tmp_path / "run", expected_roster=("wA", "wB"),
        outer_lr=0.5, outer_momentum=0.0, hub_ema_decay=0.8, keep_last_k=2, keep_every_m=3,
    )
    tkeys = {n for n, _ in base.named_parameters()}
    d_ok = {k: (torch.full_like(v, 0.2) if k in tkeys else v.clone()) for k, v in base.state_dict().items()}
    d_alt = {k: (torch.full_like(v, 0.4) if k in tkeys else v.clone()) for k, v in base.state_dict().items()}
    d_bf16 = {k: (v.to(torch.bfloat16) if k in tkeys else v.clone()) for k, v in d_ok.items()}

    with pytest.raises(DiLoCoValidationError, match="float32"):
        hub.submit_delta(worker_id="wA", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                         delta_sd=d_bf16, anchor_sha256=hub.anchor_sha256, data_rank=0, data_world=2)

    r1 = hub.submit_delta(worker_id="wA", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                          delta_sd=d_ok, anchor_sha256=hub.anchor_sha256, data_rank=0, data_world=2)
    assert r1["accepted"] and not r1["duplicate"]
    r1_dup = hub.submit_delta(worker_id="wA", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                              delta_sd=d_ok, anchor_sha256=hub.anchor_sha256, data_rank=0, data_world=2)
    assert r1_dup["accepted"] and r1_dup["duplicate"]
    with pytest.raises(DiLoCoConflictError, match="conflicting duplicate"):
        hub.submit_delta(worker_id="wA", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                         delta_sd=d_alt, anchor_sha256=hub.anchor_sha256, data_rank=0, data_world=2)
    with pytest.raises(DiLoCoConflictError, match="duplicate data_rank"):
        hub.submit_delta(worker_id="wB", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                         delta_sd=d_ok, anchor_sha256=hub.anchor_sha256, data_rank=0, data_world=2)

    # Complete 4 rounds to verify hub EMA + retention pruning (keep_last_k=2, keep_every_m=3).
    hub.submit_delta(worker_id="wB", host="h", base_round=0, local_steps=1, audio_hours=1.0, local_loss=0.5,
                     delta_sd=d_ok, anchor_sha256=hub.anchor_sha256, data_rank=1, data_world=2)
    for rnd in range(1, 5):
        sha = hub.anchor_sha256
        hub.submit_delta(worker_id="wA", host="h", base_round=rnd, local_steps=1, audio_hours=1.0, local_loss=0.5,
                         delta_sd=d_ok, anchor_sha256=sha, data_rank=0, data_world=2)
        hub.submit_delta(worker_id="wB", host="h", base_round=rnd, local_steps=1, audio_hours=1.0, local_loss=0.5,
                         delta_sd=d_ok, anchor_sha256=sha, data_rank=1, data_world=2)
    assert hub.round == 5 and hub.hub_ema is not None
    wd = tmp_path / "run"
    assert not (wd / "anchor_r1.safetensors").exists() and not (wd / "anchor_r2.safetensors").exists()
    assert (wd / "anchor_r3.safetensors").is_file()  # kept by keep_every_m=3
    assert (wd / "anchor_r4.safetensors").is_file() and (wd / "anchor_r5.safetensors").is_file()  # kept by keep_last_k=2


def test_experimental_async_quorum_quarantine_and_promote_cli(tmp_path):
    """Opt-in experimental async BF16 quorum quarantine + dev-eval gated diloco_promote.py CLI."""
    base = _TinyWithBuffer(1.0)
    hub = DiLoCoHub(
        "V3_Async", _payload(base), tmp_path / "hub", allow_async_experimental=True,
        quorum_k=3, outer_lr=0.7, outer_momentum=0.9, min_cosine=-0.1,
    )
    anchor_sd = {k: v.clone() for k, v in base.state_dict().items()}
    tkeys = {n for n, _ in base.named_parameters()}
    pack = lambda m: {k: (pack_bf16_delta(anchor_sd, m.state_dict())[k] if k in tkeys else anchor_sd[k]) for k in anchor_sd}
    hub.submit_delta(worker_id="w1", host="h1", base_round=0, local_steps=10, audio_hours=10.0, local_loss=0.4, delta_sd=pack(_TinyWithBuffer(0.8)))
    hub.submit_delta(worker_id="w2", host="h2", base_round=0, local_steps=10, audio_hours=4.0, local_loss=0.4, delta_sd=pack(_TinyWithBuffer(0.75)))
    r3 = hub.submit_delta(worker_id="w_bad", host="h3", base_round=0, local_steps=10, audio_hours=2.0, local_loss=9.9, delta_sd=pack(_TinyWithBuffer(5.0)))
    assert r3["triggered_outer"] and r3["round_record"]["quarantined"] == ["w_bad"]
    latest_sd = st_torch.load_file(str(tmp_path / "hub" / "latest_anchor.safetensors"))
    assert float(latest_sd["fc1.weight"].mean().item()) < 0.85

    # Promotion CLI: reject bad sha256 or high WER; promote only on verified passing receipt.
    cand = tmp_path / "hub" / "anchor_r1.safetensors"
    out_p = tmp_path / "promoted" / "anchor.safetensors"
    bad_rec = tmp_path / "bad.json"
    bad_rec.write_text(json.dumps({"anchor_sha256": "0" * 64, "wer": 3.5, "split": "dev-clean"}))
    with pytest.raises(PromotionError, match="anchor_sha256 mismatch"):
        promote_anchor(cand, bad_rec, out_p, max_wer=4.0)
    assert not out_p.exists()

    hi_rec = tmp_path / "hi.json"
    hi_rec.write_text(json.dumps({"anchor_sha256": hub.anchor_sha256, "wer": 5.2, "split": "dev-clean"}))
    assert promote_main(["--anchor", str(cand), "--receipt", str(hi_rec), "--out", str(out_p), "--max-wer", "4.5"]) == 2
    assert not out_p.exists()

    ok_rec = tmp_path / "ok.json"
    ok_rec.write_text(json.dumps({"anchor_sha256": hub.anchor_sha256, "wer": 3.8, "baseline_wer": 4.2, "split": "dev-clean", "ref_words": 54402}))
    assert promote_main(["--anchor", str(cand), "--receipt", str(ok_rec), "--out", str(out_p), "--max-wer", "4.0", "--min-gain-pp", "0.2"]) == 0
    assert out_p.is_file()
    prov = json.loads(out_p.with_name("anchor.safetensors.provenance.json").read_text())
    assert prov["promoted"] and prov["anchor_sha256"] == hub.anchor_sha256 and prov["gain_pp"] == 0.4

    hook = TrainerDiLoCoHook(None, 0)
    assert hook.log_fields() == {}
    hook.round = 3
    assert hook.log_fields() == {"diloco_round": 3}

    # CLI defaults: 127.0.0.1:8768, fails closed without token or roster.
    from train.diloco_http import main as hub_main
    with pytest.raises(SystemExit):
        hub_main(["--cohort", "c", "--work-dir", str(tmp_path / "cli")])


def test_load_init_anchor_safetensors_and_rejects_unsafe_pickle(tmp_path):
    """load_init_anchor supports .safetensors and weights_only=True .pt, rejecting unsafe pickle payloads."""
    global _CANARY_FIRED
    _CANARY_FIRED = False
    base = _TinyWithBuffer(1.5)
    st_path = tmp_path / "init.safetensors"
    st_torch.save_file({k: v.detach().cpu().contiguous() for k, v in base.state_dict().items()}, str(st_path))
    st_path.with_suffix(".json").write_text(json.dumps({"model_config": {"d_model": 8}, "tokenizer_sha256": "abc"}))
    loaded_st = load_init_anchor(st_path)
    assert loaded_st["tokenizer_sha256"] == "abc" and torch.equal(loaded_st["model"]["fc1.weight"], base.fc1.weight)
    safe_pt, evil_pt, bad_ext = tmp_path / "safe.pt", tmp_path / "evil.pt", tmp_path / "evil.pkl"
    torch.save(_payload(base), safe_pt)
    assert torch.equal(load_init_anchor(safe_pt)["model"]["fc1.weight"], base.fc1.weight)
    torch.save(_ExploitPickle(), evil_pt)
    with pytest.raises(DiLoCoValidationError, match="unsafe or invalid"):
        load_init_anchor(evil_pt)
    bad_ext.write_bytes(pickle.dumps(_ExploitPickle()))
    with pytest.raises(DiLoCoValidationError, match="unsupported init anchor"):
        load_init_anchor(bad_ext)
    assert not _CANARY_FIRED, "unsafe pickle init anchor MUST NEVER execute!"


