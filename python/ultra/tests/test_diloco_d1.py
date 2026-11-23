"""D1 Items 1, 3, and 7 proofs: disjoint island sharding, data_rank chunk stratification, post-anchor ckpt/eval & resume."""
from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import os
import threading
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

from train import checkpoint
from train.data import Featurizer, FixedBatches, ParquetBatches, apply_data_shard, collate, stratified_parts
from train.diloco_hub import DiLoCoHub, make_hub_app
from train.errors import CheckpointError, DataError
from train.tokenizer import Tokenizer
from train.trainer import TrainConfig, Trainer

D1_TOKEN = "h50-d1-test-token"


@pytest.fixture(scope="module")
def parquet_shard(tmp_path_factory):
    import sentencepiece as spm

    d = tmp_path_factory.mktemp("h50_d1_loader")
    words = ["hello there", "the quick brown fox", "general kenobi", "a b c d e f"]
    (d / "t.txt").write_text("\n".join(words * 40))
    spm.SentencePieceTrainer.train(
        input=str(d / "t.txt"), model_prefix=str(d / "tok"), vocab_size=30, model_type="bpe", character_coverage=1.0
    )
    rng = np.random.default_rng(0)
    rows = {"id": [], "text": [], "audio": []}
    for i in range(36):
        pcm = (rng.standard_normal(int(16000 * (0.5 + 0.13 * i))) * 3000).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm, 16000, format="FLAC")
        rows["id"].append(f"u{i}")
        rows["text"].append(words[i % 4].upper())
        rows["audio"].append({"bytes": buf.getvalue(), "path": f"u{i}.flac"})
    root = d / "parquet"
    (root / "train.clean.100").mkdir(parents=True)
    pq.write_table(pa.table(rows), root / "train.clean.100" / "0000.parquet", row_group_size=6)
    return str(root), str(d / "tok.model")


def _src(shard, rank=0, world=1, **kw):
    root, tok = shard
    t = Tokenizer(tok)
    s = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=7, budget_s=kw.pop("budget_s", 4.0),
                       max_batch=8, parquet_root=root, workers=1, window_groups=3)
    s.rank, s.world = rank, world
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _tiny_cfg_and_examples():
    from tests.test_train_checkpoint import _cfg, _examples

    base = _cfg()
    return dataclasses.replace(base, encoder=dataclasses.replace(base.encoder, dropout=0.0)), _examples(24, seed=3)


def _serve_hub(hub: DiLoCoHub):
    from aiohttp import web

    app = make_hub_app(hub, auth_token=D1_TOKEN)
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


# ---- D1 Item 1: rank-exact disjoint island data loading + zero-support step -----------------------
def test_island_reads_exactly_its_ddp_rank_share_and_is_disjoint(parquet_shard):
    for world in (2, 3):
        seen_by_rank = []
        for r in range(world):
            ddp = _src(parquet_shard, r, world, ddp_global_batches=True, budget_s=8.0)
            island = _src(parquet_shard, 0, 1, budget_s=8.0)
            apply_data_shard(island, TrainConfig(data_rank=r, data_world=world), world=1)
            uids_r = set()
            for w in range(ddp._windows(0)):
                a = [([e.uid for e in g], act) for g, act in ddp._load_window(0, w)]
                b = [([e.uid for e in g], act) for g, act in island._load_window(0, w)]
                assert a == b
                for g, act in b:
                    if act:
                        uids_r.update(g)
            seen_by_rank.append(uids_r)
        for i in range(world):
            for j in range(i + 1, world):
                assert seen_by_rank[i].isdisjoint(seen_by_rank[j])
    s = _src(parquet_shard)
    apply_data_shard(s, TrainConfig(), world=1)
    assert (s.rank, s.world, s.ddp_global_batches) == (0, 1, False)
    for bad, w in ((TrainConfig(data_rank=0, data_world=2), 2), (TrainConfig(data_rank=2, data_world=2), 1),
                   (TrainConfig(data_rank=0, data_world=2, ddp_rank_weights="1,1"), 1)):
        with pytest.raises(ValueError):
            apply_data_shard(_src(parquet_shard), bad, w)


def test_island_zero_support_step_and_duration_strata(tmp_path):
    mcfg, exs = _tiny_cfg_and_examples()
    cfg = TrainConfig(preset="TINY", warmup=2, total_steps=50, ckpt_every_steps=100, eval_every=0, log_every=100, specaug=False)
    t = Trainer(str(tmp_path), cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
    b = collate(exs[:2])
    b.active = False
    t.train_step(b)
    assert all(p.grad is None or not p.grad.any() for p in t.model.parameters()) and t.step == 1

    rng = np.random.default_rng(0)
    dur = list(rng.uniform(1.0, 20.0, 48))
    order = sorted(range(48), key=lambda i: dur[i])
    groups = [order[i:i + 4] for i in range(0, 48, 4)]
    parts = [stratified_parts(groups, dur, 3, r, [7, 0, 1]) for r in range(3)]
    assert sorted(tuple(g) for p in parts for g in p) == sorted(tuple(g) for g in groups)
    with pytest.raises(DataError):
        stratified_parts(groups[:5], dur, 2, 0, [1])


# ---- D1 Item 3: data_rank / data_world chunk stratification ---------------------------------------
def test_chunk_stratification_by_data_rank_and_data_world():
    def _chunks(steps, stratify=False, data_rank=0, data_world=1):
        cfg = TrainConfig(chunk_modes="4,8,16", stratify_chunks=stratify, seed=1234)
        fake = SimpleNamespace(cfg=cfg, data_rank=data_rank, data_world=data_world, step=0)
        out = []
        for s in range(steps):
            fake.step = s
            out.append(Trainer._step_chunk(fake))
        return out

    base = _chunks(60)
    assert _chunks(60, stratify=True, data_rank=0, data_world=1) == base
    assert _chunks(60, stratify=True, data_rank=0, data_world=3) == base
    ranks = [_chunks(60, stratify=True, data_rank=r, data_world=3) for r in range(3)]
    assert all(sorted(col) == [4, 8, 16] for col in zip(*ranks, strict=True))
    two = [_chunks(60, stratify=True, data_rank=r, data_world=2) for r in range(2)]
    assert all(a != b for a, b in zip(*two, strict=True))


# ---- D1 Item 7: post-anchor-only ckpt/eval + exact round & anchor resume validation ---------------
def test_post_anchor_only_ckpt_eval_and_exact_round_anchor_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("DILOCO_AUTH_TOKEN", D1_TOKEN)
    mcfg, exs = _tiny_cfg_and_examples()
    seed_cfg = TrainConfig(preset="TINY", total_steps=20, specaug=False)
    seed_t = Trainer(str(tmp_path / "seed"), seed_cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
    init_payload = {"model": seed_t.model.state_dict(), "ema": seed_t.model.state_dict(),
                    "trainable_keys": [n for n, _ in seed_t.model.named_parameters()],
                    "model_config": mcfg.to_dict(), "tokenizer_sha256": None, "step": 0}
    hub = DiLoCoHub("D1_Item7", init_payload, tmp_path / "hub", expected_roster=("w0",), outer_lr=0.7, outer_momentum=0.9)
    url, stop = _serve_hub(hub)
    try:
        run_dir = str(tmp_path / "island0")
        cfg = TrainConfig(
            preset="TINY", warmup=1, total_steps=20, log_every=1, specaug=False,
            eval_every=2, weights_every=2, ckpt_every_steps=2, ckpt_every_minutes=1e9,
            data_rank=0, data_world=1, diloco_hub=url, diloco_worker="w0", diloco_every=3,
        )
        t = Trainer(run_dir, cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=True)
        eval_steps: list[int] = []
        t.evaluate = lambda: (eval_steps.append(t.step) or {"wer": 5.0})  # type: ignore[method-assign]

        # Run 5 steps (H=3 -> anchor commits ONLY at step 3; steps 2, 4, 5 are mid-round!).
        t.run(max_steps=5)
        assert eval_steps == [3], f"expected eval ONLY at post-anchor step 3, got {eval_steps}"
        assert sorted(p.name for p in (tmp_path / "island0").glob("ckpt-*.pt")) == ["ckpt-00000003.pt"]
        assert sorted(p.name for p in (tmp_path / "island0").glob("model-*.pt")) == ["model-00000003.pt"]
        ck = checkpoint.load(checkpoint.latest(run_dir))
        assert ck["diloco"]["round"] == 1 and ck["diloco"]["step"] == 3
        assert ck["diloco"]["anchor_sha256"] == hub.anchor_sha256
        assert ck["diloco"]["data_rank"] == 0 and ck["diloco"]["data_world"] == 1

        # 1. Valid resume against matching hub round=1 + anchor_sha256 advances cleanly to round=2 (step 6).
        t2 = Trainer(run_dir, cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=True)
        t2.evaluate = lambda: {"wer": 4.5}  # type: ignore[method-assign]
        assert t2.resume() and t2.step == 3
        t2.run(max_steps=3)
        assert t2.step == 6 and hub.round == 2 and t2._diloco_state["round"] == 2
        assert checkpoint.latest(run_dir).endswith("ckpt-00000006.pt")

        # 2. Refuses shard mismatch on resume (data_rank/data_world changed).
        bad_shard_cfg = dataclasses.replace(cfg, data_rank=1, data_world=2)
        t_bad_shard = Trainer(run_dir, bad_shard_cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
        with pytest.raises(CheckpointError, match="shard mismatch"):
            t_bad_shard.resume()

        # 3. Refuses diloco_every mismatch on resume.
        bad_h_cfg = dataclasses.replace(cfg, diloco_every=5)
        t_bad_h = Trainer(run_dir, bad_h_cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
        with pytest.raises(CheckpointError, match="diloco_every mismatch"):
            t_bad_h.resume()

        # 4. Refuses resume when hub round/anchor differs from checkpoint (e.g. fresh round-0 hub vs round-2 ckpt).
        hub_stale = DiLoCoHub("D1_Item7", init_payload, tmp_path / "hub2", expected_roster=("w0",))
        url2, stop2 = _serve_hub(hub_stale)
        try:
            t_mismatch = Trainer(run_dir, dataclasses.replace(cfg, diloco_hub=url2),
                                 FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
            assert t_mismatch.resume()
            with pytest.raises(CheckpointError, match="hub mismatch"):
                t_mismatch.run(max_steps=3)
        finally:
            stop2()

        # 5. Refuses mid-round checkpoint without 'diloco' metadata when DiLoCo is enabled.
        mid_dir = str(tmp_path / "mid")
        t_plain = Trainer(mid_dir, dataclasses.replace(cfg, diloco_hub="", diloco_every=0),
                          FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
        t_plain.run(max_steps=2)
        t_mid = Trainer(mid_dir, cfg, FixedBatches(exs, batch_size=2, shuffle=False), model_cfg=mcfg, device="cpu", log=False)
        with pytest.raises(CheckpointError, match="post-anchor checkpoint"):
            t_mid.resume()
    finally:
        stop()
