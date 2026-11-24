"""Sealed label-only preflight and an interrupted real k2/DDP restart."""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_shard_sidecar import _write_shard
from train import checkpoint
from train.asr import PRESETS
from train.data import Featurizer, ParquetBatches
from train.data_identity import compare
from train.errors import CheckpointError, DataError
from train.label_map import LabelMap
from train.tokenizer import Tokenizer
from train.trainer import TrainConfig, Trainer, main, model_config_for


def _map(root, name, rows):
    path = root / (name + ".tsv")
    path.write_text("".join(f"{uid}\t{text}\n" for uid, text in rows.items()))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sources(tmp_path):
    import sentencepiece as spm

    book = tmp_path / "fixture.txt"
    book.write_text("the quick brown fox\nthe small horse\n" * 12)
    spm.SentencePieceTrainer.train(input=str(book), model_prefix=str(tmp_path / "tok"),
                                   vocab_size=28, model_type="bpe", num_threads=1)
    base = tmp_path / "base" / "train.clean.100"
    extra = tmp_path / "extra"
    base.mkdir(parents=True)
    extra.mkdir()
    _write_shard(str(base / "0.parquet"), ["ls-0", "ls-1"], [0.9, 1.1], row_group_size=2)
    shard = extra / "0.parquet"
    _write_shard(str(shard), ["ll-0", "ll-1"], [0.9, 1.1], row_group_size=2)
    table = pq.read_table(shard)
    table = table.append_column("src", pa.array(["small/8/book/rec"] * 2))
    table = table.append_column("t0", pa.array([0.0, 1.0])).append_column("t1", pa.array([0.9, 2.1]))
    pq.write_table(table, shard)
    _map(tmp_path, "r0", {"ll-0": "THE QUICK BROWN FOX", "ll-1": "THE QUICK BROWN FOX"})
    _map(tmp_path, "book", {"ll-0": "THE SMALL HORSE", "ll-1": "THE QUICK BROWN FOX"})
    return tmp_path


def _trainer(root, name, *, map_name="r0", rank=0, world=1):
    tok = Tokenizer(str(root / "tok.model"))
    path = root / (map_name + ".tsv")
    labels = LabelMap.load(path, hashlib.sha256(path.read_bytes()).hexdigest(), only=True)
    source = ParquetBatches(["train-clean-100"], Featurizer(tok, tok), seed=7919,
                            budget_s=5.0, max_batch=4, window_groups=2, workers=1,
                            speed_perturb=(0.9, 1.0, 1.1), parquet_root=str(root / "base"),
                            extra=[(str(root / "extra"), None)], label_map=labels)
    source.rank, source.world, source.ddp_global_batches = rank, world, True
    cfg = TrainConfig(preset="TINY", seed=7919, seal_data_identity=True, ddp_native=world > 1,
                      ema_decay=0.8, specaug_start=0, loss_trunc_q=0.8, loss_trunc_mode="cauchy",
                      inter_ctc_weight=0.2, rnnt_loss="pruned", eval_every=0, eval_each_epoch=False,
                      ckpt_every_steps=4)
    tiny = PRESETS["TINY"]
    encoder = replace(tiny.encoder, family="unet3", chunk=4, dropout=0.1,
                      lower_blocks=1, middle_blocks=1, deep_blocks=1, up_blocks=1, final_blocks=1)
    model_cfg = replace(tiny, encoder=encoder, inter_ctc=True,
                        rnnt_loss="pruned", vocab_rnnt=tok.num_outputs, vocab_ctc=tok.num_outputs)
    return Trainer(str(root / name), cfg, source, model_cfg, tokenizer=tok, device="cpu",
                   rank=rank, world=world, local_rank=rank, log=False)


def test_preflight_uses_normal_cli_and_rejects_different_audio(sources, monkeypatch):
    torch.set_num_threads(1)
    from ultra import libri

    monkeypatch.setattr(libri, "PARQUET", str(sources / "base"))
    tok = Tokenizer(str(sources / "tok.model"))
    cfg = TrainConfig(preset="TINY", rnnt_pieces=tok.vocab, ctc_pieces=tok.vocab)
    from train.asr import ASRModel

    model_cfg = model_config_for(cfg)
    seed = checkpoint.save_weights(str(sources), 1, {
        "model": ASRModel(model_cfg).state_dict(), "model_config": model_cfg.to_dict(),
        "tokenizer_sha256": tok.sha256,
    })
    seed_hash = hashlib.sha256(Path(seed).read_bytes()).hexdigest()

    def preflight(arm, labels):
        path = sources / (labels + ".tsv")
        return main([
            "--run-dir", str(sources / arm), "--preset", "TINY", "--device", "cpu",
            "--preflight-only", "--init-from", seed, "--init-sha256", seed_hash, "--set",
            f"rnnt_pieces={tok.vocab}", f"ctc_pieces={tok.vocab}", f"tokenizer={tok.path}",
            f"extra_splits={sources / 'extra'}", f"label_map={path}",
            f"label_map_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}",
            "seal_data_identity=true", "label_map_only=1",
        ])

    assert preflight("A", "r0") == preflight("B", "book") == 0
    receipt = compare(sources / "A", sources / "B")
    assert receipt["world"] == 1 and receipt["initial_checkpoint_sha256"] == seed_hash
    assert len(set(receipt["label_map_sha256"])) == 2
    assert checkpoint.latest(str(sources / "A")) is None
    shard = sources / "extra" / "0.parquet"
    rows = pq.read_table(shard)
    pq.write_table(rows.set_column(rows.schema.get_field_index("t1"), "t1", pa.array([0.8, 2.1])), shard)
    assert preflight("C", "book") == 0
    with pytest.raises(CheckpointError, match="ordered audio"):
        compare(sources / "A", sources / "C")
    _map(sources, "missing", {"ll-0": "THE FOX", "unknown": "THE HORSE"})
    with pytest.raises(DataError, match="absent from the eligible"):
        preflight("D", "missing")


def _train_steps(tr, n):
    rows = []
    step = tr.train_step

    def trace(batch):
        # The first step has unequal support. The next step has a collective-only rank.
        if tr.world > 1 and tr.rank == 1 and tr.step <= 1:
            batch.row_keep = torch.arange(len(batch.uids)) < (1 - tr.step)
        r = step(batch)
        assert "rnnt_simple" in r and "inter_ctc" in r
        selected = batch.supervised()
        rows.append((selected.uids, selected.audio_seconds, r))
        return r

    tr.train_step = trace
    try:
        tr.run(max_steps=n)
    finally:
        tr.train_step = step
    assert len(rows) == n
    return rows


def _ddp_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "gloo").as_uri(), rank=rank, world_size=2)
    try:
        a = _trainer(root, "uninterrupted", rank=rank, world=2)
        direct = _train_steps(a, 7)
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(4))
        b = _trainer(root, "interrupted", rank=rank, world=2)
        head = _train_steps(b, 4)
        dist.barrier()
        c = _trainer(root, "interrupted", rank=rank, world=2)
        assert c.resume() and c.pos == b.pos
        assert c.loss_trunc.history == b.loss_trunc.history
        tail = _train_steps(c, 3)
        assert head + tail == direct
        assert c.loss_trunc.history == a.loss_trunc.history
        assert (random.random(), float(np.random.rand())) == expected_rng[:2]
        assert torch.equal(torch.rand(4), expected_rng[2])
        for name, weight in a.model.state_dict().items():
            assert torch.equal(weight, c.model.state_dict()[name]), name
            assert torch.equal(a.ema[name], c.ema[name]), name
        for (_, p), (_, q) in zip(a.model.named_parameters(), c.model.named_parameters(), strict=True):
            if p.grad is not None:
                assert torch.equal(p.grad, q.grad)
        assert a.sched.state_dict() == c.sched.state_dict()
        for key, state in a.opt.state_dict()["state"].items():
            for field, value in state.items():
                assert torch.equal(value, c.opt.state_dict()["state"][key][field])
    finally:
        dist.destroy_process_group()


def test_pruned_uneven_rank_active_helper_resume(sources):
    pytest.importorskip("k2")
    mp.spawn(_ddp_worker, args=(sources,), nprocs=2, join=True)
    payload = checkpoint.load(checkpoint.latest(str(sources / "interrupted")), restore_rng=False)
    assert [r["layout"][0] for r in payload["objective_state"]["ranks"]] == [0, 1]
    assert payload["objective_state"]["ranks"][0]["truncation"]["history"] != (
        payload["objective_state"]["ranks"][1]["truncation"]["history"])


def test_sealed_resume_rejects_label_change_and_skips_globally_empty(sources):
    pytest.importorskip("k2")
    torch.set_num_threads(1)
    t = _trainer(sources, "resume")
    batch = next(iter(t._batches()))[1]
    batch.row_keep = torch.zeros(len(batch.uids), dtype=torch.bool)
    assert t.train_step(batch) == {"optimized": False}
    assert t.step == 0 and t.ema is None and not t.opt.state
    _train_steps(t, 1)
    with pytest.raises(CheckpointError, match="sealed data identity changed"):
        _trainer(sources, "resume", map_name="book").resume()
    t2 = _trainer(sources, "resume")
    assert t2.resume() and t2.step == 1
