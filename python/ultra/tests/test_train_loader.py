"""Process-based loader parity: ParquetBatches(loader="process") yields exactly the batches of the
thread path (same order, bit-identical log-mel features, same token ids), with and without speed
perturbation, from a small synthetic parquet shard."""
import io
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.data import Featurizer, ParquetBatches, Position  # noqa: E402
from train.tokenizer import Tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    import sentencepiece as spm

    d = tmp_path_factory.mktemp("loader")
    words = ["hello there", "the quick brown fox", "general kenobi", "a b c d e f"]
    corpus = d / "t.txt"
    corpus.write_text("\n".join(words * 40))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(d / "tok"), vocab_size=30,
                                   model_type="bpe", character_coverage=1.0)
    rng = np.random.default_rng(0)
    rows = {"id": [], "text": [], "audio": []}
    for i in range(24):
        pcm = (rng.standard_normal(int(16000 * (0.5 + 0.1 * i))) * 3000).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm, 16000, format="FLAC")
        rows["id"].append(f"u{i}")
        rows["text"].append(words[i % 4].upper())
        rows["audio"].append({"bytes": buf.getvalue(), "path": f"u{i}.flac"})
    root = d / "parquet"
    (root / "train.clean.100").mkdir(parents=True)
    pq.write_table(pa.table(rows), root / "train.clean.100" / "0000.parquet", row_group_size=6)
    return str(root), str(d / "tok.model")


def _batches(root, tok, loader, speeds, n=6):
    t = Tokenizer(tok)
    src = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=7, budget_s=4.0, max_batch=8,
                         parquet_root=root, workers=2, speed_perturb=speeds, loader=loader)
    out = []
    for i, (pos, b) in enumerate(src.iterate(Position())):
        out.append((pos, b))
        if i + 1 >= n:
            break
    return out


@pytest.mark.parametrize("speeds", [(), (0.9, 1.0, 1.1)])
def test_process_loader_matches_thread_loader(shard, speeds):
    root, tok = shard
    a = _batches(root, tok, "thread", speeds)
    b = _batches(root, tok, "process", speeds)
    assert len(a) == len(b) == 6
    for (pa_, ba), (pb, bb) in zip(a, b, strict=True):
        assert pa_ == pb and ba.uids == bb.uids
        assert torch.equal(ba.feat_lens, bb.feat_lens) and torch.equal(ba.feats, bb.feats)
        assert torch.equal(ba.rnnt, bb.rnnt) and torch.equal(ba.ctc, bb.ctc)


def test_unknown_loader_refused(shard):
    root, tok = shard
    t = Tokenizer(tok)
    with pytest.raises(Exception, match="unknown loader"):
        ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=1, parquet_root=root,
                       loader="magic")


def test_producer_failure_surfaces_as_data_error(shard, monkeypatch):
    from train.errors import DataError

    root, tok = shard
    t = Tokenizer(tok)
    src = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=1, parquet_root=root,
                         workers=1)

    def boom(epoch, window):
        raise OSError("disk went away")

    monkeypatch.setattr(src, "_load_window", boom)
    with pytest.raises(DataError, match="disk went away"):
        next(iter(src.iterate(Position())))


def test_window_width_change_waits_for_the_epoch_boundary(shard):
    """A resume with a new window width keeps the old width until the epoch ends: the units of
    the current epoch are exactly those of an uninterrupted run, and the next epoch uses the new
    width."""
    root, tok = shard
    t = Tokenizer(tok)
    old = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=3, parquet_root=root,
                         window_groups=1, workers=1)
    new = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=3, parquet_root=root,
                         window_groups=4, workers=1)
    new.carry_width = (1, 0)
    assert [new._window_units(0, w) for w in range(new._windows(0))] == \
        [old._window_units(0, w) for w in range(old._windows(0))]
    assert new._windows(1) == 1 and old._windows(1) == 4
    flat = sorted(u for w in range(new._windows(1)) for u in new._window_units(1, w))
    assert flat == sorted(new.units)  # epoch 1 covers every unit once at the new width


@pytest.mark.parametrize("speeds", [(), (0.9, 1.0, 1.1)])
def test_sharded_windows_partition_the_single_process_batches(shard, speeds):
    """world=2: rank 0 and rank 1 together get exactly the single-process groups (header-only
    durations pack identically), alternating; an odd tail goes to rank 0 and rank 1 repeats the
    window's first group, so nothing is dropped and both ranks have the same batch count."""
    root, tok = shard
    t = Tokenizer(tok)

    def src(rank, world):
        s = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=7, budget_s=4.0,
                           max_batch=8, parquet_root=root, workers=1, speed_perturb=speeds,
                           window_groups=2)
        s.rank, s.world = rank, world
        return s

    one, r0, r1 = src(0, 1), src(0, 2), src(1, 2)
    for w in range(one._windows(0)):
        full = [[e.uid for e in g] for g in one._load_window(0, w)]
        a = [[e.uid for e in g] for g in r0._load_window(0, w)]
        b = [[e.uid for e in g] for g in r1._load_window(0, w)]
        padded = full + full[: len(full) % 2]
        assert a == padded[0::2] and b == padded[1::2] and len(a) == len(b)
        feats_one = {e.uid: e.feats for g in one._load_window(0, w) for e in g}
        for g in r0._load_window(0, w):
            for e in g:
                assert torch.equal(e.feats, feats_one[e.uid])


def test_weighted_share_deals_seconds_by_weight_and_partitions_the_group():
    from train.data import weighted_share

    dur = [1.0 + 0.1 * i for i in range(23)]
    group = list(range(23))
    for world in (2, 3, 4):  # equal weights: disjoint, complete, and even in seconds
        parts = [weighted_share(group, dur, (1.0,) * world, r) for r in range(world)]
        assert sorted(i for p in parts for i in p) == group
        secs = [sum(dur[i] for i in p) for p in parts]
        assert max(secs) - min(secs) <= max(dur)
    w = (1.0, 0.5, 1.0)
    parts = [weighted_share(group, dur, w, r) for r in range(3)]
    assert sorted(i for p in parts for i in p) == group  # disjoint and complete
    secs = [sum(dur[i] for i in p) for p in parts]
    for r in range(3):
        assert abs(secs[r] / sum(secs) - w[r] / sum(w)) < 0.06
    assert max(dur[i] for i in parts[1]) > 2.5  # the slow rank still spans the duration range


def test_weighted_shards_partition_the_single_process_batches(shard):
    """world=2 with rank weights 1 : 0.5: the two ranks' parts of every global batch are disjoint
    and together are exactly the single-process batch; rank 0 carries about two thirds."""
    root, tok = shard
    t = Tokenizer(tok)

    def src(rank, world):
        s = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=7, budget_s=8.0,
                           max_batch=16, parquet_root=root, workers=1, window_groups=2)
        s.rank, s.world = rank, world
        if world > 1:
            s.ddp_global_batches, s.rank_weights = True, (1.0, 0.5)
        return s

    one, r0, r1 = src(0, 1), src(0, 2), src(1, 2)
    n0 = n1 = 0
    for w in range(one._windows(0)):
        full = [sorted(e.uid for e in g) for g in one._load_window(0, w)]
        a, b = r0._load_window(0, w), r1._load_window(0, w)
        assert len(a) == len(b) == len(full)
        for g, (ea, act_a), (eb, act_b) in zip(full, a, b, strict=True):
            ua = [e.uid for e in ea] if act_a else []
            ub = [e.uid for e in eb] if act_b else []
            assert not set(ua) & set(ub) and sorted(ua + ub) == g
            n0, n1 = n0 + len(ua), n1 + len(ub)
    assert n0 > n1 > 0

