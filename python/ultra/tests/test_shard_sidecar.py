"""H55 S4 data-loader tests: per-row FLAC frame-count sidecar parity and Option (iv) node-local
disjoint shard partitioning (disjointness, prestaged parity, statistical balance, lockstep resume)."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.data import DataError, Featurizer, ParquetBatches, Position  # noqa: E402
from train.shard_sidecar import (  # noqa: E402
    build_dur_table,
    parse_node_shards,
    partition_items,
    plan_node_manifests,
)
from train.tokenizer import Tokenizer  # noqa: E402


def _write_shard(path: str, uids: list[str], lengths_s: list[float], *, conf: list[float] | None = None,
                 row_group_size: int = 4) -> None:
    rng = np.random.default_rng(abs(hash(os.path.basename(path))) % (2**32))
    rows = []
    for i, (u, sec) in enumerate(zip(uids, lengths_s, strict=True)):
        pcm = (rng.standard_normal(int(16000 * sec)) * 2500).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm, 16000, format="FLAC")
        r = {"id": u, "text": "THE QUICK BROWN FOX", "audio": {"bytes": buf.getvalue(), "path": f"{u}.flac"}}
        if conf is not None:
            r["conf"] = float(conf[i])
        rows.append(r)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    import sentencepiece as spm

    d = tmp_path_factory.mktemp("h55_sidecar")
    corpus_txt = d / "tok.txt"
    corpus_txt.write_text("\n".join(["the quick brown fox", "hello world", "general kenobi"] * 40))
    spm.SentencePieceTrainer.train(input=str(corpus_txt), model_prefix=str(d / "tok"), vocab_size=30,
                                   model_type="bpe", character_coverage=1.0)
    ls_dir = d / "ls" / "train.clean.100"
    ll_dir = d / "ll_extra"
    ls_dir.mkdir(parents=True)
    ll_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for si in range(4):
        uids = [f"ls_{si}_{j}" for j in range(12)]
        lens = [float(x) for x in rng.uniform(0.5, 2.2, size=12)]
        _write_shard(str(ls_dir / f"{si:04d}.parquet"), uids, lens, row_group_size=4)
    for si in range(12):
        uids = [f"ll_{si}_{j}" for j in range(8)]
        lens = [float(x) for x in rng.uniform(0.6, 2.6, size=8)]
        confs = [0.2 if j % 4 == 0 else 0.9 for j in range(8)]
        _write_shard(str(ll_dir / f"{si:04d}.parquet"), uids, lens, conf=confs, row_group_size=4)
    tbl_ls = str(d / "dur_ls.jsonl")
    tbl_ll = str(d / "dur_ll.jsonl")
    r1 = build_dur_table([str(ls_dir)], tbl_ls, workers=2)
    r2 = build_dur_table([str(ll_dir)], tbl_ll, workers=2)
    assert r1["files"] == 4 and r1["scanned"] == 4
    assert r2["files"] == 12 and r2["scanned"] == 12
    # Incremental rebuild scans 0 files.
    assert build_dur_table([str(ls_dir)], tbl_ls, workers=1)["scanned"] == 0
    return str(d / "ls"), str(ll_dir), str(d / "tok.model"), tbl_ls, tbl_ll, d


def _collect(src: ParquetBatches, n: int, start: Position = Position()):
    out = []
    for pos, b in src.iterate(start):
        out.append((pos, b))
        if len(out) >= n:
            break
    return out


@pytest.mark.parametrize("world,ddp_global,speeds,pad_f", [
    (1, False, (), 0),
    (1, False, (0.9, 1.0, 1.1), 4),
    (2, True, (0.9, 1.0, 1.1), 0),
    (3, True, (0.9, 1.0, 1.1), 4),
    (2, False, (0.9, 1.0, 1.1), 0),
])
def test_sidecar_packing_bit_identical_to_header_path(corpus, monkeypatch, world, ddp_global, speeds, pad_f):
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)

    def make(dur_tables: str, rank: int):
        s = ParquetBatches(
            ["train-clean-100"], Featurizer(t, t), seed=7919, budget_s=6.0, max_batch=12, max_seconds=2.4,
            window_groups=6, parquet_root=ls_root, workers=1, speed_perturb=speeds, pad_frames=pad_f,
            extra=[(ll_dir, None)], base_weight=0.5, dur_tables=dur_tables,
        )
        s.rank, s.world, s.ddp_global_batches = rank, world, ddp_global
        return s

    for rank in range(world):
        ref = _collect(make("", rank), 8)
        # Prove sidecar path NEVER touches sf.info() during packing.
        monkeypatch.setattr(sf, "info", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("sf.info called!")))
        got = _collect(make(f"{tbl_ls},{tbl_ll}", rank), 8)
        monkeypatch.undo()
        assert len(got) == len(ref) == 8
        for (p_ref, b_ref), (p_got, b_got) in zip(ref, got, strict=True):
            assert p_got == p_ref
            assert b_got.uids == b_ref.uids
            assert b_got.active == b_ref.active
            assert torch.equal(b_got.feat_lens, b_ref.feat_lens)
            assert torch.equal(b_got.feats, b_ref.feats)
            assert torch.equal(b_got.rnnt, b_ref.rnnt) and torch.equal(b_got.ctc, b_ref.ctc)


def test_sidecar_conf_filter_and_relpath_fallback(corpus, tmp_path):
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)
    # Build a sidecar with conf column and synthetic moved paths to exercise (parent, basename) fallback.
    tbl_conf = str(tmp_path / "dur_ll_conf.jsonl")
    build_dur_table([ll_dir], tbl_conf, workers=1, include_conf=True)
    moved_tbl = tmp_path / "moved_ll.jsonl"
    lines = []
    for line in open(tbl_conf, encoding="utf-8"):
        rec = json.loads(line)
        rec["file"] = f"/other/node/mount/{os.path.basename(os.path.dirname(rec['file']))}/{os.path.basename(rec['file'])}"
        lines.append(json.dumps(rec))
    moved_tbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    s_ref = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=9, budget_s=5.0, max_batch=8,
                           window_groups=4, parquet_root=ls_root, workers=1, extra=[(ll_dir, None)],
                           min_conf=0.5, base_weight=0.5)
    s_ref.rank, s_ref.world, s_ref.ddp_global_batches = 0, 2, True
    s_got = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=9, budget_s=5.0, max_batch=8,
                           window_groups=4, parquet_root=ls_root, workers=1, extra=[(ll_dir, None)],
                           min_conf=0.5, base_weight=0.5, dur_tables=f"{tbl_ls},{moved_tbl}")
    s_got.rank, s_got.world, s_got.ddp_global_batches = 0, 2, True
    # Also test 2-column sidecar (no conf in sidecar -> reads only conf column, never audio bytes).
    s_got2 = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=9, budget_s=5.0, max_batch=8,
                            window_groups=4, parquet_root=ls_root, workers=1, extra=[(ll_dir, None)],
                            min_conf=0.5, base_weight=0.5, dur_tables=f"{tbl_ls},{tbl_ll}")
    s_got2.rank, s_got2.world, s_got2.ddp_global_batches = 0, 2, True

    ref, got, got2 = _collect(s_ref, 6), _collect(s_got, 6), _collect(s_got2, 6)
    for (p0, b0), (p1, b1), (p2, b2) in zip(ref, got, got2, strict=True):
        assert p0 == p1 == p2 and b0.uids == b1.uids == b2.uids
        assert torch.equal(b0.feats, b1.feats) and torch.equal(b0.feats, b2.feats)
    # Deterministic window-exact dropped_conf check (no async prefetch timing variance).
    for s in (s_ref, s_got, s_got2):
        s.dropped_conf = 0
        s._load_window(0, 0)
    assert s_got.dropped_conf == s_ref.dropped_conf == s_got2.dropped_conf and s_ref.dropped_conf > 0


def test_sidecar_fails_loud_on_missing_or_tampered(corpus, tmp_path):
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)
    with pytest.raises(DataError, match="lacks"):
        ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=1, parquet_root=ls_root,
                       workers=1, extra=[(ll_dir, None)], dur_tables=tbl_ls)
    # Tamper a uid in tbl_ls -> DataError on load.
    bad = tmp_path / "bad_ls.jsonl"
    recs = [json.loads(line) for line in open(tbl_ls, encoding="utf-8")]
    recs[0]["groups"][0][0][0] = "TAMPERED_UID"
    bad.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    src = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=1, budget_s=5.0, max_batch=8,
                         parquet_root=ls_root, workers=1, dur_tables=str(bad))
    with pytest.raises(DataError, match="uid mismatch"):
        for w in range(src._windows(0)):
            src._load_window(0, w)


def test_node_shards_disjoint_uniform_and_gpu_weighted_and_prestaged_parity(corpus, tmp_path):
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)
    # 1. Disjointness & completeness for uniform 1/4 and GPU-weighted 2:2:1:1.
    for weights, expected_counts in [(None, [3, 3, 3, 3]), ((2, 2, 1, 1), [4, 4, 2, 2])]:
        plan = plan_node_manifests([ll_dir], 4, weights=weights, seed=5504)
        sets = [set(n[ll_dir]) for n in plan["nodes"]]
        assert [len(s) for s in sets] == expected_counts
        assert sum(len(s) for s in sets) == 12
        assert len(set.union(*sets)) == 12
        for i in range(4):
            for j in range(i + 1, 4):
                assert not (sets[i] & sets[j])

    # 2. Online partitioning (prestaged=0) == pre-staged local disk (prestaged=1) bit-identically.
    plan = plan_node_manifests([ll_dir], 4, weights=(2, 2, 1, 1), seed=5504)
    node0_dir = tmp_path / "node0_ll"
    node0_dir.mkdir()
    for fname in plan["nodes"][0][ll_dir]:
        shutil.copy2(os.path.join(ll_dir, fname), node0_dir / fname)

    s_online = ParquetBatches(
        ["train-clean-100"], Featurizer(t, t), seed=7919, budget_s=12.0, max_batch=24, window_groups=12,
        parquet_root=ls_root, workers=1, extra=[(ll_dir, None)], base_weight=0.5,
        dur_tables=f"{tbl_ls},{tbl_ll}", node_shards="node=0/4,local=1/2,weights=2:2:1:1,prestaged=0,seed=5504",
    )
    s_online.rank, s_online.world, s_online.ddp_global_batches = 1, 6, True

    # Pre-staged dir only has node 0's 4 shards; sidecar lookup matches by (parent, basename) or build fresh.
    tbl_n0 = str(tmp_path / "dur_n0.jsonl")
    build_dur_table([str(node0_dir)], tbl_n0, workers=1)
    s_staged = ParquetBatches(
        ["train-clean-100"], Featurizer(t, t), seed=7919, budget_s=12.0, max_batch=24, window_groups=12,
        parquet_root=ls_root, workers=1, extra=[(str(node0_dir), None)], base_weight=0.5,
        dur_tables=f"{tbl_ls},{tbl_n0}", node_shards="node=0/4,local=1/2,weights=2:2:1:1,prestaged=1,seed=5504",
    )
    s_staged.rank, s_staged.world, s_staged.ddp_global_batches = 1, 6, True

    a, b = _collect(s_online, 8), _collect(s_staged, 8)
    for (pa_, ba), (pb, bb) in zip(a, b, strict=True):
        assert pa_ == pb and ba.uids == bb.uids and torch.equal(ba.feats, bb.feats)


def test_node_shards_lockstep_resume_and_intra_node_disjointness(corpus):
    """Across heterogeneous 4-node (2,2,1,1) DDP world=6: all ranks share identical Position(0,0,step),
    intra-node GPUs are disjoint, cross-node extra shards are disjoint, and mid-run resume is bit-identical."""
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)
    topo = [(0, 0, 2, 0), (0, 1, 2, 1), (1, 0, 2, 2), (1, 1, 2, 3), (2, 0, 1, 4), (3, 0, 1, 5)]

    def make_rank(nr, lr, lw, gr):
        s = ParquetBatches(
            ["train-clean-100"], Featurizer(t, t), seed=7919, budget_s=12.0, max_batch=24, window_groups=12,
            parquet_root=ls_root, workers=1, extra=[(ll_dir, None)], base_weight=0.5,
            dur_tables=f"{tbl_ls},{tbl_ll}", node_shards=f"node={nr}/4,local={lr}/{lw},weights=2:2:1:1,seed=5504",
        )
        s.rank, s.world, s.ddp_global_batches = gr, 6, True
        return s

    streams = [(_collect(make_rank(nr, lr, lw, gr), 10), nr, lr) for nr, lr, lw, gr in topo]
    # Every rank has identical lockstep Position(0, 0, step) at every step.
    for step in range(10):
        positions = {st[step][0] for st, _, _ in streams}
        assert positions == {Position(0, 0, step + 1)}
    # Intra-node GPUs on node 0 receive disjoint active utterances at every step.
    n0_r0, n0_r1 = streams[0][0], streams[1][0]
    for step in range(10):
        u0 = set(n0_r0[step][1].uids) if n0_r0[step][1].active else set()
        u1 = set(n0_r1[step][1].uids) if n0_r1[step][1].active else set()
        assert not (u0 & u1)
    # Cross-node Libri-Light UIDs are completely disjoint across all 4 nodes.
    ll_by_node = [set() for _ in range(4)]
    for st, nr, _ in streams:
        for _, b in st:
            ll_by_node[nr].update(u for u in b.uids if u.startswith("ll_"))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (ll_by_node[i] & ll_by_node[j])

    # Resume from step 5 produces bit-identical steps 6..10 and exact exposure accounting.
    for nr, lr, lw, gr in [(0, 0, 2, 0), (2, 0, 1, 4)]:
        full = _collect(make_rank(nr, lr, lw, gr), 10)
        resumed = _collect(make_rank(nr, lr, lw, gr), 5, start=full[4][0])
        for (p_exp, b_exp), (p_got, b_got) in zip(full[5:], resumed, strict=True):
            assert p_got == p_exp and b_got.uids == b_exp.uids
            assert abs(b_got.audio_seconds - b_exp.audio_seconds) < 1e-6
            assert torch.equal(b_got.feats, b_exp.feats)


def test_parse_node_shards_validation():
    assert parse_node_shards("") is None
    p = parse_node_shards("node=2/4,local=0/1,weights=2:2:1:1")
    assert p.node_rank == 2 and p.node_world == 4 and abs(p.share - 1.0 / 6.0) < 1e-9
    p2 = parse_node_shards("1/4")
    assert p2.node_rank == 1 and p2.node_world == 4 and p2.weights == (1.0, 1.0, 1.0, 1.0)
    assert partition_items(list(range(10)), 1) == [list(range(10))]
    for bad in ("node=4/4", "node=0/2,weights=1:2:3", "node=0/2,weights=1:0", "badtoken"):
        with pytest.raises(DataError):
            parse_node_shards(bad)
