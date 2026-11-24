"""train/label_map.py: sealed transcript override for extra splits (H53 Amendment 12 PL3 arm).
Default OFF; an identity map is bit-identical to no map; `only` keeps exactly the mapped extra rows
on both packing paths; two maps over the same uids differ only in their text targets."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_shard_sidecar import _collect, corpus  # noqa: E402,F401  (module-scoped fixture)
from train.data import DataError, Featurizer, ParquetBatches  # noqa: E402
from train.label_map import LabelMap  # noqa: E402
from train.tokenizer import Tokenizer  # noqa: E402

LL_UIDS = [f"ll_{si}_{j}" for si in range(12) for j in range(8)]


def _tsv(path, pairs) -> tuple[str, str]:
    data = "".join(f"{u}\t{t}\n" for u, t in pairs).encode()
    path.write_bytes(data)
    return str(path), hashlib.sha256(data).hexdigest()


def _src(corpus, dur: bool, label_map=None, world=1):
    ls_root, ll_dir, tok, tbl_ls, tbl_ll, _ = corpus
    t = Tokenizer(tok)
    s = ParquetBatches(["train-clean-100"], Featurizer(t, t), seed=7919, budget_s=6.0, max_batch=12,
                       max_seconds=2.4, window_groups=6, parquet_root=ls_root, workers=1,
                       extra=[(ll_dir, None)], base_weight=0.5,
                       dur_tables=f"{tbl_ls},{tbl_ll}" if dur else "", label_map=label_map)
    s.rank, s.world, s.ddp_global_batches = 0, world, world > 1
    return s


def test_load_validates_digest_and_lines(tmp_path):
    p, h = _tsv(tmp_path / "ok.tsv", [("a", "HELLO"), ("b", "WORLD")])
    m = LabelMap.load(p, h, only=True)
    assert m.texts == {"a": "HELLO", "b": "WORLD"} and m.keep("a") and not m.keep("z")
    with pytest.raises(DataError, match="sha256"):
        LabelMap.load(p, "0" * 64)
    with pytest.raises(DataError, match="needs label_map_sha256"):
        LabelMap.load(p, "")
    p2, h2 = _tsv(tmp_path / "dup.tsv", [("a", "X"), ("a", "Y")])
    with pytest.raises(DataError, match="duplicate"):
        LabelMap.load(p2, h2)
    p3, h3 = _tsv(tmp_path / "empty.tsv", [("a", " ")])
    with pytest.raises(DataError, match="non-empty"):
        LabelMap.load(p3, h3)


@pytest.mark.parametrize("dur", [False, True])
def test_identity_map_is_bit_identical_to_off(corpus, tmp_path, dur):
    p, h = _tsv(tmp_path / "id.tsv", [(u, "THE QUICK BROWN FOX") for u in LL_UIDS])
    ref = _collect(_src(corpus, dur), 8)
    got = _collect(_src(corpus, dur, LabelMap.load(p, h, only=False)), 8)
    for (pa_, ba), (pb, bb) in zip(ref, got, strict=True):
        assert pa_ == pb and ba.uids == bb.uids and ba.texts == bb.texts
        assert torch.equal(ba.feats, bb.feats) and torch.equal(ba.rnnt, bb.rnnt) and torch.equal(ba.ctc, bb.ctc)


def test_only_keeps_mapped_extra_rows_on_both_paths(corpus, tmp_path):
    keep = set(LL_UIDS[::3])
    p, h = _tsv(tmp_path / "sub.tsv", [(u, "THE QUICK BROWN FOX") for u in sorted(keep)])
    hdr = _collect(_src(corpus, False, LabelMap.load(p, h, only=True)), 8)
    side = _collect(_src(corpus, True, LabelMap.load(p, h, only=True)), 8)
    seen_ls = 0
    for (pa_, ba), (pb, bb) in zip(hdr, side, strict=True):
        assert pa_ == pb and ba.uids == bb.uids and torch.equal(ba.feats, bb.feats)
        for u in ba.uids:
            assert u.startswith("ls_") or u in keep  # LibriSpeech rows are never filtered
            seen_ls += u.startswith("ls_")
    assert seen_ls > 0


def test_two_arms_differ_only_in_text(corpus, tmp_path):
    keep = sorted(LL_UIDS[::2])
    pa_, ha = _tsv(tmp_path / "inc.tsv", [(u, "THE QUICK BROWN FOX") for u in keep])
    pb, hb = _tsv(tmp_path / "pl3.tsv", [(u, "HELLO WORLD") for u in keep])
    a = _collect(_src(corpus, True, LabelMap.load(pa_, ha, only=True)), 8)
    b = _collect(_src(corpus, True, LabelMap.load(pb, hb, only=True)), 8)
    changed = 0
    for (qa, ba), (qb, bb) in zip(a, b, strict=True):
        assert qa == qb and ba.uids == bb.uids and torch.equal(ba.feats, bb.feats)
        for u, ta, tb in zip(ba.uids, ba.texts, bb.texts, strict=True):
            if u.startswith("ll_"):
                assert (ta, tb) == ("THE QUICK BROWN FOX", "HELLO WORLD")
                changed += 1
            else:
                assert ta == tb
    assert changed > 0
