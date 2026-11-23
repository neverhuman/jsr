"""H52 data pilot: an extra training split of parquet shards with pseudo-label text (Libri-Light with R0
labels), given as an absolute directory with an optional mixing weight and a `conf` column filter."""
import io
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.data import DataError, ParquetBatches, parse_extra_splits  # noqa: E402


def _shard(path, uids, conf=None, rows_per_group=2):
    import soundfile as sf

    rows = []
    for i, u in enumerate(uids):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600, np.int16), 16000, format="FLAC")
        row = {"id": u, "speaker_id": 1, "text": "HELLO", "audio": {"bytes": buf.getvalue(), "path": None}}
        if conf is not None:
            row["conf"] = float(conf[i])
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=rows_per_group)


def _tree(tmp_path, n_base=12, n_extra=4, conf=None):
    base = tmp_path / "ls" / "train.clean.100"
    extra = tmp_path / "ll_small"
    base.mkdir(parents=True)
    extra.mkdir()
    _shard(str(base / "0000.parquet"), [f"b{i}" for i in range(n_base)])
    _shard(str(extra / "0000.parquet"), [f"x{i}" for i in range(n_extra)], conf=conf)
    return str(tmp_path / "ls"), str(extra)


def test_parse_extra_splits():
    assert parse_extra_splits("") == []
    assert parse_extra_splits("/d/a@0.3,/d/b") == [("/d/a", 0.3), ("/d/b", None)]
    for bad in ("rel/dir", "/d/a@1.0", "/d/a@0", "/d/a@0.6,/d/b@0.5"):
        with pytest.raises(ValueError):
            parse_extra_splits(bad)


def test_extra_split_joins_at_its_natural_share(tmp_path):
    root, extra = _tree(tmp_path)
    src = ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=root, extra=[(extra, None)])
    assert src.manifest == ["train.clean.100/0000.parquet", "ll_small/0000.parquet"]
    assert len(src._epoch_units(0)) == 6 + 2  # 12 base rows and 4 extra rows, two per row group


def test_mixing_weight_sets_the_extra_share_deterministically(tmp_path):
    root, extra = _tree(tmp_path, n_base=40, n_extra=4)
    src = ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=root, extra=[(extra, 0.5)])
    units = src._epoch_units(3)
    n_extra = sum(1 for fi, _ in units if fi == 1)
    assert n_extra == 20 and len(units) == 40  # 20 base groups, 20 extra draws (2 groups, repeated)
    assert src._epoch_units(3) == units  # same epoch, same list
    assert src._windows(3) == (40 + src.window_groups - 1) // src.window_groups


def test_conf_filter_drops_low_confidence_extra_rows_only(tmp_path):
    root, extra = _tree(tmp_path, conf=[0.9, 0.2, 0.95, 0.5])
    src = ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=root, extra=[(extra, None)],
                         min_conf=0.6)
    rows = src._read_rows(1, 0) + src._read_rows(1, 1)
    assert [r["id"] for r in rows] == ["x0", "x2"] and src.dropped_conf == 2
    assert len(src._read_rows(0, 0)) == 2  # LibriSpeech rows have no conf column and are kept
    root2, extra2 = _tree(tmp_path / "noconf")
    with pytest.raises(DataError):
        ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=root2, extra=[(extra2, None)],
                       min_conf=0.5)


def test_base_weight_sets_the_librispeech_share_and_splits_the_rest_by_size(tmp_path):
    """h52's gate arm G (LS ~50% of exposure) and h53's S3 (LS 3 : LL 1): base_weight fixes the base splits'
    share of every epoch; the extra splits share the rest in proportion to their row groups."""
    base = tmp_path / "ls" / "train.clean.100"
    base.mkdir(parents=True)
    _shard(str(base / "0000.parquet"), [f"b{i}" for i in range(40)])  # 20 groups
    small, medium = tmp_path / "small", tmp_path / "medium"
    small.mkdir()
    medium.mkdir()
    _shard(str(small / "0000.parquet"), [f"s{i}" for i in range(4)])  # 2 groups
    _shard(str(medium / "0000.parquet"), [f"m{i}" for i in range(12)])  # 6 groups
    src = ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=str(tmp_path / "ls"),
                         extra=[(str(small), None), (str(medium), None)], base_weight=0.5)
    for epoch in range(3):
        units = src._epoch_units(epoch)
        n = [sum(1 for fi, _ in units if fi == k) for k in range(3)]
        assert n == [20, 5, 15], n  # 50% base; 20 extra draws split 2 : 6
    with pytest.raises(DataError):  # explicit weights and base_weight together are ambiguous
        ParquetBatches(["train-clean-100"], object(), seed=1, parquet_root=str(tmp_path / "ls"),
                       extra=[(str(small), 0.2)], base_weight=0.5)


def test_no_extras_keeps_the_old_window_order(tmp_path):
    """Without extra splits the per-epoch unit list and window selection are the pre-H52 formula."""
    root, _ = _tree(tmp_path, n_base=40)
    src = ParquetBatches(["train-clean-100"], object(), seed=7, parquet_root=root)
    assert src._epoch_units(5) is src.units
    perm = np.random.default_rng([7, 5]).permutation(len(src.units))
    wg = src.window_groups
    assert src._window_units(5, 1) == [src.units[i] for i in sorted(perm[wg:2 * wg])]
