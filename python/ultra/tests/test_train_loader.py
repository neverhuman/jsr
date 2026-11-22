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
