"""Deterministic CPU smoke tests for H46 `train_ffnn_lm.py` and `ultra.lm-ffnn/v1` export."""
from __future__ import annotations

import gzip
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.ffnn_lm_export import (  # noqa: E402
    FfnnLmConfig,
    load_ffnn_v1,
    preset_config,
    verify_parity,
)
from train.train_ffnn_lm import (  # noqa: E402
    FfnnLmNet,
    TrainConfig,
    evaluate_ppl,
    iter_text_lines,
    seq_to_windows,
    train_ffnn_lm,
)


def _synthetic_corpus(vocab: int = 512) -> tuple[list[list[int]], list[list[int]]]:
    """Structured token sequences with strong n-gram predictability for fast CPU convergence."""
    motifs = [
        [3, 17, 42, 91, 128, 255, 7, 19, 64, 12, 88, 201],
        [5, 29, 77, 104, 199, 311, 14, 53, 82, 160, 44, 9],
        [11, 38, 66, 145, 222, 333, 27, 71, 95, 180, 61, 18],
        [2, 23, 51, 118, 240, 405, 31, 69, 102, 175, 48, 15],
    ]
    seqs = [[t % vocab for t in motifs[i % len(motifs)]] for i in range(64)]
    return seqs[:52], seqs[52:]


@pytest.mark.parametrize(
    ("preset", "byte_cap"),
    [("0.5m", 512 * 1024), ("1.4m", int(1.4 * 1024 * 1024))],
)
def test_smoke_train_loss_drop_budget_and_parity(tmp_path, preset: str, byte_cap: int):
    cfg = preset_config(preset)
    train_seqs, val_seqs = _synthetic_corpus(cfg.vocab)
    out_file = tmp_path / f"ffnn_{preset}.ulm"
    log_file = tmp_path / f"ffnn_{preset}.jsonl"
    tcfg = TrainConfig(
        steps=10,
        batch_size=64,
        lr=1.5e-2,
        weight_decay=1e-3,
        warmup_steps=2,
        val_every=5,
        seed=13,
        shuffle_buf=512,
        device="cpu",
        bf16=False,
        parity_tol=0.25,
    )
    model, summary = train_ffnn_lm(
        cfg, lambda: train_seqs, val_seqs, out_file, tcfg, log_path=log_file
    )
    assert summary["final_loss"] < summary["init_loss"]
    assert summary["val_ppl"] < float(cfg.vocab) * 0.25
    assert out_file.is_file() and log_file.is_file()
    raw = out_file.read_bytes()
    assert len(raw) == cfg.file_bytes() == summary["size_bytes"] <= byte_cap
    qlm = load_ffnn_v1(out_file)
    assert qlm.cfg == cfg
    histories = [[], [3], [3, 17, 42], [3, 17, 42, 91, 128, 255]]
    assert verify_parity(model.score_all_fp32, raw, histories, max_abs_tol=0.25)["size_bytes"] == len(raw)
    dequant = FfnnLmNet.from_quantized(qlm)
    exact = verify_parity(dequant.score_all_fp32, raw, histories, max_abs_tol=1e-4)
    assert exact["max_abs_diff"] < 1e-4
    assert exact["top1_agreement"] == 1.0


def test_tied_model_and_streaming_helpers(tmp_path):
    cfg = FfnnLmConfig(vocab=32, ctx=6, d_e=16, d_h=16, activation="silu", tied=True)
    train_seqs, val_seqs = _synthetic_corpus(cfg.vocab)
    out_file = tmp_path / "ffnn_tied.ulm"
    tcfg = TrainConfig(
        steps=10, batch_size=32, lr=2e-2, warmup_steps=1, val_every=10, seed=7,
        shuffle_buf=128, device="cpu", bf16=False,
    )
    _, summary = train_ffnn_lm(cfg, lambda: train_seqs, val_seqs, out_file, tcfg)
    assert summary["final_loss"] < summary["init_loss"]
    raw = out_file.read_bytes()
    qlm = load_ffnn_v1(raw)
    assert qlm.cfg.tied and len(raw) == cfg.file_bytes()
    dequant = FfnnLmNet.from_quantized(qlm)
    assert verify_parity(dequant.score_all_fp32, raw, [[], [3, 17, 42]], max_abs_tol=1e-4)["max_abs_diff"] < 1e-4
    val_loss, val_ppl = evaluate_ppl(dequant, val_seqs, batch_size=32)
    assert np.isfinite(val_loss) and val_ppl < float(cfg.vocab)

    xw, yw = seq_to_windows([4, 9, 12], ctx=6, vocab=32)
    assert xw.tolist() == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 4, 9]]
    assert yw.tolist() == [4, 9, 12]

    plain = tmp_path / "a.txt"
    gz = tmp_path / "b.txt.gz"
    plain.write_text("hello world\n\nsecond line\n", encoding="utf-8")
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write("third line\n")
    assert [s.strip() for s in iter_text_lines([plain, gz])] == ["hello world", "second line", "third line"]
