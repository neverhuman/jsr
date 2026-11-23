"""Tests for H46 `ultra.lm-ffnn/v1` exporter, byte budgets, and Python<->Rust numerical parity."""
from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.errors import DataError, ShapeError  # noqa: E402
from train.ffnn_lm_export import (  # noqa: E402
    PRESETS,
    FfnnLmConfig,
    export_ffnn_tensors,
    export_ffnn_v1,
    load_ffnn_v1,
    pack_ffnn_v1,
    preset_config,
    verify_parity,
)


class _TinyFfnn(nn.Module):
    def __init__(self, cfg: FfnnLmConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_e)
        self.fc1 = nn.Linear(cfg.ctx * cfg.d_e, cfg.d_h)
        if cfg.tied:
            self.fc2 = None
            self.out_bias = nn.Parameter(torch.zeros(cfg.vocab))
        else:
            self.fc2 = nn.Linear(cfg.d_h, cfg.vocab)
            self.out_bias = None

    def score_all_fp32(self, hist: list[int]) -> np.ndarray:
        c = self.cfg
        take = min(len(hist), c.ctx)
        tail = [t if 0 <= t < c.vocab else 0 for t in hist[-take:]] if take else []
        slots = torch.tensor([[0] * (c.ctx - take) + tail], dtype=torch.long)
        with torch.no_grad():
            x = self.emb(slots).flatten(-2)
            h = self.fc1(x)
            h = F.silu(h) if c.activation == "silu" else F.relu(h)
            logits = F.linear(h, self.emb.weight, self.out_bias) if c.tied else self.fc2(h)
            return F.log_softmax(logits, dim=-1)[0].numpy()


def test_presets_byte_ladder_within_budgets():
    budgets = {
        "0.5m": 512 * 1024,
        "l2": int(1.4 * 1024 * 1024),
        "1.4m": int(1.4 * 1024 * 1024),
        "2.0m": 2 * 1024 * 1024,
    }
    for name, cap in budgets.items():
        cfg = preset_config(name)
        assert cfg.file_bytes() <= cap, f"{name}: {cfg.file_bytes()} > {cap}"
        assert name in PRESETS
    model = _TinyFfnn(preset_config("l2"))
    raw = export_ffnn_v1(model)
    assert len(raw) == model.cfg.file_bytes() <= budgets["l2"]


@pytest.mark.parametrize("tied,act", [(False, "relu"), (False, "silu"), (True, "silu")])
def test_ffnn_export_load_parity_and_tamper(tmp_path, tied, act):
    torch.manual_seed(11)
    d_e = 16
    d_h = d_e if tied else 24
    cfg = FfnnLmConfig(vocab=20, ctx=4, d_e=d_e, d_h=d_h, activation=act, tied=tied)
    model = _TinyFfnn(cfg)
    out_file = tmp_path / f"lm_{act}_{tied}.uffnn"
    raw = export_ffnn_v1(model, out_file)
    assert out_file.read_bytes() == raw
    assert len(raw) == cfg.file_bytes()
    qlm = load_ffnn_v1(out_file)
    histories = [[], [1], [2, 5, 7], [19, 3, 4, 8, 12]]
    stats = verify_parity(model.score_all_fp32, raw, histories, max_abs_tol=0.15)
    assert stats["top1_agreement"] >= 0.75
    for h in histories:
        lp = qlm.score_all(h)
        assert abs(float(np.exp(lp).sum()) - 1.0) < 1e-5
        assert qlm.score_next(h, 3) == pytest.approx(float(lp[3]), abs=1e-7)
    assert qlm.score_next([1], 999) == float("-inf")
    tampered = bytearray(raw)
    tampered[12] ^= 0x40
    with pytest.raises(DataError, match="SHA-256"):
        load_ffnn_v1(bytes(tampered))


def test_export_raw_tensors_and_validation():
    rng = np.random.default_rng(5)
    raw = export_ffnn_tensors(
        rng.normal(0, 0.2, (12, 8)).astype(np.float32),
        rng.normal(0, 0.2, (16, 4 * 8)).astype(np.float32),
        np.zeros(16, dtype=np.float32),
        np.zeros(12, dtype=np.float32),
        rng.normal(0, 0.2, (12, 16)).astype(np.float32),
        ctx=4,
        activation="relu",
    )
    qlm = load_ffnn_v1(raw)
    assert qlm.cfg == FfnnLmConfig(vocab=12, ctx=4, d_e=8, d_h=16, activation="relu", tied=False)
    with pytest.raises(ShapeError):
        export_ffnn_tensors(
            np.zeros((12, 8)), np.zeros((16, 10)), np.zeros(16), np.zeros(12), ctx=4
        )
    with pytest.raises(ShapeError):
        FfnnLmConfig(vocab=12, ctx=4, d_e=8, d_h=16, tied=True).validate()


def test_python_rust_golden_binary_and_score_parity():
    """Exact same deterministic golden model constructed in `crates/ultra-engine/src/lm/ffnn.rs`."""
    vocab, ctx, d_e, d_h = 8, 4, 6, 8
    cfg = FfnnLmConfig(vocab=vocab, ctx=ctx, d_e=d_e, d_h=d_h, activation="relu", tied=False)
    emb_q = np.array([((i * 17 + 7) % 25) - 12 for i in range(vocab * d_e)], dtype=np.int8).reshape(vocab, d_e)
    emb_scale = np.array([(v + 1) / 16.0 for v in range(vocab)], dtype=np.float32)
    w1_q = np.array([((i * 29 + 3) % 27) - 13 for i in range(d_h * ctx * d_e)], dtype=np.int8).reshape(d_h, ctx * d_e)
    w1_scale = np.array([(j + 1) / 32.0 for j in range(d_h)], dtype=np.float32)
    b1 = np.array([(j - 3) / 8.0 for j in range(d_h)], dtype=np.float32)
    w2_q = np.array([((i * 13 + 5) % 23) - 11 for i in range(vocab * d_h)], dtype=np.int8).reshape(vocab, d_h)
    w2_scale = np.array([(v + 2) / 64.0 for v in range(vocab)], dtype=np.float32)
    b2 = np.array([(v - 4) / 16.0 for v in range(vocab)], dtype=np.float32)
    raw = pack_ffnn_v1(cfg, emb_q, emb_scale, w1_q, w1_scale, b1, b2, w2_q, w2_scale)
    qlm = load_ffnn_v1(raw)
    scores = qlm.score_all([1, 3, 5])
    # Exact SHA-256 and scores pinned in `crates/ultra-engine/src/lm/ffnn.rs::tests`.
    assert raw[-32:].hex() == "8f46b3295769032bcd0106fd4e5a4a97b31e8bc0aed6b70302ac54821e848b0c"
    golden = np.array(
        [-32.460968, -18.0407124, -36.5013123, -9.9910603, -38.605011, -0.0047626, -38.7720642, -5.3591938],
        dtype=np.float32,
    )
    assert np.allclose(scores, golden, atol=1e-5)
    assert hashlib.sha256(raw[:-32]).digest() == raw[-32:]
    assert abs(float(np.exp(scores).sum()) - 1.0) < 1e-6
    assert all(math.isfinite(float(x)) and x <= 0.0 for x in scores)
