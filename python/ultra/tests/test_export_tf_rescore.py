"""Torch-free unit tests for `ultra.tf-rescore/v1` (`.utfr`) exporter & reference scorer."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.export_tf_rescore import (  # noqa: E402
    TfRescoreConfig,
    export_tf_tensors,
    load_tf_v1,
    quantize_rows_i8,
)


def _synthetic_state(cfg: TfRescoreConfig, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, v = cfg.d_model, cfg.d_ff, cfg.vocab
    sd: dict[str, np.ndarray] = {
        "emb.weight": (rng.standard_normal((v, d)) * 0.2).astype(np.float32),
        "norm.weight": (1.0 + rng.standard_normal((d,)) * 0.05).astype(np.float32),
    }
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        sd[f"{p}n1.weight"] = (1.0 + rng.standard_normal((d,)) * 0.05).astype(np.float32)
        sd[f"{p}n2.weight"] = (1.0 + rng.standard_normal((d,)) * 0.05).astype(np.float32)
        sd[f"{p}qkv.weight"] = (rng.standard_normal((3 * d, d)) * 0.15).astype(np.float32)
        sd[f"{p}wo.weight"] = (rng.standard_normal((d, d)) * 0.15).astype(np.float32)
        sd[f"{p}w12.weight"] = (rng.standard_normal((2 * f, d)) * 0.15).astype(np.float32)
        sd[f"{p}w3.weight"] = (rng.standard_normal((d, f)) * 0.15).astype(np.float32)
    if not cfg.tied:
        sd["head.weight"] = (rng.standard_normal((v, d)) * 0.15).astype(np.float32)
    return sd


@pytest.mark.parametrize(("direction", "tied"), [("l2r", True), ("r2l", True), ("l2r", False)])
def test_roundtrip_tamper_and_scoring(tmp_path, direction: str, tied: bool):
    cfg = TfRescoreConfig(vocab=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, ctx=16, direction=direction, tied=tied)
    sd = _synthetic_state(cfg)
    out = tmp_path / f"tf_{direction}_{tied}.utfr"
    raw = export_tf_tensors(cfg, sd, out)
    assert len(raw) == cfg.file_bytes() == out.stat().st_size
    loaded = load_tf_v1(out)
    assert loaded.cfg.vocab == cfg.vocab and loaded.cfg.direction == cfg.direction
    assert loaded.cfg.tied == cfg.tied and loaded.cfg.n_layers == cfg.n_layers
    assert loaded.cfg.rms_eps == pytest.approx(cfg.rms_eps, rel=1e-6)
    assert loaded.cfg.rope_theta == pytest.approx(cfg.rope_theta, rel=1e-6)
    assert np.array_equal(loaded.emb, sd["emb.weight"])

    sc = loaded.score_sequence([4, 7, 11, 2])
    assert sc["n_tok"] == 4
    assert np.isfinite(sc["total_logp"]) and sc["total_logp"] < 0.0
    assert abs(sc["tok_logp"] + sc["eos_logp"] - sc["total_logp"]) < 1e-6

    empty_sc = loaded.score_sequence([])
    assert empty_sc["n_tok"] == 0 and empty_sc["tok_logp"] == 0.0 and empty_sc["eos_logp"] < 0.0

    bad = bytearray(raw)
    bad[16] ^= 0x55
    with pytest.raises(ValueError, match="SHA-256"):
        load_tf_v1(bytes(bad))


def test_r2l_differs_from_l2r_on_asymmetric_sequence():
    cfg_l = TfRescoreConfig(vocab=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, ctx=16, direction="l2r")
    cfg_r = TfRescoreConfig(vocab=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, ctx=16, direction="r2l")
    sd = _synthetic_state(cfg_l, seed=11)
    ml = load_tf_v1(export_tf_tensors(cfg_l, sd))
    mr = load_tf_v1(export_tf_tensors(cfg_r, sd))
    seq = [3, 5, 9, 12, 6]
    assert abs(ml.score_sequence(seq)["total_logp"] - mr.score_sequence(list(reversed(seq)))["total_logp"]) < 1e-5
    assert abs(ml.score_sequence(seq)["total_logp"] - mr.score_sequence(seq)["total_logp"]) > 1e-3


def test_quantize_rows_i8_bounds_and_zero_row():
    w = np.array([[0.0, 0.0, 0.0], [-1.0, 0.5, 1.0]], dtype=np.float32)
    q, s = quantize_rows_i8(w)
    assert q.dtype == np.int8 and not (q == -128).any()
    assert q[0].tolist() == [0, 0, 0] and s[0] == 0.0
    assert q[1].tolist() == [-127, 64, 127] and abs(float(s[1]) - 1.0 / 127.0) < 1e-7
