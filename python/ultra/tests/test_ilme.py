"""Unit tests for Exact Stateless-Predictor ILME (`python/ultra/train/ilme.py`)."""
from __future__ import annotations

import numpy as np
import pytest

from train.ilme import (
    ILME_SCHEMA,
    estimate_eavg_from_frames,
    load_ilme_artifact,
    save_ilme_artifact,
    stateless_ilme_logprobs,
)


def test_stateless_ilme_logprobs_normalized_over_non_blank():
    rng = np.random.default_rng(42)
    v, e, j = 10, 4, 8
    emb = rng.normal(0.0, 0.2, size=(v, e)).astype(np.float32)
    pred_w = rng.normal(0.0, 0.2, size=(j, 2 * e)).astype(np.float32)
    pred_b = rng.normal(0.0, 0.1, size=(j,)).astype(np.float32)
    join_w = rng.normal(0.0, 0.2, size=(v, j)).astype(np.float32)
    join_b = rng.normal(0.0, 0.1, size=(v,)).astype(np.float32)
    e_avg = rng.normal(0.0, 0.3, size=(j,)).astype(np.float32)
    pairs = np.array([[0, 0], [0, 3], [3, 7]], dtype=np.int64)

    lp = stateless_ilme_logprobs(emb, pred_w, pred_b, join_w, join_b, e_avg, pairs, blank=0)
    assert lp.shape == (3, v)
    assert np.all(np.isneginf(lp[:, 0]))
    sums = np.exp(lp[:, 1:]).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_eavg_estimation_and_artifact_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    true_mean = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    frames = [true_mean[None, :] + rng.normal(0.0, 0.05, size=(5, 16)).astype(np.float32) for _ in range(40)]
    counts = [np.array([0, 1, 0, 2, 0], dtype=np.int32) for _ in range(40)]

    e_avg, total_em, conv = estimate_eavg_from_frames(frames, counts, checkpoints=(10, 25))
    assert total_em == 120
    assert np.allclose(e_avg, true_mean, atol=0.05)
    assert len(conv) == 2 and conv[-1]["cos_vs_final"] > 0.99

    out = tmp_path / "ilme.json"
    art = save_ilme_artifact(out, e_avg, num_utts=40, num_emissions=total_em, model_weights_sha256="abc")
    assert art["schema"] == ILME_SCHEMA
    loaded, meta = load_ilme_artifact(out)
    assert meta["num_emissions"] == total_em
    assert np.array_equal(loaded, e_avg)

    # Tampering invalidates e_avg_sha256.
    out.write_text(out.read_text().replace(str(art["e_avg"][0]), "999.0", 1))
    with pytest.raises(ValueError, match="sha256"):
        load_ilme_artifact(out)
