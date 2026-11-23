"""Regression tests for the Wave-0 training defect closures from PR #72 (B2 KD path guard, B3 multirate
causality). B6 (the Python canonical-normaliser twin) is not taken: H49 MWER uses H43's Rust counts."""
from __future__ import annotations

import json
import numpy as np
import pytest


def test_teacher_store_get_checked_enforces_exact_emitted_path(tmp_path):
    """B2: TeacherStore.get_checked rejects mismatched subword paths even when text matches."""
    torch = pytest.importorskip("torch")
    _ = torch
    from train.kd import TeacherStore

    # Build a tiny single-utterance teacher cache where text == hyp ("ok") but emitted pieces
    # are R0 [4, 9] -> student [5, 10].
    npz_path = tmp_path / "shard_0000_sp1.0.npz"
    np.savez(
        npz_path,
        meta=json.dumps({"speed": 1.0}),
        uid=np.array(["utt-1"]),
        text=np.array(["hello"]),
        hyp=np.array(["HELLO"]),
        n_frames=np.array([4], dtype=np.int32),
        emitted=np.array([0, 4, 0, 9], dtype=np.int32),
        topk_ids=np.zeros((4, 2), dtype=np.int32),
        topk_lp=np.zeros((4, 2), dtype=np.float32),
    )
    store = TeacherStore(str(tmp_path))
    # Exact student target [5, 10] -> accepted.
    row_ok = store.get_checked("utt-1", 1.0, [5, 10])
    assert row_ok is not None and row_ok[3] is True
    # Different subword segmentation [5, 11] with same normalised text -> rejected!
    row_bad = store.get_checked("utt-1", 1.0, [5, 11])
    assert row_bad is not None and row_bad[3] is False


def test_multirate_rejects_non_causal_odd_chunk():
    """B3: multirate / unet3 reject chunks where integer division would turn c2/c4 into 0."""
    torch = pytest.importorskip("torch")
    from train.model import EncoderConfig, Encoder

    cfg_mr = EncoderConfig(
        family="multirate", d_model=32, n_heads=2, n_blocks=4, lower_blocks=1, middle_blocks=2,
        final_blocks=1, ffn_mult=2, stem_channels=16, conv_kernel=3, chunk=4, left_context=8,
        dropout=0.0,
    )
    enc_mr = Encoder(cfg_mr).eval()
    x = torch.zeros(1, 32, 80)
    lens = torch.tensor([32])
    with pytest.raises(ValueError, match="even positive chunk"):
        enc_mr(x, lens, chunk=1)
    # Legal even chunk=2 works and stays strictly causal (perturbing future frames has 0 effect).
    y1, _ = enc_mr(x, lens, chunk=2)
    x2 = x.clone()
    x2[:, 16:, :] = 5.0
    y2, _ = enc_mr(x2, lens, chunk=2)
    assert torch.allclose(y1[:, :2], y2[:, :2], atol=1e-6)

    cfg_u3 = EncoderConfig(
        family="unet3", d_model=32, n_heads=2, n_blocks=4, lower_blocks=1, middle_blocks=1,
        deep_blocks=1, up_blocks=1, final_blocks=1, ffn_mult=2, stem_channels=16, conv_kernel=3,
        chunk=4, left_context=8, dropout=0.0,
    )
    enc_u3 = Encoder(cfg_u3).eval()
    with pytest.raises(ValueError, match="divisible by 4"):
        enc_u3(x, lens, chunk=2)
