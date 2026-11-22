"""Tests for Stage-2 Student N-Best MWER (L_risk) and Zero-GPU Multi-View Decoding Assay."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from train.asr import PRESETS, ASRModel
from train.mwer import mwer_loss
from train.ops.multiview_assay import fuse_nbest_views, pad_pcm, tempo_pcm


class _DummyTok:
    WORDS = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta", 5: "echo"}

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.WORDS.get(int(i), f"w{i}") for i in ids)


def test_mwer_loss_computes_finite_gradients_and_updates_weights():
    torch.manual_seed(11)
    m = ASRModel(PRESETS["TINY"])
    feats = torch.randn(2, 40, 80)
    feat_lens = torch.tensor([40, 32])
    rnnt = torch.tensor([[1, 2, 3], [2, 4, 0]], dtype=torch.long)
    rnnt_lens = torch.tensor([3, 2])
    texts = ["alpha beta gamma", "beta delta"]
    tok = _DummyTok()

    out = mwer_loss(m, feats, feat_lens, rnnt, rnnt_lens, texts, tok, beam=3, lam=0.5, tau=1.0)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["rnnt"]) and torch.isfinite(out["mwer"])
    assert out["active_utts"] >= 1
    out["loss"].backward()
    assert m.joiner.out.weight.grad is not None
    assert torch.isfinite(m.joiner.out.weight.grad).all()
    assert m.encoder.blocks[0].att.qkv.weight.grad is not None


def test_multiview_pad_tempo_and_consensus_fusion():
    pcm = np.arange(1600, dtype=np.int16)
    p20 = pad_pcm(pcm, 20.0)
    assert len(p20) == 1600 + 320
    assert np.all(p20[:320] == 0)
    assert np.array_equal(p20[320:], pcm)

    t95 = tempo_pcm(pcm, 0.95)
    assert t95.dtype == np.int16 and len(t95) > len(pcm)
    assert np.array_equal(tempo_pcm(pcm, 1.0), pcm)

    # View 0 slightly prefers "cat sat", while views 1 and 2 strongly prefer "the cat sat".
    v0 = [("cat sat", -1.0), ("the cat sat", -1.2)]
    v1 = [("the cat sat", -0.5), ("cat sat", -3.0)]
    v2 = [("the cat sat", -0.4), ("cat sat", -2.5)]
    assert fuse_nbest_views([v0, v1, v2]) == "the cat sat"


def test_multiview_refuses_test_splits():
    from train.ops.multiview_assay import main

    with pytest.raises(ValueError, match="refused"):
        main(["--checkpoint", "nonexistent.pt", "--split", "test-clean"])
