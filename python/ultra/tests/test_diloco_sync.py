"""Tests for dev-gated cross-node weight sharing (`ops/diloco_sync.py`)."""
from __future__ import annotations

import os
import pytest
import torch

from train.ops.diloco_sync import run_sync_round, verify_cohort


def _ckpt(val: float, *, vocab_rnnt: int = 513, tok_sha: str = "abc123") -> dict:
    w = {"w": torch.full((4, 4), float(val))}
    return {
        "model": w,
        "ema": {k: v.clone() for k, v in w.items()},
        "model_config": {"d": 384, "vocab_rnnt": vocab_rnnt, "qat": "none"},
        "tokenizer_sha256": tok_sha,
        "step": 1000,
    }


def test_verify_cohort_rejects_mismatched_tokenizer_or_architecture():
    anchor = _ckpt(0.0)
    verify_cohort(anchor, [("ok", _ckpt(1.0))])
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        verify_cohort(anchor, [("bad_tok", _ckpt(1.0, tok_sha="different"))])
    with pytest.raises(ValueError, match="model_config"):
        verify_cohort(anchor, [("bad_arch", _ckpt(1.0, vocab_rnnt=4097))])


def test_sync_round_promotes_only_verified_dev_improvement(tmp_path):
    # Target optimum is w == 1.0; anchor is at 0.0, workers moved to 0.8 and 1.2.
    anchor = _ckpt(0.0)
    workers = [
        ("a100_d2e", _ckpt(0.8), 80.0),
        ("rtx3090_d2", _ckpt(1.2), 20.0),
    ]
    out_pt = str(tmp_path / "promoted_swap.pt")

    def eval_fn(weights: dict[str, torch.Tensor]) -> float:
        return 5.0 + float((weights["w"] - 1.0).abs().mean())

    outcome, outer_sd = run_sync_round(
        anchor, workers, eval_fn, out_path=out_pt, min_gain_pp=0.03,
    )
    assert outcome.promoted is True
    assert outcome.winner_wer < outcome.incumbent_wer
    assert os.path.exists(out_pt)
    saved = torch.load(out_pt, map_location="cpu", weights_only=False)
    assert saved["tokenizer_sha256"] == "abc123"
    assert outer_sd["round"] == 1


def test_sync_round_rejects_regression_and_writes_no_swap_file(tmp_path):
    # Anchor is already at optimum 0.0; noisy workers at +2.0 and +3.0 regress WER.
    anchor = _ckpt(0.0)
    workers = [("w1", _ckpt(2.0), 10.0), ("w2", _ckpt(3.0), 10.0)]
    out_pt = str(tmp_path / "should_not_exist.pt")

    def eval_fn(weights: dict[str, torch.Tensor]) -> float:
        return 5.0 + float(weights["w"].abs().mean())

    outcome, _ = run_sync_round(anchor, workers, eval_fn, out_path=out_pt, min_gain_pp=0.03)
    assert outcome.promoted is False
    assert outcome.winner_name == "incumbent"
    assert not os.path.exists(out_pt)
