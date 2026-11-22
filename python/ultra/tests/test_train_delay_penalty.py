"""H34: the pruned-loss delay_penalty key is off by default (bit-exact) and changes the loss when on."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
k2 = pytest.importorskip("k2")
from train.asr import PRESETS, ASRModel  # noqa: E402


def _batch():
    torch.manual_seed(0)
    return (torch.randn(2, 64, 80), torch.tensor([64, 50]), torch.tensor([[1, 2, 3], [4, 5, 0]]),
            torch.tensor([3, 2]), torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1]))


def test_delay_penalty_zero_is_bit_exact_and_positive_changes_the_loss():
    import dataclasses

    cfg = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned")
    torch.manual_seed(3)
    m = ASRModel(cfg)
    b = _batch()
    base = m.losses(*b)
    zero = m.losses(*b, delay_penalty=0.0)
    assert float(base["loss"]) == float(zero["loss"])
    on = m.losses(*b, delay_penalty=0.01)
    assert float(on["loss"]) != float(base["loss"])
    on["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_delay_penalty_rejects_the_exact_loss():
    torch.manual_seed(3)
    m = ASRModel(PRESETS["TINY"])
    with pytest.raises(ValueError):
        m.losses(*_batch(), delay_penalty=0.01)
