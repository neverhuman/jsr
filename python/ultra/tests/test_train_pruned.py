"""H13b pruned RNN-T option. With k2 present: at a band covering every label position the pruned
loss equals the exact loss (values and gradients of all shared parameters), and a narrow band
trains (finite loss, simple-joiner heads receive gradient). Without k2 the option fails with a
clear ImportError instead of silently falling back."""
import dataclasses
import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel  # noqa: E402

HAVE_K2 = importlib.util.find_spec("k2") is not None


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(3, 64, 80, generator=g)
    feat_lens = torch.tensor([64, 50, 33])
    rnnt = torch.randint(1, 11, (3, 6), generator=g)
    rnnt_lens = torch.tensor([6, 4, 0])
    ctc = torch.randint(1, 7, (3, 3), generator=g)
    ctc_lens = torch.tensor([3, 2, 1])
    return feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens


def _models(prune_range, simple_scale=0.0):
    torch.manual_seed(3)
    exact = ASRModel(dataclasses.replace(PRESETS["TINY"], encoder=dataclasses.replace(
        PRESETS["TINY"].encoder, dropout=0.0))).eval()
    cfg = dataclasses.replace(exact.cfg, rnnt_loss="pruned", prune_range=prune_range,
                              simple_loss_scale=simple_scale)
    pruned = ASRModel(cfg).eval()
    missing, unexpected = pruned.load_state_dict(exact.state_dict(), strict=False)
    assert sorted(missing) == ["simple_am.bias", "simple_am.weight", "simple_lm.bias",
                               "simple_lm.weight"] and not unexpected
    return exact, pruned


def test_pruned_full_band_equals_exact_or_requires_k2():
    exact, pruned = _models(prune_range=7)  # >= max U + 1: nothing is pruned
    b = _batch()
    if not HAVE_K2:
        with pytest.raises(ImportError, match="needs k2"):
            pruned.losses(*b)
        return
    le, lp = exact.losses(*b), pruned.losses(*b)
    assert torch.allclose(lp["nll"].double(), le["nll"].double(), atol=1e-4, rtol=1e-6)
    assert torch.allclose(lp["loss"].double(), le["loss"].double(), atol=1e-5, rtol=1e-6)
    ge = dict(zip([n for n, _ in exact.named_parameters()],
                  torch.autograd.grad(le["loss"], list(exact.parameters()), allow_unused=True)))
    names = [n for n, _ in pruned.named_parameters() if not n.startswith("simple_")]
    gp = torch.autograd.grad(lp["loss"], [dict(pruned.named_parameters())[n] for n in names],
                             allow_unused=True)
    for n, g in zip(names, gp, strict=True):
        if ge[n] is None:
            assert g is None or float(g.abs().max()) == 0.0, n
        else:
            assert torch.allclose(g, ge[n], atol=1e-5, rtol=1e-4), n


def test_pruned_narrow_band_trains_simple_joiner():
    if not HAVE_K2:
        with pytest.raises(ImportError, match="needs k2"):
            _models(prune_range=3, simple_scale=0.5)[1].losses(*_batch())
        return
    _, pruned = _models(prune_range=3, simple_scale=0.5)
    pruned.train()
    out = pruned.losses(*_batch(), pruned_scale=0.1)
    assert torch.isfinite(out["loss"]) and float(out["rnnt_simple"]) > 0
    out["loss"].backward()
    assert float(pruned.simple_am.weight.grad.abs().sum()) > 0
    assert float(pruned.simple_lm.weight.grad.abs().sum()) > 0
    assert float(pruned.joiner.out.weight.grad.abs().sum()) > 0


def test_unknown_loss_and_bad_range_refused():
    with pytest.raises(ValueError):
        ASRModel(dataclasses.replace(PRESETS["TINY"], rnnt_loss="fast"))
    with pytest.raises(ValueError):
        ASRModel(dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned", prune_range=1))


def test_prepare_seed_adds_only_simple_heads(tmp_path):
    from train import checkpoint
    from train.prepare_seed import make_seed

    exact, _ = _models(prune_range=5)
    src = str(tmp_path / "src.pt")
    checkpoint.save_weights(str(tmp_path), 7, {"model": exact.state_dict(),
                                               "model_config": exact.cfg.to_dict()})
    src = str(tmp_path / "model-00000007.pt")
    before = open(src, "rb").read()
    pl = make_seed(src, str(tmp_path / "seed.pt"), "pruned", 5, 1)
    assert sorted(pl["seed_from"]["added"]) == ["simple_am.bias", "simple_am.weight",
                                                "simple_lm.bias", "simple_lm.weight"]
    assert open(src, "rb").read() == before  # source untouched
    seed = checkpoint.load(str(tmp_path / "seed.pt"), restore_rng=False)
    for k, v in exact.state_dict().items():
        assert torch.equal(seed["model"][k], v), k
    assert seed["model_config"]["rnnt_loss"] == "pruned"


def test_simple_detach_keeps_simple_loss_out_of_the_model():
    if not HAVE_K2:
        with pytest.raises(ImportError, match="needs k2"):
            _models(prune_range=3)[1].losses(*_batch(), simple_detach=True)
        return
    _, pruned = _models(prune_range=3, simple_scale=0.5)
    pruned.train()
    out = pruned.losses(*_batch(), pruned_scale=0.0, simple_detach=True)
    pruned.zero_grad()
    (out["rnnt_simple"]).backward()
    assert float(pruned.simple_am.weight.grad.abs().sum()) > 0
    assert pruned.joiner.out.weight.grad is None or float(pruned.joiner.out.weight.grad.abs().sum()) == 0
    enc_grads = [p.grad for n, p in pruned.named_parameters() if n.startswith("encoder.")]
    assert all(g is None or float(g.abs().sum()) == 0 for g in enc_grads)
