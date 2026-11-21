"""Ternary QAT: projection, STE gradients, engine-equal activation fake-quant, loss-path parity."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.model import EncoderConfig  # noqa: E402
from train.qat import (TernaryLinear, fake_quant_rows, ternary_project,  # noqa: E402
                       ternary_targets, ternary_weight_ste)
from train.transducer import (StatelessPredictor, full_lattice_logprobs,  # noqa: E402
                              joiner_logprobs, rnnt_nll)


def _engine_quantize_row(x: np.ndarray) -> np.ndarray:
    """Reference of crates/ultra-engine quantize_row followed by dequantisation."""
    lo, hi = float(x.min()), float(x.max())
    c, h = (hi + lo) / 2, (hi - lo) / 2
    if h == 0:
        return np.full_like(x, c)
    q = np.clip(np.round((x - c) * 127.0 / h), -127, 127)  # numpy rounds half to even
    return (q * (h / 127.0) + c).astype(np.float32)


def test_ternary_projection_absmean():
    w = torch.tensor([[0.9, -0.1, 0.4, -1.2], [0.0, 0.0, 0.0, 0.0]])
    t, s = ternary_project(w)
    assert torch.allclose(s[0], torch.tensor([0.65]))
    assert t[0].tolist() == [1.0, 0.0, 1.0, -1.0]
    assert set(t.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert t[1].abs().sum() == 0  # all-zero row stays zero


def test_weight_ste_gradient_is_identity():
    w = torch.randn(5, 7, requires_grad=True)
    g = torch.randn(5, 7)
    (ternary_weight_ste(w) * g).sum().backward()
    assert torch.allclose(w.grad, g)


def test_fake_quant_rows_matches_engine_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, (6, 33)).astype(np.float32)
    x[2] = np.abs(x[2])  # one-sided row (post-SiLU like)
    x[3] = 1.5  # constant row
    got = fake_quant_rows(torch.from_numpy(x)).numpy()
    want = np.stack([_engine_quantize_row(r) for r in x])
    assert np.allclose(got, want, atol=1e-5)
    xt = torch.from_numpy(x).requires_grad_(True)
    fake_quant_rows(xt).sum().backward()
    assert torch.equal(xt.grad, torch.ones_like(xt))


def test_ternary_linear_loads_linear_state_and_computes_quantised_product():
    torch.manual_seed(0)
    lin = torch.nn.Linear(12, 5)
    tl = torch.nn.Linear(12, 5)
    tl.__class__ = TernaryLinear
    tl.load_state_dict(lin.state_dict())
    x = torch.randn(3, 12)
    t, s = ternary_project(lin.weight)
    want = torch.nn.functional.linear(fake_quant_rows(x), t * s, lin.bias)
    assert torch.allclose(tl(x), want, atol=1e-6)


def _tiny_qat():
    torch.manual_seed(1)
    enc = EncoderConfig(d_model=32, n_heads=2, stem_channels=8, conv_kernel=5, chunk=4,
                        left_context=8, dropout=0.0, post_block_norm=False, family="multirate",
                        lower_blocks=1, middle_blocks=1, final_blocks=1)
    return ASRModel(ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16,
                                qat="ternary"))


def test_targets_cover_encoder_and_joiner_only():
    m = _tiny_qat()
    names = ternary_targets(m)
    assert "joiner.out" in names and "joiner.enc_proj" in names
    assert "encoder.pair" in names and "encoder.restore" in names
    assert not any(n.startswith("encoder.stem") for n in names)
    assert "predictor.proj" not in names and "ctc_head" not in names
    assert all(isinstance(dict(m.named_modules())[n], TernaryLinear) for n in names)


def test_chunked_loss_matches_full_lattice_with_ternary_joiner():
    m = _tiny_qat()
    j, pred = m.joiner, m.predictor
    torch.manual_seed(2)
    enc = torch.randn(2, 6, 32)
    targets = torch.tensor([[1, 2, 3], [4, 5, 0]])
    tl, ul = torch.tensor([6, 5]), torch.tensor([3, 2])

    def run(chunked):
        m.zero_grad()
        ep, pp = j.enc_proj(enc), pred(StatelessPredictor.contexts(targets))
        f = (lambda: joiner_logprobs(j, ep, pp, targets, budget_elems=200)) if chunked else \
            (lambda: full_lattice_logprobs(j, ep, pp, targets))
        lpb, lpl = f()
        loss = rnnt_nll(lpb, lpl, tl, ul).sum()
        loss.backward()
        return float(loss), j.out.weight.grad.clone(), j.enc_proj.weight.grad.clone()

    l1, gw1, ge1 = run(True)
    l2, gw2, ge2 = run(False)
    assert abs(l1 - l2) < 1e-4
    assert torch.allclose(gw1, gw2, atol=1e-4) and torch.allclose(ge1, ge2, atol=1e-4)


def test_qat_model_trains():
    m = _tiny_qat()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    feats, fl = torch.randn(2, 64, 80), torch.tensor([64, 50])
    rt, rl = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([3, 2])
    ct, cl = torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1])
    first = None
    for _ in range(30):
        out = m.losses(feats, fl, rt, rl, ct, cl)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        first = first if first is not None else float(out["loss"])
    assert float(out["loss"]) < 0.7 * first


def test_t1_pack_matches_ultra_core_layout_and_export_round_trips(tmp_path_factory=None):
    import json
    import shutil

    from train import checkpoint
    from train.export_ternary import export, pack_t1, unpack_t1

    t = np.array([[-1, 0, 1, 1, -1]], dtype=np.int8)
    p = pack_t1(t)
    # ultra-core pack_t1: code = t + 1 at bits 2k: [-1,0,1,1] -> 0 | 1<<2 | 2<<4 | 2<<6
    assert p[0, 0] == (0 | (1 << 2) | (2 << 4) | (2 << 6))
    assert p[0, 1] == (0 | (1 << 2) | (1 << 4) | (1 << 6))  # [-1, pad 0, 0, 0]
    assert np.array_equal(unpack_t1(p, 5), t)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    d = os.path.join(repo, "target", "test-train", "ternary_export")
    shutil.rmtree(d, ignore_errors=True)
    m = _tiny_qat()
    ck = checkpoint.save(d, 1, {"model": m.state_dict(), "model_config": m.cfg.to_dict()})
    man = export(ck, os.path.join(d, "bundle"))
    assert len(man["ternary_layers"]) == len(ternary_targets(m))
    blob = open(os.path.join(d, "bundle", "ternary.bin"), "rb").read()
    lay = man["ternary_layers"][0]
    tr = np.frombuffer(blob[lay["trits"]["offset"]:lay["trits"]["offset"] + lay["trits"]["bytes"]],
                       dtype=np.uint8).reshape(lay["out"], lay["row_bytes"])
    want, _ = ternary_project(dict(m.named_modules())[lay["name"]].weight.detach())
    assert np.array_equal(unpack_t1(tr, lay["in"]), want.numpy().astype(np.int8))
    json.dumps(man)
    shutil.rmtree(d, ignore_errors=True)
