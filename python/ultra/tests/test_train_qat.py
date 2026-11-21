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
    """float32 reference of crates/ultra-engine row_params + quantize_row, then dequantise."""
    f = np.float32
    lo, hi = f(x.min()), f(x.max())
    half = f(f(0.5) * hi) - f(f(0.5) * lo)
    c = f(f(0.5) * hi) + f(f(0.5) * lo)
    if half == 0:
        return np.full_like(x, c)
    inv = f(127.0) / half
    q = np.clip(np.round((x - c) * inv), -127, 127).astype(np.float32)  # ties to even
    return (q * (half / f(127.0)) + c).astype(np.float32)


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
    assert np.array_equal(got, want)  # bit-identical, also on rounding boundaries
    # Rounding-boundary rows: values placed exactly on half-steps of the code grid.
    b = np.array([[-1.0, 1.0, 0.5 / 127, 1.5 / 127, 2.5 / 127, -0.5 / 127]], dtype=np.float32)
    assert np.array_equal(fake_quant_rows(torch.from_numpy(b)).numpy(),
                          np.stack([_engine_quantize_row(r) for r in b]))
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


def test_int8_conv_as_gemm_close_to_conv_and_weights_int8():
    from train.qat import int8_conv2d_as_gemm, int8_weight_ste

    torch.manual_seed(3)
    conv = torch.nn.Conv2d(8, 8, 3, stride=2)
    x = torch.randn(2, 8, 11, 19)
    ref = conv(x)
    got = int8_conv2d_as_gemm(x, conv)
    assert got.shape == ref.shape
    assert float((got - ref).abs().max()) < 0.05 * float(ref.abs().max())
    w = int8_weight_ste(conv.weight).reshape(8, -1)
    s = conv.weight.reshape(8, -1).abs().amax(1, keepdim=True) / 127
    codes = w / s
    assert torch.allclose(codes, codes.round(), atol=1e-4) and codes.abs().max() <= 127 + 1e-4


def test_int8_stem_model_is_causal_and_streams_like_batch():
    from train.model import EncoderConfig as EC
    from train.infer import StreamingRecognizer, run_streaming

    torch.manual_seed(4)
    enc = EC(d_model=32, n_heads=2, stem_channels=8, conv_kernel=5, chunk=4, left_context=8,
             dropout=0.0, post_block_norm=False, family="multirate", lower_blocks=1,
             middle_blocks=1, final_blocks=1)
    m = ASRModel(ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16,
                             qat_stem="int8")).eval()
    assert m.encoder.stem.int8_qat
    x = torch.randn(1, 120, 80)
    with torch.no_grad():
        base, lens = m.encode(x, torch.tensor([120]))
        y = x.clone()
        y[:, 4 * 8:] += 3.0
        pert, _ = m.encode(y, torch.tensor([120]))
    assert torch.equal(pert[:, :8], base[:, :8])

    class Tok:
        def encode(self, t):
            return [1]

        def decode(self, ids):
            return " ".join(map(str, ids))

    class U:
        def __init__(self, i):
            self.uid, self.text = f"u{i}", "x"
            self.pcm = (np.random.default_rng(i).standard_normal(5000 + 2000 * i) * 3000).astype(np.int16)

    rec = StreamingRecognizer(m, Tok(), threads=1)
    rec.keep_states = True
    made = []
    orig = rec.create_stream
    rec.create_stream = lambda: made.append(orig()) or made[-1]
    utts = [U(i) for i in range(3)]
    run_streaming(rec, utts, 2, 30)
    from train.data import Featurizer, collate
    for u, s in zip(utts, made):
        b = collate([Featurizer(Tok(), Tok()).example(u.uid, u.text, u.pcm)])
        with torch.no_grad():
            e, l = m.encode(b.feats, b.feat_lens)
        assert torch.allclose(torch.cat(s.states, 0), e[0, :int(l[0])], atol=1e-4)


def test_speed_perturb_lengths_and_determinism():
    from train.data import ParquetBatches

    pb = ParquetBatches.__new__(ParquetBatches)
    pb.seed, pb.speed_perturb = 7, (0.9, 1.0, 1.1)
    pcm = (np.random.default_rng(0).standard_normal(16000) * 1000).astype(np.int16)
    lens = {len(pb._perturb(pcm, f"u{i}", 0)) for i in range(40)}
    assert lens == {round(16000 / 0.9), 16000, round(16000 / 1.1)} or \
        all(abs(n - t) <= 1 for n, t in zip(sorted(lens), sorted([16000 / 1.1, 16000, 16000 / 0.9])))
    assert np.array_equal(pb._perturb(pcm, "u3", 2), pb._perturb(pcm, "u3", 2))
