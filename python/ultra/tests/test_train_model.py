"""Encoder causality under the chunk mask, CTC feasibility counting, tiny forward/backward."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel  # noqa: E402
from train.ctc import FeasibilityStats, adjacent_repeats, ctc_loss, feasible  # noqa: E402
from train.data import pack  # noqa: E402
from train.features import LogMel  # noqa: E402
from train.model import Encoder, EncoderConfig, chunk_mask, stem_out_len  # noqa: E402


def test_chunk_mask_shape_and_bounds():
    m = chunk_mask(10, chunk=4, left=4)
    assert m[0].tolist() == [True] * 4 + [False] * 6
    assert m[5].tolist() == [True] * 8 + [False] * 2  # chunk 1 sees chunk 0 (left 4) and itself
    assert m[9].tolist() == [False] * 4 + [True] * 6  # chunk 2 sees chunk 1 and itself


def test_future_perturbation_leaves_earlier_chunks_unchanged():
    torch.manual_seed(0)
    cfg = EncoderConfig(d_model=32, n_blocks=3, n_heads=2, stem_channels=8, conv_kernel=5,
                        chunk=3, left_context=6, dropout=0.0)
    enc = Encoder(cfg).eval()
    T = 120
    x = torch.randn(1, T, 80)
    lens = torch.tensor([T])
    with torch.no_grad():
        base, out_lens = enc(x, lens)
    n_out = int(out_lens[0])
    assert n_out == int(stem_out_len(T)) == base.shape[1]
    for c in range(n_out // cfg.chunk):
        boundary = (c + 1) * cfg.chunk  # first encoder frame of the next chunk
        y = x.clone()
        y[:, 4 * boundary:] += torch.randn_like(y[:, 4 * boundary:]) * 5.0
        with torch.no_grad():
            pert, _ = enc(y, lens)
        assert torch.equal(pert[:, :boundary], base[:, :boundary]), c
        if boundary < n_out:
            assert not torch.allclose(pert[:, boundary:], base[:, boundary:])


def test_left_context_bound_holds():
    # Frames older than the left context of a chunk do not reach it through one block.
    torch.manual_seed(1)
    cfg = EncoderConfig(d_model=16, n_blocks=1, n_heads=2, stem_channels=4, conv_kernel=3,
                        chunk=2, left_context=2, dropout=0.0)
    enc = Encoder(cfg).eval()
    x = torch.randn(1, 80, 80)
    with torch.no_grad():
        base, _ = enc(x, torch.tensor([80]))
        y = x.clone()
        y[:, :4] += 3.0  # encoder frame 0 (and only stem-reachable frames 0..1)
        pert, _ = enc(y, torch.tensor([80]))
    # Frame 10 (chunk 5) sees keys 8..11 and conv taps 8..10; frame 0/1 changes cannot reach it.
    assert torch.equal(pert[:, 10:], base[:, 10:])


def test_padding_does_not_change_valid_outputs():
    torch.manual_seed(2)
    m = ASRModel(PRESETS["TINY"]).eval()
    x = torch.randn(1, 50, 80)
    with torch.no_grad():
        a, _ = m.encode(x, torch.tensor([50]))
        xp = torch.cat([x, torch.randn(1, 30, 80)], dim=1)
        b, lb = m.encode(xp, torch.tensor([50]))
    assert torch.allclose(a, b[:, :a.shape[1]], atol=1e-6)


def test_ctc_feasibility_masks_and_counts():
    tg = torch.tensor([[1, 1, 2, 0], [3, 4, 5, 6], [2, 2, 2, 0]])
    ul = torch.tensor([3, 4, 3])
    assert adjacent_repeats(tg, ul).tolist() == [1, 0, 2]
    tl = torch.tensor([4, 3, 4])  # need 4, 4, 5 frames
    assert feasible(tl, tg, ul).tolist() == [True, False, False]
    stats = FeasibilityStats()
    lp = torch.log_softmax(torch.randn(3, 4, 7), -1)
    total, ntok, ok = ctc_loss(lp, tl, tg, ul, stats)
    assert ok.tolist() == [True, False, False] and ntok == 3 and torch.isfinite(total)
    assert stats.total == 3 and stats.infeasible == 2
    assert stats.summary()["ctc_infeasible"] == 2


def test_ctc_all_infeasible_is_zero_with_graph_and_counted():
    stats = FeasibilityStats()
    lp = torch.log_softmax(torch.randn(1, 2, 5, requires_grad=True), -1)
    total, ntok, ok = ctc_loss(lp, torch.tensor([2]), torch.tensor([[1, 1, 1]]),
                               torch.tensor([3]), stats)
    assert ntok == 0 and stats.infeasible == 1 and float(total) == 0.0
    total.backward()  # still differentiable, contributes nothing


def test_tiny_forward_backward_decreases_loss():
    torch.manual_seed(3)
    m = ASRModel(PRESETS["TINY"])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    feats = torch.randn(2, 60, 80)
    fl = torch.tensor([60, 44])
    rt, rl = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([3, 2])
    ct, cl = torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1])
    first = None
    for _ in range(40):
        out = m.losses(feats, fl, rt, rl, ct, cl)
        opt.zero_grad()
        out["loss"].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
        first = first if first is not None else float(out["loss"])
    assert float(out["loss"]) < 0.5 * first
    toks, forced = m.greedy(feats, fl)
    assert len(toks) == 2 and forced >= 0


def test_logmel_shape_and_causal_framing():
    lm = LogMel()
    wav = torch.randn(16000) * 0.1
    f = lm(wav)
    assert f.shape == (98, 80)
    g = lm(torch.cat([wav, torch.randn(800)]))
    assert torch.allclose(f, g[:98], atol=1e-5)


def test_pack_respects_budget():
    d = [1.0, 5.0, 2.0, 9.0, 3.0, 3.0, 12.0]
    batches = pack(d, budget_s=12.0, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(len(d)))
    for b in batches:
        assert len(b) <= 3 and (len(b) == 1 or max(d[i] for i in b) * len(b) <= 12.0)
