"""H48c: the MFA-anchored tilted valid-path RNN-T loss (train/anchored_rnnt.py).

Invariants (JSR final engineering plan §8.3 + the H48 brief):
  * kappa = 0 (and kappa > 0 with no anchors) reproduces k2's objective bit for bit, values,
    occupancy gradients and the model's batch-normalised loss;
  * finite-difference gradients match the k2 occupancy gradients of the tilted objective;
  * on tiny enumerated lattices: path mass, path counts, arc marginals, dL/dkappa = E_q[C] and
    dE_q[C]/dkappa = -Var_q(C);
  * frames after the MFA anchor + margin are penalised; emissions before the evidence are never
    rewarded (the uniform k2 delay penalty is shown to reward them, for contrast).
"""
import dataclasses
import itertools
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
k2 = pytest.importorskip("k2")
from train.anchored_rnnt import (AnchorLoss, anchored_pruned_rnnt, late_cost,  # noqa: E402
                                 rnnt_loss_pruned_tilted, rnnt_loss_smoothed_tilted, tilt_px)
from train.asr import PRESETS, ASRModel  # noqa: E402

NEG = float("-inf")


# ---------------------------------------------------------------------------------------------
# helpers


def _lattice(S, T, seed, dtype=torch.float64):
    """Random regular-RNN-T px (1, S, T + 1) / py (1, S + 1, T) log-probs (no label at t = T)."""
    g = torch.Generator().manual_seed(seed)
    px = torch.randn(1, S, T + 1, generator=g, dtype=dtype)
    px[:, :, T] = NEG
    py = torch.randn(1, S + 1, T, generator=g, dtype=dtype)
    bd = torch.tensor([[0, 0, S, T]], dtype=torch.int64)
    return px, py, bd


def _paths(S, T):
    """Every valid path as its label-emission frames (t_0 <= ... <= t_{S-1} <= T - 1)."""
    return list(itertools.combinations_with_replacement(range(T), S))


def _path_terms(px, py, cost, frames, S, T):
    """(score, cost) of one path given its emission frames, walking the regular lattice."""
    s, t, score, c = 0, 0, 0.0, 0.0
    for u in range(S):
        while t < frames[u]:
            score += float(py[0, s, t])
            t += 1
        score += float(px[0, s, t])
        c += float(cost[0, s, t])
        s += 1
    while t < T:
        score += float(py[0, s, t])
        t += 1
    return score, c


def _enumerate(px, py, cost, kappa, S, T):
    terms = [_path_terms(px, py, cost, f, S, T) for f in _paths(S, T)]
    w = torch.tensor([sc - kappa * c for sc, c in terms], dtype=torch.float64)
    q = torch.softmax(w, 0)
    cs = torch.tensor([c for _, c in terms], dtype=torch.float64)
    return torch.logsumexp(w, 0), q, cs, terms


def _tilted_logz(px, py, bd, cost, kappa, return_grad=False):
    return k2.mutual_information_recursion(px=tilt_px(px.clone(), cost, kappa), py=py,
                                           boundary=bd, return_grad=return_grad)


def _model(prune_range=3, simple_scale=0.5):
    torch.manual_seed(3)
    cfg = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned", prune_range=prune_range,
                              simple_loss_scale=simple_scale,
                              encoder=dataclasses.replace(PRESETS["TINY"].encoder, dropout=0.0))
    return ASRModel(cfg).eval()


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(3, 64, 80, generator=g)
    feat_lens = torch.tensor([64, 50, 33])
    rnnt = torch.randint(1, 11, (3, 6), generator=g)
    rnnt_lens = torch.tensor([6, 4, 2])
    ctc = torch.randint(1, 7, (3, 3), generator=g)
    ctc_lens = torch.tensor([3, 2, 1])
    # anchors: encoder frames (16 / 13 / 9 valid frames), -1 = padding / unanchored
    anchor = torch.tensor([[1, 3, 3, 6, 9, 12], [2, 2, 7, 10, -1, -1], [4, -1, -1, -1, -1, -1]])
    return (feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens), anchor


def _bits(a, b):
    return torch.equal(a, b) or (a.shape == b.shape and bool(
        ((a == b) | (torch.isnan(a) & torch.isnan(b))).all()))


# ---------------------------------------------------------------------------------------------
# kappa = 0 is k2 bit for bit


@pytest.mark.parametrize("dp", [0.0, 0.005])
def test_kappa_zero_equals_k2_smoothed_and_pruned_bit_for_bit(dp):
    g = torch.Generator().manual_seed(1)
    B, S, T, C = 2, 4, 7, 9
    lm, am = torch.randn(B, S + 1, C, generator=g), torch.randn(B, T, C, generator=g)
    symbols = torch.randint(1, C, (B, S), generator=g)
    bd = torch.tensor([[0, 0, 4, 7], [0, 0, 3, 5]])
    cost = late_cost(torch.tensor([[0, 1, 2, 3], [1, 1, 2, -1]]), T + 1, 1, 12.0)
    ref, (rpx, rpy) = k2.rnnt_loss_smoothed(lm=lm, am=am, symbols=symbols, termination_symbol=0,
                                            lm_only_scale=0.25, am_only_scale=0.0, boundary=bd,
                                            rnnt_type="regular", reduction="none",
                                            return_grad=True, delay_penalty=dp)
    for c, kap in ((None, 0.0), (cost, 0.0), (torch.zeros_like(cost), 0.7)):
        got, (gpx, gpy) = rnnt_loss_smoothed_tilted(lm, am, symbols, 0, 0.25, 0.0, bd,
                                                    delay_penalty=dp, cost=c, kappa=kap)
        assert _bits(got, ref) and _bits(gpx, rpx) and _bits(gpy, rpy)
    ranges = k2.get_rnnt_prune_ranges(px_grad=rpx, py_grad=rpy, boundary=bd, s_range=3)
    logits = torch.randn(B, T, 3, C, generator=g)
    ref = k2.rnnt_loss_pruned(logits=logits, symbols=symbols, ranges=ranges, termination_symbol=0,
                              boundary=bd, rnnt_type="regular", reduction="none", delay_penalty=dp)
    for c, kap in ((None, 0.0), (cost, 0.0), (torch.zeros_like(cost), 0.7)):
        got = rnnt_loss_pruned_tilted(logits, symbols, ranges, 0, bd, delay_penalty=dp, cost=c,
                                      kappa=kap)
        assert _bits(got, ref)
    # the untilted diagnostic shares px / py with the tilted loss and is k2's loss bit for bit
    tilted, plain = rnnt_loss_pruned_tilted(logits, symbols, ranges, 0, bd, delay_penalty=dp,
                                            cost=cost, kappa=0.7, return_plain=True)
    assert _bits(plain, ref) and bool((tilted >= ref).all()) and not plain.requires_grad


@pytest.mark.parametrize("dp", [0.0, 0.005])
def test_model_kappa_zero_and_unanchored_batches_reproduce_the_existing_loss(dp):
    m = _model()
    b, anchor = _batch()
    kw = {"delay_penalty": dp} if dp else {}
    base = m.losses(*b, **kw)
    # inactive spec: the unchanged call
    off = m.losses(*b, anchor=anchor, anchor_spec=AnchorLoss(kappa=0.0), **kw)
    # active spec but no anchored token: px - 0 * cost = px, the same numbers through the new path
    none = m.losses(*b, anchor=torch.full_like(anchor, -1), anchor_spec=AnchorLoss(kappa=0.9), **kw)
    for out in (off, none):
        for k in ("loss", "rnnt", "rnnt_simple", "ctc"):
            assert float(out[k]) == float(base[k]), k
        assert torch.equal(out["nll"], base["nll"])
    assert float(none["anchor_risk"]) == 0.0 and none["anchor_tokens"] == 0
    # the new code path itself at kappa = 0 (forced past the `active` switch): same NLL, gradients
    enc, lens = m.encode(b[0], b[1])
    enc = enc.float()
    from train.transducer import StatelessPredictor

    enc_p = m.joiner.enc_proj(enc).float()
    pred_p = m.predictor(StatelessPredictor.contexts(b[2], m.cfg.pred_ctx)).float()
    nll0, simple0, _ = m._pruned_rnnt(enc, enc_p, pred_p, b[2], lens, b[3], delay_penalty=dp)
    spec = AnchorLoss(kappa=0.0)
    nll1, simple1, _, diag = anchored_pruned_rnnt(m, enc, enc_p, pred_p, b[2], lens, b[3], anchor,
                                                  spec, delay_penalty=dp)
    assert torch.equal(nll0, nll1) and torch.equal(simple0, simple1)
    params = [p for p in m.parameters() if p.requires_grad]
    g0 = torch.autograd.grad((nll0.sum() + simple0.sum()), params, retain_graph=True,
                             allow_unused=True)
    g1 = torch.autograd.grad((nll1.sum() + simple1.sum()), params, allow_unused=True)
    for a, c in zip(g0, g1, strict=True):
        assert (a is None and c is None) or torch.equal(a, c)
    assert float(diag["anchor_risk"]) == 0.0


# ---------------------------------------------------------------------------------------------
# gradients


def test_finite_difference_gradients_match_for_the_tilted_pruned_and_smoothed_losses():
    g = torch.Generator().manual_seed(5)
    B, S, T, C, R = 2, 3, 5, 6, 3
    symbols = torch.randint(1, C, (B, S), generator=g)
    bd = torch.tensor([[0, 0, 3, 5], [0, 0, 2, 4]])
    cost = late_cost(torch.tensor([[0, 1, 1], [1, 2, -1]]), T + 1, 1, 3.0, dtype=torch.float64)
    lm = torch.randn(B, S + 1, C, generator=g, dtype=torch.float64, requires_grad=True)
    am = torch.randn(B, T, C, generator=g, dtype=torch.float64, requires_grad=True)

    def smoothed(lm_, am_):
        return rnnt_loss_smoothed_tilted(lm_, am_, symbols, 0, 0.25, 0.0, bd, delay_penalty=0.005,
                                         cost=cost, kappa=0.8, return_grad=False).sum()

    assert torch.autograd.gradcheck(smoothed, (lm, am), eps=1e-6, atol=1e-5, rtol=1e-4)
    with torch.no_grad():
        _, (pxg, pyg) = rnnt_loss_smoothed_tilted(lm, am, symbols, 0, 0.25, 0.0, bd, cost=cost,
                                                  kappa=0.8)
        ranges = k2.get_rnnt_prune_ranges(px_grad=pxg.float(), py_grad=pyg.float(), boundary=bd,
                                          s_range=R)
    logits = torch.randn(B, T, R, C, generator=g, dtype=torch.float64, requires_grad=True)

    def pruned(lg):
        return rnnt_loss_pruned_tilted(lg, symbols, ranges, 0, bd, delay_penalty=0.005, cost=cost,
                                       kappa=0.8).sum()

    assert torch.autograd.gradcheck(pruned, (logits,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_occupancy_gradients_are_the_finite_difference_of_the_tilted_log_partition():
    S, T = 3, 4
    px, py, bd = _lattice(S, T, seed=11)
    cost = late_cost(torch.tensor([[0, 1, 1]]), T + 1, 0, 12.0, dtype=torch.float64)
    kap = 1.3
    _, (pxg, pyg) = _tilted_logz(px, py, bd, cost, kap, return_grad=True)
    eps = 1e-6
    for s in range(S):
        for t in range(T):
            hi, lo = px.clone(), px.clone()
            hi[0, s, t] += eps
            lo[0, s, t] -= eps
            fd = (_tilted_logz(hi, py, bd, cost, kap) - _tilted_logz(lo, py, bd, cost, kap)) / (2 * eps)
            assert abs(float(fd) - float(pxg[0, s, t])) < 1e-6
    for s in range(S + 1):
        for t in range(T):
            hi, lo = py.clone(), py.clone()
            hi[0, s, t] += eps
            lo[0, s, t] -= eps
            fd = (_tilted_logz(px, hi, bd, cost, kap) - _tilted_logz(px, lo, bd, cost, kap)) / (2 * eps)
            assert abs(float(fd) - float(pyg[0, s, t])) < 1e-6


# ---------------------------------------------------------------------------------------------
# tiny enumerated lattices


@pytest.mark.parametrize("S,T,seed", [(1, 3, 0), (2, 3, 1), (3, 4, 2), (2, 5, 3), (4, 4, 4)])
def test_enumerated_lattices_mass_counts_marginals_and_kappa_moments(S, T, seed):
    px, py, bd = _lattice(S, T, seed)
    g = torch.Generator().manual_seed(100 + seed)
    anchor = torch.randint(-1, T, (1, S), generator=g)
    cost = late_cost(anchor, T + 1, int(seed % 2), 2.0 + seed, dtype=torch.float64)
    # path count: unit weights give C(S + T - 1, S) valid paths (no label at t = T)
    zero_x, zero_y = torch.zeros_like(px), torch.zeros_like(py)
    zero_x[:, :, T] = NEG
    count = k2.mutual_information_recursion(px=zero_x, py=zero_y, boundary=bd)
    assert len(_paths(S, T)) == math.comb(S + T - 1, S)
    assert abs(float(count.exp()) - math.comb(S + T - 1, S)) < 1e-9 * math.comb(S + T - 1, S)
    for kap in (0.0, 0.4, 2.5):
        logz, (pxg, pyg) = _tilted_logz(px, py, bd, cost, kap, return_grad=True)
        ref, q, cs, terms = _enumerate(px, py, cost, kap, S, T)
        assert abs(float(logz) - float(ref)) < 1e-10  # path mass
        # arc marginals of the tilted posterior = k2 occupancies
        lab = torch.zeros(S, T + 1, dtype=torch.float64)
        for qi, f in zip(q, _paths(S, T), strict=True):
            for u in range(S):
                lab[u, f[u]] += qi
        assert torch.allclose(lab, pxg[0], atol=1e-10)
        # flow conservation: one arc per anti-diagonal step carries all the mass
        occ = torch.zeros(S + T, dtype=torch.float64)
        for s in range(S + 1):
            for t in range(T + 1):
                if s < S:
                    occ[s + t] += pxg[0, s, t]
                if t < T:
                    occ[s + t] += pyg[0, s, t]
        assert torch.allclose(occ, torch.ones_like(occ), atol=1e-10)
        # dL/dkappa = E_q[C] (L = -logZ) and E_q[C] equals the occupancy-weighted cost
        e_c = float((q * cs).sum())
        assert abs(e_c - float((pxg * cost).sum())) < 1e-10
        h = 1e-5
        dl = -(float(_tilted_logz(px, py, bd, cost, kap + h)) -
               float(_tilted_logz(px, py, bd, cost, kap - h))) / (2 * h)
        assert abs(dl - e_c) < 1e-6
        # dE_q[C]/dkappa = -Var_q(C)
        var = float((q * cs * cs).sum()) - e_c ** 2

        def expect(k):
            _, (gx, _) = _tilted_logz(px, py, bd, cost, k, return_grad=True)
            return float((gx * cost).sum())

        de = (expect(kap + h) - expect(kap - h)) / (2 * h)
        assert abs(de + var) < 1e-5


# ---------------------------------------------------------------------------------------------
# late frames penalised, early frames never rewarded


def test_late_cost_is_zero_through_anchor_plus_margin_then_linear_and_capped():
    c = late_cost(torch.tensor([[3, -1, 0]]), 12, margin=2, cap=4.0)
    assert c.shape == (1, 3, 12)
    assert c[0, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4]
    assert c[0, 1].tolist() == [0] * 12  # unanchored token: no cost
    assert c[0, 2].tolist() == [0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
    assert float(c.min()) == 0.0 and not c.requires_grad


def _emission_posterior(px, py, bd, cost, kappa, dp=0.0):
    """Posterior over the emission frame of the single token of an S = 1 lattice."""
    p = px.clone()
    if dp:
        from train.anchored_rnnt import _k2_delay_penalty

        _k2_delay_penalty(p, bd, dp)
    _, (pxg, _) = _tilted_logz(p, py, bd, cost, kappa, return_grad=True)
    return pxg[0, 0, :-1]


def test_frames_after_the_anchor_are_penalised_and_earlier_frames_never_rewarded():
    T, a, m = 10, 4, 1
    px, py, bd = _lattice(1, T, seed=21)
    cost = late_cost(torch.tensor([[a]]), T + 1, m, 12.0, dtype=torch.float64)
    p = _emission_posterior(px, py, bd, cost, 0.0)
    q = _emission_posterior(px, py, bd, cost, 0.6)
    ratio = q / p
    early = ratio[: a + m + 1]
    # no frame at or before anchor + margin is preferred over another: one common factor
    assert torch.allclose(early, early[0].expand_as(early), rtol=1e-10)
    assert float(early[0]) >= 1.0  # they only gain the mass taken from late frames
    late = ratio[a + m:]
    assert bool((late[1:] < late[:-1]).all())  # each later frame strictly penalised
    # the uniform k2 delay penalty, by contrast, rewards every earlier frame, also before the word
    r_dp = _emission_posterior(px, py, bd, None, 0.0, dp=0.05) / p
    assert bool((r_dp[1: a + m + 1] < r_dp[: a + m]).all())


def test_tilt_never_lowers_the_loss_and_is_free_when_no_path_is_late():
    S, T = 3, 6
    px, py, bd = _lattice(S, T, seed=31)
    anchor = torch.tensor([[1, 2, 4]])
    cost = late_cost(anchor, T + 1, 0, 12.0, dtype=torch.float64)
    l0 = -float(_tilted_logz(px, py, bd, cost, 0.0))
    for kap in (0.1, 1.0, 5.0):
        assert -float(_tilted_logz(px, py, bd, cost, kap)) >= l0 - 1e-12  # R = L_k - L_0 >= 0
    # forbid every late emission: all remaining paths are on time and the tilt changes nothing,
    # however early they emit (no reward for earliness)
    on_time = px.clone()
    on_time[cost > 0] = NEG
    l0 = -float(_tilted_logz(on_time, py, bd, cost, 0.0))
    assert -float(_tilted_logz(on_time, py, bd, cost, 3.0)) == pytest.approx(l0, abs=1e-12)


def test_model_tilt_raises_the_pruned_loss_only_through_late_paths_and_trains():
    m = _model(simple_scale=0.5)
    b, anchor = _batch(seed=2)
    base = m.losses(*b, delay_penalty=0.005)
    spec = AnchorLoss(kappa=0.5, margin=1, cap=8.0, simple=False)  # same band as the base loss
    out = m.losses(*b, delay_penalty=0.005, anchor=anchor, anchor_spec=spec)
    assert float(out["rnnt_simple"]) == float(base["rnnt_simple"])  # untilted simple loss
    assert bool((out["nll"] >= base["nll"] - 1e-6).all())
    assert float(out["anchor_risk"]) >= 0.0 and out["anchor_tokens"] == 6 + 4 + 1
    assert abs(float(out["anchor_risk"]) - float((out["nll"] - base["nll"]).sum()) / 12) < 1e-5
    out2 = m.losses(*b, delay_penalty=0.005, anchor=anchor, anchor_spec=dataclasses.replace(spec, simple=True))
    assert float(out2["rnnt_simple"]) >= float(base["rnnt_simple"]) - 1e-6
    out2["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_anchored_loss_rejects_the_exact_loss_and_bad_settings():
    torch.manual_seed(3)
    m = ASRModel(PRESETS["TINY"])
    b, anchor = _batch()
    with pytest.raises(ValueError, match="needs rnnt_loss='pruned'"):
        m.losses(*b, anchor=anchor, anchor_spec=AnchorLoss(kappa=0.1))
    for bad in ({"kappa": -1.0}, {"margin": -1}, {"cap": 0.0}):
        with pytest.raises(ValueError):
            AnchorLoss(**bad)
    with pytest.raises(ValueError):
        late_cost(torch.zeros(3), 4, 1, 1.0)
    with pytest.raises(ValueError):
        tilt_px(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), 0.5)
