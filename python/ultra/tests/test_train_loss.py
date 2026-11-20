"""RNN-T loss: recursion vs brute-force enumeration; chunked joiner vs full-lattice autograd."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.transducer import (Joiner, StatelessPredictor, full_lattice_logprobs,  # noqa: E402
                              joiner_logprobs, rnnt_bruteforce, rnnt_nll)


def _fixture(seed=0, B=3, T=5, U=3, V=6, J=8, act="tanh"):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    joiner = Joiner(4, J, V, act).double()
    pred = StatelessPredictor(V, 4, J).double()
    enc = torch.randn(B, T, 4, generator=g, dtype=torch.float64)
    targets = torch.randint(1, V, (B, U), generator=g)
    t_lens = torch.tensor([T, T - 1, 2])[:B]
    u_lens = torch.tensor([U, 2, 1])[:B]
    return joiner, pred, enc, targets, t_lens, u_lens


def test_recursion_matches_bruteforce():
    joiner, pred, enc, targets, t_lens, u_lens = _fixture()
    with torch.no_grad():
        lpb, lpl = full_lattice_logprobs(joiner, joiner.enc_proj(enc),
                                         pred(StatelessPredictor.contexts(targets)), targets)
    nll = rnnt_nll(lpb, lpl, t_lens, u_lens)
    for b in range(3):
        ref = rnnt_bruteforce(lpb[b], lpl[b], int(t_lens[b]), int(u_lens[b]))
        assert torch.allclose(nll[b], ref, atol=1e-10), (b, nll[b], ref)


def test_probabilities_sum_to_one_over_label_sequences():
    # With V=2 (blank + one symbol) the transducer distribution over U in {0..} with T=2 frames:
    # sum over all label sequences up to a bound of exp(-nll) must approach 1.
    torch.manual_seed(1)
    joiner = Joiner(3, 5, 2).double()
    pred = StatelessPredictor(2, 3, 5).double()
    enc = torch.randn(1, 2, 3, dtype=torch.float64)
    total = 0.0
    for u in range(0, 30):
        tg = torch.ones(1, max(u, 1), dtype=torch.long)
        with torch.no_grad():
            lpb, lpl = full_lattice_logprobs(joiner, joiner.enc_proj(enc),
                                             pred(StatelessPredictor.contexts(tg)), tg)
        total += float(torch.exp(-rnnt_nll(lpb, lpl, torch.tensor([2]), torch.tensor([u]))))
    assert abs(total - 1.0) < 1e-6


@pytest.mark.parametrize("act", ["tanh", "clipped_relu"])
@pytest.mark.parametrize("budget", [1 << 20, 50])  # one chunk, and many 1-frame chunks
def test_chunked_joiner_matches_full_lattice(act, budget):
    joiner, pred, enc, targets, t_lens, u_lens = _fixture(seed=3, act=act)
    params = list(joiner.parameters()) + list(pred.parameters())

    def run(chunked):
        for p in params:
            p.grad = None
        e = enc.clone().requires_grad_(True)
        ep, pp = joiner.enc_proj(e), pred(StatelessPredictor.contexts(targets))
        if chunked:
            lpb, lpl = joiner_logprobs(joiner, ep, pp, targets, budget_elems=budget)
        else:
            lpb, lpl = full_lattice_logprobs(joiner, ep, pp, targets)
        loss = rnnt_nll(lpb, lpl, t_lens, u_lens).sum()
        loss.backward()
        return loss.detach(), e.grad.clone(), [p.grad.clone() for p in params]

    l1, ge1, gp1 = run(True)
    l2, ge2, gp2 = run(False)
    assert torch.allclose(l1, l2, atol=1e-10)
    assert torch.allclose(ge1, ge2, atol=1e-9)
    for a, b in zip(gp1, gp2, strict=True):
        assert torch.allclose(a, b, atol=1e-9)


def test_gradcheck_recursion():
    torch.manual_seed(5)
    lpb = torch.log_softmax(torch.randn(2, 4, 3, 2, dtype=torch.float64), -1)[..., 0]
    lpl = torch.randn(2, 4, 2, dtype=torch.float64).clamp(max=-0.1).requires_grad_(True)
    lpb = lpb.clone().requires_grad_(True)
    f = lambda a, b: rnnt_nll(a, b, torch.tensor([4, 3]), torch.tensor([2, 1]))  # noqa: E731
    assert torch.autograd.gradcheck(f, (lpb, lpl))


def test_torchaudio_crosscheck_if_available():
    ta = pytest.importorskip("torchaudio")
    joiner, pred, enc, targets, t_lens, u_lens = _fixture(seed=7)
    with torch.no_grad():
        ep, pp = joiner.enc_proj(enc), pred(StatelessPredictor.contexts(targets))
        lpb, lpl = full_lattice_logprobs(joiner, ep, pp, targets)
        z = joiner.logits(ep[:, :, None], pp[:, None]).float()
    ours = rnnt_nll(lpb, lpl, t_lens, u_lens)
    ref = ta.functional.rnnt_loss(z, targets.int(), t_lens.int(), u_lens.int(), blank=0,
                                  reduction="none")
    assert torch.allclose(ours.float(), ref, atol=1e-4)
