"""H12 diagnostic beam search: beam 1 equals greedy with one symbol per frame; wider beams return
valid token lists; empty-length utterances decode to nothing; a bad width is refused."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.beam import modified_beam_search  # noqa: E402
from train.transducer import Joiner, StatelessPredictor, greedy_decode  # noqa: E402


def _parts(seed=0, v=12):
    torch.manual_seed(seed)
    pred, join = StatelessPredictor(v, emb=8, joiner=16), Joiner(24, 16, v)
    enc_p = join.enc_proj(torch.randn(3, 20, 24))
    return pred.eval(), join.eval(), enc_p.detach(), torch.tensor([20, 13, 0])




def test_beam1_equals_greedy_one_symbol_per_frame():
    pred, join, enc_p, lens = _parts()
    with torch.no_grad():
        g, _ = greedy_decode(pred, join, enc_p, lens, max_symbols=1)
    assert modified_beam_search(pred, join, enc_p, lens, beam=1) == g
    assert g[2] == []


def test_wider_beam_runs_and_rejects_bad_width():
    pred, join, enc_p, lens = _parts(1)
    out = modified_beam_search(pred, join, enc_p, lens, beam=4)
    assert len(out) == 3 and out[2] == []
    assert all(isinstance(k, int) and k != 0 for s in out for k in s)
    with pytest.raises(ValueError):
        modified_beam_search(pred, join, enc_p, lens, beam=0)


def test_blank_penalty_and_length_norm():
    pred, join, enc_p, lens = _parts(2)
    base = modified_beam_search(pred, join, enc_p, lens, beam=4)
    assert modified_beam_search(pred, join, enc_p, lens, beam=4, blank_penalty=0.0) == base
    many = modified_beam_search(pred, join, enc_p, lens, beam=4, blank_penalty=50.0)
    # a huge blank penalty makes every frame emit a token
    assert [len(s) for s in many] == [int(n) for n in lens]
    ln = modified_beam_search(pred, join, enc_p, lens, beam=4, length_norm=True)
    assert len(ln) == 3 and ln[2] == []
    with pytest.raises(ValueError):
        modified_beam_search(pred, join, enc_p, lens, beam=4, blank_penalty=-1.0)
