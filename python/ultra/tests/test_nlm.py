"""H46 neural token LM: context windows reset at sentence breaks, model shapes, fusion rows."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.nlm import ARCHS, VOCAB, NeuralFusion, contexts, fake_quant_sym_rows, make_model  # noqa: E402


def test_contexts_reset_at_sentence_breaks():
    #                  0  1  2  3  4  5  6  7
    s = torch.tensor([0, 5, 6, 7, 0, 9, 10, 11], dtype=torch.int16)
    got = contexts(s, torch.tensor([3, 6, 7, 1]), 3).tolist()
    assert got[0] == [0, 5, 6]                    # before position 3: tokens 5 6, BOS-padded
    assert got[1] == [0, 0, 9]                    # the 7 and the break before 9 are masked
    assert got[2] == [0, 9, 10]
    assert got[3] == [0, 0, 0]                    # first token of the stream: all BOS


def test_every_arch_maps_contexts_to_501_log_probs():
    for arch, (ctx, _, _, _) in ARCHS.items():
        m = make_model(arch)
        lp = m(torch.zeros(3, ctx, dtype=torch.long))
        assert lp.shape == (3, VOCAB)
        assert torch.allclose(lp.exp().sum(1), torch.ones(3), atol=1e-4)
        m.qat = True  # the int8 round trip keeps the shapes and stays normalised
        assert torch.allclose(m(torch.zeros(2, ctx, dtype=torch.long)).exp().sum(1),
                              torch.ones(2), atol=1e-4)


def test_symmetric_row_quantiser_matches_the_exporter_rule():
    w = torch.tensor([[0.5, -1.0, 0.25], [0.0, 0.0, 0.0]])
    q = fake_quant_sym_rows(w)
    assert torch.allclose(q[0], torch.round(w[0] / (1.0 / 127)) * (1.0 / 127))
    assert torch.equal(q[1], w[1])


def test_fusion_rows_blank_zero_and_cached():
    m = make_model("0.5m").eval()
    f = NeuralFusion(m, 0.5)
    rows = f.stack([(), (3,), (3,)])
    assert rows.shape == (3, VOCAB)
    assert np.all(rows[:, 0] == 0)            # blank gets no LM term
    assert np.array_equal(rows[1], rows[2])   # one row per distinct context
    assert f.contexts == 2
    assert f._key((1, 2, 3, 4, 5, 6, 7)) == (2, 3, 4, 5, 6, 7)
    assert f._key((7,)) == (0, 0, 0, 0, 0, 7)
