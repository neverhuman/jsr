"""H46 +text tables: exact ids, B0 equivalence at init, gradient routing of the text loss."""
import pytest

torch = pytest.importorskip("torch")

from train.text_tables import PRIMES, TextPredictor, TextTables, ngram_ids, text_loss  # noqa: E402
from train.transducer import Joiner, StatelessPredictor  # noqa: E402


def test_ngram_ids_are_exact_base_501_numbers():
    c = torch.tensor([[0, 0, 7, 9], [500, 500, 500, 500]])
    assert ngram_ids(c[:, 2:]).tolist() == [7 * 501 + 9, 500 * 501 + 500]
    assert ngram_ids(c).tolist()[1] == 501 ** 4 - 1  # largest 4-gram id, no overflow


def test_tables_start_at_zero_output():
    t = TextTables(joiner=16, dim=8)
    assert t.width == 4
    out = t(torch.randint(0, 501, (3, 5, 4)))
    assert out.shape == (3, 5, 16) and torch.count_nonzero(out) == 0
    assert [e.num_embeddings for e in t.tables] == [PRIMES[2], PRIMES[3], PRIMES[4]]


def test_text_predictor_equals_the_plain_predictor_at_init():
    torch.manual_seed(0)
    plain = StatelessPredictor(501, emb=8, joiner=16, context_len=2)
    tp = TextPredictor(501, emb=8, joiner=16, dense_len=2, dim=8)
    tp.load_plain(plain.state_dict())
    assert tp.context_len == 4
    ctx4 = torch.randint(0, 501, (6, 4))
    assert torch.equal(tp(ctx4), plain(ctx4[:, 2:]))
    from train.errors import TextTablesError

    with pytest.raises(TextTablesError):
        tp.load_plain({"emb.weight": plain.emb.weight})  # proj missing: not a plain state


def test_text_loss_trains_only_the_tables():
    torch.manual_seed(0)
    tp = TextPredictor(501, emb=8, joiner=16, dense_len=2, dim=8)
    j = Joiner(d_enc=12, joiner=16, vocab_out=501)
    ctx, tgt = torch.randint(0, 501, (32, 4)), torch.randint(1, 501, (32,))
    loss = text_loss(tp, j, ctx, tgt, torch.zeros(16))
    loss.backward()
    assert tp.text.out.weight.grad is not None and torch.count_nonzero(tp.text.out.weight.grad)
    for p in (tp.emb.weight, tp.proj.weight, j.out.weight, j.enc_proj.weight):
        assert p.grad is None
