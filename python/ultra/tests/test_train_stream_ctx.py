"""H41 arm 11: stream-context augmentation (train/stream_ctx.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel  # noqa: E402
from train.data import Example, collate  # noqa: E402
from train.features import LogMel  # noqa: E402
from train.stream_ctx import ContextBank, add_context, shift_rows  # noqa: E402


def _exs(n=6):
    g = torch.Generator().manual_seed(3)
    return [Example(f"u{i}", "x", torch.randn(300 + 40 * i, 80, generator=g), [1, 2, 3], [1, 2])
            for i in range(n)]


def test_context_is_seeded_capped_aligned_and_keeps_the_utterance():
    bank, exs = ContextBank(LogMel(), seconds=6.0), _exs()
    a = add_context(exs, bank, seed=1, epoch=0)
    b = add_context(exs, bank, seed=1, epoch=0)
    longest = max(e.feats.shape[0] for e in exs)
    for e, x, y in zip(exs, a, b, strict=True):
        assert torch.equal(x.feats, y.feats) and x.ctx == y.ctx  # deterministic
        assert x.ctx % 4 == 0 and 0 <= x.ctx <= 512
        assert x.feats.shape[0] == e.feats.shape[0] + x.ctx <= max(int(1.5 * longest), e.feats.shape[0])
        assert torch.equal(x.feats[x.ctx:], e.feats) and x.text == e.text and x.rnnt == e.rnnt
    c = add_context(exs, bank, seed=1, epoch=1)
    assert any(x.ctx != y.ctx for x, y in zip(a, c, strict=True))  # a new draw each epoch
    kinds = set()
    for ep in range(30):
        for x in add_context(exs, bank, seed=2, epoch=ep):
            if x.ctx:
                head = x.feats[:x.ctx]
                if torch.equal(head, bank.silence.expand(x.ctx, -1)):
                    kinds.add("silence")
                elif any(torch.equal(head[-8:], e.feats[-8:]) for e in exs):
                    kinds.add("tail")
                else:
                    kinds.add("noise_or_tail_with_gap")
    assert {"silence", "tail", "noise_or_tail_with_gap"} <= kinds


def test_losses_see_only_the_utterance_frames():
    torch.manual_seed(0)
    m = ASRModel(PRESETS["TINY"]).eval()
    exs = _exs(3)
    b0 = collate(exs)
    zero = collate([Example(e.uid, e.text, e.feats, e.rnnt, e.ctc, ctx=0) for e in exs])
    assert zero.ctx is None  # no context anywhere: the unchanged path
    args = (b0.feats, b0.feat_lens, b0.rnnt, b0.rnnt_lens, b0.ctc, b0.ctc_lens)
    with torch.no_grad():
        base = m.losses(*args)
        same = m.losses(*args, ctx=torch.zeros(3, dtype=torch.long))  # zero shift: identical
    assert float(base["loss"]) == float(same["loss"])
    enc = torch.arange(2 * 10 * 1, dtype=torch.float32).view(2, 10, 1)
    out, lens = shift_rows(enc, torch.tensor([10, 7]), torch.tensor([8, 0]))
    assert out[0, :8, 0].tolist() == list(range(2, 10)) + [9.0] * 0 and lens.tolist() == [8, 7]
    assert torch.equal(out[1], enc[1])
    bank = ContextBank(LogMel(), seconds=6.0)
    bc = collate(add_context(exs, bank, seed=5, epoch=0))
    assert bc.ctx is not None and abs(bc.audio_seconds - b0.audio_seconds) < 1e-9
    with torch.no_grad():
        out = m.losses(bc.feats, bc.feat_lens, bc.rnnt, bc.rnnt_lens, bc.ctc, bc.ctc_lens, ctx=bc.ctx)
    assert torch.isfinite(out["loss"]) and out["rnnt_tokens"] == base["rnnt_tokens"]
