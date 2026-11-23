"""H44 training-side fixes (research/H44, AGENT_CHAT 2026-11-23). Each test is a witness: it fails on
origin/main e61121e and passes with the fix.

1. An explicit offline eval mode; a configured chunk of 0 keeps meaning "the model's own chunk".
2. Chunk legality checked at startup for every chunk the run will use.
3. Per-frame objectives (z-loss, CR-CTC KL, balancer) normalised by global frames under DDP.
4. Stream context dropped consistently: InterCTC sees the same utterance frames as the other heads.
5. KD cache identity (tokenizer sha256 + shard hashes via KD_META.json) and lookup counts with a
   frame-count check.
"""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel  # noqa: E402
from train.data import Example, collate  # noqa: E402
from train.trainer import Trainer, TrainConfig, model_config_for  # noqa: E402


# ---- 1. explicit offline mode ------------------------------------------------------------------
class _Greedy:
    def __init__(self):
        self.seen = []

    def eval(self):
        return self

    def greedy(self, feats, lens, chunk):
        self.seen.append(chunk)
        return [[1] for _ in range(feats.shape[0])], 0


def _stub(eval_chunks="", chunk=0):
    g = torch.Generator().manual_seed(0)
    dev = [Example(f"d{i}", "A B", torch.randn(120, 80, generator=g), [1], [1]) for i in range(2)]
    tok = types.SimpleNamespace(decode=lambda t: "A B")
    cfg = types.SimpleNamespace(chunk=chunk, eval_chunks=eval_chunks)
    stub = types.SimpleNamespace(model=_Greedy(), dev=dev, tok=tok, cfg=cfg, device=torch.device("cpu"),
                                 opt=object(), ema=None)
    stub._evaluate = lambda s, c=None: Trainer._evaluate(stub, s, c)
    stub._evaluate_ema = lambda c=None: Trainer._evaluate_ema(stub, c)
    return stub


def test_offline_is_an_explicit_eval_token_and_zero_inherits():
    stub = _stub(eval_chunks="8,offline")
    out = Trainer._evaluate_extra_chunks(stub)
    assert set(out) == {"wer_c8", "wer_offline"}
    assert stub.model.seen == [8, 0]  # 'offline' reaches the model as 0 (whole utterance)
    stub = _stub(chunk=0)
    Trainer._evaluate(stub, 300.0)
    assert stub.model.seen == [None]  # configured 0 = the model's own chunk, never offline
    stub = _stub(chunk=4)
    Trainer._evaluate(stub, 300.0)
    assert stub.model.seen == [4]


# ---- 2. chunk legality at startup --------------------------------------------------------------
def test_illegal_chunks_fail_at_startup():
    from train.chunks import check_plan

    mr = model_config_for(TrainConfig(preset="S_MR"))
    with pytest.raises(ValueError, match="even chunk"):
        check_plan(TrainConfig(preset="S_MR", chunk_modes="1,2,4"), mr)
    with pytest.raises(ValueError, match="even chunk"):
        check_plan(TrainConfig(preset="S_MR", eval_chunks="1"), mr)
    with pytest.raises(ValueError, match="offline"):
        check_plan(TrainConfig(preset="S_MR", chunk_modes="2,offline"), mr)
    check_plan(TrainConfig(preset="S_MR", chunk_modes="2,4,8", eval_chunks="16,offline"), mr)
    check_plan(TrainConfig(preset="S", chunk_modes="1,2,4,8,16", eval_chunks="1"), model_config_for(TrainConfig()))


# ---- 3. global frame normalisation -------------------------------------------------------------
def test_frame_mean_terms_are_normalised_by_global_frames(monkeypatch):
    import train.dist as dist

    totals = [40.0, 40.0, 0.0, 40.0]  # tokens (20 + 20), CTC tokens, KD frames, frames (10 + 30)
    calls = []

    def fake_reduce(values, device):
        calls.append(len(values))
        return totals[:len(values)] if len(calls) == 1 else list(values)

    monkeypatch.setattr(dist, "all_reduce_sum", fake_reduce)
    cfg = types.SimpleNamespace(kd_weight=0.0, zloss=0.0, cr_ctc_weight=0.5, balancer_weight=0.0,
                                inter_ctc_weight=0.0)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(ctc_weight=0.3, simple_loss_scale=0.5))
    stub = types.SimpleNamespace(cfg=cfg, model=model, world=2, device=torch.device("cpu"),
                                 _pruned_scale=lambda: 1.0)
    z = torch.zeros(())
    out = {"rnnt": z, "ctc": z, "rnnt_tokens": 20, "ctc_tokens": 20, "frames": 10,
           "cr_ctc": torch.tensor(1.0)}
    loss, _ = Trainer._global_loss(stub, out)
    # rank 0 holds 10 of 40 frames: its KL counts world * 10 / 40 = 0.5 of a per-rank mean
    assert float(loss) == pytest.approx(0.5 * 0.5 * 1.0)


# ---- 4. stream context on every head -----------------------------------------------------------
def test_stream_context_is_dropped_from_inter_ctc_too():
    from train.ctc import ctc_loss
    from train.stream_ctx import shift_rows

    torch.manual_seed(0)
    m = ASRModel(PRESETS["V2_TINY"]).eval()
    g = torch.Generator().manual_seed(4)
    exs = [Example(f"u{i}", "x", torch.randn(320 + 64 * i, 80, generator=g), [1, 2, 3], [1, 2], ctx=96)
           for i in range(2)]
    b = collate(exs)
    with torch.no_grad():
        out = m.losses(b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.ctc, b.ctc_lens, ctx=b.ctx,
                       inter_ctc_weight=0.3)
        _, enc_lens, aux = m.encoder(m.cmvn(b.feats), b.feat_lens, None, return_aux=True)
        il, lens = shift_rows(aux["inter_logits"].float(), enc_lens, b.ctx)
        isum, itok, _ = ctc_loss(torch.log_softmax(il, dim=-1), lens, b.ctc, b.ctc_lens, None)
    assert float(out["inter_ctc"]) == pytest.approx(float(isum / max(1, itok)), rel=1e-5)


# ---- 5. KD cache identity and counts -----------------------------------------------------------
def _cache(d, emitted=(0, 4, 0, 9)):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "shard_0000_sp1.0.npz"), meta=json.dumps({"speed": 1.0}),
             uid=np.array(["utt-1"]), text=np.array(["hello"]), hyp=np.array(["HELLO"]),
             n_frames=np.array([4], np.int32), emitted=np.array(emitted, np.int32),
             topk_ids=np.zeros((4, 2), np.int32), topk_lp=np.zeros((4, 2), np.float32))


def test_teacher_cache_must_prove_its_tokenizer(tmp_path):
    from train.kd import KdIdentityError, TeacherStore, write_meta

    cache = str(tmp_path / "cache")
    _cache(cache)
    tok_a, tok_b = tmp_path / "a.model", tmp_path / "b.model"
    tok_a.write_bytes(b"tokenizer A")
    tok_b.write_bytes(b"tokenizer B")
    import hashlib

    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tok_a, tok_b)}
    with pytest.raises(KdIdentityError):  # no side file and no tokenizer in the shard meta
        TeacherStore(cache, tokenizer_sha256=sha[tok_a])
    write_meta(cache, str(tok_a), "test teacher")
    assert len(TeacherStore(cache, tokenizer_sha256=sha[tok_a])) == 1
    with pytest.raises(KdIdentityError):
        TeacherStore(cache, tokenizer_sha256=sha[tok_b])
    _cache(cache, emitted=(0, 4, 0, 8))  # the shard changes after the side file was written
    with pytest.raises(KdIdentityError):
        TeacherStore(cache, tokenizer_sha256=sha[tok_a])


def test_teacher_lookups_are_counted_and_frame_checked(tmp_path):
    from train.kd import TeacherStore

    cache = str(tmp_path / "c")
    _cache(cache)
    st = TeacherStore(cache)
    assert st.get_checked("utt-1", 1.0, [5, 10], n_frames=4)[3] is True
    assert st.get_checked("utt-1", 1.0, [5, 10], n_frames=7)[3] is False  # 4 teacher frames vs 7
    assert st.get_checked("utt-1", 1.0, [5, 11], n_frames=4)[3] is False
    assert st.get_checked("utt-1", 0.9, [5, 10], n_frames=4) is None
    assert st.counts == {"accepted": 1, "missing": 1, "text_mismatch": 0, "path_mismatch": 1,
                         "frame_mismatch": 1}
