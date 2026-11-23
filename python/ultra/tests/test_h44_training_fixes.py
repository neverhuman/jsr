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
def _cache(d, emitted=(0, 4, 0, 9, 0, 0, 0, 0)):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "shard_0000_sp1.0.npz"), meta=json.dumps({"speed": 1.0}),
             uid=np.array(["utt-1"]), text=np.array(["hello"]), hyp=np.array(["HELLO"]),
             n_frames=np.array([8], np.int32), emitted=np.array(emitted, np.int32),
             topk_ids=np.zeros((8, 2), np.int32), topk_lp=np.zeros((8, 2), np.float32))


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
    _cache(cache, emitted=(0, 4, 0, 8, 0, 0, 0, 0))  # the shard changes after the side file was written
    with pytest.raises(KdIdentityError):
        TeacherStore(cache, tokenizer_sha256=sha[tok_a])


def test_teacher_lookups_are_counted_and_frame_checked(tmp_path):
    from train.kd import TeacherStore

    cache = str(tmp_path / "c")
    _cache(cache)
    st = TeacherStore(cache)
    assert st.get_checked("utt-1", 1.0, [5, 10], n_mel=40)[3] is True  # r0_frames(40) = 8 = the cache's
    assert st.get_checked("utt-1", 1.0, [5, 10], n_mel=50)[3] is False  # 16 expected: other audio
    assert st.get_checked("utt-1", 1.0, [5, 11], n_mel=40)[3] is False
    assert st.get_checked("utt-1", 0.9, [5, 10], n_mel=40) is None
    assert st.counts == {"accepted": 1, "missing": 1, "text_mismatch": 0, "path_mismatch": 1,
                         "frame_mismatch": 1}


# Real R0 cache rows (teacher_cache_R0 train-clean-100 shard 0, speed 1.0): "n_samples:R0 frames" from the
# parquet audio headers and the cache's n_frames (kd-cache, 2026-11-23: R0 - student frames spans -3..+4).
R0_ROWS = (
    "232480:360,257360:400,212720:336,178000:280,225280:352,218960:344,213280:336,252000:392,219520:344,25096"
    "0:392,226480:352,210960:328,249120:392,207440:328,226400:352,237440:368,245600:384,229360:360,251680:392"
    ",132880:208,229760:360,207440:328,241280:376,238400:376,244960:384,226960:352,47360:72,244640:384,240560"
    ":376,242000:376,241680:376,248880:392,220640:344,223600:352,251920:392,137680:216,221200:344,230240:360,"
    "146320:232,139680:216,252480:392,275360:432,214880:336,165840:256,251600:392,179680:280,64320:104,261040"
    ":408,261760:408,231520:360,220720:344,248640:392,253520:400,203440:320,228160:360,224640:352,213120:336,"
    "237680:368,210400:328,151520:240,135601:216,228881:360,204640:320,226960:352,242960:384,99440:152,237360"
    ":368,52640:80,44960:72,239360:376,216560:336,66320:104,246800:384,220080:344,210720:328,225760:352,18936"
    "0:296,239920:376,203440:320,238720:376,235280:368,222800:352,221600:344,235680:368,209440:328,222800:352"
    ",255040:400,250560:392,242000:376,175120:272,237760:368,213200:336,234640:368,233520:368,192400:304,2168"
    "00:336,231280:360,226160:352,244480:384,235760:368,256560:400,157040:248,243520:384,152400:240,153840:24"
    "0,242640:376,256960:400,236800:368,239520:376,123840:192,236640:368,241040:376,255680:400,239600:376,230"
    "160:360,242880:376,201840:312,200880:312,76320:120,250240:392"
)


def test_r0_frame_rule_matches_real_cache_rows_where_the_tolerance_failed():
    """kd-cache's finding: the +-1 tolerance against the student's encoder frames rejected ~63% of valid
    R0 rows. The exact rule r0_frames(n_mel) reproduces every real row."""
    from train.features import LogMel
    from train.kd import r0_frames
    from train.model import stem_out_len

    lm = LogMel()
    rows = [tuple(int(v) for v in p.split(":")) for p in "".join(R0_ROWS).split(",")]
    assert len(rows) == 120
    assert all(r0_frames(lm.num_frames(n)) == f for n, f in rows)
    old_ok = sum(abs(f - stem_out_len(lm.num_frames(n))) <= 1 for n, f in rows)
    assert old_ok < 0.5 * len(rows)  # the old check rejected most of these valid rows
    deltas = {f - stem_out_len(lm.num_frames(n)) for n, f in rows}
    assert min(deltas) >= -3 and max(deltas) <= 4
