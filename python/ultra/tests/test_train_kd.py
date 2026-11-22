"""H22 distillation: TeacherStore indexing (u_t, ok flag), KD batch collation, and the collapsed
KL: ~0 when the teacher equals the student, positive otherwise, masked frames excluded."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.data import Example, collate  # noqa: E402
from train.kd import TeacherStore, kd_loss, r0_to_student  # noqa: E402
from train.transducer import Joiner  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _cache(d):
    os.makedirs(d, exist_ok=True)
    emitted = np.array([0, 5, 0, 7, 0, 0, 3], np.int16)  # utt a: 4 frames, utt b: 3 frames
    ids = np.tile(np.arange(8, dtype=np.int16), (7, 1))
    lp = np.full((7, 8), -3.0, np.float16)
    np.savez(os.path.join(d, "train-clean-100_0000_sp1.0.npz"), uid=np.array(["a", "b"]),
             text=np.array(["HELLO THERE", "NO"]), hyp=np.array(["hello there", "yes"]),
             n_frames=np.array([4, 3], np.int32), topk_ids=ids, topk_lp=lp, emitted=emitted,
             meta=np.array(json.dumps({"speed": 1.0, "topk": 8})))


def test_store_indexes_frames_and_flags(tmp_path):
    _cache(str(tmp_path))
    st = TeacherStore(str(tmp_path))
    assert len(st) == 2 and st.topk == 8
    ids, lp, u, ok = st.get("a", 1.0)
    assert ids.shape == (4, 8) and list(u) == [0, 0, 1, 1] and ok
    _, _, u, ok = st.get("b", 1.0)
    assert list(u) == [0, 0, 0] and not ok  # R0 said "yes" for "NO": not on the reference lattice
    assert st.get("a", 0.9) is None


def test_collate_masks_teacher_frames(tmp_path):
    _cache(str(tmp_path))
    st = TeacherStore(str(tmp_path))
    exs = [Example("a", "x", torch.zeros(20, 80), [1, 2], [1], kd=st.get("a", 1.0)),
           Example("b", "y", torch.zeros(10, 80), [1], [1], kd=st.get("b", 1.0)),
           Example("c", "z", torch.zeros(10, 80), [1], [1])]
    ids, lp, u, mask = collate(exs).kd
    assert ids.shape == (3, 4, 8) and mask.tolist() == [[True] * 4, [False] * 4, [False] * 4]
    assert collate([exs[2]]).kd is None


def test_kl_zero_for_self_teacher_and_positive_otherwise():
    torch.manual_seed(0)
    V, J, B, T, U = 21, 16, 2, 6, 3
    join = Joiner(24, J, V)
    enc_p = join.enc_proj(torch.randn(B, T, 24)).detach().requires_grad_()
    pred_p = torch.randn(B, U + 1, J)
    enc_lens, rnnt_lens = torch.tensor([6, 4]), torch.tensor([3, 2])
    u = torch.tensor([[0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 2, 2]])
    with torch.no_grad():
        pp = torch.gather(pred_p, 1, u[..., None].expand(-1, -1, J))
        logp = torch.log_softmax(join.logits(enc_p, pp), -1)
    # student index 1 is R0 piece 0, which is R0's blank symbol and never a teacher token
    top = logp.masked_fill(torch.arange(V) == 1, float("-inf")).topk(8, -1)
    r0_ids = torch.where(top.indices > 0, top.indices - 1, torch.zeros_like(top.indices))
    assert torch.equal(r0_to_student(r0_ids), top.indices)
    mask = torch.ones(B, T, dtype=torch.bool)
    kl, n = kd_loss(join, enc_p, pred_p, enc_lens, rnnt_lens, r0_ids, top.values, u, mask)
    assert n == 6 + 4 and float(kl) < 1e-5
    other = torch.zeros_like(top.values) - 2.5  # a different teacher
    kl2, _ = kd_loss(join, enc_p, pred_p, enc_lens, rnnt_lens, r0_ids, other, u, mask)
    assert float(kl2) > 0.01
    kl2.backward()
    assert torch.isfinite(enc_p.grad).all() and float(enc_p.grad.abs().sum()) > 0
    none, n0 = kd_loss(join, enc_p, pred_p, enc_lens, rnnt_lens, r0_ids, other, u,
                       torch.zeros(B, T, dtype=torch.bool))
    assert n0 == 0 and float(none) == 0.0


def test_tokenizer_follows_the_model_case(tmp_path):
    import sentencepiece as spm

    from train.tokenizer import Tokenizer

    corpus = tmp_path / "t.txt"
    corpus.write_text("\n".join(["HELLO THERE GENERAL", "THE QUICK BROWN FOX", "HELLO WORLD"] * 50))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "up"),
                                   vocab_size=30, model_type="bpe", character_coverage=1.0)
    tok = Tokenizer(str(tmp_path / "up.model"))
    assert tok.upper
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "HELLO WORLD"
    assert all(i - 1 != tok.sp.unk_id() for i in ids)
