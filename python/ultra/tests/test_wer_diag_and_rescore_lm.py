"""Tests for Wave 0 WER diagnostics (`wer_diag.py`) and Wave 1B-2 rescorers (`nlm_rescore.py`)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.errors import DataError  # noqa: E402
from train.model import EncoderConfig  # noqa: E402
from train.nlm_rescore import (  # noqa: E402
    DEFAULT_COMPOUND_MAP,
    RESCORE_PRESETS,
    RescoreLmConfig,
    apply_compound_align,
    build_compound_align_map,
    build_rescore_lm,
    load_rescore_ckpt,
    rescore_nbest_records,
    save_rescore_ckpt,
    score_sequences,
    train_rescore_lm,
)
from train.wer_diag import (  # noqa: E402
    eval_greedy_nll_and_align,
    eval_nbest_oracle,
    guard_utts,
    load_split_guard,
    pruned_alignment_batch,
    score_nbest_oracle,
)
from ultra.libri import Utt  # noqa: E402


class _FakeTok:
    vocab = 10
    num_outputs = 11
    path = "fake"

    def encode(self, text: str) -> list[int]:
        return [1 + (abs(hash(w)) % 9) for w in text.lower().split()] or [1]

    def decode(self, ids: list[int]) -> str:
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
        return " ".join(words[(i - 1) % len(words)] for i in ids if i > 0)


def _tiny_pruned_model() -> ASRModel:
    torch.manual_seed(7)
    ecfg = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5, chunk=4, left_context=8, dropout=0.0)
    mcfg = ModelConfig(encoder=ecfg, vocab_rnnt=11, vocab_ctc=11, pred_emb=8, joiner=16, rnnt_loss="pruned", prune_range=5)
    return ASRModel(mcfg).eval()


def test_split_guard_blocks_confirmation_and_test_splits(tmp_path):
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps({"utterances": {"selection": ["u1", "u2"], "confirmation": ["u3", "u4"]}}), encoding="utf-8")
    sel, conf = load_split_guard(sp)
    assert sel == {"u1", "u2"} and conf == {"u3", "u4"}
    u_ok = [Utt("u1", 1, "hello", np.zeros(1600, dtype=np.int16))]
    guard_utts("dev-clean", u_ok, conf)
    with pytest.raises(DataError, match="forbidden test split"):
        guard_utts("test-clean", u_ok, conf)
    with pytest.raises(DataError, match="confirmation-half"):
        guard_utts("dev-clean", [Utt("u3", 2, "nope", np.zeros(1600, dtype=np.int16))], conf)


def test_k2_align_stats_parity_and_wave0_pipeline(tmp_path):
    model = _tiny_pruned_model()
    tok = _FakeTok()
    rng = np.random.default_rng(11)
    utts = [
        Utt("u1", 1, "alpha bravo charlie delta", rng.integers(-1000, 1000, 9600, dtype=np.int16)),
        Utt("u2", 1, "echo foxtrot golf", rng.integers(-1000, 1000, 8000, dtype=np.int16)),
    ]
    res = eval_greedy_nll_and_align(model, tok, utts, chunk=4, batch_size=2)
    ast = res["align_stats"]
    assert ast["mi_max_abs_diff"] < 1e-4
    assert ast["entropy_mean"] >= 0.0 and ast["entropy_per_tok_mean"] >= 0.0
    assert 0.0 <= ast["active_edge_occ_mean"] <= 1.0
    assert 0.0 <= ast["simple_oob_mass_mean"] <= 1.0
    assert math.isfinite(res["rnnt_nll_per_tok"]) and math.isfinite(res["ctc_nll_per_tok"])

    nb_out = tmp_path / "nb.jsonl"
    orc = eval_nbest_oracle(model, tok, utts, chunk=4, beam=4, workers=1, out_jsonl=nb_out)
    assert nb_out.is_file()
    assert orc["pooled"]["oracle_wer"]["k16"] <= orc["pooled"]["oracle_wer"]["k1"]
    assert orc["pooled"]["recoverable_edits"] >= 0

    rows = [
        {"uid": "a", "ref": "hello world", "nbest": [[[1], -1.0, "hello word"], [[2], -1.2, "hello world"]]},
    ]
    o = score_nbest_oracle(rows, ks=(1, 2))
    assert o["pooled"]["oracle_edits"] == {"k1": 1, "k2": 0}
    assert o["pooled"]["recoverable_edits"] == 1


def test_rescore_lm_presets_budgets_and_int8_sizes(tmp_path):
    m_gru = build_rescore_lm(RESCORE_PRESETS["ultra_lm_2m"])
    p_gru = sum(p.numel() for p in m_gru.parameters())
    assert 1_750_000 <= p_gru <= 1_950_000
    sz_gru = save_rescore_ckpt(m_gru, tmp_path / "gru_i8.pt", quantize_int8=True)
    assert sz_gru < 2_100_000

    m_tf = build_rescore_lm(RESCORE_PRESETS["l_ref_tf"])
    p_tf = sum(p.numel() for p in m_tf.parameters())
    assert 3_000_000 <= p_tf <= 3_800_000
    sz_tf = save_rescore_ckpt(m_tf, tmp_path / "tf_i8.pt", quantize_int8=True)
    assert sz_tf < 4_000_000


@pytest.mark.parametrize(("arch", "reverse"), [("ultra_lm_2m", False), ("l_ref_tf", False), ("l_ref_tf", True)])
def test_rescore_lm_train_int8_roundtrip_and_rerank(tmp_path, arch: str, reverse: bool):
    cfg = RescoreLmConfig(arch=arch, vocab=32, d_model=32, d_hidden=64, n_layers=2, n_heads=4, ctx=32, reverse=reverse)
    seqs = [[3, 7, 11, 15, 19, 23], [4, 8, 12, 16, 20, 24]] * 24
    ckpt = tmp_path / f"{arch}_{reverse}.pt"
    model, summary = train_rescore_lm(cfg, lambda: seqs, seqs[:4], ckpt, steps=12, batch_size=16, lr=2e-2, seed=5, device="cpu", bf16=False)
    assert summary["final_loss"] < summary["init_loss"]
    loaded, _ = load_rescore_ckpt(ckpt)
    s_fp32 = score_sequences(model, [[3, 7, 11, 15], [29, 28, 27, 26]])
    s_i8 = score_sequences(loaded, [[3, 7, 11, 15], [29, 28, 27, 26]])
    assert abs(s_fp32[0] - s_i8[0]) < 0.35
    assert s_i8[0] > s_i8[1]  # trained motif scores strictly higher than unseen tokens

    records = [
        {"uid": "u1", "ref": "good", "nbest": [[[29, 28, 27, 26], -1.0, "bad", -0.5], [[3, 7, 11, 15], -1.5, "good", -1.5]]},
    ]
    rescored, met = rescore_nbest_records(
        records,
        l2r_model=None if reverse else loaded,
        r2l_model=loaded if reverse else None,
        lm_scale=0.0 if reverse else 0.5,
        r2l_scale=0.5 if reverse else 0.0,
        ilme_scale=0.2,
    )
    assert rescored[0]["hyp"] == "good"
    assert met["wer"] == 0.0


def test_compound_alignment_map_derivation_and_application():
    train_lines = ["every one said all right"] * 12 + ["everyone said alright"] * 2 + ["death like silence"] * 8 + ["deathlike"] * 1
    cmap = build_compound_align_map(train_lines, min_split_count=5, ratio_thresh=1.5)
    assert cmap["everyone"] == "every one"
    assert cmap["alright"] == "all right"
    assert cmap["deathlike"] == "death like"
    assert apply_compound_align("Everyone felt deathlike but alright", DEFAULT_COMPOUND_MAP) == "every one felt death like but all right"
