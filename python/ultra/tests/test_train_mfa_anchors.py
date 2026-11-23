"""H48c: MFA anchor side file (train/mfa_anchors.py) and its data-path hooks.

  * the evidence-frame rule is the exact window arithmetic of LogMel + the causal stem, at every
    speed the loader uses, and matches what the real featurizer + encoder do;
  * the builder reads only allowlisted TRAIN members (dev / test members are never opened), maps
    MFA tokens onto the reference words (apostrophe pieces, <unk>) and records coverage;
  * the reader keys by (utterance id, speed), refuses unknown speeds and changed transcripts, and
    maps tokens to their word's end; collate pads anchors with -1.
"""
import dataclasses
import json
import math
import os
import sys
import zipfile
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.errors import DataError  # noqa: E402
from train.mfa_anchors import (SPEEDS, MfaAnchors, build, convert, evidence_frame,  # noqa: E402
                               speed_ratio, text_hash, token_word_index, word_start_table)

PIECES = ["<unk>", "▁the", "▁wick", "er", "'s", "▁cat", "▁z", "orb", "lax",
          "▁hello", "▁world", "▁good", "bye"]


class FakeSP:
    def get_piece_size(self):
        return len(PIECES)

    def id_to_piece(self, i):
        return PIECES[i]


def ids(*pieces):
    return [PIECES.index(p) + 1 for p in pieces]  # model output index = piece id + 1


def textgrid(words, phones, xmax):
    def tier(name, ivs):
        out = ['\t\tclass = "IntervalTier"', f'\t\tname = "{name}"', "\t\txmin = 0.0",
               f"\t\txmax = {xmax}", f"\t\tintervals: size = {len(ivs)}"]
        for i, (a, b, t) in enumerate(ivs, 1):
            out += [f"\t\t\tintervals [{i}]:", f"\t\t\t\txmin = {a}", f"\t\t\t\txmax = {b}",
                    f'\t\t\t\ttext = "{t}"']
        return out

    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0.0",
             f"xmax = {xmax}", "tiers? <exists>", "size = 2", "item []:", "\titem [1]:"]
    lines += tier("words", words) + ["\titem [2]:"] + tier("phones", phones)
    return "\n".join(lines) + "\n"


GOOD_WORDS = [(0.0, 0.31, ""), (0.31, 0.45, "the"), (0.45, 0.80, "wicker"), (0.80, 0.93, "s"),
              (0.93, 1.27, "cat"), (1.27, 1.31, ""), (1.31, 1.90, "<unk>"), (1.90, 2.30, "")]
GOOD_PHONES = [(0.0, 0.31, "sil"), (0.31, 0.37, "DH"), (0.37, 0.45, "AH0"), (0.45, 0.80, "W"),
               (0.80, 0.93, "Z"), (0.93, 1.05, "K"), (1.05, 1.27, "AE1"), (1.27, 1.31, "sp"),
               (1.31, 1.90, "spn"), (1.90, 2.30, "")]
TEXTS = {"19-198-0001": ("THE WICKER'S CAT ZORBLAX", "train-clean-100"),
         "19-198-0002": ("GOODBYE WORLD", "train-clean-100"),
         "19-198-0003": ("HELLO WORLD", "train-clean-100"),
         "84-121-0004": ("HELLO WORLD", "train-other-500")}


def _zip(tmp_path):
    p = tmp_path / "librispeech_alignments.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("train-clean-100/19/198/19-198-0001.TextGrid", textgrid(GOOD_WORDS, GOOD_PHONES, 2.3))
        z.writestr("train-clean-100/19/198/19-198-0002.TextGrid",
                   textgrid([(0.1, 0.5, "hello"), (0.5, 0.9, "world")], [(0.1, 0.9, "HH")], 1.0))
        z.writestr("train-clean-100/unaligned.txt", "19-198-0003 beam too narrow\n")
        z.writestr("train-other-500/84/121/84-121-0004.TextGrid",
                   textgrid([(0.2, 0.6, "hello"), (0.6, 1.1, "world")],
                            [(0.2, 0.6, "HH"), (0.6, 1.1, "W")], 1.2))
        # sealed / evaluation members: must never be opened by the train builder
        z.writestr("test-clean/1/2/1-2-0003.TextGrid", "SEALED: NEVER READ")
        z.writestr("dev-clean/1/2/1-2-0004.TextGrid", "EVALUATION ONLY: NEVER READ")
    return str(p)


# ---------------------------------------------------------------------------------------------
# grid


def test_evidence_frame_is_exact_window_arithmetic_at_every_loader_speed():
    assert [speed_ratio(s) for s in SPEEDS] == [(9, 10), (1, 1), (11, 10)]
    for s in SPEEDS:
        num, den = speed_ratio(s)
        assert Fraction(s).limit_denominator(100) == Fraction(num, den)
        tau = np.arange(0, 6000)
        got = evidence_frame(tau, s)
        # first encoder frame t whose stem window end 640 t + 400 samples covers 16 tau den / num
        want = [max(0, math.ceil((Fraction(16 * int(x) * den, num) - 400) / 640)) for x in tau]
        assert got.tolist() == want
    assert evidence_frame(25) == 0 and evidence_frame(26) == 1 and evidence_frame(65) == 1
    assert evidence_frame(66) == 2
    assert evidence_frame(900, 0.9) == evidence_frame(1000) == evidence_frame(1100, 1.1) == 25


def test_evidence_frame_matches_the_real_featurizer_and_causal_stem():
    """Perturb the last sample of a 'word' and find the first encoder frame that moves (chunk 1:
    attention sees no future frame): it is the evidence frame of the word end."""
    from train.asr import PRESETS, ASRModel
    from train.features import LogMel

    torch.manual_seed(0)
    cfg = dataclasses.replace(PRESETS["TINY"], encoder=dataclasses.replace(
        PRESETS["TINY"].encoder, dropout=0.0))
    m = ASRModel(cfg).eval()
    lm = LogMel()
    g = torch.Generator().manual_seed(1)
    wav = 0.1 * torch.randn(16000, generator=g)
    with torch.no_grad():
        base, _ = m.encode(lm(wav)[None], torch.tensor([lm.num_frames(16000)]), chunk=1)
        for end_ms in (30, 410, 455, 640, 641, 799):
            s = 16 * end_ms - 1  # last sample of a word that ends at end_ms
            w = wav.clone()
            w[s] += 0.5
            enc, _ = m.encode(lm(w)[None], torch.tensor([lm.num_frames(16000)]), chunk=1)
            moved = (enc - base).abs().amax(dim=(0, 2)) > 0
            assert int(moved.nonzero()[0]) == int(evidence_frame(end_ms)), end_ms


def test_text_hash_and_token_word_index():
    assert text_hash("the  Cat") == text_hash("THE CAT") != text_hash("THE CATS")
    starts = word_start_table(FakeSP())
    assert starts[0] == 0 and starts[ids("▁the")[0]] and not starts[ids("er")[0]]
    toks = ids("▁the", "▁wick", "er", "'s", "▁cat", "▁z", "orb", "lax")
    assert token_word_index(toks, starts).tolist() == [0, 1, 1, 1, 2, 3, 3, 3]
    assert token_word_index(ids("er", "▁cat"), starts) is None  # must open a word
    assert token_word_index([], starts).tolist() == []


# ---------------------------------------------------------------------------------------------
# builder + reader


def test_convert_maps_apostrophe_pieces_unk_and_phones_to_reference_words():
    a_ms, words, phones = convert(textgrid(GOOD_WORDS, GOOD_PHONES, 2.3), "THE WICKER'S CAT ZORBLAX")
    assert a_ms == 2300
    assert words == [(310, 450), (450, 930), (930, 1270), (1310, 1900)]
    assert [p for p, *_ in phones] == ["DH", "AH0", "W", "Z", "K", "AE1", "spn"]
    assert [k for *_, k in phones] == [0, 0, 1, 1, 2, 2, 3]
    assert convert(textgrid([(0.1, 0.5, "hello")], [], 1.0), "GOODBYE") is None


def test_build_reads_train_members_only_and_round_trips(tmp_path, monkeypatch):
    zp = _zip(tmp_path)
    opened = []
    real_open = zipfile.ZipFile.open

    def spy(self, name, mode="r", *a, **k):
        if mode == "r":  # member reads (np.savez writes its own zip members in mode "w")
            opened.append(name if isinstance(name, str) else name.filename)
        return real_open(self, name, mode, *a, **k)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    man = build(zp, str(tmp_path / "out"), transcripts=TEXTS)
    monkeypatch.setattr(zipfile.ZipFile, "open", real_open)
    assert opened and all(n.startswith("train-") for n in opened)  # dev / test never opened
    c100 = man["splits"]["train-clean-100"]
    assert (c100["utterances"], c100["matched"], c100["token_mismatch"], c100["no_textgrid"],
            c100["unaligned_listed"]) == (3, 1, 1, 1, 1)
    assert c100["no_textgrid_ids"] == ["19-198-0003"] and c100["token_mismatch_ids"] == ["19-198-0002"]
    assert man["splits"]["train-other-500"]["pct_words_timed"] == 100.0
    out = tmp_path / "out"
    assert json.loads((out / "MANIFEST.json").read_text())["files"].keys() == {
        "mfa_anchors_train960_v1.npz", "mfa_phones_train960_v1.npz"}
    with np.load(out / "mfa_phones_train960_v1.npz") as z:
        inv = [str(x) for x in z["phone_inventory"]]
        assert [inv[i] for i in z["phone_id"][:7]] == ["DH", "AH0", "W", "Z", "K", "AE1", "spn"]
        assert z["phone_end_frame"].shape[1] == 3
    store = MfaAnchors(str(out / "mfa_anchors_train960_v1.npz"), FakeSP())
    assert len(store) == 2 and store.speeds == SPEEDS
    ends_ms = np.array([450, 930, 1270, 1900])
    for s in SPEEDS:
        assert store.word_ends("19-198-0001", s).tolist() == evidence_frame(ends_ms, s).tolist()
    toks = ids("▁the", "▁wick", "er", "'s", "▁cat", "▁z", "orb", "lax")
    e = evidence_frame(ends_ms, 1.1)
    got = store.token_anchors("19-198-0001", 1.1, toks, "THE WICKER'S CAT ZORBLAX")
    assert got.tolist() == [e[0], e[1], e[1], e[1], e[2], e[3], e[3], e[3]]
    with pytest.raises(DataError, match="speed"):
        store.word_ends("19-198-0001", 1.05)
    with pytest.raises(DataError, match="transcript"):
        store.token_anchors("19-198-0001", 1.0, toks, "THE WICKER'S DOG ZORBLAX")
    assert store.token_anchors("19-198-0003", 1.0, ids("▁hello", "▁world"), "HELLO WORLD") is None
    assert store.token_anchors("19-198-0001", 1.0, ids("er", "▁cat"), "THE WICKER'S CAT ZORBLAX") is None
    assert store.counts == {"anchored": 1, "missing": 1, "token_mismatch": 1}


def test_build_refuses_dev_and_test_splits(tmp_path):
    zp = _zip(tmp_path)
    for split in ("dev-clean", "dev-other", "test-clean", "test-other"):
        with pytest.raises(DataError, match="training splits only"):
            build(zp, str(tmp_path / "x"), splits=(split,), transcripts=TEXTS)


# ---------------------------------------------------------------------------------------------
# data path


def test_collate_pads_anchors_and_loader_hook_attaches_them(tmp_path):
    from train.data import Example, ParquetBatches, collate

    zp = _zip(tmp_path)
    build(zp, str(tmp_path / "out"), transcripts=TEXTS)
    store = MfaAnchors(str(tmp_path / "out" / "mfa_anchors_train960_v1.npz"), FakeSP())
    toks = ids("▁hello", "▁world")
    fake = SimpleNamespace(teacher=None, anchors=store)
    ex = ParquetBatches._attach(fake, ("84-121-0004", "HELLO WORLD", np.zeros((120, 80), np.float32),
                                       toks, [1, 2], 0.9))
    assert ex.anchor.tolist() == evidence_frame([600, 1100], 0.9).tolist()
    plain = Example("u0", "HELLO", torch.zeros(50, 80), [3, 4, 5], [1])
    b = collate([ex, plain])
    assert b.anchor.tolist() == [ex.anchor.tolist() + [-1], [-1, -1, -1]]
    assert collate([plain]).anchor is None  # no anchors anywhere: field stays off
    bad = dataclasses.replace(ex, anchor=np.array([1], np.int32))
    with pytest.raises(DataError, match="anchors for"):
        collate([bad])
    assert b.to("cpu").anchor.tolist() == b.anchor.tolist()


def test_trainer_runs_the_anchored_loss_and_keeps_calls_unchanged_when_off(tmp_path):
    from train.asr import PRESETS
    from train.data import Example, FixedBatches
    from train.trainer import TrainConfig, Trainer, _anchors

    pytest.importorskip("k2")
    exs = [Example(f"u{i}", "hi there", torch.randn(48, 80), [1, 2, 3], [1, 2],
                   anchor=np.array([0, 2, 4], np.int32)) for i in range(4)]
    mc = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned", prune_range=3)

    def trainer(d, **kw):
        cfg = TrainConfig(preset="TINY", total_steps=8, warmup=1, log_every=1, eval_every=0,
                          ckpt_every_steps=100, specaug=False, rnnt_loss="pruned", prune_range=3,
                          delay_penalty=0.005, **kw)
        return Trainer(str(tmp_path / d), cfg, FixedBatches(exs, batch_size=2), model_cfg=mc,
                       device="cpu", log=False)

    off = trainer("off")
    batch = next(iter(off.source.iterate(off.pos)))[1]
    assert off._anchor_kw(batch) == {}  # kappa 0: the loss call is unchanged
    assert off._watch_sig() == trainer("off2")._watch_sig()
    on = trainer("on", anchor_kappa=0.05, anchor_margin=1)
    kw = on._anchor_kw(batch)
    assert kw["anchor_spec"].kappa == 0.05 and kw["anchor"].shape == batch.rnnt.shape
    assert on._watch_sig() != off._watch_sig()
    r = on.train_step(batch)
    assert r["anchor_risk"] >= 0.0 and r["anchor_cov"] == 1.0 and math.isfinite(r["loss"])
    on.run(max_steps=2)
    assert all(torch.isfinite(p).all() for p in on.model.parameters())
    with pytest.raises(ValueError, match="needs mfa_anchors"):
        _anchors(TrainConfig(anchor_kappa=0.1, rnnt_loss="pruned"), None)
    with pytest.raises(ValueError, match="pruned"):
        _anchors(TrainConfig(anchor_kappa=0.1, mfa_anchors="x.npz"), None)
    assert _anchors(TrainConfig(mfa_anchors="x.npz"), None) is None  # kappa 0: not even loaded
