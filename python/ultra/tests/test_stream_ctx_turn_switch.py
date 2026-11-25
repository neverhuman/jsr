"""Witness tests for Track M1 turn_switch stream context & fail-closed side-feature export guard."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.r1_to_safetensors import ConvertError, convert, sha256_file  # noqa: E402
from train.asr import PRESETS, ASRModel  # noqa: E402
from train.data import Example, collate  # noqa: E402
from train.export import ExportError, export_bundle  # noqa: E402
from train.features import LogMel  # noqa: E402
from train.stream_ctx import ContextBank, _speaker_id, add_context  # noqa: E402
from train.trainer import TrainConfig  # noqa: E402


def _make_batch(uids: list[str], n_frames: int = 400) -> list[Example]:
    exs = []
    for idx, uid in enumerate(uids):
        # Distinct per-utterance feature signature so donor identity is unambiguously verifiable.
        feats = torch.full((n_frames, 80), float(idx + 1) * 10.0)
        feats[-8:] = float(idx + 1) * 100.0
        exs.append(Example(uid=uid, text="hello world", feats=feats, rnnt=[1, 2, 3], ctc=[1, 2]))
    return exs


def test_speaker_id_parsing_librispeech_and_librilight():
    assert _speaker_id("374-180298-0000") == "374"
    assert _speaker_id("1272-128104-0001") == "1272"
    assert _speaker_id("lls-254-12345-0001") == "lls-254"
    assert _speaker_id("llm-891-999-0002") == "llm-891"
    assert _speaker_id("u0") == "u0"


def test_h41_default_bit_identical():
    bank = ContextBank(LogMel(), seconds=6.0, seed=41)
    exs = _make_batch(["374-1-0000", "374-1-0001", "888-2-0000", "lls-254-1-0000"], n_frames=360)
    a = add_context(exs, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5)
    b = add_context(exs, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5, p=1.0, mode="h41", gap_s=0.0)
    for x, y in zip(a, b, strict=True):
        assert x.ctx == y.ctx
        assert torch.equal(x.feats, y.feats)


def test_turn_switch_diff_speaker_alignment_gap_and_rate():
    bank = ContextBank(LogMel(), seconds=6.0, noise=False)
    assert bank.noise == {}  # noise FFTs skipped when noise=False

    uids = [
        "374-180298-0000",
        "374-180298-0001",
        "374-180298-0002",
        "888-100001-0000",
        "lls-254-999-0000",
        "lls-254-999-0001",
        "lls-901-888-0000",
        "1272-128104-0000",
    ]
    exs = _make_batch(uids, n_frames=600)
    sig_to_spk = {float(i + 1) * 100.0: _speaker_id(u) for i, u in enumerate(uids)}

    # 1. With p=1.0, every utterance gets a different-speaker donor tail + exact 320 ms (32 frames) silence gap.
    out_all = add_context(exs, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch", gap_s=0.32)
    longest = max(e.feats.shape[0] for e in exs)
    for orig, aug in zip(exs, out_all, strict=True):
        assert aug.ctx > 0
        assert aug.ctx % 4 == 0
        assert aug.ctx >= (12 + 8) * 4  # >= 480 ms tail + 320 ms gap = 80 frames
        assert aug.ctx <= 400
        assert aug.feats.shape[0] == orig.feats.shape[0] + aug.ctx <= int(1.5 * longest)
        # Target frames preserved verbatim
        assert torch.equal(aug.feats[aug.ctx:], orig.feats)
        # Exact 320 ms (32 mel frames) silence gap immediately preceding target
        gap_slice = aug.feats[aug.ctx - 32 : aug.ctx]
        assert torch.equal(gap_slice, bank.silence.expand(32, -1))
        # Immediately preceding the silence gap is the trailing signature of a DIFFERENT speaker
        donor_marker = float(aug.feats[aug.ctx - 33, 0].item())
        assert donor_marker in sig_to_spk
        assert sig_to_spk[donor_marker] != _speaker_id(orig.uid)

    # 2. Fallback to any j != i when all utterances share one speaker.
    same_spk = _make_batch(["374-1-0000", "374-1-0001", "374-1-0002"], n_frames=600)
    out_fb = add_context(same_spk, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch", gap_s=0.32)
    for idx, aug in enumerate(out_fb):
        assert aug.ctx > 0 and aug.ctx % 4 == 0
        donor_marker = float(aug.feats[aug.ctx - 33, 0].item())
        assert donor_marker != float(idx + 1) * 100.0

    # 3. Empirical activation rate matches p=0.20 across many epochs.
    n_aug, n_tot = 0, 0
    for ep in range(250):
        res = add_context(exs, bank, seed=7919, epoch=ep, max_s=4.0, cap=1.5, p=0.20, mode="turn_switch", gap_s=0.32)
        for orig, aug in zip(exs, res, strict=True):
            n_tot += 1
            if aug.ctx > 0:
                n_aug += 1
                assert aug.ctx % 4 == 0
                assert torch.equal(aug.feats[aug.ctx - 32 : aug.ctx], bank.silence.expand(32, -1))
            else:
                assert torch.equal(aug.feats, orig.feats)
    rate = n_aug / n_tot
    assert 0.16 <= rate <= 0.24, f"empirical turn_switch rate {rate:.4f} outside [0.16, 0.24]"

    # 4. Collated batch preserves audio_seconds excluding context and runs cleanly through ASRModel.losses.
    b = collate(out_all)
    b0 = collate(exs)
    assert b.ctx is not None and abs(b.audio_seconds - b0.audio_seconds) < 1e-9
    model = ASRModel(PRESETS["TINY"]).eval()
    with torch.no_grad():
        loss_out = model.losses(b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.ctc, b.ctc_lens, ctx=b.ctx)
    assert torch.isfinite(loss_out["loss"])


def test_turn_switch_mixed_same_diff_balance_alignment_and_skip():
    bank = ContextBank(LogMel(), seconds=6.0, noise=False)
    # Every speaker has >= 2 utterances so both same-speaker and different-speaker donors exist.
    uids = [
        "374-180298-0000",
        "374-180298-0001",
        "374-180298-0002",
        "888-100001-0000",
        "888-100001-0001",
        "lls-254-999-0000",
        "lls-254-999-0001",
        "lll-901-888-a1b2-0000",
        "lll-901-888-a1b2-0001",
    ]
    exs = _make_batch(uids, n_frames=600)
    sig_to_idx = {float(i + 1) * 100.0: i for i in range(len(uids))}
    sig_to_spk = {float(i + 1) * 100.0: _speaker_id(u) for i, u in enumerate(uids)}

    # 1. Determinism and exact 25 Hz alignment + 32-frame silence gap + target preservation.
    a = add_context(exs, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch_mixed", gap_s=0.32)
    b = add_context(exs, bank, seed=7919, epoch=0, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch_mixed", gap_s=0.32)
    longest = max(e.feats.shape[0] for e in exs)
    for idx, (orig, x, y) in enumerate(zip(exs, a, b, strict=True)):
        assert x.ctx == y.ctx and torch.equal(x.feats, y.feats)
        assert x.ctx > 0 and x.ctx % 4 == 0
        assert (12 + 8) * 4 <= x.ctx <= 400
        assert x.feats.shape[0] == orig.feats.shape[0] + x.ctx <= int(1.5 * longest)
        assert torch.equal(x.feats[x.ctx:], orig.feats)
        assert torch.equal(x.feats[x.ctx - 32 : x.ctx], bank.silence.expand(32, -1))
        donor_marker = float(x.feats[x.ctx - 33, 0].item())
        assert donor_marker in sig_to_idx
        assert sig_to_idx[donor_marker] != idx  # never self-donor

    # 2. Empirical 50/50 split of same-speaker vs different-speaker at p=0.40 (~20% same, ~20% diff, ~60% none).
    n_same = n_diff = n_none = 0
    for ep in range(300):
        res = add_context(exs, bank, seed=7919, epoch=ep, max_s=4.0, cap=1.5, p=0.40, mode="turn_switch_mixed", gap_s=0.32)
        for idx, (orig, aug) in enumerate(zip(exs, res, strict=True)):
            if aug.ctx == 0:
                n_none += 1
                assert torch.equal(aug.feats, orig.feats)
                continue
            assert aug.ctx % 4 == 0
            assert torch.equal(aug.feats[aug.ctx - 32 : aug.ctx], bank.silence.expand(32, -1))
            donor_marker = float(aug.feats[aug.ctx - 33, 0].item())
            assert sig_to_idx[donor_marker] != idx
            if sig_to_spk[donor_marker] == _speaker_id(orig.uid):
                n_same += 1
            else:
                n_diff += 1
    n_tot = n_same + n_diff + n_none
    same_rate, diff_rate = n_same / n_tot, n_diff / n_tot
    assert 0.16 <= same_rate <= 0.24, f"same_rate {same_rate:.4f} outside [0.16, 0.24]"
    assert 0.16 <= diff_rate <= 0.24, f"diff_rate {diff_rate:.4f} outside [0.16, 0.24]"

    # 3. When no same-speaker donor exists, same-speaker coin flips skip context (never leak into diff-speaker).
    all_distinct = _make_batch(["101-1-0000", "202-1-0000", "303-1-0000", "404-1-0000"], n_frames=600)
    dist_sig_to_spk = {float(i + 1) * 100.0: _speaker_id(e.uid) for i, e in enumerate(all_distinct)}
    d_aug = d_tot = 0
    for ep in range(250):
        res = add_context(all_distinct, bank, seed=7919, epoch=ep, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch_mixed", gap_s=0.32)
        for orig, aug in zip(all_distinct, res, strict=True):
            d_tot += 1
            if aug.ctx > 0:
                d_aug += 1
                donor_marker = float(aug.feats[aug.ctx - 33, 0].item())
                assert dist_sig_to_spk[donor_marker] != _speaker_id(orig.uid)
    assert 0.44 <= d_aug / d_tot <= 0.56, f"distinct-speaker rate {d_aug / d_tot:.4f} should be ~0.50"

    # 4. When all utterances share one speaker, diff-speaker coin flips skip context (never leak into same-speaker).
    all_same = _make_batch(["374-1-0000", "374-1-0001", "374-1-0002", "374-1-0003"], n_frames=600)
    s_aug = s_tot = 0
    for ep in range(250):
        res = add_context(all_same, bank, seed=7919, epoch=ep, max_s=4.0, cap=1.5, p=1.0, mode="turn_switch_mixed", gap_s=0.32)
        for idx, aug in enumerate(res):
            s_tot += 1
            if aug.ctx > 0:
                s_aug += 1
                donor_marker = float(aug.feats[aug.ctx - 33, 0].item())
                assert donor_marker != float(idx + 1) * 100.0
    assert 0.44 <= s_aug / s_tot <= 0.56, f"same-speaker-only rate {s_aug / s_tot:.4f} should be ~0.50"

    # 5. Collated batch preserves audio_seconds excluding context and runs cleanly through ASRModel.losses.
    bc = collate(a)
    b0 = collate(exs)
    assert bc.ctx is not None and abs(bc.audio_seconds - b0.audio_seconds) < 1e-9
    model = ASRModel(PRESETS["TINY"]).eval()
    with torch.no_grad():
        loss_out = model.losses(bc.feats, bc.feat_lens, bc.rnnt, bc.rnnt_lens, bc.ctc, bc.ctc_lens, ctx=bc.ctx)
    assert torch.isfinite(loss_out["loss"])

    # 6. Unknown mode raises ValueError.
    with pytest.raises(ValueError, match="unknown stream_ctx mode"):
        add_context(exs, bank, seed=7919, epoch=0, mode="bogus")


def _save_ckpt(path, payload: dict) -> str:
    torch.save(payload, path)
    marker = str(path)[: -len(".pt")] + ".sha256"
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"{sha256_file(str(path))}\n")
    return str(path)


def test_r1_to_safetensors_and_export_fail_closed_on_side_features(tmp_path):
    cfg = PRESETS["TINY"].to_dict()
    model = ASRModel(PRESETS["TINY"]).eval()
    base_sd = model.state_dict()

    # Case 1: side_mode="aux"
    cfg_aux = dict(cfg, encoder={**cfg["encoder"], "side_mode": "aux"})
    p_aux = _save_ckpt(tmp_path / "ckpt_side_aux.pt", {"model": base_sd, "model_config": cfg_aux, "step": 100})
    with pytest.raises(ConvertError, match="side_mode"):
        convert(p_aux, str(tmp_path / "out_aux"), bundle=None)
    with pytest.raises(ExportError, match="side_mode"):
        export_bundle(p_aux, str(tmp_path / "onnx_aux"), int8=False)

    # Case 2: side_film_dim > 0
    cfg_film = dict(cfg, encoder={**cfg["encoder"], "side_film_dim": 16})
    p_film = _save_ckpt(tmp_path / "ckpt_side_film.pt", {"model": base_sd, "model_config": cfg_film, "step": 100})
    with pytest.raises(ConvertError, match="side_film_dim"):
        convert(p_film, str(tmp_path / "out_film"), bundle=None)
    with pytest.raises(ExportError, match="side_film_dim"):
        export_bundle(p_film, str(tmp_path / "onnx_film"), int8=False)

    # Case 3: TrainConfig exposes stream_ctx_mode / stream_ctx_p / stream_ctx_gap_s defaults.
    tc = TrainConfig()
    assert tc.stream_ctx_mode == "h41" and tc.stream_ctx_p == 1.0 and tc.stream_ctx_gap_s == 0.0
