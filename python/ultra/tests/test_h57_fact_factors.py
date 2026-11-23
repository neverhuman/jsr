"""Unit tests and 4-step default-OFF bit-exactness witness for H57 FACT Factors B, D, and E."""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel, ModelConfig  # noqa: E402
from train.beam import modified_beam_search  # noqa: E402
from train.data import Example, FixedBatches, Position  # noqa: E402
from train.errors import CheckpointError, DataError, ShapeError, TextTablesError  # noqa: E402
from train.export import ExportError, export_bundle  # noqa: E402
from train.fact_aug import FactAugConfig, apply_fact_aug, list_train_only_audio  # noqa: E402
from train.fact_hooks import FactTextHelper, fact_aug_from_cfg, load_init_state  # noqa: E402
from train.model import Encoder, EncoderConfig  # noqa: E402
from train.recipe_opts import check_init_compatible, model_overrides  # noqa: E402
from train.text_tables import TextPredictor  # noqa: E402
from train.trainer import TrainConfig, Trainer, model_config_for  # noqa: E402


def _tiny_examples(seed: int = 0) -> list[Example]:
    g = torch.Generator().manual_seed(seed)
    return [
        Example(f"u{i}", "a b", torch.randn(48 + 8 * i, 80, generator=g), [1, 2, 3], [1, 2])
        for i in range(4)
    ]


# ---- Factor B: stochastic depth (layer_drop <= 0.1) -----------------------------------------
def test_factor_b_stochastic_depth_schedule_eval_identity_and_init_compat():
    with pytest.raises(ShapeError):
        Encoder(EncoderConfig(d_model=16, n_blocks=2, n_heads=2, layer_drop=-0.05))
    with pytest.raises(ShapeError):
        Encoder(EncoderConfig(d_model=16, n_blocks=2, n_heads=2, layer_drop=1.0))

    base_cfg = PRESETS["TINY"].encoder
    drop_cfg = dataclasses.replace(base_cfg, n_blocks=4, layer_drop=0.1)
    torch.manual_seed(7)
    enc_zero = Encoder(dataclasses.replace(drop_cfg, layer_drop=0.0))
    enc_drop = Encoder(drop_cfg)
    enc_drop.load_state_dict(enc_zero.state_dict())

    assert [round(b.drop_path, 6) for b in enc_drop.blocks] == [0.025, 0.05, 0.075, 0.1]
    x, lens = torch.randn(6, 40, 80), torch.full((6,), 40, dtype=torch.long)

    # Eval mode: exact bit-identical identity!
    enc_zero.eval()
    enc_drop.eval()
    with torch.no_grad():
        y0, _ = enc_zero(x, lens)
        y1, _ = enc_drop(x, lens)
    assert torch.equal(y0, y1)

    # Train mode with layer_drop > 0 drops residual branches stochastically.
    enc_drop.train()
    torch.manual_seed(11)
    yt_a, _ = enc_drop(x, lens)
    torch.manual_seed(12)
    yt_b, _ = enc_drop(x, lens)
    assert not torch.equal(yt_a, yt_b)

    # Multirate linear schedule across lower + middle + final blocks.
    mr = Encoder(EncoderConfig(d_model=16, n_heads=2, stem_channels=4, conv_kernel=3, chunk=4,
                               left_context=8, dropout=0.0, family="multirate",
                               lower_blocks=1, middle_blocks=2, final_blocks=1, layer_drop=0.1))
    probs = [b.drop_path for b in (*mr.blocks, *mr.mid_blocks, *mr.final_blocks)]
    assert [round(p, 6) for p in probs] == [0.025, 0.05, 0.075, 0.1]

    # check_init_compatible accepts changing layer_drop (eval-identity).
    src = PRESETS["TINY"]
    dst = dataclasses.replace(src, encoder=dataclasses.replace(src.encoder, layer_drop=0.1))
    check_init_compatible(src, dst)


# ---- Factor D: text_tables + pred_ctx=4 warm-start + export refusal -------------------------
def test_factor_d_warm_start_text_loss_beam_and_export_refusal(tmp_path):
    torch.manual_seed(3)
    src_cfg = PRESETS["TINY"]
    src_model = ASRModel(src_cfg).eval()
    x = torch.randn(2, 48, 80)
    lens = torch.tensor([48, 44])
    tgt = torch.tensor([[1, 2, 3], [2, 1, 0]])
    tgt_lens = torch.tensor([3, 2])
    ctc_tgt = torch.tensor([[1, 2], [2, 1]])
    ctc_lens = torch.tensor([2, 2])
    with torch.no_grad():
        base_out = src_model.losses(x, lens, tgt, tgt_lens, ctc_tgt, ctc_lens)

    # Warm-start into (pred_ctx=4, text_tables_dim=0), (pred_ctx=2, dim=8), and (pred_ctx=4, dim=8).
    for pred_ctx, tdim in ((4, 0), (2, 8), (4, 8)):
        dst_cfg = model_overrides(
            src_cfg, TrainConfig(preset="TINY", pred_ctx=pred_ctx, text_tables_dim=tdim)
        )
        dst_model = ASRModel(dst_cfg).eval()
        load_init_state(dst_model, src_model.state_dict(), src_cfg, dst_cfg)
        if tdim > 0:
            assert isinstance(dst_model.predictor, TextPredictor)
            assert dst_model.predictor.context_len == 4
        with torch.no_grad():
            out = dst_model.losses(x, lens, tgt, tgt_lens, ctc_tgt, ctc_lens)
        assert torch.allclose(out["rnnt"], base_out["rnnt"], atol=1e-6)
        assert torch.equal(out["nll"], base_out["nll"])

    # Beam search handles context_len == 4 (both pred_ctx=4 and TextPredictor).
    dst_cfg = model_overrides(src_cfg, TrainConfig(preset="TINY", pred_ctx=4, text_tables_dim=8))
    dst_model = ASRModel(dst_cfg).eval()
    load_init_state(dst_model, src_model.state_dict(), src_cfg, dst_cfg)
    with torch.no_grad():
        enc, elens = dst_model.encode(x, lens)
        hyps = modified_beam_search(dst_model.predictor, dst_model.joiner,
                                    dst_model.joiner.enc_proj(enc), elens, beam=2)
    assert len(hyps) == 2

    # TextStream + FactTextHelper + Trainer step route text_loss grads strictly to predictor.text.*.
    stream_path = str(tmp_path / "train_tokens.u16")
    toks = np.array([1, 2, 3, 4, 5, 0, 2, 3, 4, 1, 0] * 32, dtype=np.uint16)
    toks.tofile(stream_path)
    cfg = TrainConfig(preset="TINY", rnnt_pieces=10, ctc_pieces=6, specaug=False,
                      pred_ctx=4, text_tables_dim=8, text_loss_weight=0.3,
                      text_stream_path=stream_path, text_batch_size=32, text_enull_batches=1,
                      eval_every=0, ckpt_every_steps=100)
    mcfg = model_config_for(cfg)
    assert mcfg.pred_ctx == 4 and mcfg.text_tables_dim == 8
    src = FixedBatches(_tiny_examples(), batch_size=2)
    tr = Trainer(str(tmp_path / "run_d"), cfg, src, model_cfg=mcfg, device="cpu", log=False)
    assert tr._fact_text is not None
    w_before = tr.model.predictor.text.out.weight.detach().clone()
    _, batch = next(src.iterate(Position()))
    r = tr.train_step(batch)
    assert "text_loss" in r and np.isfinite(r["text_loss"]) and r["text_loss"] > 0.0
    assert not torch.equal(tr.model.predictor.text.out.weight, w_before)

    # Misconfigured text_loss without text_tables_dim fails closed.
    with pytest.raises(TextTablesError):
        FactTextHelper.build(src_model, TrainConfig(text_loss_weight=0.3, text_stream_path=stream_path))

    # Export (both safetensors and ONNX bundle) MUST refuse text_tables checkpoints fail-closed.
    from train import checkpoint

    tr.save(wait=True)
    ckpt_path = checkpoint.latest(tr.run_dir)
    assert ckpt_path is not None
    from tools.r1_to_safetensors import ConvertError, convert

    with pytest.raises(ConvertError, match="text_tables"):
        convert(ckpt_path, str(tmp_path / "st_out"), bundle=None)
    with pytest.raises(ExportError, match="text_tables"):
        export_bundle(ckpt_path, str(tmp_path / "onnx_out"), int8=False)

    # Reject loading a text_tables checkpoint into a plain model.
    with pytest.raises(CheckpointError):
        check_init_compatible(dst_cfg, src_cfg)


# ---- Factor E: TRAIN-only gain / noise / RIR augmentation -----------------------------------
def test_factor_e_train_only_gain_noise_rir_and_eval_leakage_guard(tmp_path):
    t = np.arange(16000, dtype=np.float32) / 16000.0
    pcm = np.clip(np.round(0.4 * np.sin(2.0 * np.pi * 440.0 * t) * 32768.0), -32768, 32767).astype(np.int16)

    # Default-OFF returns exact same array untouched.
    assert fact_aug_from_cfg(TrainConfig()) is None
    off = FactAugConfig()
    assert apply_fact_aug(pcm, "u0", 0, 1234, off) is pcm

    # Random gain reduces RMS into [-30, -10] dB range and is deterministic.
    gcfg = FactAugConfig(gain_p=1.0, gain_min_db=-30.0, gain_max_db=-10.0)
    g1 = apply_fact_aug(pcm, "u0", 0, 1234, gcfg)
    g2 = apply_fact_aug(pcm, "u0", 0, 1234, gcfg)
    assert np.array_equal(g1, g2) and g1.shape == pcm.shape and g1.dtype == np.int16
    rms_ratio = float(np.sqrt(np.mean(g1.astype(np.float32) ** 2))) / float(
        np.sqrt(np.mean(pcm.astype(np.float32) ** 2))
    )
    assert 10.0 ** (-30.5 / 20.0) <= rms_ratio <= 10.0 ** (-9.5 / 20.0)

    # Real TRAIN-only noise_dir and causal rir_dir work deterministically and preserve onset.
    ndir = tmp_path / "train_noise"
    rdir = tmp_path / "train_rir"
    ndir.mkdir()
    rdir.mkdir()
    rng = np.random.default_rng(42)
    sf.write(str(ndir / "room0.wav"), (0.05 * rng.standard_normal(8000)).astype(np.float32), 16000)
    # RIR with 50 samples of pre-delay before the direct peak -> causal alignment strips pre-delay!
    rir = np.zeros(400, dtype=np.float32)
    rir[50] = 1.0
    rir[70] = 0.3
    sf.write(str(rdir / "rir0.wav"), rir, 16000)

    rcfg = FactAugConfig(noise_p=1.0, noise_dir=str(ndir), noise_snr_min_db=10.0,
                         noise_snr_max_db=15.0, rir_p=1.0, rir_dir=str(rdir))
    rcfg.validate(("train-clean-100",))
    aug1 = apply_fact_aug(pcm, "u1", 0, 1234, rcfg)
    aug2 = apply_fact_aug(pcm, "u1", 0, 1234, rcfg)
    assert np.array_equal(aug1, aug2) and aug1.shape == pcm.shape and not np.array_equal(aug1, pcm)

    # Causal onset check on an impulse at sample 200: RIR peak-alignment keeps onset at 200 (0 before).
    imp = np.zeros(1000, dtype=np.int16)
    imp[200] = 20000
    imp_rir = apply_fact_aug(imp, "u2", 0, 1234, FactAugConfig(rir_p=1.0, rir_dir=str(rdir)))
    assert np.all(imp_rir[:200] == 0) and abs(int(imp_rir[200])) > 10000 and abs(int(imp_rir[220])) > 1000

    # In-window TRAIN donor noise mixing when noise_dir == "".
    donor = (0.1 * rng.standard_normal(4000) * 32768.0).astype(np.int16)
    dmix = apply_fact_aug(pcm, "u3", 0, 1234, FactAugConfig(noise_p=1.0), donor_pcm=donor)
    assert dmix.shape == pcm.shape and not np.array_equal(dmix, pcm)

    # Fail-closed eval-leakage guards!
    bad_dir = tmp_path / "dev-clean"
    bad_dir.mkdir()
    sf.write(str(bad_dir / "clip.wav"), np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(DataError, match="forbidden eval"):
        list_train_only_audio(str(bad_dir), "noise_dir")
    with pytest.raises(DataError, match="eval split"):
        FactAugConfig(gain_p=0.5).validate(("dev-clean",))


# ---- 4-step BIT-IDENTICAL default-OFF witness -----------------------------------------------
def test_four_step_default_off_bit_identical_witness(tmp_path):
    """All new H57 FACT flags default-OFF must produce 0-ULP bit-identical losses, params, EMA, and RNG."""
    exs = _tiny_examples(seed=99)

    def run_four(run_name: str, explicit_zero_flags: bool):
        kw = dict(preset="TINY", lr=1e-3, warmup=2, total_steps=10, seed=1234,
                  specaug=True, specaug_start=0, ema_decay=0.9, eval_every=0,
                  ckpt_every_steps=100, rnnt_pieces=10, ctc_pieces=6)
        if explicit_zero_flags:
            kw.update(layer_drop=0.0, text_tables_dim=0, text_loss_weight=0.0,
                      text_stream_path="", gain_aug_p=0.0, noise_aug_p=0.0, rir_aug_p=0.0)
        cfg = TrainConfig(**kw)
        mcfg = model_config_for(cfg)
        assert mcfg == PRESETS["TINY"]
        assert fact_aug_from_cfg(cfg) is None
        src = FixedBatches(exs, batch_size=2)
        tr = Trainer(str(tmp_path / run_name), cfg, src, model_cfg=mcfg, device="cpu", log=False)
        assert tr._fact_text is None
        assert not isinstance(tr.model.predictor, TextPredictor)
        metrics = []
        it = src.iterate(Position())
        for _ in range(4):
            _, batch = next(it)
            metrics.append(tr.train_step(batch))
        return metrics, tr.model.state_dict(), tr.ema, torch.get_rng_state()

    m_base, sd_base, ema_base, rng_base = run_four("base", False)
    m_off, sd_off, ema_off, rng_off = run_four("off", True)

    assert [r["loss"] for r in m_base] == [r["loss"] for r in m_off]
    assert [r["rnnt"] for r in m_base] == [r["rnnt"] for r in m_off]
    assert [r["grad_norm"] for r in m_base] == [r["grad_norm"] for r in m_off]
    for k in sd_base:
        assert torch.equal(sd_base[k], sd_off[k]), f"param {k} drifted"
    assert ema_base is not None and ema_off is not None
    for k in ema_base:
        assert torch.equal(ema_base[k], ema_off[k]), f"ema {k} drifted"
    assert torch.equal(rng_base, rng_off)
