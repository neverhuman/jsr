"""Tests for JSR v2 Wave H28 (unet3 25->12.5->6.25 Hz, GQA, ReLU^2) & H27 dual-mode chunking."""
from __future__ import annotations

import torch

from train.asr import PRESETS, ASRModel
from train.data import Example, FixedBatches
from train.model import Encoder, EncoderConfig
from train.trainer import TrainConfig, Trainer


def test_unet3_zero_lookahead_causality_is_exact():
    torch.manual_seed(0)
    cfg = PRESETS["V2_TINY"].encoder
    enc = Encoder(cfg).eval()
    feats = torch.randn(1, 80, cfg.n_mels)  # 20 frames at 25 Hz (5 chunks of 4 = 160 ms)
    lens = torch.tensor([80])
    with torch.no_grad():
        base, out_lens = enc(feats, lens, chunk=4)
        assert int(out_lens[0]) == 20
        # Perturb everything strictly after chunk c=1 (i.e. 25 Hz frames >= 8 -> mel frames >= 32).
        for c in (0, 1, 2, 3):
            pert = feats.clone()
            cut_mel = 4 * (c + 1) * 4
            pert[:, cut_mel:, :] += 7.5
            y, _ = enc(pert, lens, chunk=4)
            diff = (y[:, :(c + 1) * 4] - base[:, :(c + 1) * 4]).abs().max().item()
            assert diff == 0.0, f"future leakage into chunk {c}: max diff {diff}"


def test_unet3_padding_invariance_and_odd_lengths():
    torch.manual_seed(1)
    cfg = PRESETS["V2_TINY"].encoder
    enc = Encoder(cfg).eval()
    x1 = torch.randn(1, 53, cfg.n_mels)  # odd 25 Hz length (14) and odd 12.5 Hz length (7)
    x2 = torch.cat([x1, torch.randn(1, 27, cfg.n_mels)], dim=1)
    with torch.no_grad():
        y1, l1 = enc(x1, torch.tensor([53]), chunk=4)
        y2, l2 = enc(x2, torch.tensor([53]), chunk=4)
    assert int(l1[0]) == int(l2[0]) == 14
    assert torch.allclose(y1[:, :14], y2[:, :14], atol=1e-5)


def test_v2_presets_build_relu2_sparsity_and_qkv_cap():
    for name in ("V2_S", "V2_M", "V2_L", "V2_TINY"):
        mc = PRESETS[name]
        assert mc.encoder.family == "unet3"
        assert mc.encoder.kv_heads == 2
        assert mc.encoder.ffn_act == "relu2"
    m = ASRModel(PRESETS["V2_TINY"])
    x = torch.randn(2, 48, 80)
    h = torch.relu(m.encoder.blocks[0].ff1.w1(m.encoder.blocks[0].ff1.norm(m.encoder.stem(x)))).square()
    zero_frac = float((h == 0).float().mean())
    assert zero_frac > 0.35, f"expected natural ReLU^2 sparsity, got {zero_frac:.3f}"


def test_dual_mode_chunk_sampling_trains_and_default_off_is_bit_exact(tmp_path):
    g = torch.Generator().manual_seed(42)
    exs = [
        Example(uid=f"u{i}", text="a b c", feats=torch.randn(48, 80, generator=g),
                rnnt=torch.tensor([1, 2, 3]), ctc=torch.tensor([1, 2, 3]))
        for i in range(4)
    ]
    mc = PRESETS["V2_TINY"]
    base_cfg = TrainConfig(
        preset="V2_TINY", total_steps=4, warmup=1, lr=1e-3, specaug=False,
        eval_every=0, eval_each_epoch=False, ckpt_every_steps=100, log_every=2,
        qkv_cap_from=1, chunk_modes="",
    )
    t1 = Trainer(str(tmp_path / "r1"), base_cfg, FixedBatches(exs, batch_size=2), model_cfg=mc, device="cpu", log=False)
    t2 = Trainer(str(tmp_path / "r2"), base_cfg, FixedBatches(exs, batch_size=2), model_cfg=mc, device="cpu", log=False)
    t1.run(max_steps=3)
    t2.run(max_steps=3)
    for k, v in t1.model.state_dict().items():
        assert torch.equal(v, t2.model.state_dict()[k]), f"default-off mismatch in {k}"

    dual_cfg = TrainConfig(
        preset="V2_TINY", total_steps=4, warmup=1, lr=1e-3, specaug=False,
        eval_every=0, eval_each_epoch=False, ckpt_every_steps=100, log_every=2,
        qkv_cap_from=1, chunk_modes="4,8,0",
    )
    td = Trainer(str(tmp_path / "rd"), dual_cfg, FixedBatches(exs, batch_size=2), model_cfg=mc, device="cpu", log=False)
    td.run(max_steps=3)
    assert td.step == 3
    assert all(torch.isfinite(p).all() for p in td.model.parameters())


def test_encoder_config_backwards_compatible_with_v1_checkpoints():
    old_dict = {"d_model": 384, "n_blocks": 16, "n_heads": 6, "post_block_norm": False, "family": "dense"}
    cfg = EncoderConfig(**old_dict)
    assert cfg.kv_heads == 0 and cfg.ffn_act == "silu" and cfg.family == "dense"
    assert cfg.stem_type == "conv2d" and not cfg.qk_rmsnorm and cfg.norm_affine
    assert cfg.layer_scale_init == 0.0 and cfg.stage_left == () and cfg.inter_ctc_vocab == 0


def test_v2_stage1a_inter_ctc_balancer_pred_ctx4_and_json_roundtrip():
    import json

    from train.asr import ModelConfig

    mc = PRESETS["V2_TINY"]
    assert mc.pred_ctx == 4 and mc.inter_ctc
    assert mc.encoder.stem_type == "dwsep" and mc.encoder.qk_rmsnorm
    assert not mc.encoder.norm_affine and mc.encoder.layer_scale_init == 0.1
    # JSON round-trip preserves stage_left tuple equality for checkpoint resume.
    rt = ModelConfig.from_dict(json.loads(json.dumps(mc.to_dict())))
    assert rt == mc

    torch.manual_seed(3)
    m = ASRModel(mc)
    feats = torch.randn(2, 48, 80)
    feat_lens = torch.tensor([48, 40])
    rnnt = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]], dtype=torch.long)
    rnnt_lens = torch.tensor([4, 2])
    ctc = torch.tensor([[1, 2, 3], [2, 3, 0]], dtype=torch.long)
    ctc_lens = torch.tensor([3, 2])
    out = m.losses(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens,
                   inter_ctc_weight=0.2, balancer_weight=1e-3)
    assert "inter_ctc" in out and "balancer" in out
    out["loss"].backward()
    assert m.encoder.inter_ctc.weight.grad is not None
    assert m.encoder.inter_proj.weight.grad is not None
    assert m.encoder.blocks[0].att.q_scale.grad is not None
    assert m.encoder.blocks[0].ls_ff1.grad is not None
    assert m.predictor.proj.weight.shape[1] == 4 * mc.pred_emb
    # Greedy decode runs cleanly with 4-token predictor context.
    toks, forced = m.greedy(feats, feat_lens)
    assert len(toks) == 2 and forced >= 0


def test_v2_synth_aug_and_specaug_ramp(tmp_path):
    import numpy as np

    from train.data import apply_synth_aug

    rng = np.random.default_rng(0)
    pcm = rng.integers(-12000, 12000, size=16000, dtype=np.int16)
    a1 = apply_synth_aug(pcm, "utt-001", epoch=0, seed=1234)
    a2 = apply_synth_aug(pcm, "utt-001", epoch=0, seed=1234)
    assert a1.dtype == np.int16 and len(a1) == len(pcm)
    assert np.array_equal(a1, a2)
    # Across 20 distinct uids, at least one is augmented by RIR/noise.
    changed = sum(not np.array_equal(pcm, apply_synth_aug(pcm, f"u-{i}", 0, 7)) for i in range(20))
    assert changed > 0

    g = torch.Generator().manual_seed(5)
    exs = [
        Example(uid=f"u{i}", text="a b", feats=torch.randn(48, 80, generator=g),
                rnnt=torch.tensor([1, 2]), ctc=torch.tensor([1, 2]))
        for i in range(4)
    ]
    cfg = TrainConfig(
        preset="V2_TINY", total_steps=6, warmup=1, lr=1e-3, specaug=True,
        specaug_ramp_start=1, specaug_start=5, inter_ctc_weight=0.2, balancer_weight=1e-3,
        eval_every=0, eval_each_epoch=False, ckpt_every_steps=100, log_every=2, qkv_cap_from=1,
    )
    tr = Trainer(str(tmp_path / "ramp"), cfg, FixedBatches(exs, batch_size=2),
                 model_cfg=PRESETS["V2_TINY"], device="cpu", log=False)
    assert tr._step_augment() is None  # step 0 < ramp_start 1
    tr.step = 3
    ramped = tr._step_augment()
    assert ramped is not None and 0 < ramped.freq_width < tr.augment.freq_width
    tr.step = 0
    tr.run(max_steps=4)
    assert tr.step == 4 and all(torch.isfinite(p).all() for p in tr.model.parameters())
