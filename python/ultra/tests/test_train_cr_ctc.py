"""Wave H26 tests: CR-CTC two-view consistency KL, exact KD emitted-path guard, and H17 resume persistence."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel, SpecAugConfig  # noqa: E402
from train.data import Batch, Example, FixedBatches  # noqa: E402
from train.kd import verify_emitted_path  # noqa: E402
from train.trainer import TrainConfig, Trainer  # noqa: E402


def _synthetic_batch():
    torch.manual_seed(0)
    feats = torch.randn(2, 40, 80)
    feat_lens = torch.tensor([40, 32])
    rnnt = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    rnnt_lens = torch.tensor([3, 2])
    ctc = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    ctc_lens = torch.tensor([2, 1])
    return feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens


def test_cr_ctc_zero_weight_is_bit_exact_and_positive_weight_trains():
    feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens = _synthetic_batch()
    aug = SpecAugConfig(enabled=True, freq_masks=2, freq_width=8, time_masks_per_10s=2, time_width=8)

    torch.manual_seed(7)
    m = ASRModel(PRESETS["TINY"])
    g1 = torch.Generator().manual_seed(99)
    out0 = m.losses(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens, augment=aug, gen=g1, cr_ctc_weight=0.0)
    assert "cr_ctc" not in out0

    g2 = torch.Generator().manual_seed(99)
    out0_b = m.losses(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens, augment=aug, gen=g2)
    assert float(out0["loss"]) == float(out0_b["loss"])

    g3 = torch.Generator().manual_seed(99)
    out_cr = m.losses(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens, augment=aug, gen=g3, cr_ctc_weight=0.2)
    assert "cr_ctc" in out_cr and float(out_cr["cr_ctc"]) > 0.0
    # View-2 supervised CTC changes out["ctc"] relative to single-view out0["ctc"] (average of two views).
    assert abs(float(out_cr["ctc"]) - float(out0["ctc"])) > 1e-6
    out_cr["loss"].backward()
    grads = [p.grad for p in m.ctc_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # Verify View-2 receives supervised CTC gradients even with tiny cr_ctc_weight (KL negligible).
    m.zero_grad(set_to_none=True)
    g4 = torch.Generator().manual_seed(99)
    out_tiny_kl = m.losses(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens, augment=aug, gen=g4,
                           cr_ctc_weight=1e-9)
    assert abs(float(out_tiny_kl["ctc"]) - float(out0["ctc"])) > 1e-6
    out_tiny_kl["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.ctc_head.parameters() if p.grad is not None)


def test_verify_emitted_path_rejects_mismatched_subword_segmentation():
    # R0 emits piece ids [4, 6] -> student token ids [5, 7]
    emitted = np.array([0, 4, 0, 6, 0], dtype=np.int16)
    assert verify_emitted_path(emitted, [5, 7])
    assert not verify_emitted_path(emitted, [5, 8])
    assert not verify_emitted_path(emitted, [5, 7, 9])


def test_trainer_resume_persists_divergence_baselines(tmp_path):
    exs = [Example(f"u{i}", "hi", torch.randn(24, 80), [1, 2], [1, 2]) for i in range(4)]
    src = FixedBatches(exs, batch_size=2)
    cfg = TrainConfig(preset="TINY", total_steps=4, warmup=1, log_every=1, eval_every=0,
                      ckpt_every_steps=100, specaug=False)
    tr = Trainer(str(tmp_path), cfg, src, model_cfg=PRESETS["TINY"], device="cpu", log=False)
    tr.loss_ema = 1.234
    tr.best_ema = 1.111
    tr.best_wer = 5.783
    tr.save(wait=True)

    tr2 = Trainer(str(tmp_path), cfg, src, model_cfg=PRESETS["TINY"], device="cpu", log=False)
    assert tr2.resume()
    assert tr2.loss_ema == 1.234
    assert tr2.best_ema == 1.111
    assert tr2.best_wer == 5.783


def test_resume_rearms_the_guard_when_the_loss_or_dev_panel_changes(tmp_path):
    import dataclasses

    exs = [Example(f"u{i}", "hi", torch.randn(24, 80), [1, 2], [1, 2]) for i in range(4)]
    cfg = TrainConfig(preset="TINY", total_steps=4, warmup=1, log_every=1, eval_every=0,
                      ckpt_every_steps=100, specaug=False)

    def trainer(c):
        return Trainer(str(tmp_path), c, FixedBatches(exs, batch_size=2), model_cfg=PRESETS["TINY"],
                       device="cpu", log=False)

    tr = trainer(cfg)
    tr.loss_ema, tr.best_ema, tr.best_wer = 1.234, 1.111, 5.783
    tr.save(wait=True)
    kd_on = trainer(dataclasses.replace(cfg, zloss=1e-4))  # loss definition changed: baselines reset
    assert kd_on.resume() and kd_on.loss_ema is None and kd_on.best_ema is None
    assert kd_on.best_wer == 5.783  # same dev panel
    panel = trainer(dataclasses.replace(cfg, dev_limit=400))  # dev panel changed: best WER resets
    assert panel.resume() and panel.best_wer is None and panel.best_ema == 1.111
    panel.reset_watch()
    assert panel.loss_ema is None and panel.best_ema is None and panel.best_wer is None
