"""Tests for hottest-rank DDP StepThrottle, CPU ceiling, and default-off bit-exactness."""
from __future__ import annotations

import torch

from train.asr import ModelConfig
from train.data import Example
from train.dist import StepThrottle
from train.trainer import FixedBatches, TrainConfig, Trainer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        d=32, heads=2, n_blocks=2, ffn=64, kernel=7, stem_ch=8,
        chunk=4, left_context=8, vocab_rnnt=12, vocab_ctc=12,
        pred_emb=16, joiner=32,
    )


def _tiny_examples(n: int = 4) -> list[Example]:
    g = torch.Generator().manual_seed(7)
    out = []
    for i in range(n):
        feats = torch.randn(40, 80, generator=g)
        rnnt = torch.tensor([1, 2, 3], dtype=torch.long)
        ctc = torch.tensor([1, 2, 3], dtype=torch.long)
        out.append(Example(uid=f"u{i}", text="a b c", feats=feats, rnnt=rnnt, ctc=ctc))
    return out


def test_step_throttle_reacts_to_hottest_rank_gpu_and_cpu_target():
    # Disable daemon thread by setting targets to 0 in __init__, then configure targets directly.
    th = StepThrottle(target=0.0, gpu_index=0, every=1, cpu_target=0.0)
    th.target = 0.85
    th.cpu_target = 0.80
    # Hottest rank over GPU cap (0.90 > 0.85) reduces duty from 1.0 -> 0.98 and requests sleep > 0.
    nap1 = th.sleep_for(step=1, step_seconds=0.5, cluster_gpu_mean=0.90, cluster_cpu_mean=0.50)
    assert th.duty == 0.98
    assert nap1 > 0.0
    # Host CPU over cap (0.85 > 0.80) reduces duty even when GPU has headroom.
    nap2 = th.sleep_for(step=2, step_seconds=0.5, cluster_gpu_mean=0.70, cluster_cpu_mean=0.85)
    assert th.duty == 0.96
    assert nap2 > nap1
    # Both roomy (< target - 0.03) nudges duty back up smoothly.
    th.sleep_for(step=3, step_seconds=0.5, cluster_gpu_mean=0.75, cluster_cpu_mean=0.70)
    assert th.duty == 0.98


def test_default_off_throttle_is_bit_exact(tmp_path):
    exs = _tiny_examples(4)
    mc = _tiny_cfg()
    cfg = TrainConfig(
        preset="S", total_steps=5, warmup=2, lr=1e-3, specaug=False,
        eval_every=0, eval_each_epoch=False, ckpt_every_steps=100, log_every=5,
        ddp_util_target=0.0, cpu_util_target=0.0,
    )
    t1 = Trainer(str(tmp_path / "r1"), cfg, FixedBatches(exs, batch_size=2), model_cfg=mc, device="cpu", log=False)
    t2 = Trainer(str(tmp_path / "r2"), cfg, FixedBatches(exs, batch_size=2), model_cfg=mc, device="cpu", log=False)
    t1.run(max_steps=4)
    t2.run(max_steps=4)
    for k, v1 in t1.model.state_dict().items():
        assert torch.equal(v1, t2.model.state_dict()[k]), f"mismatch in {k}"
