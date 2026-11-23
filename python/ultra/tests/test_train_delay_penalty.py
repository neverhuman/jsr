"""H34: the pruned-loss delay_penalty key is off by default (bit-exact) and changes the loss when on."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
k2 = pytest.importorskip("k2")
from train.asr import PRESETS, ASRModel  # noqa: E402


def _batch():
    torch.manual_seed(0)
    return (torch.randn(2, 64, 80), torch.tensor([64, 50]), torch.tensor([[1, 2, 3], [4, 5, 0]]),
            torch.tensor([3, 2]), torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1]))


def test_delay_penalty_zero_is_bit_exact_and_positive_changes_the_loss():
    import dataclasses

    cfg = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned")
    torch.manual_seed(3)
    m = ASRModel(cfg)
    b = _batch()
    base = m.losses(*b)
    zero = m.losses(*b, delay_penalty=0.0)
    assert float(base["loss"]) == float(zero["loss"])
    on = m.losses(*b, delay_penalty=0.01)
    assert float(on["loss"]) != float(base["loss"])
    on["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_delay_penalty_rejects_the_exact_loss():
    torch.manual_seed(3)
    m = ASRModel(PRESETS["TINY"])
    with pytest.raises(ValueError):
        m.losses(*_batch(), delay_penalty=0.01)


def test_model_config_records_the_trained_chunk():
    from train.trainer import TrainConfig, model_config_for

    assert model_config_for(TrainConfig(preset="S")).encoder.chunk == PRESETS["S"].encoder.chunk
    assert model_config_for(TrainConfig(preset="S", chunk=2)).encoder.chunk == 2


def test_chunk_modes_draw_is_deterministic_per_step_and_trains(tmp_path):
    from train.data import Example, FixedBatches
    from train.trainer import TrainConfig, Trainer

    exs = [Example(f"u{i}", "hi", torch.randn(48, 80), [1, 2], [1, 2]) for i in range(4)]

    def trainer(d, **kw):
        cfg = TrainConfig(preset="TINY", total_steps=8, warmup=1, log_every=1, eval_every=0,
                          ckpt_every_steps=100, specaug=False, **kw)
        return Trainer(str(tmp_path / d), cfg, FixedBatches(exs, batch_size=2), model_cfg=PRESETS["TINY"],
                       device="cpu", log=False)

    off = trainer("off")
    assert off._step_chunk() is None  # "" = the model's own chunk, unchanged path
    a, b = trainer("a", chunk_modes="2,4"), trainer("b", chunk_modes="2,4")
    seen = set()
    for step in range(40):
        a.step = b.step = step
        assert a._step_chunk() == b._step_chunk()  # same draw on every rank / rerun
        seen.add(a._step_chunk())
    assert seen == {2, 4}
    a.step = 0
    a.run(max_steps=4)
    assert a.step == 4 and all(torch.isfinite(p).all() for p in a.model.parameters())
    assert a._evaluate_extra_chunks() == {}  # no dev set: nothing scored


def test_recipe_opts_widen_the_predictor_function_preservingly_and_set_masks():
    import dataclasses

    from train.recipe_opts import check_init_compatible, model_overrides, specaug, widen_state
    from train.trainer import TrainConfig, model_config_for

    assert model_config_for(TrainConfig(preset="S")) == model_config_for(TrainConfig(preset="S", pred_ctx=0))
    cfg = TrainConfig(preset="TINY", pred_ctx=4, left_context=8, specaug_time_masks=4, specaug_time_width=60)
    dst = model_overrides(PRESETS["TINY"], cfg)
    assert dst.pred_ctx == 4 and dst.encoder.left_context == 8
    sa = specaug(cfg)
    assert sa.time_masks_per_10s == 4 and sa.time_width == 60 and sa.freq_width == 27
    assert specaug(TrainConfig()).time_masks_per_10s == 2  # default unchanged
    src_cfg = PRESETS["TINY"]
    check_init_compatible(src_cfg, dst)  # chunk / left context / longer predictor context: allowed
    with pytest.raises(Exception):
        check_init_compatible(dataclasses.replace(src_cfg, pred_ctx=4), dataclasses.replace(dst, pred_ctx=2))
    torch.manual_seed(0)
    a = ASRModel(src_cfg).eval()
    b = ASRModel(dst).eval()
    b.load_state_dict(widen_state(a.state_dict(), src_cfg, dst))
    tgt = torch.tensor([[1, 2, 3, 4]])
    from train.transducer import StatelessPredictor

    pa = a.predictor(StatelessPredictor.contexts(tgt, 2))
    pb = b.predictor(StatelessPredictor.contexts(tgt, 4))
    assert torch.allclose(pa, pb, atol=1e-6)  # the widened predictor computes what the source did
