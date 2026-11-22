"""Tests for Wave H23 speed_pack and DiLoCo multi-node coordination modules."""
from __future__ import annotations

import json

import numpy as np
import torch

from train.asr import PRESETS, ASRModel, SpecAugConfig
from train.diloco import (
    DiLoCoOuterOptimizer,
    StepBoundaryThrottle,
    greedy_model_soup,
    pack_bf16_delta,
    unpack_bf16_delta,
)
from train.model import chunk_mask, rope_tables
from train.speed_pack import (
    CachedQkvCap,
    ForeachEMA,
    MaskRopeCache,
    build_adamw,
    measure_packing_efficiency,
    recommended_window_groups,
    vectorized_spec_augment,
)


def test_vectorized_spec_augment_determinism_and_valid_bounds():
    x = torch.ones(8, 250, 80)
    lens = torch.tensor([60, 120, 180, 250, 90, 210, 150, 200])
    cfg = SpecAugConfig(enabled=True, freq_masks=2, freq_width=20, time_masks_per_10s=2, time_width=30)
    y1 = vectorized_spec_augment(x, lens, cfg, torch.Generator().manual_seed(42))
    y2 = vectorized_spec_augment(x, lens, cfg, torch.Generator().manual_seed(42))
    assert torch.equal(y1, y2)
    # Padded tail beyond lens[b] must remain untouched (all ones).
    for b, n in enumerate(lens.tolist()):
        if n < 250:
            assert torch.all(y1[b, n:] == 1.0)
    # Some valid entries must actually be masked to 0.
    assert float((y1 == 0.0).float().mean()) > 0.05


def test_foreach_ema_matches_scalar_lerp():
    m = ASRModel(PRESETS["TINY"])
    ema = ForeachEMA(m, decay=0.99)
    ref = {k: v.detach().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.25)
    ema.update(m)
    for k, v in m.state_dict().items():
        if v.is_floating_point():
            ref[k].lerp_(v, 0.01)
            assert torch.allclose(ema.state_dict()[k], ref[k], atol=1e-6)
        else:
            assert torch.equal(ema.state_dict()[k], v)


def test_cached_qkv_cap_clamps_rows_and_norm_gains():
    m = ASRModel(PRESETS["TINY"])
    cap = CachedQkvCap(m, cap_ratio=1.5)
    assert cap.ref_norm > 0.0
    # Inflate qkv weights and norm gains beyond 1.5x.
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "qkv"):
                mod.qkv.weight.mul_(4.0)
                mod.norm.weight.mul_(4.0)
    new_max = cap.apply()
    assert new_max <= cap.ref_norm * 1.501


def test_mask_rope_cache_exact_parity():
    cache = MaskRopeCache()
    for t in (17, 64, 300, 45):
        m_c, e_c = cache.get_mask_and_eye(t, 4, 16, torch.device("cpu"))
        assert torch.equal(m_c, chunk_mask(t, 4, 16, device=torch.device("cpu")))
        assert torch.equal(e_c, torch.eye(t, dtype=torch.bool))
        cos_c, sin_c = cache.get_rope(t, 16, torch.device("cpu"), torch.float32)
        cos_r, sin_r = rope_tables(t, 16, torch.device("cpu"), torch.float32)
        assert torch.equal(cos_c, cos_r)
        assert torch.equal(sin_c, sin_r)


def test_window_groups_improves_packing_efficiency():
    assert recommended_window_groups(300.0, 64) == 4
    assert recommended_window_groups(2000.0, 400) >= 20
    rng = np.random.default_rng(7)
    # Realistic LibriSpeech 960h duration distribution (2s..17s).
    small_win = rng.uniform(2.0, 17.0, size=380).tolist()
    large_win = rng.uniform(2.0, 17.0, size=1900).tolist()
    eff_small = measure_packing_efficiency(small_win, budget_s=2000.0, max_batch=400)
    eff_large = measure_packing_efficiency(large_win, budget_s=2000.0, max_batch=400)
    assert eff_large > eff_small + 0.05
    assert eff_large > 0.90


def test_step_boundary_throttle_duty_and_broadcast(tmp_path):
    ctrl = tmp_path / "throttle.json"
    ctrl.write_text(json.dumps({"target_duty": 0.80, "t": 1.0}))
    th = StepBoundaryThrottle(str(ctrl), max_sleep_s=1.0)
    # At duty 0.80 and 0.40s step, need 0.40 * (1/0.8 - 1) = 0.10s sleep.
    assert abs(th.compute_sleep_s(0.40) - 0.10) < 1e-6
    seen = []
    slept = th.maybe_sleep(0.0, broadcast_fn=lambda s: (seen.append(s), 0.0)[1])
    assert slept == 0.0 and seen == [0.0]


def test_diloco_two_workers_converge_and_resume_bit_exact():
    torch.manual_seed(0)
    target = torch.tensor([3.0, -2.0, 1.5, -0.5])
    anchor = {"w": torch.zeros(4), "step_count": torch.tensor(0)}
    outer = DiLoCoOuterOptimizer(anchor, lr=0.7, momentum=0.9, nesterov=True)

    # Check BF16 delta pack/unpack round-trip.
    delta = pack_bf16_delta(anchor, {"w": target, "step_count": torch.tensor(1)})
    assert delta["w"].dtype == torch.bfloat16
    assert torch.allclose(unpack_bf16_delta(delta)["w"], -target, atol=2e-2)

    # Run 30 outer rounds with 2 heterogeneous workers (weights 3:1, e.g. A100 vs 3090).
    init_err = float((outer.anchor["w"] - target).norm())
    for r in range(30):
        w1 = {"w": outer.anchor["w"] - 0.15 * (outer.anchor["w"] - target), "step_count": torch.tensor(r + 1)}
        w2 = {"w": outer.anchor["w"] - 0.10 * (outer.anchor["w"] - target), "step_count": torch.tensor(r + 1)}
        new_sd = outer.step_from_states([w1, w2], weights=[3.0, 1.0], bf16_compress=True)
    final_err = float((new_sd["w"] - target).norm())
    assert final_err < init_err * 0.15

    # Verify bit-exact checkpoint/resume of outer optimizer state.
    saved = outer.state_dict()
    outer2 = DiLoCoOuterOptimizer(anchor)
    outer2.load_state_dict(saved)
    w_next = [{"w": outer.anchor["w"] - 0.1 * (outer.anchor["w"] - target), "step_count": torch.tensor(9)}]
    s1 = outer.step_from_states(w_next)
    s2 = outer2.step_from_states(w_next)
    assert torch.equal(s1["w"], s2["w"])


def test_greedy_model_soup_selects_only_improving_models():
    target = torch.tensor([1.0, 1.0])
    cands = [
        ("a", {"w": torch.tensor([1.4, 0.8])}, 0.45),  # dist ~0.447
        ("b", {"w": torch.tensor([0.6, 1.2])}, 0.45),  # avg(a,b) = [1.0, 1.0] -> dist 0.0!
        ("bad", {"w": torch.tensor([9.0, 9.0])}, 11.3),
    ]
    avg, best, picked = greedy_model_soup(cands, lambda sd: float((sd["w"] - target).norm()))
    assert picked == ["a", "b"]
    assert best < 1e-6
    assert torch.allclose(avg["w"], target)


def test_build_adamw_splits_decay_and_no_decay():
    m = ASRModel(PRESETS["TINY"])
    opt = build_adamw(m, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.98), device=torch.device("cpu"))
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-2
    assert opt.param_groups[1]["weight_decay"] == 0.0
